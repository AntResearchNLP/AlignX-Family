"""
Qwen2.5-7B base model + ppo

debug running command in single node:

DEBUG_MODE=True python -m playground.orz_7b_grpo

"""
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional
import copy
import json
import re
import numpy as np
import ray
import torch
from typing_extensions import override
from torch import nn

from collections import defaultdict
from functools import cached_property, partial
from itertools import islice, zip_longest
from typing import Any, Awaitable, Callable, List, Optional, Tuple

from orz.exps.examples.ppo.ppo_base_exp import BasePPOExp, BasePPOExpConfig
from orz.ppo.tools.math_utils import is_equal, solution2answer, compute_bleu
from orz.ppo.utils import check_reflection_pattern

from loguru import logger
from omegaconf.listconfig import ListConfig
from orz.ppo import RayPPOTrainer

from orz.exps.examples.ppo.ppo_base_exp import BasePPOExpConfig
from playground.orz_7b_ppo import PPOExp

from playground.zero_setting_base import CustomDataset, EvalCustomDataset, GenCustomDataset

from orz.ppo.utils import (
    Timer,
    compute_approx_kl,
    compute_reward,
    get_advantages_and_returns,
    masked_mean,
    normalize_advantages,
    compute_reward_no_kl,
)
from orz.ppo.replay_buffer import Experience, NaiveReplayBuffer


DEBUG_MODE = False if os.environ.get("DEBUG_MODE", "False") == "False" else True  # Global debug flag

file_name = f"{'debug_' if DEBUG_MODE else ''}{os.path.splitext(os.path.basename(__file__))[0]}"

executor = ThreadPoolExecutor(max_workers=64)
IGNORE_INDEX = -100


def repeatness(s: str):
    def ranks(l):
        index = {v: i for i, v in enumerate(sorted(set(l)))}
        return [index[v] for v in l]

    def suffixArray(s):
        line = ranks(s)
        n, k, ans, sa = len(s), 1, line, [0] * len(s)
        while k < n - 1:
            line = ranks(list(zip_longest(line, islice(line, k, None), fillvalue=-1)))
            ans, k = line, k << 1
        for i, k in enumerate(ans):
            sa[k] = i
        return ans, sa

    def lcp(arr, suffixArr, inv_suff):
        n, ans, k = len(arr), [0] * len(arr), 0

        for i in range(n):
            if inv_suff[i] == n - 1:
                k = 0
                continue

            j = suffixArr[inv_suff[i] + 1]
            while i + k < n and j + k < n and arr[i + k] == arr[j + k]:
                k += 1

            ans[inv_suff[i]] = k
            if k > 0:
                k -= 1

        return ans

    arr = [ord(i) for i in s]
    n = len(arr)
    if n <= 1:
        return 0
    c, sa = suffixArray(arr)
    cnt = sum(lcp(arr, sa, c))

    return cnt * 2 / (n * (n + 1))


@dataclass
class PPOExpConfig(BasePPOExpConfig):
    use_compute_reward_fn: bool = True
    use_orm_score: bool = False

    # Conditional settings with production values first
    # total_num_nodes: int = 32 if not DEBUG_MODE else 8
    total_num_nodes: int = 16 if not DEBUG_MODE else 8

    # resource related settings
    ref_num_nodes: int = total_num_nodes
    ref_num_gpus_per_node: int = 1
    actor_num_nodes: int = total_num_nodes
    actor_num_gpus_per_node: int = 1
    critic_num_nodes: int = total_num_nodes
    critic_num_gpus_per_node: int = 1
    reward_num_nodes: int = total_num_nodes
    reward_num_gpus_per_node: int = 1
    colocate_all: bool = True
    colocate_critic_reward: bool = True
    colocate_actor_ref: bool = True
    vllm_num_engines: int = total_num_nodes
    vllm_tensor_parallel_size: int = 1
    adam_offload: bool = False
    zero_stage: int = 3

    # path related settings
    pretrain: Optional[str] = "your pretrain model"
    reward_pretrain: Optional[str] = "your reward model"
    save_interval: int = 50
    ckpt_path: str = f"your ckpt path"
    save_path: str = f"your save path"
    tensorboard_log_dir: str = f"your tensorboard log dir"

    # MathTrain dataset and Math500 eval dataset
    # data related settings
    prompt_data: ListConfig = ListConfig(
        [
            "rl_train.json",
        ]
    )
    eval_prompt_data: ListConfig = ListConfig(
        [
            "eval data",
        ]
    )
    prompt_data_probs: ListConfig = ListConfig([1.0])

    # ppo related settings
    actor_learning_rate: float = 1e-6
    critic_learning_rate: float = 5e-6
    num_warmup_steps: int = 50
    prompt_max_len: int = 8192
    enable_prefix_caching: bool = True
    update_ref_every_epoch: bool = True
    advantage_normalize: bool = False

    num_episodes: int = 4
    rollout_batch_size: int = 128 if not DEBUG_MODE else 16
    n_samples_per_prompt: int = 4 if not DEBUG_MODE else 2
    micro_rollout_batch_size: int = 128 # 128

    policy_update_steps: int = 1
    critic_update_steps: int = 12 if not DEBUG_MODE else 1
    micro_train_batch_size: int = 1
    micro_forward_batch_size: int = 1
    freezing_actor_steps: int = -1
    init_kl_coef: float = 0
   
    kl_loss_coef: float = 0.0
    use_kl_loss: bool = False
    use_kl: bool = False
    use_kl_estimator_k3: bool = False

    enable_eval: bool = False
    eval_interval: int = 10

    # generate related settings
    packing_max_len: int = 12000
    generate_max_len: int = 2048  # TODO: change to larger later
    max_len: int = 8192  # TODO: change to larger later
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop: ListConfig = ListConfig(["User:", "Human:", "Assistant:", "</answer>", "<|im_end|>"])

    # grpo related settings
    use_grpo: bool = True

    gpu_memory_utilization: float = 0.4 if not DEBUG_MODE else 0.4
    critic_pretrain: Optional[str] = "" if use_grpo else pretrain

    gamma: float = 1.0
    lambd: float = 1.0

    tao_margin: float = 0.03



class RewardModelTrainer(RayPPOTrainer):
    @override
    @torch.no_grad()
    async def custom_reward_fn(
        self,
        prompts: List[str],
        outputs: List[Any],
        extras: List[dict],
        reward_model_fn: Callable[[List[int], int], Awaitable[float]],
    ) -> Tuple[List[str], List[str], List[torch.Tensor]]:
        # make log metrics
        scores = []
        responses = []
        avg_non_stop_count = 0
        pass_at_n_dict = defaultdict(list)
        num_tokens: List[int] = []
        tasks = []
        chosens = []
        rejecteds = []
        chosen_logs = []
        rejected_logs = []
        final_answers = []

        @ray.remote(num_cpus=1)
        def get_repeat_score(res):
            return repeatness(res)

        @ray.remote(num_cpus=1)
        def get_reflection_pattern_score(res):
            reflection_pattern_dict = check_reflection_pattern(res)
            reflection_pattern_num = sum(reflection_pattern_dict.values())
            return reflection_pattern_num

        rep_tasks = []
        for output in outputs:
            response = output["response"]
            # calculate repeat score for log
            rep_tasks.extend([get_repeat_score.remote(response), get_reflection_pattern_score.remote(response)])
        rep_task_results = ray.get(rep_tasks)

        repeat_scores = []
        reflection_pattern_scores = []
        for idx in range(len(outputs)):
            repeat_scores.append(rep_task_results[idx * 2])
            reflection_pattern_scores.append(rep_task_results[idx * 2 + 1])

        for output in outputs:
            responses.append(output["response"])
            # final_answers.append("User Preferences: " + output["final_answer"])
            final_answers.append(output["final_answer"])
        output_tokens = self._tokenize(responses, self.cfg.generate_max_len, padding=False)["input_ids"]

        for extra in extras:
            # tasks.append("User: " + extra["task"] + "\n\n" + "Assistant: ")
            tasks.append(extra["task"])
            chosens.append(extra["chosen"])
            rejecteds.append(extra["rejected"])
            chosen_logs.append(extra["chosen_log"])
            rejected_logs.append(extra["rejected_log"])

        assert len(tasks) == len(chosens) == len(rejecteds) == len(final_answers)


        ###
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.reward_pretrain, padding_side="left")
        preference_prompt = []
        for final_answer, task in zip(final_answers, tasks):
            reward_prompt = (
                f"***User Preferences***\n\n{final_answer}"
                f"***Task***\n\n{task}"
            )

            messages = [
                {"role": "system", "content": "Generate a task-specific response based on user preferences."},
                {"role": "user", "content": reward_prompt}
            ]

            reward_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            preference_prompt.append(reward_prompt)
        ###

        self.writer.add_text(
            "generated_raws",
            f"prompts: {prompts[0]}\n\noutputs: {outputs[0]['response']}\n\nfinal_answer: {outputs[0]['final_answer']}\n\nis_correct: {outputs[0]['iscorrect']}\n\nstop_reason: {outputs[0]['stop_reason']}\n\nresponse_token: {len(output_tokens[0])}",
            self.global_step,
        )

        prompt_all = preference_prompt + preference_prompt
        responses_all = chosens + rejecteds

        (
            sequences,
            attention_mask,
            num_actions,
            packed_seq_lens,
            _,
        ) = self._convert_prompts_outputs_to_batch_tensors_packing(
            prompt_all, responses_all, None, self.cfg.packing_max_len
        )

        log_all = await reward_model_fn(sequences, attention_mask, num_actions, packed_seq_lens)

        preference_chosen_log, preference_rejected_log = log_all[:len(log_all)//2], log_all[len(log_all)//2:]

        for idx, (n_c, n_r, p_c, p_r) in enumerate(zip(chosen_logs, rejected_logs, preference_chosen_log, preference_rejected_log)):
            if outputs[idx]["stop_reason"] != "stop" or outputs[idx]["final_answer"] == "":
                avg_non_stop_count += 1
                scores.append(0.0)
            else:
                if (p_c/p_r)/(n_c/n_r) > 1 + self.cfg.tao_margin:
                    scores.append(1.0)
                else:
                    scores.append(0.0)
            
        for idx in range(len(outputs)):
            prompt, output, out_token = prompts[idx], outputs[idx], output_tokens[idx]

            rep_score, reflection_pattern_score = repeat_scores[idx], reflection_pattern_scores[idx]
            iscorrect = output["iscorrect"]
            stop_reason = output["stop_reason"]
            response_token = len(out_token)
            output["repeat_score"] = rep_score
            output["reflection_pattern_score"] = reflection_pattern_score

            # calculate pass@n
            pass_at_n_dict[prompt].append(scores[idx])
            # log num_tokens
            num_tokens.append(response_token)

        # must before grpo, for grpo will change scores
        num_tokens_arr = np.array(num_tokens, dtype=np.float32)  # must be float to calculate mean and std
        scores_arr = np.array(scores)
        # correct_tokens_arr = np.array([]) if np.all(scores_arr == 0) else np.array(num_tokens_arr[scores_arr == 1])
        # incorrect_tokens_arr = np.array([]) if np.all(scores_arr == 1) else np.array(num_tokens_arr[scores_arr == 0])

        # GRPO
        if self.cfg.use_grpo:
            self.writer.add_scalar("grpo_raw_reward", np.mean(scores), self.global_step)
            # grpo reward normalization
            for i, prompt in enumerate(prompts):
                scores[i] -= np.mean(pass_at_n_dict[prompt])
                if std := np.std(pass_at_n_dict[prompt]) > 0:
                    scores[i] /= std


        def dump_results(prompts, outputs, scores):
            saved = []
            for prompt, output, score in zip(prompts, outputs, scores):
                saved.append(dict(prompt=prompt, score=score, outputs=output))
            json.dump(
                saved,
                open(os.path.join(self.cfg.save_path, f"iter{self.global_step}_generation_results.json"), "w"),
                ensure_ascii=False,
                indent=2,
            )

        global executor
        asyncio.get_event_loop().run_in_executor(
            executor, dump_results, copy.deepcopy(prompts), copy.deepcopy(outputs), copy.deepcopy(scores)
        )

        log_dict = {
            "avg_non_stop_count": avg_non_stop_count / len(prompts),
            "avg_repeat_score": sum(repeat_scores) / len(prompts),
            "avg_reflection_pattern_score": sum(reflection_pattern_scores) / len(prompts),
            "avg_pass_at_n": sum(1 for v in pass_at_n_dict.values() if np.sum(v) > 0) / len(pass_at_n_dict),
            "avg_num_tokens": np.mean(num_tokens_arr).item(),
            "std_num_tokens": np.std(num_tokens_arr).item(),
            # "avg_correct_num_tokens": 0 if len(correct_tokens_arr) == 0 else np.mean(correct_tokens_arr).item(),
            # "std_correct_num_tokens": 0 if len(correct_tokens_arr) == 0 else np.std(correct_tokens_arr).item(),
            # "avg_incorrect_num_tokens": 0 if len(incorrect_tokens_arr) == 0 else np.mean(incorrect_tokens_arr).item(),
            # "std_incorrect_num_tokens": 0 if len(incorrect_tokens_arr) == 0 else np.std(incorrect_tokens_arr).item(),
        }
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, self.global_step)
        logging_str = ",".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])
        logger.info(logging_str)

        # # make histogram for correct and incorrect response length
        # if len(correct_tokens_arr) > 0:
        #     self.writer.add_histogram("correct_response_length", correct_tokens_arr, self.global_step)
        # if len(incorrect_tokens_arr) > 0:
        #     self.writer.add_histogram("incorrect_response_length", incorrect_tokens_arr, self.global_step)

        # make a pre-token score tensor for each output, for example: [0, 0, 0, 0, r]
        score_tensors = []
        for score, output_token in zip(scores, output_tokens):
            score_tensor = torch.zeros(len(output_token))
            if len(output_token) > 0:
                score_tensor[-1] = score
            score_tensors.append(score_tensor)

        # rm empty response
        res_prompts = []
        res_responses = []
        res_score_tensors = []
        for prompt, response, score_tensor in zip(prompts, responses, score_tensors):
            if len(response) > 0:
                res_prompts.append(prompt)
                res_responses.append(response)
                res_score_tensors.append(score_tensor)

        return res_prompts, res_responses, res_score_tensors

    @override
    @torch.no_grad()
    def _custom_reward_model_fn(self):
        if self.reward_model:
            num_reward_dp_groups = self.cfg.reward_num_nodes * self.cfg.reward_num_gpus_per_node
            async def custom_reward_model_fn(sequences, attention_mask, num_actions, packed_seq_lens):
                async def micro_infer_model(num_dps, model_type, sequences, num_actions, attention_mask, packed_seq_lens):
                    dp_iterator = self._split_dp_batch(
                        (sequences, num_actions, attention_mask, packed_seq_lens),
                        num_dps,
                    )
                    dp_tasks = []
                    for dp_rank, (
                        micro_sequences,
                        micro_num_actions,
                        micro_attention_mask,
                        micro_packed_seq_lens,
                    ) in enumerate(dp_iterator):
                        model = self._get_dp_group_models(dp_rank, model_type)

                        async def forward_fn(
                            local_model, fwd_sequences, fwd_num_actions, fwd_attention_mask, fwd_packed_seq_lens
                        ):
                            return await local_model.forward.remote(
                                sequences=fwd_sequences,
                                num_actions=fwd_num_actions,
                                attention_mask=fwd_attention_mask,
                                packed_seq_lens=fwd_packed_seq_lens,
                            )

                        dp_tasks.append(
                            self._split_and_run_micro_batch(
                                partial(forward_fn, model),
                                (micro_sequences, micro_num_actions, micro_attention_mask, micro_packed_seq_lens),
                                self.cfg.micro_forward_batch_size,
                            )
                        )
                    results = await asyncio.gather(*dp_tasks)
                    results = sum(results, [])
                    return results
                
                reward_refs = micro_infer_model(
                    num_reward_dp_groups, "reward_model", sequences, num_actions, attention_mask, packed_seq_lens
                )

                rewards_logs = await reward_refs
                empty_cache_tasks = [rm.async_run_method("empty_cache") for rm in self.reward_model]
                await asyncio.gather(*empty_cache_tasks)

                rewards_logs = rewards_logs[: len(sequences)]

                log_all = []

                for i in range(len(rewards_logs)):
                    offset = 0
                    num_action = num_actions[i]
                    for num_action_i in num_action:
                        cur_log = rewards_logs[i][0, offset:offset + num_action_i]
                        cur_log_sum = cur_log.sum().item()
                        log_all.append(cur_log_sum)
                        offset += num_action_i

                return log_all
            
            return custom_reward_model_fn
        else:
            return None
        

    @override
    @torch.no_grad()
    async def generate_vllm(
        self,
        gen_func: Callable[[List[str]], Awaitable[List[str | Any]]],
        prompts: List[str],
        extras: List[dict],
        **kwargs,
    ) -> List[str | Any]:
        from vllm import SamplingParams

        # read sampling params from self.cfg

        sampling_params = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            max_tokens=self.cfg.generate_max_len,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
            stop=self.cfg.stop,
        )
        responses, stop_reasons = await gen_func(
            prompts=prompts, sampling_params=sampling_params, use_tqdm=False, truncate_prompt=True
        )

        @ray.remote(num_cpus=1)
        def extract_final_answers_batch(responses: List[str]) -> List[str]:
            # pattern = re.compile(r"(\\boxed{.*})")
            # pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
            # results = []
            # for response in responses:
            #     matches = re.findall(pattern, response)
            #     results.append(matches[-1] if matches else "")

            pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
            results = []
            for response in responses:
                matches = re.findall(pattern, response)
                if matches:
                    result = matches[-1].replace("answer here", "")
                else:
                    result = ""
                results.append(result)

            return results

        BATCH_SIZE = 16
        num_batches = (len(responses) + BATCH_SIZE - 1) // BATCH_SIZE

        extract_tasks = []
        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(responses))
            batch = responses[start_idx:end_idx]
            extract_tasks.append(extract_final_answers_batch.remote(batch))
        batched_results = await asyncio.gather(*[asyncio.to_thread(ray.get, task) for task in extract_tasks])
        final_answers = [answer for batch in batched_results for answer in batch]

        global executor
        equal_tasks = []
        for extra, final_answer in zip(extras, final_answers):
            equal_tasks.append(compute_bleu(solution2answer(extra["answer"]), solution2answer(final_answer), executor))
        equal_results = await asyncio.gather(*equal_tasks)

        results = []
        for extra, response, final_answer, stop_reason, iscorrect in zip(
            extras, responses, final_answers, stop_reasons, equal_results
        ):
            results.append(
                dict(
                    extra=extra,
                    response=response,
                    iscorrect=iscorrect,
                    stop_reason=stop_reason,
                    final_answer=final_answer,
                )
            )

        return results
    
    @override
    @torch.no_grad()
    async def _calc_advantages_and_returns(self, experience: Experience):
        num_actions = experience.info["num_actions"]
        reward = await compute_reward_no_kl.remote(
            experience.info["reward"],
            self.cfg.init_kl_coef,
            experience.kl,
            custom_rewards=experience.info["custom_rewards"],
            action_mask=experience.action_mask,
            num_actions=num_actions,
            reward_clip_range=self.cfg.reward_clip_range,
            use_kl_loss=self.cfg.use_kl_loss,
            use_kl=self.cfg.use_kl,
        )

        experience.advantages, experience.returns = await get_advantages_and_returns.remote(
            experience.values,
            reward,
            experience.action_mask,
            num_actions,
            self.cfg.gamma,
            self.cfg.lambd,
            packing=True,
        )
        
        return_sums = reward.sum(dim=-1)
        return_sums /= len(num_actions)
        experience.info["return"] = return_sums
        experience.kl = None

        avg_rewards = return_sums.mean().item()
        avg_kl = experience.info["kl"].mean().item()
        avg_kl_max = experience.info["kl_max"].mean().item()

        avg_response_length = experience.info["response_length"].mean().item()
        if experience.info["reward"] is not None:
            avg_orm_score = experience.info["reward"].mean().item()
        else:
            avg_orm_score = 0

        if experience.info["custom_rewards"] is not None:

            def func(x):
                return [r.sum() for r in x]

            avg_custom_rewards = torch.stack(func(experience.info["custom_rewards"])).mean().item()
        else:
            avg_custom_rewards = 0

        del experience.info["num_actions"]
        del experience.info["custom_rewards"]
        del experience.info["reward"]
        del experience.info["kl_max"]
        experience.to_device("cpu")

        num_packed_samples = len(num_actions)
        return_sums /= num_packed_samples
        experience.info["response_length"] = torch.Tensor(experience.info["response_length"]).mean().unsqueeze(0)
        experience.info["total_length"] = torch.Tensor(experience.info["total_length"]).mean().unsqueeze(0)

        metrics = {
            "avg_rewards": avg_rewards,
            "avg_kl": avg_kl,
            "avg_kl_max": avg_kl_max,
            "avg_response_length": avg_response_length,
            "avg_orm_score": avg_orm_score,
            "avg_custom_rewards": avg_custom_rewards,
            "avg_advantages": experience.advantages.mean().item(),
            "avg_advantages_abs": experience.advantages.abs().mean().item(),
        }

        return experience, metrics


class GRPOExp(PPOExp):
    @cached_property
    def trainer(self):
        vllm_engines = self.create_inference_engine()
        return RewardModelTrainer(
            cfg=self.cfg,
            strategy=self.strategy,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            vllm_engines=vllm_engines,
            colocate_pg=self.get_colocate_pg,
        )

    @override
    @cached_property
    def train_dataset(self):
        dialogues = []
        for file_path in self.cfg.prompt_data:
            with open(file_path, "r") as f:
                dialogues.extend(json.load(f))
        logger.info(f"Start processing {len(dialogues)} dialogues")
        prompts_dataset = GenCustomDataset(
            dialogues,
            self.tokenizer,
            self.cfg.prompt_max_len,
            self.strategy,
            pretrain_mode=False,
            num_processors=1,
        )
        logger.info(f"Finished processing {len(prompts_dataset)} dialogues")
        return prompts_dataset


if __name__ == "__main__":
    exp = GRPOExp().set_cfg(PPOExpConfig())
    logger.info(exp.get_cfg_as_str(exp.cfg))
    if not os.path.exists(exp.cfg.save_path):
        os.makedirs(exp.cfg.save_path, exist_ok=True)
    if not os.path.exists(exp.cfg.tensorboard_log_dir):
        os.makedirs(exp.cfg.tensorboard_log_dir, exist_ok=True)
    if not os.path.exists(exp.cfg.ckpt_path):
        os.makedirs(exp.cfg.ckpt_path, exist_ok=True)
    asyncio.run(exp.run())
