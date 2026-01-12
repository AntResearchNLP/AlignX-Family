from __future__ import annotations

import os
import uuid
from collections import defaultdict
from copy import deepcopy
from typing import Optional, Type, Iterable
from pprint import pprint

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    Role,
    ResourcePoolManager,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_response_mask,
    compute_advantage,
    _timer,
)
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.metric import reduce_metrics
import verl.utils.torch_functional as verl_F
from verl.single_controller.ray import RayWorkerGroup
from verl.utils.model import compute_position_id_with_mask
from verl.utils.dataset.upi_dataset import process_dialogue


class StreamingGRPOTrainer(RayPPOTrainer):
    """
    两阶段 GRPO 训练器： 阶段一：用 prompt1 rollout n 次，得到 raw_reward1 随机挑一条阶段一回复，灌入 prompt2 的 "Not Provided." 字段 阶段二：对新的 prompt2 再 rollout 一次，得到 reward2 优化：将阶段一与阶段二作为独立样本参与优化，但阶段一奖励按 reward1 = raw_reward1 + λ·reward2 修正
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, Type],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )
        # extra hyper-parameter: reward1 = raw_reward1 + lambda * reward2
        self.two_stage_lambda = 0.9

    # ---------- 内部工具 ----------

    def _build_prompt2_text(self, example_extra_info: dict, filled_text: str):
        """
        构造阶段二 prompt2:
        - 将 extra_info['prompt2'] 中的 "Not Provided" 替换为 filled_text
        - 返回 str 或 messages(list)；调用者会统一转文本编码
        """

        # 用第一阶段随机挑选的回复 resp1_text 替换 prompt2 中的 "Not provided"
        def extract_preference(response):
            return response.strip().split("</think>")[-1].strip()

        p2 = example_extra_info.get("prompt2", "")
        filled_text = extract_preference(filled_text).strip()
        if isinstance(p2, str):
            return p2.replace("Not Provided.", filled_text)
        if isinstance(p2, list):
            new_msgs = deepcopy(p2)
            for msg in new_msgs:
                if (
                    isinstance(msg.get("content", ""), str)
                    and "Not Provided." in msg["content"]
                ):
                    msg["content"] = msg["content"].replace(
                        "Not Provided.", filled_text
                    )
            return new_msgs
        p2_str = str(p2)
        return p2_str.replace("Not Provided.", filled_text)

    def _encode_prompts(self, prompts_list):
        """
        将 prompt 文本批量编码成 DataProto（对齐 RLHFDataset 的后处理：left_pad=True、max_length、position_ids）
        """
        # 优先用数据集的 max_prompt_length；若无，回退到 config.data.max_prompt_length 或 1024
        max_len = getattr(self.train_dataset, "max_prompt_length", None)
        if max_len is None:
            max_len = OmegaConf.select(
                self.config, "data.max_prompt_length", default=8192
            )
        self.tokenizer.padding_side = "left"
        encoded = self.tokenizer(
            prompts_list,
            return_tensors="pt",
            padding=True,
            truncation=False,  # 交给 postprocess 控制策略
            add_special_tokens=False,
        )
        # breakpoint()
        input_ids = encoded.pop("input_ids")
        attention_mask = encoded.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_len,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.config.data.get("truncation", "error"),
        )
        # breakpoint()
        position_ids = compute_position_id_with_mask(attention_mask)

        return DataProto.from_single_dict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )

    @staticmethod
    def _concat_dataproto(a: DataProto, b: DataProto) -> DataProto:
        """
        安全拼接两个 DataProto（样本维拼接）：
        - 对 non_tensor_batch 的键做对齐填充，保证键集合一致；
        - 断言 TensorDict 的 keys 完全一致；
        - 使用框架自带的 DataProto.concat 进行拼接；
        - 合并 meta_info（b 覆盖 a 的同名键）。
        """
        assert isinstance(a, DataProto) and isinstance(
            b, DataProto
        ), "inputs must be DataProto"

        # 1) 对齐 non_tensor_batch 键集合
        keys_a = set(a.non_tensor_batch.keys())
        keys_b = set(b.non_tensor_batch.keys())
        all_nt_keys = keys_a | keys_b

        def _ensure_nt_keys(dp: DataProto, keys: Iterable[str]):
            length = len(dp)
            for k in keys:
                if k not in dp.non_tensor_batch:
                    # 为缺失键补占位，长度对齐
                    if k == "selected_first_stage":
                        # 该键在 Stage-1 存在，Stage-2 不一定有，默认 False
                        dp.non_tensor_batch[k] = np.zeros(length, dtype=bool)
                    else:
                        dp.non_tensor_batch[k] = np.array([None] * length, dtype=object)

        _ensure_nt_keys(a, all_nt_keys)
        _ensure_nt_keys(b, all_nt_keys)

        # 2) 断言 TensorDict 的 keys 一致
        keys_batch_a = set(a.batch.keys()) if a.batch is not None else set()
        keys_batch_b = set(b.batch.keys()) if b.batch is not None else set()
        missing_in_b = keys_batch_a - keys_batch_b
        missing_in_a = keys_batch_b - keys_batch_a
        assert not missing_in_a and not missing_in_b, (
            f"Tensor keys mismatch: missing_in_a={missing_in_a}, missing_in_b={missing_in_b}. "
            "Ensure both DataProto.batch have identical keys before concatenation."
        )

        # 3) 使用 DataProto.concat 实现样本维拼接（参见 verl/protocol.py: DataProto.concat）
        merged = DataProto.concat([a, b])

        # 4) 合并 meta_info（后者覆盖前者）
        # merged.meta_info = {**a.meta_info, **b.meta_info}

        return merged

    def _uniform_add_scalar_on_response_tokens(
        self, data: DataProto, add_scalar: torch.Tensor
    ):
        """
        将 add_scalar (shape: [B]) 均匀分配到每个样本的 response 有效 token 上，
        在 data.batch["token_level_rewards"] 上就地相加，使其在 response 区间的总和增加 add_scalar。
        """
        response_mask = data.batch["response_mask"].to(
            device=add_scalar.device
        )  # [B, L]
        resp_len = response_mask.sum(dim=1).clamp(min=1)  # [B]
        per_token_inc = (add_scalar / resp_len).unsqueeze(-1)  # [B, 1]
        inc = per_token_inc * response_mask  # [B, L]
        data.batch["token_level_rewards"] = (
            data.batch["token_level_rewards"].to(inc.device) + inc
        )

    def _add_scalar_on_sequence_position(
        self, data: DataProto, add_scalar: torch.Tensor
    ):
        """
        将 add_scalar (shape: [B]) 仅加到每条样本响应区间的一个 token 上：
        - 若该样本当前 token_level_rewards 在响应区间内已存在非零位，则加到“最后一个非零位”；
        - 否则回退为“最后一个有效响应 token”。
        注意：
        - data.batch["response_mask"] 需为 [B, L]；
        - data.batch["token_level_rewards"] 需为 [B, L]；
        - add_scalar 为 [B]，需与 data 的样本数一致。
        """
        rewards = data.batch["token_level_rewards"]
        resp_mask = data.batch["response_mask"].to(add_scalar.device)  # [B, L]
        rewards = rewards.to(add_scalar.device)

        B, L = resp_mask.shape

        # 当前响应区间是否已有非零 reward
        # nz_mask[i, j] = True 表示第 i 条样本的第 j 个响应 token 的 reward 非 0
        nz_mask = (rewards * resp_mask) != 0

        # 标记每条样本是否存在非零位
        has_nz = nz_mask.any(dim=1)  # [B]

        # 找“最后一个非零位”的下标：先横向翻转后用 argmax 再换回正向下标
        rev = torch.flip(nz_mask, dims=[1])  # [B, L]
        last_nz_from_end = torch.argmax(rev.int(), dim=1)  # [B]，从结尾数起的偏移
        last_nz_pos = (L - 1) - last_nz_from_end  # [B]，正向下标（但对 has_nz=False 的行无意义）

        # 计算“最后一个有效响应 token”的下标（回退位点）
        resp_len = resp_mask.sum(dim=1).clamp(min=1).long()  # [B]
        last_resp_pos = resp_len - 1  # [B]

        # 选择位置：有非零位则用最后一个非零位，否则用最后一个有效响应 token
        pos = torch.where(has_nz, last_nz_pos, last_resp_pos).long()  # [B]

        # 构造仅在该位置加数的增量矩阵 inc
        inc = torch.zeros_like(rewards, device=add_scalar.device)  # [B, L]
        # scatter_add 到 dim=1 的列位置
        inc.scatter_add_(dim=1, index=pos.unsqueeze(1), src=add_scalar.unsqueeze(1))

        # 叠加到 token_level_rewards
        data.batch["token_level_rewards"] = rewards + inc

    def _annotate_stage_into_extra_info(self, batch: DataProto, stage_id: int):
        """
        将阶段标识写入每个样本的 extra_info 中，键为 'stage_id'。
        这样 RewardManager / compute_score 能稳定读取到该字段。
        """
        extra = batch.non_tensor_batch.get("reward_model", None)
        num = len(batch.batch["attention_mask"])
        if extra is None:
            extra = np.array([{} for _ in range(num)], dtype=object)
        else:
            # 确保是可修改的 dict（避免共享引用导致的副作用）
            extra = np.array(
                [dict(e) if isinstance(e, dict) else {} for e in extra.tolist()],
                dtype=object,
            )
        for i in range(num):
            extra[i]["stage_id"] = int(stage_id)
        batch.non_tensor_batch["reward_model"] = extra

    def _annotate_prompt_into_reward_info(self, batch: DataProto, prompt_text):
        # print(batch)
        # print(prompt_text)
        extra = batch.non_tensor_batch.get("reward_model", None)
        num = len(batch.batch["attention_mask"])
        if extra is None:
            extra = np.array([{} for _ in range(num)], dtype=object)
        else:
            # 确保是可修改的 dict（避免共享引用导致的副作用）
            extra = np.array(
                [dict(e) if isinstance(e, dict) else {} for e in extra.tolist()],
                dtype=object,
            )
        for i in range(num):
            extra[i]["current_prompt"] = prompt_text[i]
        batch.non_tensor_batch["reward_model"] = extra

    def fit(self):
        """
        两阶段 GRPO 训练循环（加入计时器）：
        1) 第一阶段：每条样本生成 n 个 rollout；
        2) 随机选择第一阶段的第 k 个 rollout，用其回复填充 prompt2；
        3) 第二阶段：对每条样本的新 prompt2 生成 n 个 rollout；
        4) 将第二阶段 n 个 rollout 的 reward 做均值 mean_r2，仅加到第一阶段被选中的那一行：reward += λ * mean_r2；
            对第一阶段未被选中的其余 n-1 行，reward 统一加常数 0.5；
        5) 第一阶段与第二阶段的 n 个 rollout 各自独立计算优势（num_repeat=n），然后合并为一个批次做一次更新。
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # 常数：未被选中的第一阶段样本奖励加成
        non_selected_bonus = 0.45  # =0.5 * 0.9

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        total_steps = self.total_training_steps
        progress_bar = tqdm(
            total=total_steps, initial=self.global_steps, desc="Streaming Training"
        )
        self.global_steps += 1
        last_val_metrics = None

        n = self.config.actor_rollout_ref.rollout.n
        multi_turn = self.config.actor_rollout_ref.rollout.multi_turn.enable
        norm_adv = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch_base = DataProto.from_single_dict(batch_dict)
                # breakpoint()
                B = len(batch_base.batch["attention_mask"])

                with _timer("step", timing_raw):
                    # =================================
                    # 阶段一：prompt1 rollout（n 次）
                    # =================================
                    print("****************Stage1****************")

                    # 1) 取生成输入
                    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                    non_tensor_batch_keys_to_pop = []
                    for k in [
                        "raw_prompt_ids",
                        "multi_modal_data",
                        "raw_prompt",
                        "tools_kwargs",
                        "interaction_kwargs",
                    ]:
                        if k in batch_base.non_tensor_batch:
                            non_tensor_batch_keys_to_pop.append(k)

                    prompts_1 = self.tokenizer.batch_decode(
                        batch_base.batch["input_ids"], skip_special_tokens=False
                    )
                    self._annotate_prompt_into_reward_info(batch_base, prompts_1)
                    # breakpoint()
                    # print("*"*20)
                    # print(
                    #     self.tokenizer.decode(
                    #         batch_base.batch["input_ids"][0], skip_special_tokens=False
                    #     )
                    # )
                    # breakpoint()
                    gen_batch1 = batch_base.pop(
                        batch_keys=batch_keys_to_pop,
                        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                    )
                    gen_batch1.meta_info = {
                        "eos_token_id": self.tokenizer.eos_token_id,
                        "pad_token_id": self.tokenizer.pad_token_id,
                        "recompute_log_prob": False,
                        "do_sample": True,
                    }

                    # 2) 生成
                    with _timer("gen1", timing_raw):
                        if not self.async_rollout_mode:
                            out1 = self.actor_rollout_wg.generate_sequences(gen_batch1)
                        else:
                            self.async_rollout_manager.wake_up()
                            out1 = self.async_rollout_manager.generate_sequences(
                                gen_batch1
                            )
                            self.async_rollout_manager.sleep()
                        # 合并生成端 timing
                        if "timing" in out1.meta_info:
                            timing_raw.update(out1.meta_info["timing"])
                            out1.meta_info.pop("timing", None)

                    # 3) 组装 batch1 并重复为 n 次
                    batch1 = batch_base
                    # breakpoint()
                    batch1.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch1.batch))],
                        dtype=object,
                    )
                    batch1 = batch1.repeat(repeat_times=n, interleave=True)
                    batch1 = batch1.union(out1)
                    batch1.batch["response_mask"] = compute_response_mask(batch1)
                    # logging response length
                    resp1_mask = batch1.batch["response_mask"]
                    resp1_lens = resp1_mask.sum(dim=1).to(torch.float32)
                    metrics[
                        "stage1/rollout/avg_response_length"
                    ] = resp1_lens.mean().item()

                    # 4) 为每个原始样本选择一个被选中的 rollout 行（balance 前记录标记）
                    rng = np.random.default_rng()
                    ks = rng.integers(low=0, high=n, size=B)  # 每个原始样本的选择 [0..n-1]

                    # 标记被选中行（shape: [B*n]，balance 时会随 DataProto.reorder 对齐）
                    selected_flags = np.zeros(B * n, dtype=bool)
                    for i in range(B):
                        selected_flags[i * n + ks[i]] = True
                    batch1.non_tensor_batch["selected_first_stage"] = selected_flags

                    # 可选：序列长度均衡
                    if self.config.trainer.balance_batch:
                        self._balance_batch(
                            batch1,
                            metrics=metrics,
                            logging_prefix="stage1/global_seqlen",
                        )

                    # 5) old_log_prob / ref / values
                    with _timer("old_log_prob1", timing_raw):
                        old_log_prob1 = self.actor_rollout_wg.compute_log_prob(batch1)
                        ent1 = old_log_prob1.batch["entropys"]
                        entropy_agg1 = agg_loss(
                            loss_mat=ent1,
                            loss_mask=batch1.batch["response_mask"],
                            loss_agg_mode=self.config.actor_rollout_ref.actor.loss_agg_mode,
                        )
                        metrics["stage1/actor/entropy"] = float(
                            entropy_agg1.detach().item()
                        )
                        old_log_prob1.batch.pop("entropys")
                        batch1 = batch1.union(old_log_prob1)

                    if self.use_reference_policy:
                        with _timer("ref1", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob1 = self.ref_policy_wg.compute_ref_log_prob(
                                    batch1
                                )
                            else:
                                ref_log_prob1 = (
                                    self.actor_rollout_wg.compute_ref_log_prob(batch1)
                                )
                            batch1 = batch1.union(ref_log_prob1)

                    if self.use_critic:
                        with _timer("values1", timing_raw):
                            values1 = self.critic_wg.compute_values(batch1)
                            batch1 = batch1.union(values1)

                    # 6) reward 1
                    self._annotate_stage_into_extra_info(batch1, stage_id=1)
                    # print("==============Reward===============")
                    # print(batch1.non_tensor_batch["reward_model"])
                    with _timer("reward1", timing_raw):
                        if self.use_rm:
                            rm_scores1 = self.rm_wg.compute_rm_score(batch1)
                            batch1 = batch1.union(rm_scores1)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward1 = compute_reward_async.remote(
                                batch1, self.config, self.tokenizer
                            )
                            token_level_scores1, reward_info1 = ray.get(future_reward1)
                        else:
                            token_level_scores1, reward_info1 = compute_reward(
                                batch1, self.reward_fn
                            )
                            for key in reward_info1:
                                if key != "score":
                                    this_val = np.array(reward_info1[key])
                                    metrics.update(
                                        {
                                            f"stage1/critic/rewards/{key}": np.mean(
                                                this_val
                                            )
                                        }
                                    )
                        batch1.batch["token_level_scores"] = token_level_scores1

                        if self.config.algorithm.use_kl_in_reward:
                            batch1, klm1 = apply_kl_penalty(
                                batch1,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            for k, v in klm1.items():
                                metrics[f"stage1/{k}"] = v
                        else:
                            batch1.batch["token_level_rewards"] = batch1.batch[
                                "token_level_scores"
                            ]

                    # =================================
                    # 阶段二：用被选中回复填充 prompt2，并 rollout（n 次）
                    # =================================
                    print("****************Stage2****************")

                    # 1) 找到每个原始样本“被选中的”第一阶段行，取其回复文本
                    responses_ids_1 = batch1.batch["responses"]
                    resp_texts_1 = self.tokenizer.batch_decode(
                        responses_ids_1, skip_special_tokens=True
                    )
                    # print("*" * 20)
                    # print(resp_texts_1[0])

                    idx_arr_b1 = batch1.non_tensor_batch["index"]  # [B*n]
                    selected_mask_b1 = batch1.non_tensor_batch[
                        "selected_first_stage"
                    ]  # [B*n]

                    # index -> 在 batch1 中被选中行的位置
                    index_to_selected_pos = {}
                    for row_pos, (idx_val, is_sel) in enumerate(
                        zip(idx_arr_b1.tolist(), list(selected_mask_b1))
                    ):
                        if is_sel:
                            index_to_selected_pos[int(idx_val)] = row_pos

                    filled_p2_texts = []
                    extra_info_list = batch_base.non_tensor_batch.get(
                        "extra_info", np.array([{} for _ in range(B)], dtype=object)
                    )
                    for i in range(B):
                        raw_index = int(batch_base.non_tensor_batch["index"][i])
                        pos = index_to_selected_pos[raw_index]
                        chosen_text = resp_texts_1[pos]
                        p2_text = self._build_prompt2_text(
                            extra_info_list[i], chosen_text
                        )  # 仅做纯文本替换
                        filled_p2_texts.append(str(p2_text))

                    # 2) 用 UPI 模板渲染，再编码
                    rendered_p2_texts = [
                        process_dialogue(self.tokenizer, t) for t in filled_p2_texts
                    ]

                    enc2 = self._encode_prompts(rendered_p2_texts)

                    # 3) 第二阶段生成 n 次并组装 batch2
                    base_batch2 = enc2
                    required_nt_keys = [
                        "index",
                        "data_source",
                        "reward_model",
                        "extra_info",
                        "ability",
                    ]
                    for k in required_nt_keys:
                        if k in batch_base.non_tensor_batch:
                            base_batch2.non_tensor_batch[
                                k
                            ] = batch_base.non_tensor_batch[k]
                    # add prompts to batch
                    self._annotate_prompt_into_reward_info(
                        base_batch2, rendered_p2_texts
                    )

                    gen_batch2 = base_batch2.pop(
                        batch_keys=batch_keys_to_pop,
                        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                    )
                    gen_batch2.meta_info = {
                        "eos_token_id": self.tokenizer.eos_token_id,
                        "pad_token_id": self.tokenizer.pad_token_id,
                        "recompute_log_prob": False,
                        "do_sample": True,
                    }
                    with _timer("gen2", timing_raw):
                        if not self.async_rollout_mode:
                            out2 = self.actor_rollout_wg.generate_sequences(gen_batch2)
                        else:
                            self.async_rollout_manager.wake_up()
                            out2 = self.async_rollout_manager.generate_sequences(
                                gen_batch2
                            )
                            self.async_rollout_manager.sleep()
                        if "timing" in out2.meta_info:
                            timing_raw.update(out2.meta_info["timing"])
                            out2.meta_info.pop("timing", None)

                    batch2 = base_batch2
                    batch2.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch2.batch))],
                        dtype=object,
                    )
                    batch2 = batch2.repeat(
                        repeat_times=n, interleave=True
                    )  # [B*n, ...]

                    batch2 = batch2.union(out2)
                    batch2.batch["response_mask"] = compute_response_mask(batch2)
                    # logging response length
                    resp2_mask = batch2.batch["response_mask"]
                    resp2_lens = resp2_mask.sum(dim=1).to(torch.float32)
                    metrics[
                        "stage2/rollout/avg_response_length"
                    ] = resp2_lens.mean().item()

                    if self.config.trainer.balance_batch:
                        self._balance_batch(
                            batch2,
                            metrics=metrics,
                            logging_prefix="stage2/global_seqlen",
                        )

                    with _timer("old_log_prob2", timing_raw):
                        old_log_prob2 = self.actor_rollout_wg.compute_log_prob(batch2)
                        ent2 = old_log_prob2.batch["entropys"]
                        entropy_agg2 = agg_loss(
                            loss_mat=ent2,
                            loss_mask=batch2.batch["response_mask"],
                            loss_agg_mode=self.config.actor_rollout_ref.actor.loss_agg_mode,
                        )
                        metrics["stage2/actor/entropy"] = float(
                            entropy_agg2.detach().item()
                        )
                        old_log_prob2.batch.pop("entropys")
                        batch2 = batch2.union(old_log_prob2)

                    if self.use_reference_policy:
                        with _timer("ref2", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob2 = self.ref_policy_wg.compute_ref_log_prob(
                                    batch2
                                )
                            else:
                                ref_log_prob2 = (
                                    self.actor_rollout_wg.compute_ref_log_prob(batch2)
                                )
                            batch2 = batch2.union(ref_log_prob2)

                    if self.use_critic:
                        with _timer("values2", timing_raw):
                            values2 = self.critic_wg.compute_values(batch2)
                            batch2 = batch2.union(values2)

                    self._annotate_stage_into_extra_info(batch2, stage_id=2)
                    with _timer("reward2", timing_raw):
                        if self.use_rm:
                            rm_scores2 = self.rm_wg.compute_rm_score(batch2)
                            batch2 = batch2.union(rm_scores2)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward2 = compute_reward_async.remote(
                                batch2, self.config, self.tokenizer
                            )
                            token_level_scores2, reward_info2 = ray.get(future_reward2)
                        else:
                            token_level_scores2, reward_info2 = compute_reward(
                                batch2, self.reward_fn
                            )
                            for key in reward_info2:
                                if key != "score":
                                    this_val = np.array(reward_info2[key])
                                    metrics.update(
                                        {
                                            f"stage2/critic/rewards/{key}": np.mean(
                                                this_val
                                            )
                                        }
                                    )
                        batch2.batch["token_level_scores"] = token_level_scores2

                        if self.config.algorithm.use_kl_in_reward:
                            batch2, klm2 = apply_kl_penalty(
                                batch2,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            for k, v in klm2.items():
                                metrics[f"stage2/{k}"] = v
                        else:
                            batch2.batch["token_level_rewards"] = batch2.batch[
                                "token_level_scores"
                            ]

                    # 4) 计算每个原始样本的 mean_reward2（对 index 分组求均值）
                    print("****************Nested Reward****************")
                    with _timer("couple", timing_raw):
                        resp_mask2 = batch2.batch["response_mask"]  # [B*n, L]
                        scalar_r2_rows = (
                            batch2.batch["token_level_rewards"] * resp_mask2
                        ).sum(
                            dim=1
                        )  # [B*n]
                        idx_arr_b2 = batch2.non_tensor_batch["index"]  # np.array [B*n]

                        sum_r2 = defaultdict(float)
                        cnt_r2 = defaultdict(int)
                        for r, idx in zip(scalar_r2_rows.tolist(), idx_arr_b2.tolist()):
                            sum_r2[int(idx)] += float(r)
                            cnt_r2[int(idx)] += 1
                        idx_to_mean_r2 = {
                            k: (sum_r2[k] / max(1, cnt_r2[k])) for k in sum_r2.keys()
                        }

                        # —— 仅给“一阶段被选中的那一行”加 λ·mean_r2[index]
                        idx_arr_b1 = batch1.non_tensor_batch["index"]  # [B*n]
                        selected_b1 = batch1.non_tensor_batch[
                            "selected_first_stage"
                        ]  # [B*n]
                        device = batch1.batch["token_level_rewards"].device
                        dtype = batch1.batch["token_level_rewards"].dtype

                        add_scalar_vec = torch.zeros(
                            len(idx_arr_b1), dtype=dtype, device=device
                        )
                        selected_mask = (
                            selected_b1
                            if isinstance(selected_b1, np.ndarray)
                            else np.array(selected_b1, dtype=bool)
                        )

                        for i_row, idx in enumerate(idx_arr_b1.tolist()):
                            if selected_mask[i_row]:
                                add_scalar_vec[i_row] = self.two_stage_lambda * float(
                                    idx_to_mean_r2[int(idx)]
                                )
                            else:
                                add_scalar_vec[i_row] = non_selected_bonus

                        self._add_scalar_on_sequence_position(batch1, add_scalar_vec)  # use this

                    # =================================
                    # 两阶段各自独立计算优势（num_repeat = n）
                    # =================================
                    print("****************Calculate Advantage****************")
                    with _timer("adv1", timing_raw):
                        batch1 = compute_advantage(
                            batch1,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=n,
                            norm_adv_by_std_in_grpo=norm_adv,
                            multi_turn=multi_turn,
                            config=self.config.algorithm,
                        )
                    with _timer("adv2", timing_raw):
                        batch2 = compute_advantage(
                            batch2,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=n,
                            norm_adv_by_std_in_grpo=norm_adv,
                            multi_turn=multi_turn,
                            config=self.config.algorithm,
                        )

                    # =================================
                    # 更新（合并两个 batch 做一次 step）
                    # =================================
                    print("****************Merge to Update****************")
                    merged = self._concat_dataproto(batch1, batch2)
                    merged.meta_info["global_token_num"] = torch.sum(
                        merged.batch["attention_mask"], dim=-1
                    ).tolist()

                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_out = self.critic_wg.update_critic(merged)
                        metrics.update(reduce_metrics(critic_out.meta_info["metrics"]))

                    # Update actor
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            merged.meta_info["multi_turn"] = multi_turn
                            actor_out = self.actor_rollout_wg.update_actor(merged)
                        metrics.update(reduce_metrics(actor_out.meta_info["metrics"]))

                    # =================================
                    # 验证 / 保存
                    # =================================
                    is_last_step = self.global_steps >= self.total_training_steps
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (
                            is_last_step
                            or self.global_steps % self.config.trainer.test_freq == 0
                        )
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # 训练指标
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # 数据、耗时、吞吐指标
                metrics.update(
                    compute_data_metrics(batch=merged, use_critic=self.use_critic)
                )
                metrics.update(
                    compute_timing_metrics(batch=merged, timing_raw=timing_raw)
                )
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(
                        batch=merged, timing_raw=timing_raw, n_gpus=n_gpus
                    )
                )

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
