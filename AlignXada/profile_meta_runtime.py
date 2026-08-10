import argparse
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency at runtime
    OpenAI = None

from profile_meta_core import (
    EvaluationResult,
    FeedbackExperience,
    PredictionResult,
    SupportExample,
    average,
    compression_stats,
    normalize_label,
    parse_json_object,
    pretty_text,
    profile_length,
)
from profile_meta_prompts import build_rewrite_prompt
from profile_meta_tasks import TaskAdapter


BACKEND_OPENAI = "openai"
BACKEND_VLLM = "vllm"
BACKEND_CHOICES = [BACKEND_OPENAI, BACKEND_VLLM]


@dataclass
class BackendSpec:
    """Configuration for one model role.

    The same abstraction is used for the meta teacher, rewrite model, and
    downstream target model so the three roles can be mixed freely across
    OpenAI-compatible APIs and local vLLM models.
    """

    backend: str
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    tensor_parallel_size: int = 2
    max_model_len: int = None
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = False
    dtype: Optional[str] = None
    use_chat_completions: bool = False
    enable_thinking: bool = False


@dataclass
class GenerationConfig:
    """Decoding knobs shared by both API-backed and local generation."""

    temperature: float
    max_output_tokens: int
    top_p: float = 1.0
    top_k: int = -1


class ModelCaller:
    """Minimal duck-typed interface expected by TaskAdapter.predict."""

    def generate_text(
        self,
        model: str,
        instructions: str,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 800,
        top_p: float = 1.0,
        top_k: int = -1,
    ) -> str:
        raise NotImplementedError

    def generate_texts(
        self,
        model: str,
        instructions: str,
        prompts: List[str],
        temperature: float = 0.0,
        max_output_tokens: int = 800,
        top_p: float = 1.0,
        top_k: int = -1,
    ) -> List[str]:
        """Batch generation. Default: sequential fallback."""
        return [
            self.generate_text(
                model=model,
                instructions=instructions,
                prompt=p,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                top_k=top_k,
            )
            for p in prompts
        ]

    def generate_json(
        self,
        model: str,
        instructions: str,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 800,
        top_p: float = 1.0,
        top_k: int = -1,
    ) -> Dict[str, Any]:
        raw_text = self.generate_text(
            model=model,
            instructions=instructions,
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
        )
        # Strip thinking content before parsing
        # breakpoint()
        content = raw_text.strip().split("</think>")[-1].strip()
        return parse_json_object(content)


class OpenAICompatibleCaller(ModelCaller):
    """OpenAI-compatible caller used for API-backed model roles."""

    def __init__(self, spec: BackendSpec) -> None:
        if OpenAI is None:
            raise ImportError(
                "The 'openai' package is required for --*-backend openai."
            )
        api_key = spec.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key is required for OpenAI-compatible backends. "
                "Pass --*-api-key or set OPENAI_API_KEY."
            )
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if spec.base_url:
            kwargs["base_url"] = spec.base_url
        self.client = OpenAI(**kwargs)
        self.use_chat_completions = spec.use_chat_completions
        self.enable_thinking = spec.enable_thinking

    @staticmethod
    def _response_debug_summary(response: Any) -> str:
        try:
            if hasattr(response, "model_dump_json"):
                text = response.model_dump_json()
            elif hasattr(response, "model_dump"):
                text = str(response.model_dump())
            else:
                text = repr(response)
        except Exception:
            text = repr(response)
        return text[:2000]

    @classmethod
    def _extract_chat_completion_text(cls, response: Any) -> Optional[str]:
        choices = getattr(response, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text", part))
                    if isinstance(part, dict)
                    else str(part)
                    for part in content
                )
            if content is not None and str(content).strip():
                return str(content).strip()

        data = getattr(response, "data", None)
        if isinstance(data, dict):
            try:
                content = data["choices"][0]["message"]["content"]
                if content is not None and str(content).strip():
                    return str(content).strip()
            except (KeyError, IndexError, TypeError):
                pass

        if isinstance(response, dict):
            try:
                content = response["choices"][0]["message"]["content"]
                if content is not None and str(content).strip():
                    return str(content).strip()
            except (KeyError, IndexError, TypeError):
                pass
        return None

    def generate_text(
        self,
        model: str,
        instructions: str,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 800,
        top_p: float = 1.0,
        top_k: int = -1,
    ) -> str:
        if self.use_chat_completions:
            messages: List[Dict[str, str]] = []
            kwargs = {"stream": False}
            if self.enable_thinking:
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": True}
                }
            if instructions.strip():
                messages.append({"role": "system", "content": instructions.strip()})
            messages.append({"role": "user", "content": prompt.strip()})
            for attempt in range(5):
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        # temperature=temperature,
                        max_tokens=max_output_tokens,
                        **kwargs,
                    )
                    content = self._extract_chat_completion_text(response)
                    if content is not None:
                        return content
                    raise RuntimeError(
                        "Chat completions returned no usable text content. "
                        f"Response summary: {self._response_debug_summary(response)}"
                    )
                except Exception as e:
                    err_str = str(e).lower()
                    if (
                        "rate" in err_str
                        or "429" in err_str
                        or "quota" in err_str
                        or "503" in err_str
                        or "internalservererror" in err_str
                        or "internal server error" in err_str
                        or "auth_unavailable" in err_str
                        or "temporarily unavailable" in err_str
                    ) and attempt < 2:
                        wait = min(2 ** attempt * 10, 60)
                        logger.warning("Rate limited (attempt %d/5), waiting %ds", attempt + 1, wait)
                        time.sleep(wait)
                        continue
                    raise
            raise RuntimeError("Chat completions failed after retries.")
        for attempt in range(3):
            try:
                response = self.client.responses.create(
                    model=model,
                    instructions=instructions,
                    input=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_output_tokens,
                )
                return response.output_text.strip()
            except Exception as e:
                err_str = str(e).lower()
                if (
                    "rate" in err_str
                    or "429" in err_str
                    or "quota" in err_str
                ) and attempt < 2:
                    wait = min(2 ** attempt * 5, 60)
                    logger.warning("Rate limited (attempt %d/5), waiting %ds", attempt + 1, wait)
                    time.sleep(wait)
                    continue
                raise

    def generate_texts(
        self,
        model: str,
        instructions: str,
        prompts: List[str],
        temperature: float = 0.0,
        max_output_tokens: int = 800,
        top_p: float = 1.0,
        top_k: int = -1,
    ) -> List[str]:
        """Concurrent API calls via ThreadPoolExecutor."""
        if not prompts:
            return []
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(len(prompts), 8)) as pool:
            futures = [
                pool.submit(
                    self.generate_text,
                    model,
                    instructions,
                    p,
                    temperature,
                    max_output_tokens,
                    top_p,
                    top_k,
                )
                for p in prompts
            ]
            return [f.result() for f in futures]


class VLLMTextCaller(ModelCaller):
    """Local vLLM wrapper that mimics the OpenAI caller surface."""

    def __init__(self, spec: BackendSpec) -> None:
        try:
            from transformers import AutoTokenizer
            from vllm import LLM
        except ImportError as exc:  # pragma: no cover - runtime dependency only
            raise ImportError(
                "The 'transformers' and 'vllm' packages are required for "
                "--*-backend vllm."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.model_name,
            trust_remote_code=spec.trust_remote_code,
        )
        llm_kwargs: Dict[str, Any] = {
            "model": spec.model_name,
            "tensor_parallel_size": spec.tensor_parallel_size,
            # "max_model_len": spec.max_model_len,
            "gpu_memory_utilization": spec.gpu_memory_utilization,
            "trust_remote_code": spec.trust_remote_code,
        }
        if spec.dtype:
            llm_kwargs["dtype"] = spec.dtype
        self.llm = LLM(**llm_kwargs)
        self.enable_thinking = spec.enable_thinking

    def _format_chat_prompt(self, instructions: str, prompt: str) -> str:
        messages = []
        if instructions.strip():
            messages.append({"role": "system", "content": instructions.strip()})
        messages.append({"role": "user", "content": prompt.strip()})

        if getattr(self.tokenizer, "chat_template", None):
            chat_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    enable_thinking=self.enable_thinking,
                    **chat_kwargs,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(messages, **chat_kwargs)

        return (
            "<SYSTEM>\n"
            f"{instructions.strip()}\n"
            "</SYSTEM>\n\n"
            "<USER>\n"
            f"{prompt.strip()}\n"
            "</USER>\n\n"
            "<ASSISTANT>\n"
        )

    def generate_text(
        self,
        model: str,
        instructions: str,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 800,
        top_p: float = 1.0,
        top_k: int = -1,
    ) -> str:
        from vllm import SamplingParams

        del model
        formatted_prompt = self._format_chat_prompt(
            instructions=instructions,
            prompt=prompt,
        )
        sampling_params = SamplingParams(
            temperature=temperature,
            # top_p=top_p,
            # top_k=top_k,
            max_tokens=max_output_tokens,
            n=1,
        )
        outputs = self.llm.generate([formatted_prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

    def generate_texts(
        self,
        model: str,
        instructions: str,
        prompts: List[str],
        temperature: float = 0.0,
        max_output_tokens: int = 800,
        top_p: float = 1.0,
        top_k: int = -1,
    ) -> List[str]:
        """Native vLLM batch inference."""
        from vllm import SamplingParams

        if not prompts:
            return []
        del model
        formatted = [self._format_chat_prompt(instructions, p) for p in prompts]
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_output_tokens,
            n=1,
        )
        outputs = self.llm.generate(formatted, sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]


def build_model_caller(spec: BackendSpec) -> ModelCaller:
    """Instantiate the correct generation backend for one model role."""

    if spec.backend == BACKEND_OPENAI:
        return OpenAICompatibleCaller(spec)
    if spec.backend == BACKEND_VLLM:
        return VLLMTextCaller(spec)
    raise ValueError(
        f"Unsupported backend '{spec.backend}'. Expected one of {BACKEND_CHOICES}."
    )


def build_role_spec_from_args(args: argparse.Namespace, prefix: str) -> BackendSpec:
    """Collect one role's backend configuration from argparse flags."""

    return BackendSpec(
        backend=getattr(args, f"{prefix}_backend"),
        model_name=getattr(args, f"{prefix}_model"),
        api_key=getattr(args, f"{prefix}_api_key"),
        base_url=getattr(args, f"{prefix}_base_url"),
        tensor_parallel_size=getattr(args, f"{prefix}_tensor_parallel_size"),
        max_model_len=getattr(args, f"{prefix}_max_model_len"),
        gpu_memory_utilization=getattr(args, f"{prefix}_gpu_memory_utilization"),
        trust_remote_code=getattr(args, f"{prefix}_trust_remote_code"),
        dtype=getattr(args, f"{prefix}_dtype"),
        use_chat_completions=(getattr(args, f"{prefix}_backend") == "openai"),
        enable_thinking=getattr(args, f"{prefix}_enable_thinking"),
    )


def build_generation_config_from_args(
    args: argparse.Namespace,
    prefix: str,
) -> GenerationConfig:
    """Collect one role's sampling configuration from argparse flags."""

    return GenerationConfig(
        temperature=getattr(args, f"{prefix}_temperature"),
        max_output_tokens=getattr(args, f"{prefix}_max_output_tokens"),
        top_p=getattr(args, f"{prefix}_top_p"),
        top_k=getattr(args, f"{prefix}_top_k"),
    )


def add_role_args(
    parser: argparse.ArgumentParser,
    prefix: str,
    default_model: str,
    default_backend: str,
    default_temperature: float,
    default_max_output_tokens: int,
) -> None:
    """Register CLI flags for one model role."""

    parser.add_argument(
        f"--{prefix}-backend",
        choices=BACKEND_CHOICES,
        default=default_backend,
        help=f"Backend used by the {prefix} role.",
    )
    parser.add_argument(
        f"--{prefix}-model",
        default=default_model,
        help=f"Model name or local path used by the {prefix} role.",
    )
    parser.add_argument(
        f"--{prefix}-api-key",
        default=None,
        help=f"API key for the {prefix} role when using an OpenAI-compatible API.",
    )
    parser.add_argument(
        f"--{prefix}-base-url",
        default=None,
        help=f"Optional base URL for the {prefix} role when using an OpenAI-compatible API.",
    )
    parser.add_argument(
        f"--{prefix}-tensor-parallel-size",
        type=int,
        default=1,
        help=f"Tensor parallel size for the {prefix} role when using vLLM.",
    )
    parser.add_argument(
        f"--{prefix}-max-model-len",
        type=int,
        default=None,
        help=f"Maximum model length for the {prefix} role when using vLLM.",
    )
    parser.add_argument(
        f"--{prefix}-gpu-memory-utilization",
        type=float,
        default=0.9,
        help=f"GPU memory utilization cap for the {prefix} role when using vLLM.",
    )
    parser.add_argument(
        f"--{prefix}-trust-remote-code",
        action="store_true",
        help=f"Pass trust_remote_code=True when loading the {prefix} role locally.",
    )
    parser.add_argument(
        f"--{prefix}-dtype",
        default=None,
        help=f"Optional dtype override for the {prefix} role when using vLLM.",
    )
    parser.add_argument(
        f"--{prefix}-temperature",
        type=float,
        default=default_temperature,
        help=f"Sampling temperature for the {prefix} role.",
    )
    parser.add_argument(
        f"--{prefix}-top-p",
        type=float,
        default=1.0,
        help=f"top_p for the {prefix} role.",
    )
    parser.add_argument(
        f"--{prefix}-top-k",
        type=int,
        default=-1,
        help=f"top_k for the {prefix} role.",
    )
    parser.add_argument(
        f"--{prefix}-max-output-tokens",
        type=int,
        default=default_max_output_tokens,
        help=f"Maximum output tokens for the {prefix} role.",
    )
    parser.add_argument(
        f"--{prefix}-enable-thinking",
        action="store_true",
        default=False,
        help=f"Thinking mode of the {prefix} role.",
    )


def flatten_instruction(instructions: str, prompt: str) -> str:
    """Flatten system and user content into one trainable instruction string."""

    return (
        "<SYSTEM>\n"
        f"{instructions.strip()}\n"
        "</SYSTEM>\n\n"
        "<USER>\n"
        f"{prompt.strip()}\n"
        "</USER>"
    )


def build_feedback_payload(
    feedbacks: Sequence[FeedbackExperience],
) -> List[Dict[str, Any]]:
    """Materialize the exact feedback keys used by the current Phase 1 code."""

    return [
        {
            "idx": item.idx,
            "prompt": item.prompt,
            "reference": item.reference,
            # "original_profile": item.profile, # original profile ablation
            "optimized_profile": item.optimized_profile,
            "original_prediction": item.original_prediction,
            "optimized_prediction": item.optimized_prediction,
            "original_primary_score": item.original_primary_score,
            "optimized_primary_score": item.optimized_primary_score,
            "score_delta": item.optimized_primary_score - item.original_primary_score,
            "original_metrics": item.original_metrics,
            "optimized_metrics": item.optimized_metrics,
            "meta_summary": item.meta_summary,
            "original_profile_words": item.original_profile_words,
            "optimized_profile_words": item.optimized_profile_words,
            "word_reduction": item.original_profile_words
            - item.optimized_profile_words,
        }
        for item in feedbacks
    ]


def summarize_original_baseline(
    examples: Sequence[SupportExample],
) -> Dict[str, Any]:
    """Summarize original-profile baselines on one held-out set."""

    if not examples:
        return {
            "primary_metric_value": 0.0,
            "avg_profile_words": 0.0,
            "total": 0,
        }

    scores = []
    lengths = []
    for example in examples:
        if example.original_evaluation is None:
            raise ValueError("original_evaluation must be populated before summary.")
        scores.append(example.original_evaluation.primary_score)
        lengths.append(profile_length(example.profile)["words"])

    return {
        "primary_metric_value": average(scores),
        "avg_profile_words": average(lengths),
        "total": len(examples),
    }


def rewrite_profile(
    profile: Any,
    pattern: str,
    task_adapter: TaskAdapter,
    rewrite_caller: ModelCaller,
    rewrite_model: str,
    rewrite_generation_config: GenerationConfig,
) -> str:
    """Execute one rewrite step using the Phase 1 field conventions."""

    instructions, prompt = build_rewrite_prompt(
        profile=profile,
        pattern=pattern,
        task_description=task_adapter.task_description,
    )
    raw = rewrite_caller.generate_text(
        model=rewrite_model,
        instructions=instructions,
        prompt=prompt,
        temperature=rewrite_generation_config.temperature,
        max_output_tokens=rewrite_generation_config.max_output_tokens,
        top_p=rewrite_generation_config.top_p,
        top_k=rewrite_generation_config.top_k,
    )
    return raw.strip().split("</think>")[-1].strip()


def evaluate_with_profile(
    example: SupportExample,
    profile_text: str,
    task_adapter: TaskAdapter,
    target_caller: ModelCaller,
    target_model: str,
    target_generation_config: GenerationConfig,
) -> Tuple[PredictionResult, EvaluationResult]:
    """Run the downstream model and evaluate its prediction."""

    prediction = task_adapter.predict(
        text_caller=target_caller,
        model_name=target_model,
        example=example,
        profile_text=profile_text,
        generation_config=target_generation_config,
    )
    evaluation = task_adapter.evaluate(example, prediction)
    return prediction, evaluation


def populate_original_baselines(
    examples: Sequence[SupportExample],
    task_adapter: TaskAdapter,
    target_caller: ModelCaller,
    target_model: str,
    target_generation_config: GenerationConfig,
) -> None:
    """Compute the original-profile baseline once so episodes can reuse it."""

    for example in tqdm(examples, desc="Running original-profile baselines"):
        if example.original_evaluation is not None:
            continue
        prediction, evaluation = evaluate_with_profile(
            example=example,
            profile_text=pretty_text(example.profile),
            task_adapter=task_adapter,
            target_caller=target_caller,
            target_model=target_model,
            target_generation_config=target_generation_config,
        )
        example.original_prediction = prediction
        example.original_evaluation = evaluation


def collect_feedback(
    examples: Sequence[SupportExample],
    pattern: str,
    task_adapter: TaskAdapter,
    rewrite_caller: ModelCaller,
    rewrite_model: str,
    rewrite_generation_config: GenerationConfig,
    target_caller: ModelCaller,
    target_model: str,
    target_generation_config: GenerationConfig,
) -> Dict[str, Any]:
    """Reproduce the Phase 1 feedback object using the current field names."""

    for example in examples:
        if example.original_prediction is None or example.original_evaluation is None:
            raise ValueError(
                "original_prediction and original_evaluation must be populated "
                "before collecting feedback."
            )

    # --- Step 1: batch rewrite all examples at once ---
    rewrite_instructions_list = []
    rewrite_prompts = []
    for example in examples:
        instructions, prompt = build_rewrite_prompt(
            profile=example.profile,
            pattern=pattern,
            task_description=task_adapter.task_description,
        )
        rewrite_instructions_list.append(instructions)
        rewrite_prompts.append(prompt)

    # All rewrite prompts share the same instructions
    raw_profiles = rewrite_caller.generate_texts(
        model=rewrite_model,
        instructions=rewrite_instructions_list[0],
        prompts=rewrite_prompts,
        temperature=rewrite_generation_config.temperature,
        max_output_tokens=rewrite_generation_config.max_output_tokens,
        top_p=rewrite_generation_config.top_p,
        top_k=rewrite_generation_config.top_k,
    )
    optimized_profiles = [
        text.strip().split("</think>")[-1].strip() for text in raw_profiles
    ]

    # --- Step 2: batch evaluate all rewrite results ---
    target_prompts = [
        task_adapter.render_prompt(example.prompt, optimized_profile)
        for example, optimized_profile in zip(examples, optimized_profiles)
    ]
    raw_predictions = target_caller.generate_texts(
        model=target_model,
        instructions=task_adapter.prediction_instruction,
        prompts=target_prompts,
        temperature=target_generation_config.temperature,
        max_output_tokens=target_generation_config.max_output_tokens,
        top_p=target_generation_config.top_p,
        top_k=target_generation_config.top_k,
    )

    # Parse target results individually. Some generation tasks intentionally
    # return raw text instead of JSON.
    expects_json_prediction = getattr(task_adapter, "expects_json_prediction", True)
    parsed_results: List[Dict[str, Any]] = []
    for i, raw_text in enumerate(raw_predictions):
        content = raw_text.strip().split("</think>")[-1].strip()
        if not expects_json_prediction:
            parsed_results.append({"_raw_text": content})
            continue
        try:
            parsed_results.append(parse_json_object(content))
        except ValueError:
            logger.warning(
                f"Target prediction {i} JSON parse failed, using raw fallback"
            )
            parsed_results.append({"_raw_text": content})

    # Assemble feedback
    feedbacks: List[FeedbackExperience] = []
    primary_score_total = 0.0
    num_errors = 0

    for example, optimized_profile, result in zip(
        examples, optimized_profiles, parsed_results
    ):
        original_prediction = example.original_prediction
        original_evaluation = example.original_evaluation

        prediction_label = task_adapter.parse_prediction(result)
        reason = result.get("reason")
        if reason is not None:
            reason = str(reason).strip() or None
        optimized_prediction = PredictionResult(
            output=prediction_label,
            reason=reason,
            prompt=task_adapter.render_prompt(example.prompt, optimized_profile),
        )
        optimized_evaluation = task_adapter.evaluate(example, optimized_prediction)

        primary_score_total += optimized_evaluation.primary_score
        if optimized_evaluation.passed is False:
            num_errors += 1

        length_metrics = compression_stats(example.profile, optimized_profile)
        feedbacks.append(
            FeedbackExperience(
                idx=example.idx,
                profile=example.profile,
                optimized_profile=optimized_profile,
                prompt=task_adapter.redact_prompt(example.prompt),
                reference=example.reference,
                original_prediction=original_prediction.output,
                optimized_prediction=optimized_prediction.output,
                original_reason=original_prediction.reason,
                optimized_reason=optimized_prediction.reason,
                original_primary_score=original_evaluation.primary_score,
                optimized_primary_score=optimized_evaluation.primary_score,
                original_passed=original_evaluation.passed,
                optimized_passed=optimized_evaluation.passed,
                original_metrics=original_evaluation.metrics,
                optimized_metrics=optimized_evaluation.metrics,
                meta_summary=task_adapter.build_meta_summary(
                    example=example,
                    original_prediction=original_prediction,
                    original_evaluation=original_evaluation,
                    optimized_prediction=optimized_prediction,
                    optimized_evaluation=optimized_evaluation,
                ),
                original_profile_words=length_metrics["original_profile_words"],
                optimized_profile_words=length_metrics["optimized_profile_words"],
            )
        )

    primary_metric_value = primary_score_total / len(examples) if examples else 0.0
    avg_original_profile_words = average(
        [item.original_profile_words for item in feedbacks]
    )
    avg_optimized_profile_words = average(
        [item.optimized_profile_words for item in feedbacks]
    )
    avg_word_reduction = avg_original_profile_words - avg_optimized_profile_words

    return {
        "feedbacks": feedbacks,
        "primary_metric_name": task_adapter.primary_metric_name,
        "primary_metric_value": primary_metric_value,
        "metrics": {task_adapter.primary_metric_name: primary_metric_value},
        "total": len(examples),
        "num_errors": num_errors,
        "avg_original_profile_words": avg_original_profile_words,
        "avg_optimized_profile_words": avg_optimized_profile_words,
        "avg_word_reduction": avg_word_reduction,
    }


class ProfileExecutionEngine:
    """Shared rewrite/evaluate/feedback engine used across offline and online flows."""

    def __init__(
        self,
        rewrite_caller: ModelCaller,
        rewrite_model: str,
        rewrite_generation_config: GenerationConfig,
        target_caller: ModelCaller,
        target_model: str,
        target_generation_config: GenerationConfig,
        task_adapter: Optional[TaskAdapter] = None,
    ) -> None:
        self.task_adapter = task_adapter
        self.rewrite_caller = rewrite_caller
        self.rewrite_model = rewrite_model
        self.rewrite_generation_config = rewrite_generation_config
        self.target_caller = target_caller
        self.target_model = target_model
        self.target_generation_config = target_generation_config
        self._task_adapters: Dict[str, TaskAdapter] = {}
        if task_adapter is not None:
            self._task_adapters[task_adapter.name] = task_adapter

    def _get_adapter(self, task: Optional[str] = None) -> TaskAdapter:
        """Get TaskAdapter by name. None returns default self.task_adapter."""
        if task is None:
            if self.task_adapter is None:
                raise ValueError(
                    "No default task_adapter set. Pass task= explicitly."
                )
            return self.task_adapter
        if task not in self._task_adapters:
            from profile_meta_tasks import build_task_adapter

            self._task_adapters[task] = build_task_adapter(task)
        return self._task_adapters[task]

    @classmethod
    def from_config(
        cls,
        config_path: str = "rl/reward_backend.yaml",
        stage: str = "train",
    ) -> "ProfileExecutionEngine":
        """Create Engine from YAML config.

        Args:
            config_path: Path to reward_backend.yaml.
            stage: "data_builder" or "train" — selects which backend/URL section to use.
        """
        import yaml

        from pathlib import Path

        config = yaml.safe_load(Path(config_path).read_text())
        stage_config = config.get(stage, {})

        rewrite_spec = BackendSpec(
            backend=stage_config.get("rewrite_backend", "openai"),
            model_name=config["rewrite_model"],
            base_url=stage_config.get("rewrite_base_url") or None,
            use_chat_completions=bool(stage_config.get("rewrite_base_url")),
            enable_thinking=stage_config.get("rewrite_enable_thinking", True),
        )
        target_spec = BackendSpec(
            backend=stage_config.get("target_backend", "openai"),
            model_name=config["target_model"],
            base_url=stage_config.get("target_base_url") or None,
            use_chat_completions=bool(stage_config.get("target_base_url")),
            enable_thinking=stage_config.get("target_enable_thinking", True),
        )
        return cls(
            rewrite_caller=build_model_caller(rewrite_spec),
            rewrite_model=config["rewrite_model"],
            rewrite_generation_config=GenerationConfig(
                temperature=config.get("rewrite_temperature", 0.0),
                max_output_tokens=config.get("rewrite_max_output_tokens", 4096),
            ),
            target_caller=build_model_caller(target_spec),
            target_model=config["target_model"],
            target_generation_config=GenerationConfig(
                temperature=config.get("target_temperature", 0.0),
                max_output_tokens=config.get("target_max_output_tokens", 512),
            ),
        )

    def evaluate_pattern_batch(
        self,
        pattern: str,
        query_example_dicts: List[Dict[str, Any]],
        task: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Batch evaluate a pattern on serialized query examples.

        Used by RL reward computation. task param routes to the correct
        TaskAdapter. Rewrite is batched; evaluation remains per-example.
        """
        adapter = self._get_adapter(task)
        profiles = [pretty_text(d["profile"]) for d in query_example_dicts]

        # Build rewrite prompts (all share the same instructions)
        rewrite_prompts = []
        rewrite_instructions = ""
        for prof in profiles:
            instructions, prompt = build_rewrite_prompt(
                profile=prof,
                pattern=pattern,
                task_description=adapter.task_description,
            )
            rewrite_instructions = instructions
            rewrite_prompts.append(prompt)

        # Batch rewrite
        raw_profiles = self.rewrite_caller.generate_texts(
            model=self.rewrite_model,
            instructions=rewrite_instructions,
            prompts=rewrite_prompts,
            temperature=self.rewrite_generation_config.temperature,
            max_output_tokens=self.rewrite_generation_config.max_output_tokens,
            top_p=self.rewrite_generation_config.top_p,
            top_k=self.rewrite_generation_config.top_k,
        )
        optimized_profiles = [
            text.strip().split("</think>")[-1].strip() for text in raw_profiles
        ]

        # Batch target prediction
        target_prompts = [
            adapter.render_prompt(d["prompt"], optimized_profiles[i])
            for i, d in enumerate(query_example_dicts)
        ]
        raw_predictions = self.target_caller.generate_texts(
            model=self.target_model,
            instructions=adapter.prediction_instruction,
            prompts=target_prompts,
            temperature=self.target_generation_config.temperature,
            max_output_tokens=self.target_generation_config.max_output_tokens,
            top_p=self.target_generation_config.top_p,
            top_k=self.target_generation_config.top_k,
        )

        scores: List[float] = []
        opt_word_counts: List[int] = []
        for i, d in enumerate(query_example_dicts):
            content = raw_predictions[i].strip().split("</think>")[-1].strip()
            if getattr(adapter, "expects_json_prediction", True):
                try:
                    result = parse_json_object(content)
                except ValueError:
                    result = {"_raw_text": content}
            else:
                result = {"_raw_text": content}
            prediction_label = adapter.parse_prediction(result)
            example = SupportExample(
                profile=d["profile"],
                prompt=d["prompt"],
                reference=d["reference"],
                idx=d.get("idx", i),
            )
            prediction = PredictionResult(
                output=prediction_label, reason=None, prompt=target_prompts[i],
            )
            evaluation = adapter.evaluate(example, prediction)
            scores.append(evaluation.primary_score)
            opt_word_counts.append(profile_length(optimized_profiles[i])["words"])

        return {
            "primary_metric_value": average(scores),
            "avg_profile_words": average(opt_word_counts),
            "total": len(query_example_dicts),
        }

    def rewrite_profile(self, profile: Any, pattern: str) -> str:
        return rewrite_profile(
            profile=profile,
            pattern=pattern,
            task_adapter=self.task_adapter,
            rewrite_caller=self.rewrite_caller,
            rewrite_model=self.rewrite_model,
            rewrite_generation_config=self.rewrite_generation_config,
        )

    def evaluate_with_profile(
        self,
        example: SupportExample,
        profile_text: str,
    ) -> Tuple[PredictionResult, EvaluationResult]:
        return evaluate_with_profile(
            example=example,
            profile_text=profile_text,
            task_adapter=self.task_adapter,
            target_caller=self.target_caller,
            target_model=self.target_model,
            target_generation_config=self.target_generation_config,
        )

    def evaluate_original_example(
        self,
        example: SupportExample,
    ) -> Tuple[PredictionResult, EvaluationResult]:
        return self.evaluate_with_profile(example, pretty_text(example.profile))

    def populate_original_baselines(self, examples: Sequence[SupportExample]) -> None:
        populate_original_baselines(
            examples=examples,
            task_adapter=self.task_adapter,
            target_caller=self.target_caller,
            target_model=self.target_model,
            target_generation_config=self.target_generation_config,
        )

    def collect_feedback(
        self,
        examples: Sequence[SupportExample],
        pattern: str,
    ) -> Dict[str, Any]:
        return collect_feedback(
            examples=examples,
            pattern=pattern,
            task_adapter=self.task_adapter,
            rewrite_caller=self.rewrite_caller,
            rewrite_model=self.rewrite_model,
            rewrite_generation_config=self.rewrite_generation_config,
            target_caller=self.target_caller,
            target_model=self.target_model,
            target_generation_config=self.target_generation_config,
        )

    def summarize_original_baseline(
        self,
        examples: Sequence[SupportExample],
    ) -> Dict[str, Any]:
        return summarize_original_baseline(examples)
