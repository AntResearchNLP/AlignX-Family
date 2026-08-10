import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from profile_meta_core import (
    DEFAULT_TARGET_MODEL,
    EvaluationResult,
    PredictionResult,
    SupportExample,
    approximate_token_count,
    compression_stats,
    normalize_label,
    serialize_evaluation,
    serialize_prediction,
    set_seed,
    word_count,
)
from profile_meta_runtime import (
    BACKEND_OPENAI,
    GenerationConfig,
    ModelCaller,
    add_role_args,
    build_generation_config_from_args,
    build_model_caller,
    build_role_spec_from_args,
    evaluate_with_profile,
)
from profile_meta_tasks import TaskAdapter, build_task_adapter


def redacted_spec(spec: Any) -> Dict[str, Any]:
    payload = asdict(spec)
    if payload.get("api_key"):
        payload["api_key"] = "<redacted>"
    return payload


@dataclass
class EvaluationConfig:
    """Top-level evaluation configuration unrelated to backend internals."""

    task: str
    input_file: str
    output_file: Optional[str]
    seed: int

    @property
    def output_path(self) -> str:
        if self.output_file:
            return self.output_file
        input_path = Path(self.input_file)
        return str(
            input_path.with_name(
                f"{input_path.stem}_preference_eval{input_path.suffix}"
            )
        )


@dataclass
class PreparedEvaluationExample:
    """One rewritten example plus the metadata needed for comparison."""

    source_idx: Optional[int]
    example: SupportExample
    optimized_profile: str
    optimization_pattern: Optional[str]
    target: Any


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate rewritten profiles with the current shared task/runtime "
            "stack. The default evaluator backend is OpenAI-compatible, but the "
            "script now follows the same role-config style as profile_meta_*.py."
        )
    )
    parser.add_argument(
        "--task",
        default="pairwise",
        help="TaskAdapter key used to interpret prompts, labels, and metrics.",
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help=(
            "JSON file containing rewritten examples, typically "
            "*_rewritten_query.json emitted by profile_meta_optimizer.py."
        ),
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional output path for the evaluation result JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by shared utilities.",
    )
    add_role_args(
        parser=parser,
        prefix="evaluator",
        default_model=DEFAULT_TARGET_MODEL,
        default_backend=BACKEND_OPENAI,
        default_temperature=0.0,
        default_max_output_tokens=128,
    )
    return parser


def load_rewritten_examples(
    path: str,
    task_adapter: TaskAdapter,
) -> List[PreparedEvaluationExample]:
    """Load rewritten-query records and align them to the shared SupportExample schema."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Input file must contain a JSON array.")

    prepared_examples: List[PreparedEvaluationExample] = []
    for row_idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Row {row_idx} is not a JSON object.")

        original_profile = item.get("original_profile", item.get("profile"))
        optimized_profile = item.get("optimized_profile")
        prompt_template = item.get("prompt")
        reference = item.get("reference")
        metadata = item.get("metadata") or {}

        if original_profile is None:
            raise ValueError(f"Missing original profile at row {row_idx}.")
        if optimized_profile is None:
            raise ValueError(f"Missing optimized_profile at row {row_idx}.")
        if prompt_template is None:
            raise ValueError(f"Missing prompt at row {row_idx}.")
        if reference is None:
            raise ValueError(f"Missing reference at row {row_idx}.")

        example = SupportExample(
            profile=original_profile,
            prompt=str(prompt_template),
            reference=reference,
            idx=item.get("idx", item.get("source_idx", row_idx)),
            metadata=metadata,
        )
        prepared_examples.append(
            PreparedEvaluationExample(
                source_idx=example.idx,
                example=example,
                optimized_profile=str(optimized_profile),
                optimization_pattern=item.get("optimization_pattern"),
                target=metadata.get("raw_input", item.get("target")),
            )
        )

    return prepared_examples


def build_failed_evaluation(
    example: SupportExample,
    profile_text: str,
    task_adapter: TaskAdapter,
    error_message: str,
) -> Tuple[PredictionResult, EvaluationResult]:
    """Convert one model-call failure into a deterministic failed evaluation record."""

    rendered_prompt = task_adapter.render_prompt(example.prompt, profile_text)
    prediction = PredictionResult(
        output=None,
        reason=None,
        prompt=rendered_prompt,
        metadata={"error": error_message},
    )
    evaluation = EvaluationResult(
        primary_metric_name=task_adapter.primary_metric_name,
        primary_score=0.0,
        metrics={task_adapter.primary_metric_name: 0.0},
        passed=False,
        summary={
            "reference_label": normalize_label(example.reference),
            "predicted_label": None,
            "outcome": "error",
            "error": error_message,
        },
    )
    return prediction, evaluation


def safe_evaluate_profile(
    example: SupportExample,
    profile_text: str,
    task_adapter: TaskAdapter,
    evaluator_caller: ModelCaller,
    evaluator_model: str,
    evaluator_generation_config: GenerationConfig,
) -> Tuple[PredictionResult, EvaluationResult, Optional[str]]:
    """Evaluate one profile while keeping the overall run robust to parse failures."""

    try:
        prediction, evaluation = evaluate_with_profile(
            example=example,
            profile_text=profile_text,
            task_adapter=task_adapter,
            target_caller=evaluator_caller,
            target_model=evaluator_model,
            target_generation_config=evaluator_generation_config,
        )
        return prediction, evaluation, None
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        prediction, evaluation = build_failed_evaluation(
            example=example,
            profile_text=profile_text,
            task_adapter=task_adapter,
            error_message=error_message,
        )
        return prediction, evaluation, error_message


def calculate_accuracy(results: List[Dict[str, Any]], key: str) -> Tuple[float, int]:
    """Compute accuracy-style metrics from boolean per-example fields."""

    total = len(results)
    correct = sum(1 for item in results if item.get(key) is True)
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct


def text_length(text: str) -> Dict[str, int]:
    """Return the same approximate length units used for profile compression."""

    return {
        "chars": len(text),
        "tokens": approximate_token_count(text),
        "words": word_count(text),
    }


def context_compression_stats(
    original_prompt: Optional[str],
    optimized_prompt: Optional[str],
    profile_metrics: Dict[str, float],
) -> Dict[str, float]:
    """Measure compression on the full downstream input prompt.

    `profile_metrics` is the existing profile-only compression result. The
    context metrics here count the entire rendered prompt sent to the downstream
    evaluator: instructions, profile, scenario, query, options, and output
    schema.
    """

    original = text_length(original_prompt or "")
    optimized = text_length(optimized_prompt or "")
    original_profile_tokens = float(profile_metrics["original_profile_tokens"])

    context_token_reduction = original["tokens"] - optimized["tokens"]
    context_char_reduction = original["chars"] - optimized["chars"]
    context_word_reduction = original["words"] - optimized["words"]
    non_profile_context_tokens = max(
        original["tokens"] - int(original_profile_tokens),
        0,
    )
    return {
        "original_context_chars": original["chars"],
        "original_context_tokens": original["tokens"],
        "original_context_words": original["words"],
        "optimized_context_chars": optimized["chars"],
        "optimized_context_tokens": optimized["tokens"],
        "optimized_context_words": optimized["words"],
        "non_profile_context_tokens": non_profile_context_tokens,
        "context_char_reduction": context_char_reduction,
        "context_token_reduction": context_token_reduction,
        "context_word_reduction": context_word_reduction,
        "context_char_compression_ratio": (
            optimized["chars"] / original["chars"] if original["chars"] else 0.0
        ),
        "context_token_compression_ratio": (
            optimized["tokens"] / original["tokens"] if original["tokens"] else 0.0
        ),
        "context_word_compression_ratio": (
            optimized["words"] / original["words"] if original["words"] else 0.0
        ),
        "original_profile_context_token_share": (
            original_profile_tokens / original["tokens"] if original["tokens"] else 0.0
        ),
    }


class PreferenceEvaluator:
    """Evaluate original vs optimized profiles with the shared task/runtime stack."""

    def __init__(
        self,
        config: EvaluationConfig,
        task_adapter: TaskAdapter,
        evaluator_caller: ModelCaller,
        evaluator_spec,
        evaluator_generation_config: GenerationConfig,
    ) -> None:
        self.config = config
        self.task_adapter = task_adapter
        self.evaluator_caller = evaluator_caller
        self.evaluator_spec = evaluator_spec
        self.evaluator_generation_config = evaluator_generation_config

    def run_evaluation(self) -> Dict[str, Any]:
        prepared_examples = load_rewritten_examples(
            path=self.config.input_file,
            task_adapter=self.task_adapter,
        )

        results: List[Dict[str, Any]] = []
        total_original_profile_tokens = 0.0
        total_optimized_profile_tokens = 0.0
        total_original_profile_words = 0.0
        total_optimized_profile_words = 0.0
        total_original_context_tokens = 0.0
        total_optimized_context_tokens = 0.0
        total_original_context_words = 0.0
        total_optimized_context_words = 0.0
        total_original_profile_context_token_share = 0.0
        accuracy_preserved_count = 0
        no_regression_count = 0
        preserved_token_reduction_total = 0.0
        no_regression_token_reduction_total = 0.0
        preserved_context_token_reduction_total = 0.0
        no_regression_context_token_reduction_total = 0.0

        for item in tqdm(prepared_examples, desc="Evaluating profiles"):
            original_prediction, original_evaluation, original_error = (
                safe_evaluate_profile(
                    example=item.example,
                    profile_text=str(item.example.profile),
                    task_adapter=self.task_adapter,
                    evaluator_caller=self.evaluator_caller,
                    evaluator_model=self.evaluator_spec.model_name,
                    evaluator_generation_config=self.evaluator_generation_config,
                )
            )
            optimized_prediction, optimized_evaluation, optimized_error = (
                safe_evaluate_profile(
                    example=item.example,
                    profile_text=item.optimized_profile,
                    task_adapter=self.task_adapter,
                    evaluator_caller=self.evaluator_caller,
                    evaluator_model=self.evaluator_spec.model_name,
                    evaluator_generation_config=self.evaluator_generation_config,
                )
            )

            original_correct = bool(original_evaluation.passed)
            optimized_correct = bool(optimized_evaluation.passed)
            length_metrics = compression_stats(
                item.example.profile, item.optimized_profile
            )
            original_context_prompt = (
                original_prediction.prompt
                or self.task_adapter.render_prompt(
                    item.example.prompt,
                    str(item.example.profile),
                )
            )
            optimized_context_prompt = (
                optimized_prediction.prompt
                or self.task_adapter.render_prompt(
                    item.example.prompt,
                    item.optimized_profile,
                )
            )
            context_metrics = context_compression_stats(
                original_prompt=original_context_prompt,
                optimized_prompt=optimized_context_prompt,
                profile_metrics=length_metrics,
            )

            total_original_profile_tokens += length_metrics["original_profile_tokens"]
            total_optimized_profile_tokens += length_metrics["optimized_profile_tokens"]
            total_original_profile_words += length_metrics["original_profile_words"]
            total_optimized_profile_words += length_metrics["optimized_profile_words"]
            total_original_context_tokens += context_metrics["original_context_tokens"]
            total_optimized_context_tokens += context_metrics["optimized_context_tokens"]
            total_original_context_words += context_metrics["original_context_words"]
            total_optimized_context_words += context_metrics["optimized_context_words"]
            total_original_profile_context_token_share += context_metrics[
                "original_profile_context_token_share"
            ]

            if original_correct == optimized_correct:
                accuracy_preserved_count += 1
                preserved_token_reduction_total += length_metrics["token_reduction"]
                preserved_context_token_reduction_total += context_metrics[
                    "context_token_reduction"
                ]
            if int(optimized_correct) >= int(original_correct):
                no_regression_count += 1
                no_regression_token_reduction_total += length_metrics["token_reduction"]
                no_regression_context_token_reduction_total += context_metrics[
                    "context_token_reduction"
                ]

            results.append(
                {
                    "idx": item.source_idx,
                    "reference": item.example.reference,
                    "metadata": item.example.metadata,
                    "prompt": item.example.prompt,
                    "target": item.target,
                    "profile": item.example.profile,
                    "optimized_profile": item.optimized_profile,
                    "optimization_pattern": item.optimization_pattern,
                    "original_selection": original_prediction.output,
                    "optimized_selection": optimized_prediction.output,
                    "original_correct": original_correct,
                    "optimized_correct": optimized_correct,
                    "original_error": original_error,
                    "optimized_error": optimized_error,
                    "original_prediction": serialize_prediction(original_prediction),
                    "optimized_prediction": serialize_prediction(optimized_prediction),
                    "original_evaluation": serialize_evaluation(original_evaluation),
                    "optimized_evaluation": serialize_evaluation(optimized_evaluation),
                    **length_metrics,
                    **context_metrics,
                }
            )

        original_accuracy, original_correct = calculate_accuracy(
            results, "original_correct"
        )
        optimized_accuracy, optimized_correct = calculate_accuracy(
            results, "optimized_correct"
        )

        evaluation_results = {
            "task": self.task_adapter.name,
            "primary_metric_name": self.task_adapter.primary_metric_name,
            "input_file": self.config.input_file,
            "output_file": self.config.output_path,
            "seed": self.config.seed,
            "evaluator_spec": redacted_spec(self.evaluator_spec),
            "evaluator_generation_config": asdict(self.evaluator_generation_config),
            "total": len(results),
            "original_accuracy": original_accuracy,
            "optimized_accuracy": optimized_accuracy,
            "accuracy_gain": optimized_accuracy - original_accuracy,
            "original_correct": original_correct,
            "optimized_correct": optimized_correct,
            "avg_original_profile_tokens": (
                total_original_profile_tokens / len(results) if results else 0.0
            ),
            "avg_optimized_profile_tokens": (
                total_optimized_profile_tokens / len(results) if results else 0.0
            ),
            "avg_original_profile_words": (
                total_original_profile_words / len(results) if results else 0.0
            ),
            "avg_optimized_profile_words": (
                total_optimized_profile_words / len(results) if results else 0.0
            ),
            "avg_token_reduction": (
                (total_original_profile_tokens - total_optimized_profile_tokens)
                / len(results)
                if results
                else 0.0
            ),
            "avg_word_reduction": (
                (total_original_profile_words - total_optimized_profile_words)
                / len(results)
                if results
                else 0.0
            ),
            "avg_token_compression_ratio": (
                total_optimized_profile_tokens / total_original_profile_tokens
                if total_original_profile_tokens
                else 0.0
            ),
            "avg_original_context_tokens": (
                total_original_context_tokens / len(results) if results else 0.0
            ),
            "avg_optimized_context_tokens": (
                total_optimized_context_tokens / len(results) if results else 0.0
            ),
            "avg_original_context_words": (
                total_original_context_words / len(results) if results else 0.0
            ),
            "avg_optimized_context_words": (
                total_optimized_context_words / len(results) if results else 0.0
            ),
            "avg_context_token_reduction": (
                (total_original_context_tokens - total_optimized_context_tokens)
                / len(results)
                if results
                else 0.0
            ),
            "avg_context_word_reduction": (
                (total_original_context_words - total_optimized_context_words)
                / len(results)
                if results
                else 0.0
            ),
            "avg_context_token_compression_ratio": (
                total_optimized_context_tokens / total_original_context_tokens
                if total_original_context_tokens
                else 0.0
            ),
            "avg_original_profile_context_token_share": (
                total_original_profile_context_token_share / len(results)
                if results
                else 0.0
            ),
            "accuracy_preserved_count": accuracy_preserved_count,
            "avg_token_reduction_when_accuracy_preserved": (
                preserved_token_reduction_total / accuracy_preserved_count
                if accuracy_preserved_count
                else 0.0
            ),
            "avg_context_token_reduction_when_accuracy_preserved": (
                preserved_context_token_reduction_total / accuracy_preserved_count
                if accuracy_preserved_count
                else 0.0
            ),
            "no_regression_count": no_regression_count,
            "avg_token_reduction_when_no_regression": (
                no_regression_token_reduction_total / no_regression_count
                if no_regression_count
                else 0.0
            ),
            "avg_context_token_reduction_when_no_regression": (
                no_regression_context_token_reduction_total / no_regression_count
                if no_regression_count
                else 0.0
            ),
            "results": results,
        }

        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(evaluation_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return evaluation_results


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    config = EvaluationConfig(
        task=args.task,
        input_file=args.input_file,
        output_file=args.output_file,
        seed=args.seed,
    )
    task_adapter = build_task_adapter(config.task)
    evaluator_spec = build_role_spec_from_args(args, prefix="evaluator")
    evaluator_generation_config = build_generation_config_from_args(
        args,
        prefix="evaluator",
    )
    evaluator_caller = build_model_caller(evaluator_spec)

    evaluator = PreferenceEvaluator(
        config=config,
        task_adapter=task_adapter,
        evaluator_caller=evaluator_caller,
        evaluator_spec=evaluator_spec,
        evaluator_generation_config=evaluator_generation_config,
    )
    result = evaluator.run_evaluation()
    # print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
