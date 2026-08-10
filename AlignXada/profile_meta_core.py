import json
import os
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_REWRITE_MODEL = "gpt-4o-mini"
DEFAULT_TARGET_MODEL = "gpt-4.1-nano"
DEFAULT_META_MODEL = "gpt-4o-mini"
DEFAULT_TASK_DESCRIPTION = (
    "Given a user profile, a query, and two candidate responses, predict which "
    "response the user prefers."
)
DEFAULT_INITIAL_PATTERN = (
    "Rewrite the profile into concise, stable preference rules that preserve "
    "task-relevant user signals while reducing unnecessary detail."
)
PROFILE_PLACEHOLDER = "[PROFILE_PLACEHOLDER]"
PAIRWISE_LABEL_SPACE = ["Item A", "Item B"]


@dataclass
class SupportExample:
    profile: Any
    prompt: str
    reference: Any
    idx: Optional[int]
    original_prediction: Optional["PredictionResult"] = None
    original_evaluation: Optional["EvaluationResult"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    output: Any
    reason: Optional[str] = None
    prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    primary_metric_name: str
    primary_score: float
    metrics: Dict[str, float]
    passed: Optional[bool] = None
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackExperience:
    idx: Optional[int]
    profile: Any
    optimized_profile: str
    prompt: str
    reference: Any
    original_prediction: Any
    optimized_prediction: Any
    original_reason: Optional[str]
    optimized_reason: Optional[str]
    original_primary_score: float
    optimized_primary_score: float
    original_passed: Optional[bool]
    optimized_passed: Optional[bool]
    original_metrics: Dict[str, float]
    optimized_metrics: Dict[str, float]
    meta_summary: Dict[str, Any]
    original_profile_words: int = 0
    optimized_profile_words: int = 0


@dataclass
class IterationTrace:
    iteration: int
    epoch: int
    batch_index: int
    pattern: str
    metrics: Dict[str, float]
    total_examples: int
    num_errors: int
    next_pattern: Optional[str]


def pretty_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=2).strip()


def normalize_label(label: Any) -> str:
    return str(label).strip()


def approximate_token_count(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def profile_length(profile: Any) -> Dict[str, int]:
    text = pretty_text(profile)
    return {
        "chars": len(text),
        "tokens": approximate_token_count(text),
        "words": word_count(text),
    }


def compression_stats(
    original_profile: Any, optimized_profile: Any
) -> Dict[str, float]:
    original = profile_length(original_profile)
    optimized = profile_length(optimized_profile)
    token_reduction = original["tokens"] - optimized["tokens"]
    char_reduction = original["chars"] - optimized["chars"]
    word_reduction = original["words"] - optimized["words"]
    return {
        "original_profile_chars": original["chars"],
        "original_profile_tokens": original["tokens"],
        "original_profile_words": original["words"],
        "optimized_profile_chars": optimized["chars"],
        "optimized_profile_tokens": optimized["tokens"],
        "optimized_profile_words": optimized["words"],
        "char_reduction": char_reduction,
        "token_reduction": token_reduction,
        "word_reduction": word_reduction,
        "char_compression_ratio": (
            optimized["chars"] / original["chars"] if original["chars"] else 0.0
        ),
        "token_compression_ratio": (
            optimized["tokens"] / original["tokens"] if original["tokens"] else 0.0
        ),
        "word_compression_ratio": (
            optimized["words"] / original["words"] if original["words"] else 0.0
        ),
    }


def parse_json_object(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip().split("</think>")[-1].strip()
    try:
        return _loads_json_object(text)
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced_match:
        try:
            return _loads_json_object(fenced_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try each '{' from the end — thinking content may contain stray braces
    end = text.rfind("}")
    if end == -1:
        raise ValueError(f"Model output is not valid JSON: {raw_text}")

    pos = end
    while pos >= 0:
        pos = text.rfind("{", 0, pos)
        if pos == -1:
            break
        candidate = text[pos : end + 1]
        try:
            return _loads_json_object(candidate)
        except json.JSONDecodeError:
            pos -= 1
            continue

    raise ValueError(
        f"Failed to parse model JSON output. Raw output: {raw_text}"
    )


def _loads_json_object(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(text, strict=False)


def chunk_list(items: Sequence[Any], chunk_size: int) -> List[List[Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    return [
        list(items[idx : idx + chunk_size]) for idx in range(0, len(items), chunk_size)
    ]


def average(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def serialize_prediction(
    prediction: Optional[PredictionResult],
) -> Optional[Dict[str, Any]]:
    if prediction is None:
        return None
    return asdict(prediction)


def serialize_evaluation(
    evaluation: Optional[EvaluationResult],
) -> Optional[Dict[str, Any]]:
    if evaluation is None:
        return None
    return asdict(evaluation)


def set_seed(seed: int = 42) -> None:
    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    print(f"Random seed set to {seed}")
