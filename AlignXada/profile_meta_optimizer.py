import re
import argparse
import json
import logging
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from profile_meta_core import (
    DEFAULT_INITIAL_PATTERN,
    DEFAULT_META_MODEL,
    DEFAULT_REWRITE_MODEL,
    DEFAULT_TARGET_MODEL,
    EvaluationResult,
    FeedbackExperience,
    IterationTrace,
    PredictionResult,
    SupportExample,
    average,
    chunk_list,
    compression_stats,
    set_seed,
)
from profile_meta_prompts import build_meta_prompt
from profile_meta_tasks import TaskAdapter, build_task_adapter, load_examples
from profile_meta_runtime import (
    BACKEND_OPENAI,
    ProfileExecutionEngine,
    add_role_args,
    build_feedback_payload,
    build_generation_config_from_args,
    build_model_caller,
    build_role_spec_from_args,
)


def log_json(logger: logging.Logger, payload: Dict[str, Any]) -> None:
    logger.info(json.dumps(payload, ensure_ascii=False, indent=2))


SUPPORT_SAMPLING_STRATEGIES = ["random", "pref_type_stratified", "domain_stratified"]


def _redacted_spec(spec: Any) -> Dict[str, Any]:
    payload = asdict(spec)
    if payload.get("api_key"):
        payload["api_key"] = "<redacted>"
    return payload


def _example_pref_type(example: SupportExample) -> str:
    return str((example.metadata or {}).get("pref_type") or "unknown")


def _example_domain(example: SupportExample) -> str:
    return str((example.metadata or {}).get("domain") or "unknown")


def _example_persona_id(example: SupportExample) -> str:
    metadata = example.metadata or {}
    value = metadata.get("persona_id")
    if value is None:
        value = metadata.get("user_id")
    return str(value if value is not None else f"idx:{example.idx}")


def _example_counts(examples: Sequence[SupportExample]) -> Dict[str, Any]:
    pref_type_counts = Counter(_example_pref_type(item) for item in examples)
    domain_counts = Counter(_example_domain(item) for item in examples)
    persona_counts = Counter(_example_persona_id(item) for item in examples)
    return {
        "pref_type_counts": dict(sorted(pref_type_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "num_personas": len(persona_counts),
        "repeated_persona_count": sum(
            1 for count in persona_counts.values() if count > 1
        ),
    }


def _select_metadata_stratified_examples(
    candidates: Sequence[SupportExample],
    size: int,
    rng: random.Random,
    key_fn: Any,
    avoid_persona_ids: Optional[set[str]] = None,
) -> List[SupportExample]:
    if size <= 0:
        return []
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    random_rank = {id(item): rank for rank, item in enumerate(shuffled)}
    selected: List[SupportExample] = []
    selected_ids = set()
    selected_personas: set[str] = set()
    avoid_persona_ids = avoid_persona_ids or set()
    current_group_counts: Counter = Counter()
    global_group_counts = Counter(key_fn(item) for item in shuffled)

    def eligible_items(mode: str) -> List[SupportExample]:
        output = []
        for item in shuffled:
            if id(item) in selected_ids:
                continue
            persona_id = _example_persona_id(item)
            if mode in {"strict", "allow_avoid"} and persona_id in selected_personas:
                continue
            if mode == "strict" and persona_id in avoid_persona_ids:
                continue
            output.append(item)
        return output

    def choose_one(pool: Sequence[SupportExample]) -> SupportExample:
        available_group_counts = Counter(key_fn(item) for item in pool)
        target_group = min(
            available_group_counts,
            key=lambda group: (
                current_group_counts[group],
                global_group_counts[group],
                group,
            ),
        )
        group_pool = [
            item for item in pool if key_fn(item) == target_group
        ]
        return min(group_pool, key=lambda item: random_rank[id(item)])

    for mode in ("strict", "allow_avoid", "allow_duplicate_persona"):
        while len(selected) < size:
            pool = eligible_items(mode)
            if not pool:
                break
            item = choose_one(pool)
            selected.append(item)
            selected_ids.add(id(item))
            selected_personas.add(_example_persona_id(item))
            current_group_counts.update([key_fn(item)])
        if len(selected) == size:
            break

    return selected


def _select_pref_type_stratified_examples(
    candidates: Sequence[SupportExample],
    size: int,
    rng: random.Random,
    avoid_persona_ids: Optional[set[str]] = None,
) -> List[SupportExample]:
    return _select_metadata_stratified_examples(
        candidates,
        size,
        rng,
        key_fn=_example_pref_type,
        avoid_persona_ids=avoid_persona_ids,
    )


def _select_domain_stratified_examples(
    candidates: Sequence[SupportExample],
    size: int,
    rng: random.Random,
    avoid_persona_ids: Optional[set[str]] = None,
) -> List[SupportExample]:
    return _select_metadata_stratified_examples(
        candidates,
        size,
        rng,
        key_fn=_example_domain,
        avoid_persona_ids=avoid_persona_ids,
    )


def _remaining_examples(
    candidates: Sequence[SupportExample],
    selected: Sequence[SupportExample],
) -> List[SupportExample]:
    selected_ids = {id(item) for item in selected}
    return [item for item in candidates if id(item) not in selected_ids]


def _split_pref_type_stratified_train_dev(
    candidates: Sequence[SupportExample],
    train_size: int,
    dev_size: int,
    rng: random.Random,
) -> Tuple[List[SupportExample], List[SupportExample]]:
    return _split_metadata_stratified_train_dev(
        candidates,
        train_size,
        dev_size,
        rng,
        key_fn=_example_pref_type,
    )


def _split_domain_stratified_train_dev(
    candidates: Sequence[SupportExample],
    train_size: int,
    dev_size: int,
    rng: random.Random,
) -> Tuple[List[SupportExample], List[SupportExample]]:
    return _split_metadata_stratified_train_dev(
        candidates,
        train_size,
        dev_size,
        rng,
        key_fn=_example_domain,
    )


def _split_metadata_stratified_train_dev(
    candidates: Sequence[SupportExample],
    train_size: int,
    dev_size: int,
    rng: random.Random,
    key_fn: Any,
) -> Tuple[List[SupportExample], List[SupportExample]]:
    combined = _select_metadata_stratified_examples(
        candidates, train_size + dev_size, rng, key_fn=key_fn
    )
    if not combined:
        return [], []

    grouped: Dict[str, List[SupportExample]] = {}
    for item in combined:
        grouped.setdefault(key_fn(item), []).append(item)

    quotas: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []
    total = len(combined)
    for pref_type, items in grouped.items():
        raw_quota = train_size * len(items) / total
        quota = min(int(raw_quota), len(items))
        quotas[pref_type] = quota
        remainders.append((raw_quota - quota, pref_type))

    remaining = train_size - sum(quotas.values())
    for _, pref_type in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        if quotas[pref_type] < len(grouped[pref_type]):
            quotas[pref_type] += 1
            remaining -= 1

    train: List[SupportExample] = []
    dev: List[SupportExample] = []
    for pref_type, items in grouped.items():
        shuffled_items = list(items)
        rng.shuffle(shuffled_items)
        quota = quotas[pref_type]
        train.extend(shuffled_items[:quota])
        dev.extend(shuffled_items[quota:])

    rng.shuffle(train)
    rng.shuffle(dev)
    return train, dev[:dev_size]


def _example_user_split(example: SupportExample) -> str:
    return str((example.metadata or {}).get("user_split") or "").strip()


def _select_examples_for_split(
    candidates: Sequence[SupportExample],
    size: int,
    rng: random.Random,
    support_sampling_strategy: str,
) -> List[SupportExample]:
    if size <= 0:
        return []
    if support_sampling_strategy == "pref_type_stratified":
        return _select_pref_type_stratified_examples(candidates, size, rng)
    if support_sampling_strategy == "domain_stratified":
        return _select_domain_stratified_examples(candidates, size, rng)
    return list(candidates[:size])


def _sample_explicit_user_split_sets(
    result_examples: Sequence[SupportExample],
    support_size: int,
    dev_support_size: int,
    query_size: int,
    rng: random.Random,
    support_sampling_strategy: str,
) -> Optional[Dict[str, Any]]:
    split_values = {_example_user_split(item) for item in result_examples}
    if (
        "query" not in split_values
        or not ({"support", "train_support", "selection"} & split_values)
    ):
        return None

    shuffled_examples = list(result_examples)
    rng.shuffle(shuffled_examples)
    train_candidates = [
        item for item in shuffled_examples
        if _example_user_split(item) in {"train_support", "support"}
    ]
    selection_candidates = [
        item for item in shuffled_examples if _example_user_split(item) == "selection"
    ]
    query_candidates = [
        item for item in shuffled_examples if _example_user_split(item) == "query"
    ]

    if selection_candidates:
        support_set = _select_examples_for_split(
            train_candidates, support_size, rng, support_sampling_strategy
        )
        dev_support_set = _select_examples_for_split(
            selection_candidates, dev_support_size, rng, support_sampling_strategy
        )
    else:
        combined_support = _select_examples_for_split(
            train_candidates,
            support_size + dev_support_size,
            rng,
            support_sampling_strategy,
        )
        support_set = combined_support[:support_size]
        dev_support_set = combined_support[support_size:]
    query_set = query_candidates[:query_size]

    if len(support_set) < support_size:
        raise ValueError(
            "Not enough train_support examples for support set: "
            f"need {support_size}, found {len(support_set)}."
        )
    if len(dev_support_set) < dev_support_size:
        raise ValueError(
            "Not enough selection examples for dev support set: "
            f"need {dev_support_size}, found {len(dev_support_set)}."
        )
    if len(query_set) < query_size:
        raise ValueError(
            "Not enough query examples for query set: "
            f"need {query_size}, found {len(query_set)}."
        )

    rng.shuffle(support_set)
    rng.shuffle(dev_support_set)
    return {
        "support_set": support_set,
        "dev_support_set": dev_support_set,
        "query_set": query_set,
        "summary": {
            "split_protocol": "explicit_user_split",
            "available_split_counts": dict(sorted(Counter(
                _example_user_split(item) or "unspecified"
                for item in result_examples
            ).items())),
            "train_support_candidate_count": len(train_candidates),
            "selection_candidate_count": len(selection_candidates),
            "query_candidate_count": len(query_candidates),
        },
    }


def sample_support_and_query_sets(
    result_examples: Sequence[SupportExample],
    support_size: int,
    support_positive_count: int,
    dev_support_size: int,
    dev_support_positive_count: Optional[int],
    query_size: int,
    seed: int,
    task_adapter: TaskAdapter,
    use_explicit_user_split: bool = False,
    use_debug_recorded_sampling: bool = False,
    support_sampling_strategy: str = "random",
) -> Dict[str, Any]:
    if support_sampling_strategy not in SUPPORT_SAMPLING_STRATEGIES:
        raise ValueError(
            f"support_sampling_strategy must be one of "
            f"{SUPPORT_SAMPLING_STRATEGIES}, got {support_sampling_strategy!r}."
        )
    if support_size <= 0:
        raise ValueError("support_size must be positive.")
    if support_positive_count < 0 or support_positive_count > support_size:
        raise ValueError("support_positive_count must be within [0, support_size].")
    if dev_support_size < 0:
        raise ValueError("dev_support_size must be non-negative.")
    if dev_support_positive_count is None:
        dev_support_positive_count = round(
            dev_support_size * support_positive_count / support_size
        )
    if (
        dev_support_positive_count < 0
        or dev_support_positive_count > dev_support_size
    ):
        raise ValueError(
            "dev_support_positive_count must be within [0, dev_support_size]."
        )
    if query_size <= 0:
        raise ValueError("query_size must be positive.")

    support_negative_count = support_size - support_positive_count
    dev_support_negative_count = dev_support_size - dev_support_positive_count

    rng = random.Random(seed)
    shuffled_examples = list(result_examples)
    rng.shuffle(shuffled_examples)

    if use_explicit_user_split:
        explicit_split_sample = _sample_explicit_user_split_sets(
            result_examples=shuffled_examples,
            support_size=support_size,
            dev_support_size=dev_support_size,
            query_size=query_size,
            rng=rng,
            support_sampling_strategy=support_sampling_strategy,
        )
        if explicit_split_sample is not None:
            explicit_split_sample["summary"].update(
                {
                    "seed": seed,
                    "task": task_adapter.name,
                    "total_examples": len(result_examples),
                    "sampling_source": "explicit_user_split",
                    "support_sampling_strategy": support_sampling_strategy,
                    "original_baselines_precomputed": False,
                    "positive_bucket": task_adapter.positive_bucket,
                    "negative_bucket": task_adapter.negative_bucket,
                    "support_size": support_size,
                    "support_positive_count": support_positive_count,
                    "support_negative_count": support_size - support_positive_count,
                    "dev_support_size": dev_support_size,
                    "dev_support_positive_count": dev_support_positive_count,
                    "dev_support_negative_count": (
                        dev_support_size - dev_support_positive_count
                    ),
                    "query_size": query_size,
                    "support_indices": [
                        item.idx for item in explicit_split_sample["support_set"]
                    ],
                    "dev_support_indices": [
                        item.idx for item in explicit_split_sample["dev_support_set"]
                    ],
                    "query_indices": [
                        item.idx for item in explicit_split_sample["query_set"]
                    ],
                    "support_distribution": _example_counts(
                        explicit_split_sample["support_set"]
                    ),
                    "dev_support_distribution": _example_counts(
                        explicit_split_sample["dev_support_set"]
                    ),
                    "query_distribution": _example_counts(
                        explicit_split_sample["query_set"]
                    ),
                }
            )
            return explicit_split_sample

    positive_candidates: List[SupportExample] = []
    negative_candidates: List[SupportExample] = []
    sampling_desc = (
        "Sampling from recorded pairwise answer/selection"
        if use_debug_recorded_sampling
        else "Sampling from original-profile buckets"
    )
    sampling_source = (
        "recorded_pairwise_answer_selection"
        if use_debug_recorded_sampling
        else "original_profile_baseline"
    )
    for item in tqdm(shuffled_examples, desc=sampling_desc):
        if use_debug_recorded_sampling:
            sampling_evaluation = task_adapter.build_recorded_sampling_evaluation(item)
            if sampling_evaluation is None:
                raise ValueError(
                    "Debug recorded sampling was requested, but the current example "
                    "does not contain usable answer/selection fields."
                )
        else:
            if item.original_evaluation is None:
                raise ValueError(
                    "original_evaluation must be populated before sampling. "
                    "Call populate_original_baselines(...) first."
                )
            sampling_evaluation = item.original_evaluation

        bucket = task_adapter.sampling_bucket(sampling_evaluation)
        if bucket == task_adapter.positive_bucket:
            positive_candidates.append(item)
        elif bucket == task_adapter.negative_bucket:
            negative_candidates.append(item)

    if support_sampling_strategy in {"pref_type_stratified", "domain_stratified"}:
        split_fn = (
            _split_domain_stratified_train_dev
            if support_sampling_strategy == "domain_stratified"
            else _split_pref_type_stratified_train_dev
        )
        support_positive, dev_support_positive = split_fn(
            positive_candidates,
            support_positive_count,
            dev_support_positive_count,
            rng,
        )

        support_negative, dev_support_negative = (
            split_fn(
                negative_candidates,
                support_negative_count,
                dev_support_negative_count,
                rng,
            )
        )
        selected_negative = support_negative + dev_support_negative
        query_candidates = _remaining_examples(
            negative_candidates, selected_negative
        )
        query_set = query_candidates[:query_size]
    else:
        support_positive = positive_candidates[:support_positive_count]
        remaining_positive = positive_candidates[support_positive_count:]
        dev_support_positive = remaining_positive[:dev_support_positive_count]

        support_negative = negative_candidates[:support_negative_count]
        remaining_negative = negative_candidates[support_negative_count:]
        dev_support_negative = remaining_negative[:dev_support_negative_count]
        query_set = remaining_negative[dev_support_negative_count:][
            :query_size
        ]

    if len(support_positive) < support_positive_count:
        raise ValueError(
            f"Not enough {task_adapter.positive_bucket} examples for support set: "
            f"need {support_positive_count}, found {len(support_positive)}."
        )
    if len(support_negative) < support_negative_count:
        raise ValueError(
            f"Not enough {task_adapter.negative_bucket} examples for support set: "
            f"need {support_negative_count}, found {len(support_negative)}."
        )
    if len(dev_support_positive) < dev_support_positive_count:
        raise ValueError(
            f"Not enough {task_adapter.positive_bucket} examples for dev support set: "
            f"need {dev_support_positive_count}, found {len(dev_support_positive)}."
        )
    if len(dev_support_negative) < dev_support_negative_count:
        raise ValueError(
            f"Not enough {task_adapter.negative_bucket} examples for dev support set: "
            f"need {dev_support_negative_count}, found {len(dev_support_negative)}."
        )
    if len(query_set) < query_size:
        raise ValueError(
            f"Not enough {task_adapter.negative_bucket} examples for query set: "
            f"need {query_size}, found {len(query_set)}."
        )

    support_set = support_positive + support_negative
    dev_support_set = dev_support_positive + dev_support_negative
    rng.shuffle(support_set)
    rng.shuffle(dev_support_set)

    return {
        "support_set": support_set,
        "dev_support_set": dev_support_set,
        "query_set": query_set,
        "summary": {
            "seed": seed,
            "task": task_adapter.name,
            "total_examples": len(result_examples),
            "sampling_source": sampling_source,
            "support_sampling_strategy": support_sampling_strategy,
            "original_baselines_precomputed": not use_debug_recorded_sampling,
            "positive_bucket": task_adapter.positive_bucket,
            "negative_bucket": task_adapter.negative_bucket,
            "support_size": support_size,
            "support_positive_count": support_positive_count,
            "support_negative_count": support_negative_count,
            "dev_support_size": dev_support_size,
            "dev_support_positive_count": dev_support_positive_count,
            "dev_support_negative_count": dev_support_negative_count,
            "query_size": query_size,
            "support_indices": [item.idx for item in support_set],
            "dev_support_indices": [item.idx for item in dev_support_set],
            "query_indices": [item.idx for item in query_set],
            "support_distribution": _example_counts(support_set),
            "dev_support_distribution": _example_counts(dev_support_set),
            "query_distribution": _example_counts(query_set),
        },
    }

_PATTERN_SECTIONS = ["Goal", "Preserve", "Compress", "Avoid", "Output Style", "Priority"]
def _flatten_pattern_value(value):
    if value is None: return ""
    if isinstance(value, str): return value.strip()
    if isinstance(value, dict):
        normalized = {re.sub(r"\s+|_", "", str(k)).lower(): k for k in value.keys()}
        out = []
        for sec in _PATTERN_SECTIONS:
            k = normalized.get(re.sub(r"\s+|_", "", sec).lower())
            if k is None: continue
            v = value[k]
            if v is None: continue
            if isinstance(v, list):
                bullets = "\n".join(f"- {str(x).strip()}" for x in v if str(x).strip())
                if bullets: out.append(f"{sec}\n{bullets}")
            else:
                t = str(v).strip()
                if t: out.append(f"{sec}\n{t}")
        return "\n\n".join(out) if out else json.dumps(value, ensure_ascii=False, indent=2).strip()
    return str(value).strip()


def _feedback_passed(passed: Optional[bool], score: float) -> bool:
    if passed is not None:
        return bool(passed)
    return score > 0.0


def _feedback_transition_summary(
    feedbacks: Sequence[FeedbackExperience],
    prefix: str,
) -> Dict[str, Any]:
    counts: Counter[str] = Counter()
    for feedback in feedbacks:
        original_passed = _feedback_passed(
            feedback.original_passed,
            feedback.original_primary_score,
        )
        optimized_passed = _feedback_passed(
            feedback.optimized_passed,
            feedback.optimized_primary_score,
        )
        if original_passed and optimized_passed:
            counts["RR"] += 1
        elif original_passed and not optimized_passed:
            counts["RW"] += 1
        elif not original_passed and optimized_passed:
            counts["WR"] += 1
        else:
            counts["WW"] += 1

    return {
        f"{prefix}_transition_counts": dict(sorted(counts.items())),
        f"{prefix}_stable_success": counts["RR"],
        f"{prefix}_right_to_wrong": counts["RW"],
        f"{prefix}_wrong_to_right": counts["WR"],
        f"{prefix}_persistent_failure": counts["WW"],
    }


class ProfileMetaOptimizer:
    def __init__(
        self,
        task_adapter: TaskAdapter,
        execution_engine: ProfileExecutionEngine,
        meta_caller: Any,
        meta_model: str = DEFAULT_META_MODEL,
        meta_generation_config: Optional[Any] = None,
        feedback_batch_size: int = 5,
        best_pattern_strategy: str = "best_epoch",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.task_adapter = task_adapter
        self.execution_engine = execution_engine
        self.rewrite_model = execution_engine.rewrite_model
        self.target_model = execution_engine.target_model
        self.meta_model = meta_model
        self.meta_caller = meta_caller
        self.meta_generation_config = meta_generation_config
        self.feedback_batch_size = feedback_batch_size
        self.best_pattern_strategy = best_pattern_strategy
        self.logger = logger

    def rewrite_profile(self, profile: Any, pattern: str) -> str:
        return self.execution_engine.rewrite_profile(profile, pattern)

    def evaluate_with_profile(
        self,
        example: SupportExample,
        profile_text: str,
    ) -> Tuple[PredictionResult, EvaluationResult]:
        return self.execution_engine.evaluate_with_profile(example, profile_text)

    def evaluate_original_example(
        self,
        example: SupportExample,
    ) -> Tuple[PredictionResult, EvaluationResult]:
        return self.execution_engine.evaluate_original_example(example)

    def collect_feedback(
        self,
        support_set: Sequence[SupportExample],
        pattern: str,
    ) -> Dict[str, Any]:
        return self.execution_engine.collect_feedback(support_set, pattern)

    def textual_gradient_step(
        self,
        current_pattern: str,
        feedbacks: Sequence[FeedbackExperience],
    ) -> Dict[str, Optional[str]]:
        if not feedbacks:
            raise ValueError(
                "No feedback examples available for textual gradient step."
            )

        feedback_payload = build_feedback_payload(feedbacks)
        instructions, prompt = build_meta_prompt(
            current_pattern=current_pattern,
            feedback_payload=feedback_payload,
            task_description=self.task_adapter.task_description,
            primary_metric_name=self.task_adapter.primary_metric_name,
        )
        meta_temperature = (
            self.meta_generation_config.temperature
            if self.meta_generation_config is not None
            else 0.4
        )
        meta_max_output_tokens = (
            self.meta_generation_config.max_output_tokens
            if self.meta_generation_config is not None
            else 800
        )
        meta_top_p = (
            self.meta_generation_config.top_p
            if self.meta_generation_config is not None
            else 1.0
        )
        meta_top_k = (
            self.meta_generation_config.top_k
            if self.meta_generation_config is not None
            else -1
        )
        result = self.meta_caller.generate_json(
            model=self.meta_model,
            instructions=instructions,
            prompt=prompt,
            temperature=meta_temperature,
            max_output_tokens=meta_max_output_tokens,
            top_p=meta_top_p,
            top_k=meta_top_k,
        )
        # next_pattern = _flatten_pattern_value(result.get("next_pattern", ""))
        next_pattern = str(result.get("next_pattern", "")).strip()
        if not next_pattern:
            raise ValueError("Meta model returned an empty next_pattern.")
        return {"next_pattern": next_pattern}

    def log_batch_update(
        self,
        epoch_idx: int,
        batch_idx: int,
        current_pattern: str,
        next_pattern: str,
        forward_result: Dict[str, Any],
    ) -> None:
        if self.logger is None:
            return

        payload = {
            "epoch": epoch_idx,
            "batch_index": batch_idx,
            "task": self.task_adapter.name,
            "batch_primary_metric_name": forward_result["primary_metric_name"],
            "batch_primary_metric_value": forward_result["primary_metric_value"],
            "batch_metrics": forward_result["metrics"],
            "batch_total": forward_result["total"],
            "batch_errors": forward_result["num_errors"],
            "avg_original_profile_words": forward_result["avg_original_profile_words"],
            "avg_optimized_profile_words": forward_result[
                "avg_optimized_profile_words"
            ],
            "avg_word_reduction": forward_result["avg_word_reduction"],
            "current_pattern": current_pattern,
            "next_pattern": next_pattern,
        }
        log_json(self.logger, payload)

    def evaluate_pattern_on_dev_support(
        self,
        dev_support_set: Sequence[SupportExample],
        pattern: str,
    ) -> Optional[Dict[str, Any]]:
        if not dev_support_set:
            return None
        dev_result = self.collect_feedback(dev_support_set, pattern)
        transition_summary = _feedback_transition_summary(
            dev_result["feedbacks"],
            prefix="dev",
        )
        return {
            "dev_primary_metric_name": dev_result["primary_metric_name"],
            "dev_primary_metric_value": dev_result["primary_metric_value"],
            "dev_metrics": dev_result["metrics"],
            "dev_total_examples": dev_result["total"],
            "dev_num_errors": dev_result["num_errors"],
            "dev_avg_original_profile_words": dev_result[
                "avg_original_profile_words"
            ],
            "dev_avg_optimized_profile_words": dev_result[
                "avg_optimized_profile_words"
            ],
            "dev_avg_word_reduction": dev_result["avg_word_reduction"],
            **transition_summary,
        }

    def run(
        self,
        support_set: Sequence[SupportExample],
        initial_pattern: str,
        num_epochs: int = 3,
        dev_support_set: Optional[Sequence[SupportExample]] = None,
    ) -> Dict[str, Any]:
        if not support_set:
            raise ValueError("Support set must not be empty.")

        current_pattern = initial_pattern
        traces: List[IterationTrace] = []
        global_step = 0
        epoch_summaries: List[Dict[str, Any]] = []
        total_batches = 0
        for _ in range(num_epochs):
            shuffled_support = list(support_set)
            support_batches = chunk_list(shuffled_support, self.feedback_batch_size)
            total_batches += len(support_batches)

        progress_bar = tqdm(total=total_batches, desc="Batches", unit="batch")
        for epoch_idx in range(num_epochs):
            shuffled_support = list(support_set)
            random.shuffle(shuffled_support)
            support_batches = chunk_list(shuffled_support, self.feedback_batch_size)

            epoch_primary_score_total = 0.0
            epoch_total = 0
            epoch_errors = 0
            epoch_original_profile_words = 0.0
            epoch_optimized_profile_words = 0.0

            for batch_idx, support_batch in enumerate(support_batches):
                forward_result = self.collect_feedback(support_batch, current_pattern)
                feedbacks = forward_result["feedbacks"]
                num_errors = forward_result["num_errors"]
                next_pattern = current_pattern

                update_result = self.textual_gradient_step(current_pattern, feedbacks)
                next_pattern = update_result["next_pattern"]

                traces.append(
                    IterationTrace(
                        iteration=global_step,
                        epoch=epoch_idx,
                        batch_index=batch_idx,
                        pattern=current_pattern,
                        metrics=forward_result["metrics"],
                        total_examples=forward_result["total"],
                        num_errors=num_errors,
                        next_pattern=next_pattern,
                    )
                )
                self.log_batch_update(
                    epoch_idx=epoch_idx,
                    batch_idx=batch_idx,
                    current_pattern=current_pattern,
                    next_pattern=next_pattern,
                    forward_result=forward_result,
                )

                current_pattern = next_pattern
                global_step += 1
                epoch_primary_score_total += (
                    forward_result["primary_metric_value"] * forward_result["total"]
                )
                epoch_total += forward_result["total"]
                epoch_errors += num_errors
                epoch_original_profile_words += (
                    forward_result["avg_original_profile_words"]
                    * forward_result["total"]
                )
                epoch_optimized_profile_words += (
                    forward_result["avg_optimized_profile_words"]
                    * forward_result["total"]
                )
                progress_bar.update(1)
                progress_bar.set_postfix(
                    metric=f"{forward_result['primary_metric_name']}={forward_result['primary_metric_value']:.4f}",
                    avg_opt_words=f"{forward_result['avg_optimized_profile_words']:.1f}",
                    errors=num_errors,
                )

            epoch_summary = {
                "epoch": epoch_idx,
                "num_batches": len(support_batches),
                "primary_metric_name": self.task_adapter.primary_metric_name,
                "primary_metric_value": (
                    epoch_primary_score_total / epoch_total if epoch_total else 0.0
                ),
                "total_examples": epoch_total,
                "num_errors": epoch_errors,
                "avg_original_profile_words": (
                    epoch_original_profile_words / epoch_total if epoch_total else 0.0
                ),
                "avg_optimized_profile_words": (
                    epoch_optimized_profile_words / epoch_total if epoch_total else 0.0
                ),
                "avg_word_reduction": (
                    (epoch_original_profile_words - epoch_optimized_profile_words)
                    / epoch_total
                    if epoch_total
                    else 0.0
                ),
                "pattern_after_epoch": current_pattern,
            }
            dev_summary = self.evaluate_pattern_on_dev_support(
                dev_support_set or [], current_pattern
            )
            if dev_summary is not None:
                epoch_summary.update(dev_summary)
            epoch_summaries.append(epoch_summary)
        progress_bar.close()

        if self.best_pattern_strategy == "last_epoch":
            selected_epoch = epoch_summaries[-1]
        elif dev_support_set:
            selected_epoch = max(
                epoch_summaries,
                key=lambda item: (
                    item["dev_primary_metric_value"],
                    item["epoch"],
                ),
            )
        else:
            selected_epoch = max(
                epoch_summaries,
                key=lambda item: (
                    item["primary_metric_value"],
                    item["epoch"],
                ),
            )
        last_next_pattern = traces[-1].next_pattern if traces else None
        return {
            "task": self.task_adapter.name,
            "task_description": self.task_adapter.task_description,
            "primary_metric_name": self.task_adapter.primary_metric_name,
            "rewrite_model": self.rewrite_model,
            "target_model": self.target_model,
            "meta_model": self.meta_model,
            "initial_pattern": initial_pattern,
            "final_pattern": current_pattern,
            "next_pattern_suggestion": last_next_pattern,
            "best_pattern_strategy": self.best_pattern_strategy,
            "best_pattern_selection_source": (
                "last_epoch"
                if self.best_pattern_strategy == "last_epoch"
                else ("dev_support" if dev_support_set else "train_support")
            ),
            "best_pattern": selected_epoch["pattern_after_epoch"],
            "best_support_primary_score": selected_epoch["primary_metric_value"],
            "best_support_avg_optimized_profile_words": selected_epoch[
                "avg_optimized_profile_words"
            ],
            "best_support_avg_word_reduction": selected_epoch["avg_word_reduction"],
            "best_dev_support_primary_score": selected_epoch.get(
                "dev_primary_metric_value"
            ),
            "best_dev_support_avg_optimized_profile_words": selected_epoch.get(
                "dev_avg_optimized_profile_words"
            ),
            "best_dev_support_avg_word_reduction": selected_epoch.get(
                "dev_avg_word_reduction"
            ),
            "epochs": epoch_summaries,
            "iterations": [asdict(trace) for trace in traces],
        }

    def rewrite_examples(
        self,
        examples: Sequence[SupportExample],
        pattern: str,
    ) -> Dict[str, Any]:
        rewritten_examples: List[Dict[str, Any]] = []

        for example in examples:
            optimized_profile = self.rewrite_profile(example.profile, pattern)
            length_metrics = compression_stats(example.profile, optimized_profile)
            rewritten_examples.append(
                {
                    **self.task_adapter.serialize_example(
                        example=example,
                        optimized_profile=optimized_profile,
                        pattern=pattern,
                    ),
                    **length_metrics,
                }
            )

        return {
            "pattern": pattern,
            "total_examples": len(examples),
            "avg_original_profile_words": average(
                [item["original_profile_words"] for item in rewritten_examples]
            ),
            "avg_optimized_profile_words": average(
                [item["optimized_profile_words"] for item in rewritten_examples]
            ),
            "avg_word_reduction": average(
                [item["word_reduction"] for item in rewritten_examples]
            ),
            "data": rewritten_examples,
        }


def setup_logger(log_file: str) -> logging.Logger:
    logger_name = f"profile_meta_optimizer.{Path(log_file).resolve()}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s\n%(message)s\n"))
    logger.addHandler(handler)
    return logger


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AlignXada profile optimization with role-based backend configuration."
    )
    parser.add_argument("--initial-pattern", default=DEFAULT_INITIAL_PATTERN)
    parser.add_argument(
        "--task",
        required=True,
        help="Downstream task type. Current implementation: pairwise.",
    )
    parser.add_argument(
        "--profile-input",
        required=True,
        help="Path to a result JSON file used to sample support/query examples.",
    )
    parser.add_argument("--support-size", type=int, default=6)
    parser.add_argument(
        "--support-positive-count",
        type=int,
        default=0,
        help="Number of positive-bucket examples to place in the sampled support set.",
    )
    parser.add_argument(
        "--dev-support-size",
        type=int,
        default=0,
        help=(
            "Optional held-out support examples used only to select best_pattern. "
            "They are not used for meta updates."
        ),
    )
    parser.add_argument(
        "--dev-support-positive-count",
        type=int,
        default=None,
        help=(
            "Number of positive-bucket examples in the dev support set. "
            "Defaults to the same positive ratio as the train support set."
        ),
    )
    parser.add_argument("--query-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=2,
        help="Number of passes over the sampled support set.",
    )
    parser.add_argument(
        "--feedback-batch-size",
        type=int,
        default=3,
        help="Mini-batch size for support-set updates.",
    )
    parser.add_argument(
        "--best-pattern-strategy",
        choices=["best_epoch", "last_epoch"],
        default="best_epoch",
        help="How to choose the final best_pattern from epoch-level patterns.",
    )
    parser.add_argument(
        "--support-sampling-strategy",
        choices=SUPPORT_SAMPLING_STRATEGIES,
        default="random",
        help=(
            "How to sample support/dev examples from the candidate pool. "
            "pref_type_stratified balances pref_type, domain_stratified "
            "balances metadata.domain, and both avoid repeated "
            "personas within each sampled set when possible."
        ),
    )
    parser.add_argument(
        "--use-explicit-user-split",
        action="store_true",
        help=(
            "Use metadata.user_split to sample train support, dev support, and "
            "query sets. This is the normal split path for MemoryCD-style "
            "pre-built datasets."
        ),
    )
    parser.add_argument(
        "--debug-use-recorded-pairwise-sampling",
        action="store_true",
        help=(
            "Debug-only shortcut for pairwise tasks: during sampling, use each "
            "sample's answer/selection to determine positive/negative buckets "
            "instead of running original-profile downstream predictions. "
            "Original predictions for feedback are still recomputed later on the "
            "sampled support/query subsets."
        ),
    )
    add_role_args(
        parser=parser,
        prefix="meta",
        default_model=DEFAULT_META_MODEL,
        default_backend=BACKEND_OPENAI,
        default_temperature=0.4,
        default_max_output_tokens=800,
    )
    add_role_args(
        parser=parser,
        prefix="rewrite",
        default_model=DEFAULT_REWRITE_MODEL,
        default_backend=BACKEND_OPENAI,
        default_temperature=0.6,
        default_max_output_tokens=1024,
    )
    add_role_args(
        parser=parser,
        prefix="target",
        default_model=DEFAULT_TARGET_MODEL,
        default_backend=BACKEND_OPENAI,
        default_temperature=0.0,
        default_max_output_tokens=800,
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    profile_input_path = Path(args.profile_input)
    suffix = profile_input_path.suffix or ".json"
    output_dir = profile_input_path.parent / "res"
    output_dir.mkdir(parents=True, exist_ok=True)
    rewritten_query_output = (
        output_dir / f"{profile_input_path.stem}_rewritten_query{suffix}"
    )
    process_output_file = output_dir / f"{profile_input_path.stem}_process{suffix}"
    log_file = output_dir / f"{profile_input_path.stem}_optimizer.log"

    task_adapter = build_task_adapter(args.task)
    meta_spec = build_role_spec_from_args(args, prefix="meta")
    rewrite_spec = build_role_spec_from_args(args, prefix="rewrite")
    target_spec = build_role_spec_from_args(args, prefix="target")
    meta_generation_config = build_generation_config_from_args(args, prefix="meta")
    rewrite_generation_config = build_generation_config_from_args(
        args, prefix="rewrite"
    )
    target_generation_config = build_generation_config_from_args(args, prefix="target")

    meta_caller = build_model_caller(meta_spec)
    rewrite_caller = build_model_caller(rewrite_spec)
    target_caller = build_model_caller(target_spec)
    execution_engine = ProfileExecutionEngine(
        task_adapter=task_adapter,
        rewrite_caller=rewrite_caller,
        rewrite_model=rewrite_spec.model_name,
        rewrite_generation_config=rewrite_generation_config,
        target_caller=target_caller,
        target_model=target_spec.model_name,
        target_generation_config=target_generation_config,
    )
    optimizer = ProfileMetaOptimizer(
        task_adapter=task_adapter,
        execution_engine=execution_engine,
        meta_caller=meta_caller,
        meta_model=meta_spec.model_name,
        meta_generation_config=meta_generation_config,
        feedback_batch_size=args.feedback_batch_size,
        best_pattern_strategy=args.best_pattern_strategy,
    )

    result_examples = load_examples(
        str(profile_input_path),
        task_adapter=task_adapter,
        seed=args.seed,
    )
    split_sampling_without_full_baseline = (
        args.use_explicit_user_split or args.debug_use_recorded_pairwise_sampling
    )
    if not split_sampling_without_full_baseline:
        execution_engine.populate_original_baselines(result_examples)

    sampled = sample_support_and_query_sets(
        result_examples=result_examples,
        support_size=args.support_size,
        support_positive_count=args.support_positive_count,
        dev_support_size=args.dev_support_size,
        dev_support_positive_count=args.dev_support_positive_count,
        query_size=args.query_size,
        seed=args.seed,
        task_adapter=task_adapter,
        use_explicit_user_split=args.use_explicit_user_split,
        use_debug_recorded_sampling=args.debug_use_recorded_pairwise_sampling,
        support_sampling_strategy=args.support_sampling_strategy,
    )
    support_set = sampled["support_set"]
    dev_support_set = sampled["dev_support_set"]
    query_set = sampled["query_set"]
    sampled_summary = sampled["summary"]

    if split_sampling_without_full_baseline:
        execution_engine.populate_original_baselines(support_set)
        if dev_support_set:
            execution_engine.populate_original_baselines(dev_support_set)
        execution_engine.populate_original_baselines(query_set)

    logger = setup_logger(str(log_file))
    log_json(
        logger,
        {
            "event": "run_started",
            "task": task_adapter.name,
            "profile_input": args.profile_input,
            "debug_use_recorded_pairwise_sampling": (
                args.debug_use_recorded_pairwise_sampling
            ),
            "use_explicit_user_split": args.use_explicit_user_split,
            "support_size": len(support_set),
            "dev_support_size": len(dev_support_set),
            "query_size": len(query_set),
            "num_epochs": args.num_epochs,
            "feedback_batch_size": args.feedback_batch_size,
            "best_pattern_strategy": args.best_pattern_strategy,
            "support_sampling_strategy": args.support_sampling_strategy,
            "meta_spec": _redacted_spec(meta_spec),
            "rewrite_spec": _redacted_spec(rewrite_spec),
            "target_spec": _redacted_spec(target_spec),
        },
    )

    optimizer.logger = logger
    result = optimizer.run(
        support_set=support_set,
        initial_pattern=args.initial_pattern,
        num_epochs=args.num_epochs,
        dev_support_set=dev_support_set,
    )

    result["sampling"] = sampled_summary
    result["sampled_support_set"] = [
        task_adapter.serialize_example(item) for item in support_set
    ]
    result["sampled_dev_support_set"] = [
        task_adapter.serialize_example(item) for item in dev_support_set
    ]
    result["pattern_used_for_query_evaluation"] = result["best_pattern"]
    rewritten_query_result = optimizer.rewrite_examples(
        examples=query_set,
        pattern=result["best_pattern"],
    )

    result["rewritten_query_summary"] = {
        "pattern": rewritten_query_result["pattern"],
        "total_examples": rewritten_query_result["total_examples"],
        "avg_original_profile_words": rewritten_query_result[
            "avg_original_profile_words"
        ],
        "avg_optimized_profile_words": rewritten_query_result[
            "avg_optimized_profile_words"
        ],
        "avg_word_reduction": rewritten_query_result["avg_word_reduction"],
    }
    rewritten_query_output.write_text(
        json.dumps(rewritten_query_result["data"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result["rewritten_query_output"] = str(rewritten_query_output)
    result["log_file"] = str(log_file)
    log_json(
        logger,
        {
            "event": "run_finished",
            "task": task_adapter.name,
            "best_pattern": result["best_pattern"],
            "best_support_primary_score": result["best_support_primary_score"],
            "final_pattern": result["final_pattern"],
            "rewritten_query_output": result.get("rewritten_query_output"),
        },
    )

    Path(process_output_file).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

if __name__ == "__main__":
    main()
