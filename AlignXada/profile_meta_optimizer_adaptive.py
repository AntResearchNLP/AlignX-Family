import argparse
import json
import logging
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from profile_meta_core import (
    DEFAULT_INITIAL_PATTERN,
    DEFAULT_META_MODEL,
    DEFAULT_REWRITE_MODEL,
    DEFAULT_TARGET_MODEL,
    FeedbackExperience,
    SupportExample,
    average,
    set_seed,
)
from profile_meta_optimizer import (
    ProfileMetaOptimizer,
    _example_counts,
    _example_persona_id,
    _example_pref_type,
    _feedback_transition_summary,
    _redacted_spec,
    log_json,
    sample_support_and_query_sets,
    setup_logger,
)
from profile_meta_runtime import (
    BACKEND_OPENAI,
    ProfileExecutionEngine,
    add_role_args,
    build_generation_config_from_args,
    build_model_caller,
    build_role_spec_from_args,
)
from profile_meta_tasks import TaskAdapter, build_task_adapter, load_examples


ADAPTIVE_SELECTION_STRATEGIES = [
    "transition_balanced",
    "raw_role_weighted",
    "score_role_weighted",
]
ADAPTIVE_PROBE_SCOPES = ["pool", "remaining"]
ADAPTIVE_DEV_GATE_STRATEGIES = ["off", "strict_improve"]
ADAPTIVE_BUCKETS = [
    "policy_regression",
    "raw_regression",
    "persistent_failure",
    "improved",
    "stable_success",
]
RAW_ROLE_BUCKETS = [
    "regressed",
    "improved",
    "persistent_failure",
    "stable_success",
]
SCORE_ROLE_BUCKETS = [
    "score_regressed",
    "stable_low",
    "score_improved",
    "stable_high",
    "stable_mid",
]


@dataclass
class AdaptiveExampleState:
    key: str
    idx: Optional[int]
    pref_type: str
    persona_id: str
    original_passed: bool
    original_score: float
    current_passed: Optional[bool] = None
    current_score: Optional[float] = None
    raw_score_delta: Optional[float] = None
    policy_score_delta: Optional[float] = None
    score_level: str = "unknown"
    score_bucket: str = "unknown"
    raw_transition: str = "unknown"
    policy_transition: str = "unknown"
    selected_count: int = 0
    last_selected_step: int = -1
    last_evaluated_step: int = -1
    optimized_profile_words: Optional[int] = None


def _example_key(example: SupportExample, position: int) -> str:
    if example.idx is not None:
        return str(example.idx)
    return f"position:{position}"


def _passed(value: Any) -> bool:
    return bool(value)


def _transition_label(before_passed: bool, after_passed: bool) -> str:
    if not before_passed and after_passed:
        return "WR"
    if before_passed and not after_passed:
        return "RW"
    if before_passed and after_passed:
        return "RR"
    return "WW"


def _raw_transition_label(original_passed: bool, optimized_passed: bool) -> str:
    if not original_passed and optimized_passed:
        return "improved"
    if original_passed and not optimized_passed:
        return "regressed"
    if original_passed and optimized_passed:
        return "stable_success"
    return "persistent_failure"


def _score_transition_label(delta: Optional[float], eps: float) -> str:
    if delta is None:
        return "unknown"
    if delta > eps:
        return "score_improved"
    if delta < -eps:
        return "score_regressed"
    return "stable"


def _percentile(sorted_values: Sequence[float], percentile: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return float(sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction)


def _score_quantiles(states: Sequence[AdaptiveExampleState]) -> Dict[str, Optional[float]]:
    scores = sorted(
        float(item.current_score)
        for item in states
        if item.current_score is not None
    )
    return {
        "q_low": _percentile(scores, 0.25),
        "q_high": _percentile(scores, 0.75),
    }


def _refresh_score_buckets(
    states: Sequence[AdaptiveExampleState],
    score_eps: float,
) -> Dict[str, Optional[float]]:
    quantiles = _score_quantiles(states)
    q_low = quantiles["q_low"]
    q_high = quantiles["q_high"]
    for state in states:
        transition = _score_transition_label(state.raw_score_delta, score_eps)
        if state.current_score is None or q_low is None or q_high is None:
            state.score_level = "unknown"
            state.score_bucket = "unknown"
            continue
        if state.current_score <= q_low:
            state.score_level = "low"
        elif state.current_score >= q_high:
            state.score_level = "high"
        else:
            state.score_level = "mid"

        if transition != "stable":
            state.score_bucket = transition
        else:
            state.score_bucket = f"stable_{state.score_level}"
    return quantiles


def _default_score_eps(primary_metric_name: str) -> float:
    if primary_metric_name == "rating_score":
        return 0.025
    if primary_metric_name.startswith("rouge"):
        return 0.005
    return 1e-6


def _init_adaptive_states(
    examples: Sequence[SupportExample],
) -> Tuple[Dict[str, AdaptiveExampleState], Dict[str, SupportExample]]:
    states: Dict[str, AdaptiveExampleState] = {}
    examples_by_key: Dict[str, SupportExample] = {}
    for position, example in enumerate(examples):
        if example.original_evaluation is None:
            raise ValueError(
                "original_evaluation must be populated before adaptive training."
            )
        key = _example_key(example, position)
        if key in states:
            key = f"{key}#{position}"
        states[key] = AdaptiveExampleState(
            key=key,
            idx=example.idx,
            pref_type=_example_pref_type(example),
            persona_id=_example_persona_id(example),
            original_passed=_passed(example.original_evaluation.passed),
            original_score=example.original_evaluation.primary_score,
        )
        examples_by_key[key] = example
    return states, examples_by_key


def _feedback_key(feedback: FeedbackExperience, key_by_idx: Dict[Optional[int], str]) -> str:
    if feedback.idx in key_by_idx:
        return key_by_idx[feedback.idx]
    raise KeyError(f"Feedback example idx={feedback.idx!r} is not in adaptive pool.")


def _update_states_from_feedbacks(
    states: Dict[str, AdaptiveExampleState],
    feedbacks: Sequence[FeedbackExperience],
    key_by_idx: Dict[Optional[int], str],
    step: int,
) -> Counter:
    policy_transitions: Counter = Counter()
    for feedback in feedbacks:
        key = _feedback_key(feedback, key_by_idx)
        state = states[key]
        old_passed = state.current_passed
        old_score = state.current_score
        new_passed = _passed(feedback.optimized_passed)
        if old_passed is not None:
            policy_transitions.update([_transition_label(old_passed, new_passed)])
        state.current_passed = new_passed
        state.current_score = feedback.optimized_primary_score
        state.raw_score_delta = state.current_score - state.original_score
        state.policy_score_delta = (
            None if old_score is None else state.current_score - old_score
        )
        state.raw_transition = _raw_transition_label(
            state.original_passed, new_passed
        )
        state.policy_transition = (
            "initial"
            if old_passed is None
            else _transition_label(old_passed, new_passed)
        )
        state.last_evaluated_step = step
        state.optimized_profile_words = feedback.optimized_profile_words
    return policy_transitions


def _state_counts(states: Sequence[AdaptiveExampleState]) -> Dict[str, Any]:
    return {
        "raw_transition_counts": dict(
            sorted(Counter(item.raw_transition for item in states).items())
        ),
        "policy_transition_counts": dict(
            sorted(Counter(item.policy_transition for item in states).items())
        ),
        "score_bucket_counts": dict(
            sorted(Counter(item.score_bucket for item in states).items())
        ),
        "score_level_counts": dict(
            sorted(Counter(item.score_level for item in states).items())
        ),
        "pref_type_counts": dict(
            sorted(Counter(item.pref_type for item in states).items())
        ),
        "num_personas": len({item.persona_id for item in states}),
        "repeated_persona_count": sum(
            1
            for count in Counter(item.persona_id for item in states).values()
            if count > 1
        ),
        "avg_current_score": average(
            [
                item.current_score
                for item in states
                if item.current_score is not None
            ]
        ),
        "avg_raw_score_delta": average(
            [
                item.raw_score_delta
                for item in states
                if item.raw_score_delta is not None
            ]
        ),
        "avg_policy_score_delta": average(
            [
                item.policy_score_delta
                for item in states
                if item.policy_score_delta is not None
            ]
        ),
        "avg_optimized_profile_words": average(
            [
                item.optimized_profile_words
                for item in states
                if item.optimized_profile_words is not None
            ]
        ),
    }


def _weighted_quota_plan(
    batch_size: int,
    weights: Dict[str, float],
    bucket_order: Sequence[str],
) -> Dict[str, int]:
    raw_allocations = {
        name: batch_size * weights[name] for name in bucket_order
    }
    quotas = {name: int(raw_allocations[name]) for name in bucket_order}
    remaining = batch_size - sum(quotas.values())
    ranked_by_remainder = sorted(
        bucket_order,
        key=lambda name: (
            -(raw_allocations[name] - quotas[name]),
            bucket_order.index(name),
        ),
    )
    for name in ranked_by_remainder:
        if remaining <= 0:
            break
        quotas[name] += 1
        remaining -= 1
    return quotas


def _quota_plan(batch_size: int, selection_strategy: str) -> Dict[str, int]:
    if selection_strategy == "raw_role_weighted":
        return _weighted_quota_plan(
            batch_size=batch_size,
            weights={
                "regressed": 0.30,
                "improved": 0.25,
                "persistent_failure": 0.25,
                "stable_success": 0.20,
            },
            bucket_order=RAW_ROLE_BUCKETS,
        )
    if selection_strategy == "score_role_weighted":
        return _weighted_quota_plan(
            batch_size=batch_size,
            weights={
                "score_regressed": 0.40,
                "stable_low": 0.35,
                "score_improved": 0.20,
                "stable_high": 0.05,
                "stable_mid": 0.0,
            },
            bucket_order=SCORE_ROLE_BUCKETS,
        )
    if selection_strategy != "transition_balanced":
        raise ValueError(
            f"selection_strategy must be one of {ADAPTIVE_SELECTION_STRATEGIES}, "
            f"got {selection_strategy!r}."
        )
    return _weighted_quota_plan(
        batch_size=batch_size,
        weights={
            "policy_regression": 0.25,
            "raw_regression": 0.25,
            "persistent_failure": 0.3,
            "improved": 0.1,
            "stable_success": 0.1,
        },
        bucket_order=ADAPTIVE_BUCKETS,
    )


def _adaptive_bucket(state: AdaptiveExampleState) -> str:
    if state.policy_transition == "RW":
        return "policy_regression"
    if state.raw_transition == "regressed":
        return "raw_regression"
    if state.raw_transition in {"persistent_failure", "improved", "stable_success"}:
        return state.raw_transition
    return "unknown"


def _bucket_for_strategy(
    state: AdaptiveExampleState,
    selection_strategy: str,
) -> str:
    if selection_strategy == "raw_role_weighted":
        if state.raw_transition in RAW_ROLE_BUCKETS:
            return state.raw_transition
        return "unknown"
    if selection_strategy == "score_role_weighted":
        if state.score_bucket in SCORE_ROLE_BUCKETS:
            return state.score_bucket
        return "unknown"
    if selection_strategy == "transition_balanced":
        return _adaptive_bucket(state)
    raise ValueError(
        f"selection_strategy must be one of {ADAPTIVE_SELECTION_STRATEGIES}, "
        f"got {selection_strategy!r}."
    )


def _bucket_order_for_strategy(selection_strategy: str) -> List[str]:
    if selection_strategy == "raw_role_weighted":
        return list(RAW_ROLE_BUCKETS)
    if selection_strategy == "score_role_weighted":
        return list(SCORE_ROLE_BUCKETS)
    if selection_strategy == "transition_balanced":
        return list(ADAPTIVE_BUCKETS)
    raise ValueError(
        f"selection_strategy must be one of {ADAPTIVE_SELECTION_STRATEGIES}, "
        f"got {selection_strategy!r}."
    )


def _choose_balanced_states(
    candidates: Sequence[AdaptiveExampleState],
    quota: int,
    selected: List[AdaptiveExampleState],
    step: int,
    rng: random.Random,
) -> List[AdaptiveExampleState]:
    if quota <= 0:
        return []
    selected_keys = {item.key for item in selected}
    selected_personas = {item.persona_id for item in selected}
    selected_pref_counts = Counter(item.pref_type for item in selected)
    random_rank = {item.key: rng.random() for item in candidates}
    output: List[AdaptiveExampleState] = []

    def rank_item(item: AdaptiveExampleState) -> Tuple[int, int, float]:
        age = step - item.last_selected_step if item.last_selected_step >= 0 else 10_000
        return (item.selected_count, -age, random_rank[item.key])

    for allow_duplicate_persona in (False, True):
        while len(output) < quota:
            pool = [
                item
                for item in candidates
                if item.key not in selected_keys
                and item.key not in {chosen.key for chosen in output}
                and (
                    allow_duplicate_persona
                    or item.persona_id not in selected_personas
                )
            ]
            if not pool:
                break
            available_pref_counts = Counter(item.pref_type for item in pool)
            target_pref = min(
                available_pref_counts,
                key=lambda pref_type: (
                    selected_pref_counts[pref_type],
                    available_pref_counts[pref_type],
                    pref_type,
                ),
            )
            pref_pool = [item for item in pool if item.pref_type == target_pref]
            chosen = min(pref_pool, key=rank_item)
            output.append(chosen)
            selected_pref_counts.update([chosen.pref_type])
            selected_personas.add(chosen.persona_id)
        if len(output) == quota:
            break
    return output


def select_adaptive_batch(
    states: Dict[str, AdaptiveExampleState],
    examples_by_key: Dict[str, SupportExample],
    batch_size: int,
    step: int,
    rng: random.Random,
    selection_strategy: str,
    score_eps: float,
) -> Tuple[List[SupportExample], List[AdaptiveExampleState], Dict[str, Any]]:
    score_quantiles = (
        _refresh_score_buckets(list(states.values()), score_eps)
        if selection_strategy == "score_role_weighted"
        else {}
    )
    quotas = _quota_plan(batch_size, selection_strategy)
    bucket_order = _bucket_order_for_strategy(selection_strategy)
    selected_states: List[AdaptiveExampleState] = []
    all_states = list(states.values())

    for adaptive_bucket in bucket_order:
        bucket = [
            item
            for item in all_states
            if _bucket_for_strategy(item, selection_strategy) == adaptive_bucket
        ]
        selected_states.extend(
            _choose_balanced_states(
                candidates=bucket,
                quota=quotas[adaptive_bucket],
                selected=selected_states,
                step=step,
                rng=rng,
            )
        )

    if len(selected_states) < batch_size:
        remaining_quota = batch_size - len(selected_states)
        priority = {name: index for index, name in enumerate(bucket_order)}
        priority["unknown"] = len(priority)
        remaining_candidates = sorted(
            [
                item
                for item in all_states
                if item.key not in {selected.key for selected in selected_states}
            ],
            key=lambda item: (
                priority.get(_bucket_for_strategy(item, selection_strategy), 9),
                item.selected_count,
                item.last_selected_step if item.last_selected_step >= 0 else -10_000,
                rng.random(),
            ),
        )
        selected_states.extend(
            _choose_balanced_states(
                candidates=remaining_candidates,
                quota=remaining_quota,
                selected=selected_states,
                step=step,
                rng=rng,
            )
        )

    selected_states = selected_states[:batch_size]
    for state in selected_states:
        state.selected_count += 1
        state.last_selected_step = step

    selected_examples = [examples_by_key[state.key] for state in selected_states]
    summary = {
        "selection_strategy": selection_strategy,
        "quota_plan": quotas,
        "selected_keys": [state.key for state in selected_states],
        "selected_indices": [state.idx for state in selected_states],
        "selected_transition_counts": dict(
            sorted(Counter(state.raw_transition for state in selected_states).items())
        ),
        "selected_policy_transition_counts": dict(
            sorted(
                Counter(state.policy_transition for state in selected_states).items()
            )
        ),
        "selected_score_bucket_counts": dict(
            sorted(Counter(state.score_bucket for state in selected_states).items())
        ),
        "selected_score_level_counts": dict(
            sorted(Counter(state.score_level for state in selected_states).items())
        ),
        "selected_adaptive_bucket_counts": dict(
            sorted(
                Counter(
                    _bucket_for_strategy(state, selection_strategy)
                    for state in selected_states
                ).items()
            )
        ),
        "selected_pref_type_counts": dict(
            sorted(Counter(state.pref_type for state in selected_states).items())
        ),
        "selected_persona_count": len({state.persona_id for state in selected_states}),
        "selected_repeated_persona_count": sum(
            1
            for count in Counter(state.persona_id for state in selected_states).values()
            if count > 1
        ),
        "score_eps": score_eps,
        "score_quantiles": score_quantiles,
    }
    return selected_examples, selected_states, summary


def _serialize_state(state: AdaptiveExampleState) -> Dict[str, Any]:
    return asdict(state)


def _dev_primary_score(dev_summary: Optional[Dict[str, Any]]) -> Optional[float]:
    if dev_summary is None:
        return None
    value = dev_summary.get("dev_primary_metric_value")
    if value is None:
        return None
    return float(value)


def _dev_gate_decision(
    strategy: str,
    current_dev_summary: Optional[Dict[str, Any]],
    candidate_dev_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if strategy not in ADAPTIVE_DEV_GATE_STRATEGIES:
        raise ValueError(
            "dev_gate_strategy must be one of "
            f"{ADAPTIVE_DEV_GATE_STRATEGIES}, got {strategy!r}."
        )

    current_score = _dev_primary_score(current_dev_summary)
    candidate_score = _dev_primary_score(candidate_dev_summary)
    enabled = (
        strategy != "off"
        and current_score is not None
        and candidate_score is not None
    )

    if not enabled:
        return {
            "strategy": strategy,
            "enabled": False,
            "accepted": True,
            "reason": "disabled_or_no_dev_support",
            "current_dev_primary_metric_value": current_score,
            "candidate_dev_primary_metric_value": candidate_score,
            "score_delta": (
                candidate_score - current_score
                if current_score is not None and candidate_score is not None
                else None
            ),
        }

    if strategy == "strict_improve":
        accepted = candidate_score > current_score
        return {
            "strategy": strategy,
            "enabled": True,
            "accepted": accepted,
            "reason": "strict_improvement" if accepted else "no_strict_improvement",
            "current_dev_primary_metric_value": current_score,
            "candidate_dev_primary_metric_value": candidate_score,
            "score_delta": candidate_score - current_score,
        }

    raise ValueError(f"Unsupported dev_gate_strategy {strategy!r}.")


def _candidate_bank_selection_key(
    item: Dict[str, Any],
) -> Tuple[float, int, int, float, int]:
    primary_score = item.get("dev_primary_metric_value")
    if primary_score is None:
        primary_score = item.get(
            "probe_primary_metric_value",
            item.get("pool_primary_metric_value", 0.0),
        )

    right_to_wrong = item.get("dev_right_to_wrong")
    if right_to_wrong is None:
        right_to_wrong = item.get("probe_right_to_wrong", 0)
    wrong_to_right = item.get("dev_wrong_to_right")
    if wrong_to_right is None:
        wrong_to_right = item.get("probe_wrong_to_right", 0)
    avg_words = item.get("dev_avg_optimized_profile_words")
    if avg_words is None:
        avg_words = item.get("probe_avg_optimized_profile_words", 1_000_000.0)

    return (
        float(primary_score),
        -int(right_to_wrong or 0),
        int(wrong_to_right or 0),
        -float(avg_words),
        int(item.get("step", -1)),
    )


class AdaptiveProfileMetaOptimizer(ProfileMetaOptimizer):
    def evaluate_adaptive_pool(
        self,
        examples: Sequence[SupportExample],
        pattern: str,
        states: Dict[str, AdaptiveExampleState],
        key_by_idx: Dict[Optional[int], str],
        step: int,
        score_eps: float,
    ) -> Dict[str, Any]:
        result = self.collect_feedback(examples, pattern)
        policy_transitions = _update_states_from_feedbacks(
            states=states,
            feedbacks=result["feedbacks"],
            key_by_idx=key_by_idx,
            step=step,
        )
        score_quantiles = _refresh_score_buckets(list(states.values()), score_eps)
        result_summary = {
            "primary_metric_name": result["primary_metric_name"],
            "primary_metric_value": result["primary_metric_value"],
            "metrics": result["metrics"],
            "total": result["total"],
            "num_errors": result["num_errors"],
            "avg_original_profile_words": result["avg_original_profile_words"],
            "avg_optimized_profile_words": result["avg_optimized_profile_words"],
            "avg_word_reduction": result["avg_word_reduction"],
            "policy_transition_counts": dict(sorted(policy_transitions.items())),
            "score_eps": score_eps,
            "score_quantiles": score_quantiles,
            "score_bucket_counts": dict(
                sorted(Counter(item.score_bucket for item in states.values()).items())
            ),
            "score_level_counts": dict(
                sorted(Counter(item.score_level for item in states.values()).items())
            ),
            **_feedback_transition_summary(result["feedbacks"], prefix="probe"),
        }
        return result_summary

    def run_adaptive(
        self,
        adaptive_pool_set: Sequence[SupportExample],
        initial_pattern: str,
        num_updates: int,
        dev_support_set: Optional[Sequence[SupportExample]] = None,
        selection_strategy: str = "raw_role_weighted",
        probe_scope: str = "pool",
        dev_gate_strategy: str = "strict_improve",
        score_eps: Optional[float] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        if selection_strategy not in ADAPTIVE_SELECTION_STRATEGIES:
            raise ValueError(
                "selection_strategy must be one of "
                f"{ADAPTIVE_SELECTION_STRATEGIES}, got {selection_strategy!r}."
            )
        if probe_scope not in ADAPTIVE_PROBE_SCOPES:
            raise ValueError(
                f"probe_scope must be one of {ADAPTIVE_PROBE_SCOPES}, got {probe_scope!r}."
            )
        if dev_gate_strategy not in ADAPTIVE_DEV_GATE_STRATEGIES:
            raise ValueError(
                "dev_gate_strategy must be one of "
                f"{ADAPTIVE_DEV_GATE_STRATEGIES}, got {dev_gate_strategy!r}."
            )
        if not adaptive_pool_set:
            raise ValueError("adaptive_pool_set must not be empty.")
        if num_updates <= 0:
            raise ValueError("num_updates must be positive.")

        rng = random.Random(seed)
        current_pattern = initial_pattern
        effective_score_eps = (
            _default_score_eps(self.task_adapter.primary_metric_name)
            if score_eps is None
            else score_eps
        )
        states, examples_by_key = _init_adaptive_states(adaptive_pool_set)
        key_by_idx = {
            example.idx: key for key, example in examples_by_key.items()
        }
        adaptive_steps: List[Dict[str, Any]] = []
        candidate_patterns: List[Dict[str, Any]] = []

        initial_pool_summary = self.evaluate_adaptive_pool(
            examples=adaptive_pool_set,
            pattern=current_pattern,
            states=states,
            key_by_idx=key_by_idx,
            step=-1,
            score_eps=effective_score_eps,
        )
        initial_dev_summary = self.evaluate_pattern_on_dev_support(
            dev_support_set or [], current_pattern
        )
        current_dev_summary = initial_dev_summary
        candidate_patterns.append(
            {
                "step": -1,
                "pattern_after_step": current_pattern,
                "probe_primary_metric_value": initial_pool_summary[
                    "primary_metric_value"
                ],
                "probe_avg_optimized_profile_words": initial_pool_summary[
                    "avg_optimized_profile_words"
                ],
                "probe_avg_word_reduction": initial_pool_summary[
                    "avg_word_reduction"
                ],
                "probe_transition_counts": initial_pool_summary.get(
                    "probe_transition_counts", {}
                ),
                "probe_right_to_wrong": initial_pool_summary.get(
                    "probe_right_to_wrong", 0
                ),
                "probe_wrong_to_right": initial_pool_summary.get(
                    "probe_wrong_to_right", 0
                ),
                "pool_primary_metric_value": initial_pool_summary[
                    "primary_metric_value"
                ],
                "pool_avg_optimized_profile_words": initial_pool_summary[
                    "avg_optimized_profile_words"
                ],
                "pool_avg_word_reduction": initial_pool_summary[
                    "avg_word_reduction"
                ],
                **(initial_dev_summary or {}),
            }
        )

        progress_bar = tqdm(total=num_updates, desc="Adaptive updates", unit="step")
        for step in range(num_updates):
            pattern_before_step = current_pattern
            batch, batch_states, batch_summary = select_adaptive_batch(
                states=states,
                examples_by_key=examples_by_key,
                batch_size=self.feedback_batch_size,
                step=step,
                rng=rng,
                selection_strategy=selection_strategy,
                score_eps=effective_score_eps,
            )
            forward_result = self.collect_feedback(batch, current_pattern)
            _update_states_from_feedbacks(
                states=states,
                feedbacks=forward_result["feedbacks"],
                key_by_idx=key_by_idx,
                step=step,
            )
            update_result = self.textual_gradient_step(
                current_pattern, forward_result["feedbacks"]
            )
            next_pattern = update_result["next_pattern"]
            candidate_dev_summary: Optional[Dict[str, Any]] = None
            if dev_support_set and dev_gate_strategy != "off":
                candidate_dev_summary = self.evaluate_pattern_on_dev_support(
                    dev_support_set, next_pattern
                )
            gate_decision = _dev_gate_decision(
                strategy=dev_gate_strategy,
                current_dev_summary=current_dev_summary,
                candidate_dev_summary=candidate_dev_summary,
            )
            if gate_decision["accepted"]:
                current_pattern = next_pattern
                if dev_support_set:
                    current_dev_summary = (
                        candidate_dev_summary
                        if candidate_dev_summary is not None
                        else self.evaluate_pattern_on_dev_support(
                            dev_support_set, current_pattern
                        )
                    )
            deployed_dev_summary = current_dev_summary

            if probe_scope == "remaining":
                batch_keys = {state.key for state in batch_states}
                probe_examples = [
                    example
                    for key, example in examples_by_key.items()
                    if key not in batch_keys
                ]
            else:
                probe_examples = list(adaptive_pool_set)
            probe_summary = self.evaluate_adaptive_pool(
                examples=probe_examples,
                pattern=current_pattern,
                states=states,
                key_by_idx=key_by_idx,
                step=step,
                score_eps=effective_score_eps,
            )
            dev_summary = deployed_dev_summary
            pool_state_summary = _state_counts(list(states.values()))

            step_record = {
                "step": step,
                "pattern_before_step": pattern_before_step,
                "candidate_pattern": next_pattern,
                "batch": batch_summary,
                "batch_primary_metric_name": forward_result["primary_metric_name"],
                "batch_primary_metric_value": forward_result["primary_metric_value"],
                "batch_metrics": forward_result["metrics"],
                "batch_total": forward_result["total"],
                "batch_num_errors": forward_result["num_errors"],
                "batch_avg_original_profile_words": forward_result[
                    "avg_original_profile_words"
                ],
                "batch_avg_optimized_profile_words": forward_result[
                    "avg_optimized_profile_words"
                ],
                "batch_avg_word_reduction": forward_result["avg_word_reduction"],
                "probe_scope": probe_scope,
                "probe_indices": [example.idx for example in probe_examples],
                "probe": probe_summary,
                "pool_state": pool_state_summary,
                "dev_gate": gate_decision,
                "candidate_dev_support": candidate_dev_summary,
                "dev_support": dev_summary,
                "next_pattern": next_pattern,
                "pattern_after_step": current_pattern,
            }
            adaptive_steps.append(step_record)
            candidate_patterns.append(
                {
                    "step": step,
                    "pattern_after_step": current_pattern,
                    "probe_primary_metric_value": probe_summary[
                        "primary_metric_value"
                    ],
                    "probe_avg_optimized_profile_words": probe_summary[
                        "avg_optimized_profile_words"
                    ],
                    "probe_avg_word_reduction": probe_summary[
                        "avg_word_reduction"
                    ],
                    "probe_transition_counts": probe_summary.get(
                        "probe_transition_counts", {}
                    ),
                    "probe_right_to_wrong": probe_summary.get(
                        "probe_right_to_wrong", 0
                    ),
                    "probe_wrong_to_right": probe_summary.get(
                        "probe_wrong_to_right", 0
                    ),
                    "pool_primary_metric_value": pool_state_summary[
                        "avg_current_score"
                    ],
                    "pool_avg_optimized_profile_words": pool_state_summary[
                        "avg_optimized_profile_words"
                    ],
                    "pool_avg_word_reduction": probe_summary[
                        "avg_word_reduction"
                    ],
                    "candidate_pattern": next_pattern,
                    "dev_gate": gate_decision,
                    **(dev_summary or {}),
                }
            )

            if self.logger is not None:
                log_json(
                    self.logger,
                    {
                        "event": "adaptive_step",
                        "step": step,
                        "batch": batch_summary,
                        "batch_primary_metric_value": forward_result[
                            "primary_metric_value"
                        ],
                        "probe_primary_metric_value": probe_summary[
                            "primary_metric_value"
                        ],
                        "dev_gate": gate_decision,
                        "pool_state": pool_state_summary,
                    },
                )

            progress_bar.update(1)
            progress_bar.set_postfix(
                batch=f"{forward_result['primary_metric_value']:.4f}",
                probe=f"{probe_summary['primary_metric_value']:.4f}",
            )
        progress_bar.close()

        if self.best_pattern_strategy == "last_epoch":
            selected = candidate_patterns[-1]
            selection_source = "last_update"
        elif self.best_pattern_strategy == "candidate_bank":
            selected = max(candidate_patterns, key=_candidate_bank_selection_key)
            selection_source = "candidate_bank"
        elif dev_support_set:
            selected = max(
                candidate_patterns,
                key=lambda item: (
                    item.get("dev_primary_metric_value", 0.0),
                    item["step"],
                ),
            )
            selection_source = "dev_support"
        else:
            selected = max(
                candidate_patterns,
                key=lambda item: (
                    item.get(
                        "probe_primary_metric_value",
                        item.get("pool_primary_metric_value", 0.0),
                    ),
                    item["step"],
                ),
            )
            selection_source = "adaptive_probe"

        return {
            "task": self.task_adapter.name,
            "task_description": self.task_adapter.task_description,
            "primary_metric_name": self.task_adapter.primary_metric_name,
            "rewrite_model": self.rewrite_model,
            "target_model": self.target_model,
            "meta_model": self.meta_model,
            "initial_pattern": initial_pattern,
            "final_pattern": current_pattern,
            "next_pattern_suggestion": adaptive_steps[-1]["next_pattern"]
            if adaptive_steps
            else None,
            "best_pattern_strategy": self.best_pattern_strategy,
            "best_pattern_selection_source": selection_source,
            "best_pattern": selected["pattern_after_step"],
            "best_adaptive_pool_primary_score": selected.get(
                "pool_primary_metric_value"
            ),
            "best_adaptive_pool_avg_optimized_profile_words": selected.get(
                "pool_avg_optimized_profile_words"
            ),
            "best_adaptive_pool_avg_word_reduction": selected.get(
                "pool_avg_word_reduction"
            ),
            "best_dev_support_primary_score": selected.get(
                "dev_primary_metric_value"
            ),
            "best_dev_support_avg_optimized_profile_words": selected.get(
                "dev_avg_optimized_profile_words"
            ),
            "best_dev_support_avg_word_reduction": selected.get(
                "dev_avg_word_reduction"
            ),
            "best_dev_support_transition_counts": selected.get(
                "dev_transition_counts"
            ),
            "best_dev_support_right_to_wrong": selected.get("dev_right_to_wrong"),
            "best_dev_support_wrong_to_right": selected.get("dev_wrong_to_right"),
            "best_probe_transition_counts": selected.get("probe_transition_counts"),
            "best_probe_right_to_wrong": selected.get("probe_right_to_wrong"),
            "best_probe_wrong_to_right": selected.get("probe_wrong_to_right"),
            "adaptive_config": {
                "selection_strategy": selection_strategy,
                "probe_scope": probe_scope,
                "dev_gate_strategy": dev_gate_strategy,
                "score_eps": effective_score_eps,
                "num_updates": num_updates,
                "feedback_batch_size": self.feedback_batch_size,
                "quota_plan": _quota_plan(
                    self.feedback_batch_size,
                    selection_strategy,
                ),
            },
            "initial_adaptive_pool_summary": initial_pool_summary,
            "initial_dev_support_summary": initial_dev_summary,
            "accepted_update_count": sum(
                1
                for step in adaptive_steps
                if step["dev_gate"]["accepted"]
            ),
            "rejected_update_count": sum(
                1
                for step in adaptive_steps
                if step["dev_gate"]["enabled"]
                and not step["dev_gate"]["accepted"]
            ),
            "candidate_patterns": candidate_patterns,
            "adaptive_steps": adaptive_steps,
            "adaptive_pool_state": [
                _serialize_state(state)
                for state in sorted(states.values(), key=lambda item: item.key)
            ],
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive support sampling for the AlignXada profile optimizer. "
            "This entrypoint is isolated from profile_meta_optimizer.py."
        )
    )
    parser.add_argument("--initial-pattern", default=DEFAULT_INITIAL_PATTERN)
    parser.add_argument(
        "--task",
        required=True,
        help="Downstream task type.",
    )
    parser.add_argument(
        "--profile-input",
        required=True,
        help="Path to a result JSON file used to sample adaptive pool/query examples.",
    )
    parser.add_argument(
        "--support-size",
        type=int,
        default=40,
        help="Adaptive support pool size.",
    )
    parser.add_argument(
        "--support-positive-count",
        type=int,
        default=0,
        help="Number of positive-bucket examples to place in the adaptive pool.",
    )
    parser.add_argument(
        "--dev-support-size",
        type=int,
        default=0,
        help="Optional held-out support examples used only to select best_pattern.",
    )
    parser.add_argument(
        "--dev-support-positive-count",
        type=int,
        default=None,
        help="Number of positive-bucket examples in the dev support set.",
    )
    parser.add_argument("--query-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help=(
            "Default number of adaptive update rounds. For this adaptive entrypoint, "
            "this is not a full pass over support_size."
        ),
    )
    parser.add_argument(
        "--adaptive-num-updates",
        type=int,
        default=None,
        help="Explicit number of adaptive update rounds. Defaults to --num-epochs.",
    )
    parser.add_argument(
        "--feedback-batch-size",
        type=int,
        default=10,
        help="Adaptive batch size for each support update.",
    )
    parser.add_argument(
        "--adaptive-probe-scope",
        choices=ADAPTIVE_PROBE_SCOPES,
        default="pool",
        help=(
            "Examples to re-evaluate after each update. 'pool' keeps state fresh; "
            "'remaining' follows the held-out-within-pool proposal."
        ),
    )
    parser.add_argument(
        "--adaptive-selection-strategy",
        choices=ADAPTIVE_SELECTION_STRATEGIES,
        default="raw_role_weighted",
        help=(
            "How to sample each adaptive feedback batch. "
            "'transition_balanced' is the original mixed raw/policy-transition "
            "strategy. 'raw_role_weighted' uses raw transitions only with "
            "regressed/improved/persistent_failure/stable_success weights "
            "0.30/0.25/0.25/0.20. 'score_role_weighted' uses primary-score "
            "deltas and score quantiles for continuous metrics."
        ),
    )
    parser.add_argument(
        "--adaptive-score-eps",
        type=float,
        default=None,
        help=(
            "Tolerance for score_role_weighted transitions. Defaults to a "
            "metric-specific value: 0.025 for rating_score, 0.005 for rouge, "
            "and 1e-6 otherwise."
        ),
    )
    parser.add_argument(
        "--adaptive-dev-gate",
        choices=ADAPTIVE_DEV_GATE_STRATEGIES,
        default="strict_improve",
        help=(
            "Whether to gate each candidate policy on the held-out dev support "
            "set before deploying it. 'strict_improve' accepts a candidate only "
            "when its dev primary metric is strictly higher than the currently "
            "deployed policy. 'off' preserves the previous unconditional update "
            "behavior."
        ),
    )
    parser.add_argument(
        "--best-pattern-strategy",
        choices=["best_epoch", "last_epoch", "candidate_bank"],
        default="best_epoch",
        help=(
            "How to choose final best_pattern from adaptive updates. "
            "'candidate_bank' ranks all candidates by dev/probe accuracy, "
            "then fewer raw-correct-to-optimized-wrong regressions, then more "
            "wrong-to-right fixes, then shorter rewritten profiles."
        ),
    )
    parser.add_argument(
        "--support-sampling-strategy",
        choices=["random", "pref_type_stratified", "domain_stratified"],
        default="pref_type_stratified",
        help="How to sample the initial adaptive pool/dev/query sets.",
    )
    parser.add_argument(
        "--use-explicit-user-split",
        action="store_true",
        help=(
            "Use metadata.user_split to sample adaptive pool, dev support, "
            "and query sets from pre-built three-way datasets."
        ),
    )
    parser.add_argument(
        "--debug-use-recorded-pairwise-sampling",
        action="store_true",
        help="Debug-only shortcut that samples buckets from recorded answer/selection.",
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
        output_dir / f"{profile_input_path.stem}_adaptive_rewritten_query{suffix}"
    )
    process_output_file = (
        output_dir / f"{profile_input_path.stem}_adaptive_process{suffix}"
    )
    log_file = output_dir / f"{profile_input_path.stem}_adaptive_optimizer.log"

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
    optimizer = AdaptiveProfileMetaOptimizer(
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
    adaptive_pool_set = sampled["support_set"]
    dev_support_set = sampled["dev_support_set"]
    query_set = sampled["query_set"]
    sampled_summary = sampled["summary"]

    if split_sampling_without_full_baseline:
        execution_engine.populate_original_baselines(adaptive_pool_set)
        if dev_support_set:
            execution_engine.populate_original_baselines(dev_support_set)
        execution_engine.populate_original_baselines(query_set)

    logger = setup_logger(str(log_file))
    log_json(
        logger,
        {
            "event": "adaptive_run_started",
            "task": task_adapter.name,
            "profile_input": args.profile_input,
            "debug_use_recorded_pairwise_sampling": (
                args.debug_use_recorded_pairwise_sampling
            ),
            "use_explicit_user_split": args.use_explicit_user_split,
            "adaptive_pool_size": len(adaptive_pool_set),
            "adaptive_pool_distribution": _example_counts(adaptive_pool_set),
            "dev_support_size": len(dev_support_set),
            "query_size": len(query_set),
            "num_epochs": args.num_epochs,
            "adaptive_num_updates": args.adaptive_num_updates,
            "feedback_batch_size": args.feedback_batch_size,
            "best_pattern_strategy": args.best_pattern_strategy,
            "support_sampling_strategy": args.support_sampling_strategy,
            "adaptive_probe_scope": args.adaptive_probe_scope,
            "adaptive_selection_strategy": args.adaptive_selection_strategy,
            "adaptive_score_eps": args.adaptive_score_eps,
            "adaptive_dev_gate": args.adaptive_dev_gate,
            "meta_spec": _redacted_spec(meta_spec),
            "rewrite_spec": _redacted_spec(rewrite_spec),
            "target_spec": _redacted_spec(target_spec),
        },
    )

    optimizer.logger = logger
    num_updates = args.adaptive_num_updates or args.num_epochs
    result = optimizer.run_adaptive(
        adaptive_pool_set=adaptive_pool_set,
        initial_pattern=args.initial_pattern,
        num_updates=num_updates,
        dev_support_set=dev_support_set,
        selection_strategy=args.adaptive_selection_strategy,
        probe_scope=args.adaptive_probe_scope,
        dev_gate_strategy=args.adaptive_dev_gate,
        score_eps=args.adaptive_score_eps,
        seed=args.seed,
    )

    result["sampling"] = sampled_summary
    result["sampled_adaptive_pool_set"] = [
        task_adapter.serialize_example(item) for item in adaptive_pool_set
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
            "event": "adaptive_run_finished",
            "task": task_adapter.name,
            "best_pattern": result["best_pattern"],
            "best_pattern_selection_source": result[
                "best_pattern_selection_source"
            ],
            "best_adaptive_pool_primary_score": result[
                "best_adaptive_pool_primary_score"
            ],
            "final_pattern": result["final_pattern"],
            "rewritten_query_output": result.get("rewritten_query_output"),
        },
    )

    Path(process_output_file).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
