"""MemoryCD loader/adapter for profile-refinement experiments.

MemoryCD is an Amazon-review memory benchmark with four downstream task forms:
rating prediction, item ranking, review-title generation, and review generation.
This adapter treats a user's historical reviews as the raw universal profile and
tests whether a refined profile helps a frozen downstream model solve the target
task.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from profile_meta_core import (
    PROFILE_PLACEHOLDER,
    EvaluationResult,
    PredictionResult,
    SupportExample,
    normalize_label,
    parse_json_object,
    serialize_evaluation,
    serialize_prediction,
)
from profile_meta_tasks import TaskAdapter, compute_rouge


MEMORYCD_TASKS = (
    "rating_prediction",
    "item_ranking",
    "review_title",
    "review_generation",
)


_MEMORYCD_DESCRIPTION = (
    "The downstream benchmark is MemoryCD, where a user's past product-review "
    "history is used as memory for personalized product tasks. A rewritten "
    "profile should preserve source-supported purchase/review preferences, "
    "rating tendencies, category interests, item attributes, style cues, and "
    "cross-domain preference signals that can help future rating prediction, "
    "item ranking, or review generation. Domain labels describe where history "
    "and target items come from; they are not themselves rewrite instructions."
)

_MEMORYCD_RATING_DESCRIPTION = (
    _MEMORYCD_DESCRIPTION
    + "\nCurrent MemoryCD task: rating_prediction. The downstream model predicts "
    "the user's 1-5 star rating for a target product, so source-supported rating "
    "calibration evidence is task-relevant."
)


@dataclass
class _MemoryCDTarget:
    task: str
    user_id: str
    domain: str
    target_item: Dict[str, Any]
    target_interaction: Dict[str, Any]
    candidate_items: List[Dict[str, Any]]


def _compact_text(value: Any, max_chars: int = 320) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,.;:") + "..."


def _first_present(mapping: Dict[str, Any], keys: Sequence[str], default: Any = "") -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", [], {}):
            return value
    return default


def _item_id(item: Dict[str, Any]) -> str:
    return str(
        _first_present(
            item,
            ("item_id", "asin", "parent_asin", "product_id", "id"),
            default="",
        )
    )


def _item_title(item: Dict[str, Any]) -> str:
    return _compact_text(
        _first_present(item, ("title", "item_title", "name", "product_title"), "")
    )


def _item_description(item: Dict[str, Any]) -> str:
    description = _first_present(
        item,
        ("description", "features", "feature", "details", "categories"),
        default="",
    )
    if isinstance(description, (list, tuple)):
        description = "; ".join(str(x) for x in description[:5])
    return _compact_text(description, max_chars=420)


def _rating(value: Any) -> Optional[float]:
    try:
        rating = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(rating):
        return None
    return min(5.0, max(1.0, rating))


def _interaction_rating(interaction: Dict[str, Any]) -> Optional[float]:
    return _rating(
        _first_present(interaction, ("rating", "overall", "stars", "score"), None)
    )


def _interaction_review(interaction: Dict[str, Any]) -> str:
    return _compact_text(
        _first_present(
            interaction,
            ("review_text", "text", "review", "content", "body"),
            default="",
        ),
        max_chars=900,
    )


def _interaction_title(interaction: Dict[str, Any]) -> str:
    return _compact_text(
        _first_present(
            interaction,
            ("review_title", "summary", "title", "headline"),
            default="",
        ),
        max_chars=180,
    )


def _format_item(item: Dict[str, Any], index: Optional[int] = None) -> str:
    item_id = _item_id(item) or "unknown"
    title = _item_title(item)
    domain = str(_first_present(item, ("domain", "category", "main_category"), ""))
    desc = _item_description(item)
    prefix = f"{index}. " if index is not None else ""
    lines = [f"{prefix}ID: {item_id}"]
    if title:
        lines.append(f"Title: {title}")
    if domain:
        lines.append(f"Domain: {domain}")
    if desc:
        lines.append(f"Details: {desc}")
    return "\n".join(lines)


def _extract_target(item: Dict[str, Any], adapter_task: Optional[str]) -> _MemoryCDTarget:
    target = item.get("target")
    if not isinstance(target, dict):
        raise ValueError("MemoryCD record requires a target object.")
    task = str(target.get("task") or item.get("task") or adapter_task or "").strip()
    if task not in MEMORYCD_TASKS:
        raise ValueError(f"Unsupported MemoryCD task {task!r}; expected {MEMORYCD_TASKS}.")
    target_item = target.get("target_item") or target.get("item") or {}
    if not isinstance(target_item, dict):
        raise ValueError("MemoryCD target.target_item must be an object.")
    target_interaction = (
        target.get("target_interaction")
        or target.get("interaction")
        or item.get("target_interaction")
        or {}
    )
    if not isinstance(target_interaction, dict):
        target_interaction = {}
    candidates = target.get("candidate_items") or []
    if not isinstance(candidates, list):
        candidates = []
    return _MemoryCDTarget(
        task=task,
        user_id=str(target.get("user_id") or item.get("user_id") or ""),
        domain=str(target.get("domain") or item.get("domain") or ""),
        target_item=target_item,
        target_interaction=target_interaction,
        candidate_items=[c for c in candidates if isinstance(c, dict)],
    )


class MemoryCDAdapter(TaskAdapter):
    """TaskAdapter for MemoryCD.

    The adapter can be bound to a specific task via task keys such as
    ``memorycd_rating_prediction``. The generic ``memorycd`` key reads the task
    from each record's target object.
    """

    name = "memorycd"
    task_description = _MEMORYCD_DESCRIPTION
    positive_bucket = "support"
    negative_bucket = "query"

    def __init__(self, task: Optional[str] = None) -> None:
        if task and task not in MEMORYCD_TASKS:
            raise ValueError(f"Unsupported MemoryCD task {task!r}.")
        self.task = task
        if task:
            self.name = f"memorycd_{task}"
        self.task_description = (
            _MEMORYCD_RATING_DESCRIPTION
            if task == "rating_prediction"
            else _MEMORYCD_DESCRIPTION
        )
        self.primary_metric_name = {
            "rating_prediction": "rating_score",
            "item_ranking": "hit@1",
            "review_title": "rouge-l",
            "review_generation": "rouge-l",
        }.get(task or "", "score")
        self.expects_json_prediction = task != "review_generation"
        if task == "review_generation":
            self.prediction_instruction = (
                "You are the frozen downstream model for a MemoryCD personalized "
                "product-review generation task. Use the user memory/profile and "
                "the target item information. Return only the generated review "
                "text, with no JSON, markdown, or extra commentary."
            )
        else:
            self.prediction_instruction = (
                "You are the frozen downstream model for a MemoryCD personalized "
                "product-memory task. Use the user memory/profile and the target "
                "item information. Return only the JSON object requested by the user "
                "message, with no markdown or extra text."
            )

    def format_prompt(self, target: _MemoryCDTarget, profile_text: str) -> str:
        if target.task == "rating_prediction":
            return (
                "Predict how this user would rate the target product on a 1-5 "
                "star scale.\n\n"
                f"<UserMemory>\n{profile_text}\n</UserMemory>\n\n"
                f"<TargetItem>\n{_format_item(target.target_item)}\n</TargetItem>\n\n"
                'Return ONLY JSON: {"rating": <integer from 1 to 5>}.'
            )

        if target.task == "item_ranking":
            if not target.candidate_items:
                raise ValueError("item_ranking requires candidate_items.")
            candidates = "\n\n".join(
                _format_item(candidate, index=i + 1)
                for i, candidate in enumerate(target.candidate_items)
            )
            candidate_ids = [_item_id(candidate) for candidate in target.candidate_items]
            return (
                "Rank the candidate products from most to least likely to match "
                "this user's preferences or positive engagement.\n\n"
                f"<UserMemory>\n{profile_text}\n</UserMemory>\n\n"
                f"<CandidateItems>\n{candidates}\n</CandidateItems>\n\n"
                f"Candidate IDs: {json.dumps(candidate_ids, ensure_ascii=False)}\n"
                'Return ONLY JSON: {"ranked_item_ids": ["<candidate id>", "..."]}. '
                "Use exact IDs from Candidate IDs, not numbers or titles."
            )

        if target.task == "review_title":
            review_text = _interaction_review(target.target_interaction)
            return (
                "Write the short review title this user would likely use for the "
                "given review and product.\n\n"
                f"<UserMemory>\n{profile_text}\n</UserMemory>\n\n"
                f"<TargetItem>\n{_format_item(target.target_item)}\n</TargetItem>\n\n"
                f"<ReviewBody>\n{review_text}\n</ReviewBody>\n\n"
                'Return ONLY JSON: {"title": "<review title>"}.'
            )

        if target.task == "review_generation":
            rating = _interaction_rating(target.target_interaction)
            rating_line = f"\nExpected rating: {rating:g} stars" if rating else ""
            return (
                "Generate a product review that matches this user's likely review "
                "style and preferences for the target product.\n\n"
                f"<UserMemory>\n{profile_text}\n</UserMemory>\n\n"
                f"<TargetItem>\n{_format_item(target.target_item)}{rating_line}\n"
                "</TargetItem>\n\n"
                "Keep the review length close to the user's historical review "
                "style and the typical target-review length for this benchmark; "
                "do not expand unnecessarily.\n"
                "Return only the review text. Do not wrap it in JSON."
            )

        raise ValueError(f"Unsupported MemoryCD task {target.task!r}.")

    def load_example(
        self,
        item: Dict[str, Any],
        row_idx: int,
        rng: Any,
    ) -> SupportExample:
        del rng
        if "profile" not in item:
            raise ValueError(f"Missing profile in MemoryCD row {row_idx}.")
        target = _extract_target(item, self.task)
        prompt_template = self.format_prompt(target, PROFILE_PLACEHOLDER)
        reference = self._reference_from_target(target)
        return SupportExample(
            profile=item["profile"],
            prompt=prompt_template,
            reference=reference,
            idx=int(item.get("idx", row_idx)),
            metadata={
                "task": self.name,
                "memorycd_task": target.task,
                "user_id": target.user_id,
                "user_split": (item.get("metadata") or {}).get("user_split"),
                "memory_source": (item.get("metadata") or {}).get("memory_source"),
                "domain": target.domain,
                "target_item_id": _item_id(target.target_item),
                "target_item_title": _item_title(target.target_item),
                "candidate_item_ids": [
                    _item_id(candidate) for candidate in target.candidate_items
                ],
            },
        )

    def _reference_from_target(self, target: _MemoryCDTarget) -> Any:
        if target.task == "rating_prediction":
            rating = _interaction_rating(target.target_interaction)
            if rating is None:
                raise ValueError("rating_prediction requires a target rating.")
            return int(round(rating))
        if target.task == "item_ranking":
            target_id = _item_id(target.target_item)
            if not target_id:
                raise ValueError("item_ranking requires a target item id.")
            return target_id
        if target.task == "review_title":
            title = _interaction_title(target.target_interaction)
            if not title:
                raise ValueError("review_title requires a target review title.")
            return title
        if target.task == "review_generation":
            review = _interaction_review(target.target_interaction)
            if not review:
                raise ValueError("review_generation requires a target review body.")
            return review
        raise ValueError(f"Unsupported MemoryCD task {target.task!r}.")

    def _generate_jsonish(
        self,
        text_caller: Any,
        model_name: str,
        instructions: str,
        prompt: str,
        generation_config: Optional[Any],
        default_max_tokens: int,
    ) -> Dict[str, Any]:
        temperature = (
            generation_config.temperature if generation_config is not None else 0.0
        )
        max_output_tokens = (
            generation_config.max_output_tokens
            if generation_config is not None
            else default_max_tokens
        )
        top_p = generation_config.top_p if generation_config is not None else 1.0
        top_k = generation_config.top_k if generation_config is not None else -1
        raw = text_caller.generate_text(
            model=model_name,
            instructions=instructions,
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
        )
        try:
            parsed = parse_json_object(raw)
            if not isinstance(parsed, dict):
                return {"_raw_text": raw}
            parsed["_raw_text"] = raw
            return parsed
        except ValueError:
            return {"_raw_text": raw}

    def predict(
        self,
        text_caller: Any,
        model_name: str,
        example: SupportExample,
        profile_text: str,
        generation_config: Optional[Any] = None,
    ) -> PredictionResult:
        prompt = self.render_prompt(example.prompt, profile_text)
        task = str(example.metadata.get("memorycd_task") or self.task or "")
        default_tokens = 512 if task in {"review_title", "review_generation"} else 128
        result = self._generate_jsonish(
            text_caller=text_caller,
            model_name=model_name,
            instructions=self.prediction_instruction,
            prompt=prompt,
            generation_config=generation_config,
            default_max_tokens=default_tokens,
        )
        output = self.parse_task_prediction(
            task,
            result,
            candidate_item_ids=example.metadata.get("candidate_item_ids") or [],
        )
        return PredictionResult(
            output=output,
            prompt=prompt,
            metadata={"raw_response": result.get("_raw_text")},
        )

    @staticmethod
    def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    @staticmethod
    def _candidate_from_text(value: Any, candidate_item_ids: Sequence[str]) -> Optional[str]:
        text = str(value or "").strip().strip("\"'` ,.;:")
        if not text:
            return None

        candidate_ids = [str(item_id) for item_id in candidate_item_ids]
        lower_map = {item_id.lower(): item_id for item_id in candidate_ids}
        if text.lower() in lower_map:
            return lower_map[text.lower()]

        numeric = re.fullmatch(
            r"(?:candidate|item|option|rank|#)?\s*([1-9])",
            text,
            flags=re.IGNORECASE,
        )
        if numeric:
            idx = int(numeric.group(1)) - 1
            if 0 <= idx < len(candidate_ids):
                return candidate_ids[idx]

        letter = re.fullmatch(
            r"(?:candidate|item|option)?\s*([a-d])",
            text,
            flags=re.IGNORECASE,
        )
        if letter:
            idx = ord(letter.group(1).lower()) - ord("a")
            if 0 <= idx < len(candidate_ids):
                return candidate_ids[idx]

        lowered = text.lower()
        matches = [
            (lowered.find(item_id.lower()), item_id)
            for item_id in candidate_ids
            if item_id and item_id.lower() in lowered
        ]
        if matches:
            return sorted(matches, key=lambda item: item[0])[0][1]
        return None

    def _parse_item_ranking_prediction(
        self,
        result: Dict[str, Any],
        candidate_item_ids: Sequence[str],
    ) -> List[str]:
        candidate_ids = [str(item_id) for item_id in candidate_item_ids]
        parsed: List[str] = []

        for key in ("top_item_id", "item_id", "choice", "answer", "selection"):
            value = result.get(key)
            if value is None:
                continue
            candidate = self._candidate_from_text(value, candidate_ids)
            if candidate:
                parsed.append(candidate)

        ranking = result.get("ranked_item_ids")
        if ranking is None:
            ranking = result.get("ranking")
        if isinstance(ranking, list):
            for value in ranking:
                candidate = self._candidate_from_text(value, candidate_ids)
                if candidate:
                    parsed.append(candidate)
        elif isinstance(ranking, str):
            candidate = self._candidate_from_text(ranking, candidate_ids)
            if candidate:
                parsed.append(candidate)

        raw = str(result.get("_raw_text") or "")
        if raw:
            lowered = raw.lower()
            raw_matches = [
                (lowered.find(item_id.lower()), item_id)
                for item_id in candidate_ids
                if item_id and item_id.lower() in lowered
            ]
            parsed.extend(item_id for _, item_id in sorted(raw_matches))
            if not raw_matches:
                numeric = re.search(r"\b(?:candidate|item|option)?\s*([1-9])\b", raw)
                if numeric:
                    idx = int(numeric.group(1)) - 1
                    if 0 <= idx < len(candidate_ids):
                        parsed.append(candidate_ids[idx])

        return self._dedupe_keep_order(parsed)

    def parse_prediction(self, result_dict: Dict[str, Any]) -> Any:
        """Parse batched downstream results.

        The shared feedback path calls TaskAdapter.parse_prediction directly
        after JSON parsing instead of going through predict(), so MemoryCD must
        handle its task-specific output schemas here as well.
        """
        if any(
            key in result_dict
            for key in (
                "top_item_id",
                "item_id",
                "ranked_item_ids",
                "ranking",
                "choice",
                "answer",
                "selection",
            )
        ):
            parsed = self._parse_item_ranking_prediction(result_dict, ())
            if parsed:
                return parsed
            for key in ("top_item_id", "item_id", "choice", "answer", "selection"):
                value = result_dict.get(key)
                if value is not None and str(value).strip():
                    return [str(value).strip()]
            ranking = result_dict.get("ranking")
            if ranking is None:
                ranking = result_dict.get("ranked_item_ids")
            if isinstance(ranking, list):
                return [str(item).strip() for item in ranking if str(item).strip()]
            if isinstance(ranking, str) and ranking.strip():
                return [ranking.strip()]
            return []

        if "rating" in result_dict:
            rating = _rating(result_dict.get("rating"))
            return int(round(rating)) if rating is not None else ""

        if "title" in result_dict:
            return str(result_dict.get("title") or "").strip()
        if "review" in result_dict:
            return str(result_dict.get("review") or "").strip()
        if "output" in result_dict:
            return str(result_dict.get("output") or "").strip()

        raw = str(result_dict.get("_raw_text") or "").strip()
        if raw:
            if self.task in {"review_title", "review_generation"}:
                return raw
            if self.task == "rating_prediction":
                match = re.search(r"\b([1-5])(?:\.0)?\b", raw)
                if match:
                    return int(match.group(1))

        return ""

    def parse_task_prediction(
        self,
        task: str,
        result: Dict[str, Any],
        candidate_item_ids: Sequence[str] = (),
    ) -> Any:
        raw = str(result.get("_raw_text") or "")
        if task == "rating_prediction":
            value = result.get("rating")
            if value is None:
                match = re.search(r"\b([1-5])(?:\.0)?\b", raw)
                value = match.group(1) if match else None
            rating = _rating(value)
            return int(round(rating)) if rating is not None else ""
        if task == "item_ranking":
            return self._parse_item_ranking_prediction(result, candidate_item_ids)
        if task == "review_title":
            return str(result.get("title") or result.get("output") or raw).strip()
        if task == "review_generation":
            return str(result.get("review") or result.get("output") or raw).strip()
        return str(result)

    def evaluate(
        self,
        example: SupportExample,
        prediction: PredictionResult,
    ) -> EvaluationResult:
        task = str(example.metadata.get("memorycd_task") or self.task or "")
        if task == "rating_prediction":
            pred = _rating(prediction.output)
            ref = _rating(example.reference)
            abs_error = 4.0 if pred is None or ref is None else abs(pred - ref)
            score = max(0.0, 1.0 - abs_error / 4.0)
            passed = abs_error <= 0.5
            return EvaluationResult(
                primary_metric_name="rating_score",
                primary_score=score,
                metrics={"rating_score": score, "mae": abs_error},
                passed=passed,
                summary={
                    "reference_rating": ref,
                    "predicted_rating": pred,
                    "abs_error": abs_error,
                    "outcome": "correct" if passed else "incorrect",
                },
            )

        if task == "item_ranking":
            ranking = list(prediction.output) if isinstance(prediction.output, list) else []
            ref = normalize_label(example.reference)
            top1 = normalize_label(ranking[0]) if ranking else ""
            hit1 = 1.0 if top1.lower() == ref.lower() else 0.0
            rank = None
            for i, item_id in enumerate(ranking, start=1):
                if normalize_label(item_id).lower() == ref.lower():
                    rank = i
                    break
            mrr = 1.0 / rank if rank else 0.0
            return EvaluationResult(
                primary_metric_name="hit@1",
                primary_score=hit1,
                metrics={"hit@1": hit1, "mrr": mrr},
                passed=bool(hit1),
                summary={
                    "reference_item_id": ref,
                    "top1_item_id": top1,
                    "rank": rank,
                    "outcome": "correct" if hit1 else "incorrect",
                },
            )

        if task in {"review_title", "review_generation"}:
            rouge = compute_rouge(str(prediction.output), str(example.reference))
            primary = float(rouge["rouge-l"])
            passed = primary >= 0.30
            return EvaluationResult(
                primary_metric_name="rouge-l",
                primary_score=primary,
                metrics=rouge,
                passed=passed,
                summary={
                    "reference_text": str(example.reference)[:300],
                    "predicted_text": str(prediction.output)[:300],
                    "rouge_pass_threshold": 0.30,
                },
            )

        raise ValueError(f"Unsupported MemoryCD task {task!r}.")

    def _transition(
        self,
        original_evaluation: EvaluationResult,
        optimized_evaluation: EvaluationResult,
    ) -> str:
        delta = optimized_evaluation.primary_score - original_evaluation.primary_score
        if delta > 1e-6:
            return "improved"
        if delta < -1e-6:
            return "regressed"
        return "stable"

    def build_meta_summary(
        self,
        example: SupportExample,
        original_prediction: PredictionResult,
        original_evaluation: EvaluationResult,
        optimized_prediction: PredictionResult,
        optimized_evaluation: EvaluationResult,
    ) -> Dict[str, Any]:
        task = str(example.metadata.get("memorycd_task") or self.task or "")
        payload: Dict[str, Any] = {
            "transition": self._transition(original_evaluation, optimized_evaluation),
            "memorycd_task": task,
            "memory_source": example.metadata.get("memory_source"),
            "domain": example.metadata.get("domain"),
            "target_item_id": example.metadata.get("target_item_id"),
            "target_item_title": example.metadata.get("target_item_title"),
            "original_primary_score": original_evaluation.primary_score,
            "optimized_primary_score": optimized_evaluation.primary_score,
            "score_delta": optimized_evaluation.primary_score
            - original_evaluation.primary_score,
        }
        if task == "rating_prediction":
            ref_rating = _rating(example.reference)
            original_rating = _rating(original_prediction.output)
            optimized_rating = _rating(optimized_prediction.output)

            def signed_error(pred: Optional[float], ref: Optional[float]) -> Optional[float]:
                if pred is None or ref is None:
                    return None
                return pred - ref

            def error_direction(error: Optional[float]) -> str:
                if error is None:
                    return "invalid"
                if error < 0:
                    return "under_predict"
                if error > 0:
                    return "over_predict"
                return "exact"

            original_error = signed_error(original_rating, ref_rating)
            optimized_error = signed_error(optimized_rating, ref_rating)
            original_mae = original_evaluation.metrics.get("mae")
            optimized_mae = optimized_evaluation.metrics.get("mae")
            mae_delta = (
                optimized_mae - original_mae
                if original_mae is not None and optimized_mae is not None
                else None
            )
            rating_shift = (
                optimized_rating - original_rating
                if original_rating is not None and optimized_rating is not None
                else None
            )
            payload.update(
                {
                    "reference_rating": ref_rating,
                    "original_prediction": original_prediction.output,
                    "optimized_prediction": optimized_prediction.output,
                    "original_rating_error": original_error,
                    "optimized_rating_error": optimized_error,
                    "original_error_direction": error_direction(original_error),
                    "optimized_error_direction": error_direction(optimized_error),
                    "mae_delta": mae_delta,
                    "rating_shift": rating_shift,
                    "gold_signal": f"user would rate target item {example.reference} stars",
                    "predicted_signal": (
                        f"optimized prediction: {optimized_prediction.output} stars"
                    ),
                }
            )
        elif task == "item_ranking":
            payload.update(
                {
                    "reference_item_id": example.reference,
                    "original_top_item": (
                        original_prediction.output[0]
                        if isinstance(original_prediction.output, list)
                        and original_prediction.output
                        else ""
                    ),
                    "optimized_top_item": (
                        optimized_prediction.output[0]
                        if isinstance(optimized_prediction.output, list)
                        and optimized_prediction.output
                        else ""
                    ),
                    "candidate_item_ids": example.metadata.get("candidate_item_ids"),
                    "gold_signal": (
                        "target item should rank first according to user memory"
                    ),
                    "predicted_signal": "optimized top-ranked item",
                }
            )
        else:
            original_rouge_l = original_evaluation.metrics.get("rouge-l")
            optimized_rouge_l = optimized_evaluation.metrics.get("rouge-l")
            original_rouge_1 = original_evaluation.metrics.get("rouge-1")
            optimized_rouge_1 = optimized_evaluation.metrics.get("rouge-1")
            original_rouge_2 = original_evaluation.metrics.get("rouge-2")
            optimized_rouge_2 = optimized_evaluation.metrics.get("rouge-2")
            reference_word_count = len(str(example.reference).split())
            original_word_count = len(str(original_prediction.output).split())
            optimized_word_count = len(str(optimized_prediction.output).split())
            payload.update(
                {
                    "reference_text": str(example.reference)[:300],
                    "original_prediction": str(original_prediction.output)[:300],
                    "optimized_prediction": str(optimized_prediction.output)[:300],
                    "reference_word_count": reference_word_count,
                    "original_word_count": original_word_count,
                    "optimized_word_count": optimized_word_count,
                    "word_count_delta": optimized_word_count - original_word_count,
                    "original_rouge_l": original_rouge_l,
                    "optimized_rouge_l": optimized_rouge_l,
                    "rouge_l_delta": (
                        optimized_rouge_l - original_rouge_l
                        if original_rouge_l is not None
                        and optimized_rouge_l is not None
                        else None
                    ),
                    "original_rouge_1": original_rouge_1,
                    "optimized_rouge_1": optimized_rouge_1,
                    "rouge_1_delta": (
                        optimized_rouge_1 - original_rouge_1
                        if original_rouge_1 is not None
                        and optimized_rouge_1 is not None
                        else None
                    ),
                    "original_rouge_2": original_rouge_2,
                    "optimized_rouge_2": optimized_rouge_2,
                    "rouge_2_delta": (
                        optimized_rouge_2 - original_rouge_2
                        if original_rouge_2 is not None
                        and optimized_rouge_2 is not None
                        else None
                    ),
                }
            )
        return payload

    def build_recorded_sampling_evaluation(
        self,
        example: SupportExample,
    ) -> Optional[EvaluationResult]:
        split = (example.metadata or {}).get("user_split")
        if split == "support":
            return EvaluationResult(
                primary_metric_name=self.primary_metric_name,
                primary_score=1.0,
                metrics={self.primary_metric_name: 1.0},
                passed=True,
                summary={"sampling_source": "memorycd_user_split_support"},
            )
        if split == "query":
            return EvaluationResult(
                primary_metric_name=self.primary_metric_name,
                primary_score=0.0,
                metrics={self.primary_metric_name: 0.0},
                passed=False,
                summary={"sampling_source": "memorycd_user_split_query"},
            )
        return None

    def redact_prompt(self, prompt: str) -> str:
        if not prompt:
            return ""
        return re.sub(
            r"(<UserMemory>\s*)(.*?)(\s*</UserMemory>)",
            r"\1[PROFILE_PLACEHOLDER]\3",
            prompt,
            flags=re.DOTALL,
        )

    def serialize_example(
        self,
        example: SupportExample,
        optimized_profile: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "profile": example.profile,
            "prompt": self.redact_prompt(example.prompt),
            "reference": example.reference,
            "idx": example.idx,
            "metadata": example.metadata,
            "original_prediction": serialize_prediction(example.original_prediction),
            "original_evaluation": serialize_evaluation(example.original_evaluation),
        }
        if optimized_profile is not None:
            payload.update(
                {
                    "optimized_profile": optimized_profile,
                    "optimization_pattern": pattern,
                }
            )
        return payload
