import json
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from profile_meta_core import (
    DEFAULT_TASK_DESCRIPTION,
    PAIRWISE_LABEL_SPACE,
    PROFILE_PLACEHOLDER,
    EvaluationResult,
    PredictionResult,
    SupportExample,
    first_present,
    normalize_label,
    serialize_evaluation,
    serialize_prediction,
)


class TaskAdapter(ABC):
    name = "unknown"
    task_description = ""
    primary_metric_name = "score"
    positive_bucket = "positive"
    negative_bucket = "negative"
    prediction_instruction = ""

    def render_prompt(self, prompt_template: str, profile_text: str) -> str:
        return prompt_template.replace(PROFILE_PLACEHOLDER, profile_text)

    def redact_prompt(self, prompt: str) -> str:
        return prompt

    def sampling_bucket(self, evaluation: EvaluationResult) -> str:
        if evaluation.passed is None:
            raise ValueError(
                f"Task '{self.name}' must override sampling_bucket when passed is None."
            )
        return self.positive_bucket if evaluation.passed else self.negative_bucket

    def build_recorded_sampling_evaluation(
        self,
        example: SupportExample,
    ) -> Optional[EvaluationResult]:
        return None

    def parse_prediction(self, result_dict: Dict[str, Any]) -> str:
        """Extract a prediction from a parsed model JSON response.

        The pairwise adapter historically used {"selection": ...}. Other
        adapters override this when their prompt schema uses a different key.
        """
        return normalize_label(result_dict.get("selection", ""))

    @abstractmethod
    def load_example(
        self,
        item: Dict[str, Any],
        row_idx: int,
        rng: random.Random,
    ) -> SupportExample:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        text_caller: Any,
        model_name: str,
        example: SupportExample,
        profile_text: str,
        generation_config: Optional[Any] = None,
    ) -> PredictionResult:
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        example: SupportExample,
        prediction: PredictionResult,
    ) -> EvaluationResult:
        raise NotImplementedError

    @abstractmethod
    def build_meta_summary(
        self,
        example: SupportExample,
        original_prediction: PredictionResult,
        original_evaluation: EvaluationResult,
        optimized_prediction: PredictionResult,
        optimized_evaluation: EvaluationResult,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def serialize_example(
        self,
        example: SupportExample,
        optimized_profile: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class PairwiseTaskAdapter(TaskAdapter):
    name = "pairwise"
    task_description = DEFAULT_TASK_DESCRIPTION
    primary_metric_name = "accuracy"
    positive_bucket = "correct"
    negative_bucket = "incorrect"

    def __init__(self, label_space: Sequence[str] = PAIRWISE_LABEL_SPACE) -> None:
        self.label_space = [normalize_label(item) for item in label_space]
        # self.prediction_instruction = (
        #     "You are the frozen downstream target model for pairwise preference "
        #     "selection. Output JSON only with the schema "
        #     '{"selection":"<one label from [Item A / Item B]>","reason":"<optional short explanation>"} '
        #     "and keep reason brief."
        # )
        self.prediction_instruction = (
            "You are the frozen downstream target model for pairwise preference "
            "selection. Output JSON only with the schema "
            '{"selection":"<one label from [Item A / Item B]>"} '
            "and keep reason brief."
        )

    def format_prompt(
        self,
        query: str,
        response_a: str,
        response_b: str,
        persona: str,
    ) -> str:
        return (
            "Determine which response the user prefers based on the user's preferences. "
            "Please output your selection below in a json format by filling in the "
            'placeholders in []:{"selection": "[Item A / Item B]"}\n'
            f"<Prompt>\n{query}\n</Prompt>\n\n"
            f"<Preference>\n{persona}\n</Preference>\n\n"
            f"<Item A>\n{response_a}\n</Item A>\n\n"
            f"<Item B>\n{response_b}\n</Item B>\n\n"
            "Now, ONLY output your selection without any other text outside of this "
            "specified structure."
        )

    def materialize_example(
        self,
        target: Dict[str, Any],
        label: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ) -> Dict[str, str]:
        if not isinstance(target, dict):
            raise ValueError("Each pairwise target must be a JSON object.")

        if {"query", "chosen", "rejected"} <= set(target):
            query = str(target["query"])
            chosen = str(target["chosen"])
            rejected = str(target["rejected"])
            if label is None:
                if rng is None:
                    raise ValueError(
                        "rng is required to randomize labels for pairwise targets."
                    )
                label = "Item A" if rng.randint(0, 1) else "Item B"
            normalized_label = normalize_label(label)
            if normalized_label == "Item A":
                item_a, item_b = chosen, rejected
            elif normalized_label == "Item B":
                item_a, item_b = rejected, chosen
            else:
                raise ValueError(
                    f"Unsupported pairwise label '{normalized_label}'. "
                    f"Expected one of {self.label_space}."
                )
            return {
                "query": query,
                "item_a": item_a,
                "item_b": item_b,
                "label": normalized_label,
            }

        raise ValueError(
            "Unsupported target schema. Expected keys {'query','chosen','rejected'} or "
            "{'query','item_a','item_b'} or {'candidate_A','candidate_B'}."
        )

    def load_example(
        self,
        item: Dict[str, Any],
        row_idx: int,
        rng: random.Random,
    ) -> SupportExample:
        if "profile" not in item:
            raise ValueError(f"Missing 'profile' in example at row {row_idx}.")
        if "target" not in item:
            raise ValueError(f"Missing 'target' in example at row {row_idx}.")

        target = item["target"]
        recorded_label = first_present(
            item.get("reference"), item.get("answer"), item.get("label")
        )
        pairwise = self.materialize_example(
            target=target, label=recorded_label, rng=rng
        )
        prompt_template = self.format_prompt(
            query=pairwise["query"],
            response_a=pairwise["item_a"],
            response_b=pairwise["item_b"],
            persona=PROFILE_PLACEHOLDER,
        )
        example = SupportExample(
            profile=item["profile"],
            prompt=prompt_template,
            reference=pairwise["label"],
            idx=int(item.get("idx", row_idx)),
            metadata={
                "task": self.name,
                "raw_input": target,
                "recorded_answer": first_present(
                    item.get("answer"),
                    item.get("label"),
                    item.get("reference"),
                ),
                "recorded_selection": first_present(
                    item.get("selection"),
                    item.get("recorded_prediction"),
                    item.get("source_prediction"),
                ),
            },
        )
        recorded_selection = example.metadata.get("recorded_selection")
        if recorded_selection is not None:
            pred_label = normalize_label(recorded_selection)
            ref_label = normalize_label(pairwise["label"])
            passed = pred_label == ref_label
            example.original_prediction = PredictionResult(output=pred_label)
            example.original_evaluation = EvaluationResult(
                primary_metric_name=self.primary_metric_name,
                primary_score=1.0 if passed else 0.0,
                metrics={self.primary_metric_name: 1.0 if passed else 0.0},
                passed=passed,
            )
        return example

    def predict(
        self,
        text_caller: Any,
        model_name: str,
        example: SupportExample,
        profile_text: str,
        generation_config: Optional[Any] = None,
    ) -> PredictionResult:
        prompt = self.render_prompt(example.prompt, profile_text)
        temperature = (
            generation_config.temperature if generation_config is not None else 0.0
        )
        max_output_tokens = (
            generation_config.max_output_tokens
            if generation_config is not None
            else 800
        )
        top_p = generation_config.top_p if generation_config is not None else 1.0
        top_k = generation_config.top_k if generation_config is not None else -1
        result = text_caller.generate_json(
            model=model_name,
            instructions=self.prediction_instruction,
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
        )
        prediction = normalize_label(result.get("selection", ""))
        # if prediction not in self.label_space:
        #     breakpoint()
        #     raise ValueError(
        #         f"Target model returned label '{prediction}', which is outside "
        #         f"label_space={self.label_space}"
        #     )
        reason = result.get("reason")
        if reason is not None:
            reason = str(reason).strip() or None
        return PredictionResult(
            output=prediction,
            reason=reason,
            prompt=prompt,
        )

    def evaluate(
        self,
        example: SupportExample,
        prediction: PredictionResult,
    ) -> EvaluationResult:
        predicted_label = normalize_label(prediction.output)
        reference_label = normalize_label(example.reference)
        passed = predicted_label == reference_label
        score = 1.0 if passed else 0.0
        return EvaluationResult(
            primary_metric_name=self.primary_metric_name,
            primary_score=score,
            metrics={self.primary_metric_name: score},
            passed=passed,
            summary={
                "reference_label": reference_label,
                "predicted_label": predicted_label,
                "outcome": "correct" if passed else "incorrect",
            },
        )

    def build_meta_summary(
        self,
        example: SupportExample,
        original_prediction: PredictionResult,
        original_evaluation: EvaluationResult,
        optimized_prediction: PredictionResult,
        optimized_evaluation: EvaluationResult,
    ) -> Dict[str, Any]:
        original_passed = bool(original_evaluation.passed)
        optimized_passed = bool(optimized_evaluation.passed)
        if not original_passed and optimized_passed:
            transition = "improved"
        elif original_passed and not optimized_passed:
            transition = "regressed"
        elif not original_passed and not optimized_passed:
            transition = "persistent_failure"
        else:
            transition = "stable_success"
        return {
            "transition": transition,
            "reference_label": normalize_label(example.reference),
            "original_prediction": normalize_label(original_prediction.output),
            "optimized_prediction": normalize_label(optimized_prediction.output),
            "prediction_changed": normalize_label(original_prediction.output)
            != normalize_label(optimized_prediction.output),
        }

    def build_recorded_sampling_evaluation(
        self,
        example: SupportExample,
    ) -> Optional[EvaluationResult]:
        recorded_answer = example.metadata.get("recorded_answer")
        recorded_selection = example.metadata.get("recorded_selection")
        if recorded_answer is None or recorded_selection is None:
            return None

        passed = normalize_label(recorded_selection) == normalize_label(recorded_answer)
        score = 1.0 if passed else 0.0
        return EvaluationResult(
            primary_metric_name=self.primary_metric_name,
            primary_score=score,
            metrics={self.primary_metric_name: score},
            passed=passed,
            summary={
                "reference_label": normalize_label(recorded_answer),
                "predicted_label": normalize_label(recorded_selection),
                "outcome": "correct" if passed else "incorrect",
                "sampling_source": "recorded_pairwise_answer_selection",
            },
        )

    def redact_prompt(self, prompt: str) -> str:
        if not prompt:
            return ""
        return re.sub(
            r"(<Preference>\s*)(.*?)(\s*</Preference>)",
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


class _LaMPAdapterBase(TaskAdapter):
    """Shared plumbing for LaMP classification / generation adapters.

    Each LaMP example feeds in as a dict with the schema:
        {
          "profile": <str>,                  # already-summarized text profile
          "target":  {"query": <str>, "gold": <Any>},
          "task_id": <int>,                  # optional, defaults to adapter.task_id
          "user_id": <str>,                  # optional, used as idx fallback
          "label_space": [<str>, ...],       # optional, classification only
        }
    """

    def __init__(self, task_id: int) -> None:
        self.task_id = task_id

    def _extract_target(self, item: Dict[str, Any], row_idx: int) -> Dict[str, Any]:
        if "profile" not in item:
            raise ValueError(f"Missing 'profile' in LaMP example at row {row_idx}.")
        if "target" not in item or not isinstance(item["target"], dict):
            raise ValueError(f"Missing/invalid 'target' in LaMP example at row {row_idx}.")
        target = item["target"]
        for key in ("query", "gold"):
            if key not in target:
                raise ValueError(
                    f"LaMP target missing '{key}' at row {row_idx}; got keys {list(target)}."
                )
        return target


class LaMPClassificationAdapter(_LaMPAdapterBase):
    """LaMP-1 / LaMP-2 / LaMP-3: personalized classification."""

    name = "lamp_classification"
    task_description = "Personalized text classification (LaMP-1/2/3)."
    primary_metric_name = "accuracy"
    positive_bucket = "correct"
    negative_bucket = "incorrect"

    def __init__(self, task_id: int, label_space: Optional[Sequence[str]] = None) -> None:
        super().__init__(task_id)
        self.label_space: List[str] = (
            [normalize_label(x) for x in label_space] if label_space else []
        )
        self.prediction_instruction = (
            "You are the frozen downstream classifier for a personalized text "
            "classification task. Output JSON only with the schema "
            '{"label": "<one of the allowed labels>"}.'
        )

    def format_prompt(self, query: str, persona: str, label_space: Sequence[str]) -> str:
        labels_str = (
            " | ".join(str(l) for l in label_space) if label_space else "<see Task>"
        )
        return (
            f"<UserProfile>\n{persona}\n</UserProfile>\n\n"
            f"<Task>\n{query}\n</Task>\n\n"
            f"Allowed labels: [{labels_str}]\n\n"
            'Return ONLY a JSON object: {"label": "<one allowed label>"}.'
        )

    def load_example(
        self,
        item: Dict[str, Any],
        row_idx: int,
        rng: random.Random,
    ) -> SupportExample:
        del rng  # classification has no label-randomization step
        target = self._extract_target(item, row_idx)
        label_space = item.get("label_space") or self.label_space
        prompt_template = self.format_prompt(
            query=str(target["query"]),
            persona=PROFILE_PLACEHOLDER,
            label_space=label_space,
        )
        return SupportExample(
            profile=item["profile"],
            prompt=prompt_template,
            reference=normalize_label(target["gold"]),
            idx=int(item.get("idx", row_idx)),
            metadata={
                "task": self.name,
                "task_id": int(item.get("task_id", self.task_id)),
                "user_id": item.get("user_id"),
                "label_space": list(label_space) if label_space else [],
            },
        )

    def parse_prediction(self, result_dict: Dict[str, Any]) -> str:
        return normalize_label(result_dict.get("label", ""))

    def predict(
        self,
        text_caller: Any,
        model_name: str,
        example: SupportExample,
        profile_text: str,
        generation_config: Optional[Any] = None,
    ) -> PredictionResult:
        prompt = self.render_prompt(example.prompt, profile_text)
        temperature = (
            generation_config.temperature if generation_config is not None else 0.0
        )
        max_output_tokens = (
            generation_config.max_output_tokens
            if generation_config is not None
            else 200
        )
        top_p = generation_config.top_p if generation_config is not None else 1.0
        top_k = generation_config.top_k if generation_config is not None else -1
        result = text_caller.generate_json(
            model=model_name,
            instructions=self.prediction_instruction,
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
        )
        label = self.parse_prediction(result)
        return PredictionResult(output=label, prompt=prompt)

    def evaluate(
        self,
        example: SupportExample,
        prediction: PredictionResult,
    ) -> EvaluationResult:
        predicted = normalize_label(prediction.output)
        reference = normalize_label(example.reference)
        passed = predicted == reference
        score = 1.0 if passed else 0.0
        return EvaluationResult(
            primary_metric_name=self.primary_metric_name,
            primary_score=score,
            metrics={self.primary_metric_name: score},
            passed=passed,
            summary={
                "reference_label": reference,
                "predicted_label": predicted,
                "outcome": "correct" if passed else "incorrect",
            },
        )

    def build_meta_summary(
        self,
        example: SupportExample,
        original_prediction: PredictionResult,
        original_evaluation: EvaluationResult,
        optimized_prediction: PredictionResult,
        optimized_evaluation: EvaluationResult,
    ) -> Dict[str, Any]:
        original_passed = bool(original_evaluation.passed)
        optimized_passed = bool(optimized_evaluation.passed)
        if not original_passed and optimized_passed:
            transition = "improved"
        elif original_passed and not optimized_passed:
            transition = "regressed"
        elif not original_passed and not optimized_passed:
            transition = "persistent_failure"
        else:
            transition = "stable_success"
        return {
            "transition": transition,
            "reference_label": normalize_label(example.reference),
            "original_prediction": normalize_label(original_prediction.output),
            "optimized_prediction": normalize_label(optimized_prediction.output),
            "prediction_changed": (
                normalize_label(original_prediction.output)
                != normalize_label(optimized_prediction.output)
            ),
        }

    def redact_prompt(self, prompt: str) -> str:
        if not prompt:
            return ""
        return re.sub(
            r"(<UserProfile>\s*)(.*?)(\s*</UserProfile>)",
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


# ---------- ROUGE (minimal, dep-free) for LaMP generation ----------

def _tokenize_for_rouge(text: str) -> List[str]:
    return re.findall(r"\w+", str(text).lower(), flags=re.UNICODE)


def _rouge_n_f1(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
    if not pred_tokens or not ref_tokens or n <= 0:
        return 0.0
    if len(pred_tokens) < n or len(ref_tokens) < n:
        return 0.0
    from collections import Counter

    pred_ngrams = Counter(zip(*[pred_tokens[i:] for i in range(n)]))
    ref_ngrams = Counter(zip(*[ref_tokens[i:] for i in range(n)]))
    overlap = sum((pred_ngrams & ref_ngrams).values())
    pred_total = sum(pred_ngrams.values())
    ref_total = sum(ref_ngrams.values())
    if pred_total == 0 or ref_total == 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / ref_total
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _rouge_l_f1(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    m, n = len(pred_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if pred_tokens[i] == ref_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[m][n]
    p = lcs / m
    r = lcs / n
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def compute_rouge(prediction_text: str, reference_text: str) -> Dict[str, float]:
    """Minimal ROUGE-1 / ROUGE-2 / ROUGE-L F1 (dep-free).

    For paper-grade reporting, prefer the `rouge_score` package to match LaMP
    paper conventions. This implementation is for in-loop reward signals.
    """
    pred = _tokenize_for_rouge(prediction_text)
    ref = _tokenize_for_rouge(reference_text)
    return {
        "rouge-1": _rouge_n_f1(pred, ref, 1),
        "rouge-2": _rouge_n_f1(pred, ref, 2),
        "rouge-l": _rouge_l_f1(pred, ref),
    }


class LaMPGenerationAdapter(_LaMPAdapterBase):
    """LaMP-4 / 5 / 6 / 7 (and LongLaMP): personalized text generation."""

    name = "lamp_generation"
    task_description = "Personalized text generation (LaMP-4/5/6/7, LongLaMP)."
    primary_metric_name = "rouge-1"
    positive_bucket = "good"
    negative_bucket = "poor"

    def __init__(
        self,
        task_id: int,
        max_output_tokens: int = 256,
        rouge_pass_threshold: float = 0.30,
    ) -> None:
        super().__init__(task_id)
        self.max_output_tokens = max_output_tokens
        self.rouge_pass_threshold = rouge_pass_threshold
        self.prediction_instruction = (
            "You are the frozen downstream generator for a personalized text "
            "generation task. Output JSON only with the schema "
            '{"output": "<your generated text>"}.'
        )

    def format_prompt(self, query: str, persona: str) -> str:
        return (
            f"<UserProfile>\n{persona}\n</UserProfile>\n\n"
            f"<Task>\n{query}\n</Task>\n\n"
            'Return ONLY a JSON object: {"output": "<generated text>"}.'
        )

    def load_example(
        self,
        item: Dict[str, Any],
        row_idx: int,
        rng: random.Random,
    ) -> SupportExample:
        del rng  # generation has no label-randomization step
        target = self._extract_target(item, row_idx)
        prompt_template = self.format_prompt(
            query=str(target["query"]),
            persona=PROFILE_PLACEHOLDER,
        )
        return SupportExample(
            profile=item["profile"],
            prompt=prompt_template,
            reference=str(target["gold"]),
            idx=int(item.get("idx", row_idx)),
            metadata={
                "task": self.name,
                "task_id": int(item.get("task_id", self.task_id)),
                "user_id": item.get("user_id"),
            },
        )

    def parse_prediction(self, result_dict: Dict[str, Any]) -> str:
        return str(result_dict.get("output", "")).strip()

    def predict(
        self,
        text_caller: Any,
        model_name: str,
        example: SupportExample,
        profile_text: str,
        generation_config: Optional[Any] = None,
    ) -> PredictionResult:
        prompt = self.render_prompt(example.prompt, profile_text)
        temperature = (
            generation_config.temperature if generation_config is not None else 0.0
        )
        max_output_tokens = (
            generation_config.max_output_tokens
            if generation_config is not None
            else self.max_output_tokens
        )
        top_p = generation_config.top_p if generation_config is not None else 1.0
        top_k = generation_config.top_k if generation_config is not None else -1
        result = text_caller.generate_json(
            model=model_name,
            instructions=self.prediction_instruction,
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
        )
        output = self.parse_prediction(result)
        return PredictionResult(output=output, prompt=prompt)

    def evaluate(
        self,
        example: SupportExample,
        prediction: PredictionResult,
    ) -> EvaluationResult:
        rouge_scores = compute_rouge(prediction.output, example.reference)
        primary = float(rouge_scores[self.primary_metric_name])
        passed = primary >= self.rouge_pass_threshold
        return EvaluationResult(
            primary_metric_name=self.primary_metric_name,
            primary_score=primary,
            metrics=rouge_scores,
            passed=passed,
            summary={
                "reference_text": str(example.reference)[:300],
                "predicted_text": str(prediction.output)[:300],
                "rouge_pass_threshold": self.rouge_pass_threshold,
            },
        )

    def build_meta_summary(
        self,
        example: SupportExample,
        original_prediction: PredictionResult,
        original_evaluation: EvaluationResult,
        optimized_prediction: PredictionResult,
        optimized_evaluation: EvaluationResult,
    ) -> Dict[str, Any]:
        delta = (
            optimized_evaluation.primary_score - original_evaluation.primary_score
        )
        if delta > 1e-6:
            transition = "improved"
        elif delta < -1e-6:
            transition = "regressed"
        else:
            transition = "stable"
        return {
            "transition": transition,
            "primary_metric": self.primary_metric_name,
            "original_primary_score": original_evaluation.primary_score,
            "optimized_primary_score": optimized_evaluation.primary_score,
            "delta": delta,
            "reference_text": str(example.reference)[:300],
            "original_prediction": str(original_prediction.output)[:300],
            "optimized_prediction": str(optimized_prediction.output)[:300],
        }

    def redact_prompt(self, prompt: str) -> str:
        if not prompt:
            return ""
        return re.sub(
            r"(<UserProfile>\s*)(.*?)(\s*</UserProfile>)",
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


def _make_lamp_classification(task_id: int):
    def _factory() -> TaskAdapter:
        return LaMPClassificationAdapter(task_id=task_id)

    return _factory


def _make_lamp_generation(task_id: int):
    def _factory() -> TaskAdapter:
        return LaMPGenerationAdapter(task_id=task_id)

    return _factory


def _make_personamem_v2(scenario: Optional[str] = None):
    """Lazy factory. Pass scenario to bind a per-scenario task_description.

    Returns a zero-arg callable so it slots into TASK_ADAPTERS uniformly.
    """

    def _factory() -> TaskAdapter:
        # Lazy import: personamem_v2.py imports TaskAdapter from this module,
        # so we resolve PersonaMemV2Adapter at call time to avoid a cycle.
        from personamem_v2 import PersonaMemV2Adapter

        return PersonaMemV2Adapter(scenario=scenario)

    return _factory


def _make_memorycd(task: Optional[str] = None):
    """Lazy factory for MemoryCD task adapters."""

    def _factory() -> TaskAdapter:
        from memorycd import MemoryCDAdapter

        return MemoryCDAdapter(task=task)

    return _factory


# PersonaMem-v2 ships 9 conversation_scenario values; each gets its own task
# key so --task carries the scenario, and task_description on the adapter
# matches the cell exactly. The bare "personamem_v2" key is the cross-scenario
# fallback (generic description).
_PERSONAMEM_V2_SCENARIOS = (
    "personal_email",
    "professional_email",
    "creative_writing",
    "professional_writing",
    "chat_message",
    "translation",
    "trouble_consult",
    "social_media_post",
    "knowledge_query",
)

_MEMORYCD_TASKS = (
    "rating_prediction",
    "item_ranking",
    "review_title",
    "review_generation",
)


TASK_ADAPTERS = {
    "pairwise": PairwiseTaskAdapter,
    "lamp1": _make_lamp_classification(1),
    "lamp2": _make_lamp_classification(2),
    "lamp3": _make_lamp_classification(3),
    "lamp4": _make_lamp_generation(4),
    "lamp5": _make_lamp_generation(5),
    "lamp6": _make_lamp_generation(6),
    "lamp7": _make_lamp_generation(7),
    "personamem_v2": _make_personamem_v2(),
    **{
        f"personamem_v2_{sc}": _make_personamem_v2(sc)
        for sc in _PERSONAMEM_V2_SCENARIOS
    },
    "memorycd": _make_memorycd(),
    **{f"memorycd_{task}": _make_memorycd(task) for task in _MEMORYCD_TASKS},
}


def build_task_adapter(task: str) -> TaskAdapter:
    normalized_task = task.strip().lower()
    adapter_cls = TASK_ADAPTERS.get(normalized_task)
    if adapter_cls is not None:
        return adapter_cls()
    if normalized_task in {"generation", "agent"}:
        raise NotImplementedError(
            f"Task '{normalized_task}' is a category, not a concrete task. "
            f"Pick a registered task: {sorted(TASK_ADAPTERS)}."
        )
    raise ValueError(
        f"Unsupported task '{task}'. Available tasks: {sorted(TASK_ADAPTERS)}."
    )


def load_examples(
    path: str,
    task_adapter: TaskAdapter,
    seed: int,
) -> List[SupportExample]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Result file must contain a JSON array.")
    rng = random.Random(seed)
    return [
        task_adapter.load_example(item, row_idx, rng)
        for row_idx, item in enumerate(payload)
    ]
