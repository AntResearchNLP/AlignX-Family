"""PersonaMem-v2 (Jiang et al. 2025, arXiv 2512.06688) loader + adapter.

Pipeline for TextProRL:
    chat_history_32k/<persona>.json
        │  M_sum  (profile_summarizer.summarize_profile, cached)
        ▼
    tilde_P_u  (comprehensive text profile)
        │  Rewrite Model R(., φ)
        ▼
    optimized P_u*  →  Downstream F_m  →  multi-choice prediction (a/b/c/d)

The adapter is a multi-choice classifier: each benchmark row carries
(correct_answer, incorrect_answers); we shuffle deterministically into 4
lettered options and ask the downstream to pick a letter.

Register in profile_meta_tasks.TASK_ADAPTERS like so:

    from personamem_v2 import PersonaMemV2Adapter
    TASK_ADAPTERS["personamem_v2"] = PersonaMemV2Adapter
"""

from __future__ import annotations

import ast
import csv
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from profile_meta_core import (
    PROFILE_PLACEHOLDER,
    EvaluationResult,
    PredictionResult,
    SupportExample,
    normalize_label,
    serialize_evaluation,
    serialize_prediction,
)
from profile_meta_tasks import TaskAdapter


_LETTER_CHOICES = ["a", "b", "c", "d"]


# PersonaMem-v2's `conversation_scenario` indicates where the relevant
# preference evidence was revealed in the user's history. It is not a reliable
# label for the held-out query category, so this description is used only by the
# meta-model as evidence-source context. The rewrite executor intentionally does
# not receive it.
_PERSONAMEM_V2_PROFILE_REFINEMENT_DESCRIPTION = (
    "The downstream task is personalized multiple-choice response selection. "
    "Given a user profile, a current user query, and four candidate responses, "
    "the downstream model chooses the response that best matches the user's "
    "source-supported preferences, constraints, habits, and history. The "
    "preference evidence source describes where relevant preferences were "
    "revealed in the user's history; it does not necessarily describe the "
    "surface category of the held-out query."
)


def _preference_source_description(
    scenario_label: str,
    source_context: str,
    typical_signals: str,
) -> str:
    return (
        f"{_PERSONAMEM_V2_PROFILE_REFINEMENT_DESCRIPTION} "
        f'For this data slice, the scenario label "{scenario_label}" means that '
        f"the relevant preference evidence was revealed through {source_context}. "
        f"Such interactions may expose signals such as {typical_signals}. "
        "Use this as context for learning a rewrite policy from feedback, not as "
        "an instruction to specialize the rewritten profile to that scenario."
    )


_SCENARIO_DESCRIPTIONS: Dict[str, str] = {
    "personal_email": _preference_source_description(
        "personal_email",
        "informal personal-email interactions with family, friends, or acquaintances",
        "relationship context, interpersonal tone, personal routines, cultural references, concrete constraints, and privacy behavior",
    ),
    "professional_email": _preference_source_description(
        "professional_email",
        "work-related email interactions with colleagues, clients, managers, or external contacts",
        "professional role, workplace relationships, formality, obligations, communication norms, and work constraints",
    ),
    "creative_writing": _preference_source_description(
        "creative_writing",
        "creative-writing interactions involving fiction, poetry, essays, or other expressive material",
        "expressive style, imagery, motifs, aesthetic taste, cultural influences, and constraints that shape creative choices",
    ),
    "professional_writing": _preference_source_description(
        "professional_writing",
        "work-oriented writing interactions such as reports, memos, proposals, summaries, or briefs",
        "professional domain, audience expectations, register, formatting preferences, decision criteria, and occupational constraints",
    ),
    "chat_message": _preference_source_description(
        "chat_message",
        "short conversational interactions such as instant messages, SMS, or group-chat replies",
        "casual register, brevity, humor, emotional tone, relationship context, everyday routines, and personal constraints",
    ),
    "translation": _preference_source_description(
        "translation",
        "translation interactions across languages the user uses or requests",
        "preferred register, terminology, language pair habits, localization choices, cultural nuance, and tone preservation",
    ),
    "trouble_consult": _preference_source_description(
        "trouble_consult",
        "advice-seeking interactions about personal, emotional, relational, health-related, or situational difficulties",
        "coping preferences, sensitive constraints, relationship context, boundaries, emotional tone, and support-seeking style",
    ),
    "social_media_post": _preference_source_description(
        "social_media_post",
        "public or semi-public social-media writing interactions",
        "public persona, platform style, audience awareness, topics, self-presentation, tone, and privacy boundaries",
    ),
    "knowledge_query": _preference_source_description(
        "knowledge_query",
        "information-seeking interactions where the user asks for explanations, facts, recommendations, or analysis",
        "interests, expertise level, preferred explanation style, curiosity patterns, decision criteria, and domain-specific constraints",
    ),
}

_DEFAULT_PERSONAMEM_V2_DESCRIPTION = _PERSONAMEM_V2_PROFILE_REFINEMENT_DESCRIPTION


# ─────────────────────────────────────────────────────────────────────────────
# CSV / chat-history loaders
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PersonaMemRow:
    persona_id: int
    question_id: str
    expanded_persona: str
    user_query: str
    correct_answer: str
    incorrect_answers: List[str]
    topic_query: str
    topic_preference: str
    preference: str
    conversation_scenario: str
    pref_type: str
    chat_history_32k_link: Optional[str]
    chat_history_128k_link: Optional[str]
    num_relevant_32k: Optional[int]
    num_irrelevant_32k: Optional[int]
    num_relevant_128k: Optional[int]
    num_irrelevant_128k: Optional[int]
    distance_32k: Optional[int]
    distance_128k: Optional[int]
    total_tokens_32k: Optional[int] = None
    total_tokens_128k: Optional[int] = None
    raw_csv_row: Dict[str, Any] = field(default_factory=dict)


def _try_int(value: Any) -> Optional[int]:
    if value in (None, "", "None"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_incorrect_answers(raw: Any) -> List[str]:
    """CSV may store as JSON array or Python list literal."""
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if not raw:
        return []
    text = str(raw).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except json.JSONDecodeError:
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed]
    except (SyntaxError, ValueError):
        pass
    return [text]


def load_benchmark_csv(
    csv_path: str | Path,
    max_rows: Optional[int] = None,
) -> List[PersonaMemRow]:
    rows: List[PersonaMemRow] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, raw in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(
                PersonaMemRow(
                    persona_id=int(raw["persona_id"]),
                    question_id=str(raw.get("question_id", f"row_{i}")),
                    expanded_persona=raw.get("expanded_persona", ""),
                    user_query=raw["user_query"],
                    correct_answer=raw["correct_answer"],
                    incorrect_answers=_parse_incorrect_answers(
                        raw.get("incorrect_answers", "[]")
                    ),
                    topic_query=raw.get("topic_query", ""),
                    topic_preference=raw.get("topic_preference", ""),
                    preference=raw.get("preference", ""),
                    conversation_scenario=raw.get("conversation_scenario", ""),
                    pref_type=raw.get("pref_type", ""),
                    chat_history_32k_link=raw.get("chat_history_32k_link") or None,
                    chat_history_128k_link=raw.get("chat_history_128k_link") or None,
                    num_relevant_32k=_try_int(
                        raw.get("num_persona_relevant_tokens_32k")
                    ),
                    num_irrelevant_32k=_try_int(
                        raw.get("num_persona_irrelevant_tokens_32k")
                    ),
                    num_relevant_128k=_try_int(
                        raw.get("num_persona_relevant_tokens_128k")
                    ),
                    num_irrelevant_128k=_try_int(
                        raw.get("num_persona_irrelevant_tokens_128k")
                    ),
                    distance_32k=_try_int(
                        raw.get("distance_from_related_snippet_to_query_32k")
                    ),
                    distance_128k=_try_int(
                        raw.get("distance_from_related_snippet_to_query_128k")
                    ),
                    total_tokens_32k=_try_int(
                        raw.get("total_tokens_in_chat_history_32k")
                    ),
                    total_tokens_128k=_try_int(
                        raw.get("total_tokens_in_chat_history_128k")
                    ),
                    raw_csv_row=dict(raw),
                )
            )
    return rows


def load_chat_history(
    persona_id: int,
    chat_history_dir: str | Path,
) -> List[Dict[str, str]]:
    """Load a persona's chat history.

    Files are named `chat_history_<ts>_persona<N>.json`. If multiple timestamps
    exist for the same persona, use the most recent.
    """
    chat_history_dir = Path(chat_history_dir)
    candidates = sorted(chat_history_dir.glob(f"*_persona{persona_id}.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No chat_history file for persona_id={persona_id} under {chat_history_dir}"
        )
    path = candidates[-1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "chat_history" in payload:
        return list(payload["chat_history"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unexpected chat_history shape in {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────────────────────────────────────


class PersonaMemV2Adapter(TaskAdapter):
    """Multi-choice classification adapter for PersonaMem-v2.

    Each example carries (user_query, correct_answer, [incorrect_answers]). At
    load time we shuffle these into 4 lettered options (deterministic per
    question_id) and store the gold *letter* as reference. Downstream model is
    expected to emit JSON {"choice": "<a|b|c|d>"}.
    """

    name = "personamem_v2"
    primary_metric_name = "accuracy"
    positive_bucket = "correct"
    negative_bucket = "incorrect"

    def __init__(self, scenario: Optional[str] = None) -> None:
        # `scenario` is one of the 9 conversation_scenario values. It is used
        # only to name the adapter/dataset slice; the description stays generic
        # because PersonaMem-v2 has no explicit downstream-task label per row.
        self.scenario = scenario
        if scenario and scenario in _SCENARIO_DESCRIPTIONS:
            self.task_description = _SCENARIO_DESCRIPTIONS[scenario]
            self.name = f"personamem_v2_{scenario}"
        else:
            self.task_description = _DEFAULT_PERSONAMEM_V2_DESCRIPTION
        self.prediction_instruction = (
            "You are the frozen downstream model for personalized response "
            "selection. Read the user's profile and the four candidate "
            "responses, then output the letter of the response best aligned "
            "with the user's preferences. Output ONLY the JSON object "
            "specified in the user message — no thinking, prose, markdown, "
            "or any text outside the JSON structure."
        )

    def format_prompt(
        self,
        user_query: str,
        options: Sequence[str],
        persona: str,
        scenario: Optional[str] = None,
    ) -> str:
        if len(options) != 4:
            raise ValueError(
                f"format_prompt expects exactly 4 options, got {len(options)}"
            )
        # `scenario` intentionally is not rendered into the downstream prompt:
        # it is an evidence-source/data-slice label, not the current query type.
        return (
            "Determine which of the four candidate responses best matches "
            "this user's preferences and habits.\n"
            "Please output your selection below in a json format by filling "
            "in the placeholder in []:\n"
            '{"choice": "[a / b / c / d]"}\n\n'
            f"<UserProfile>\n{persona}\n</UserProfile>\n\n"
            f"<UserQuery>\n{user_query}\n</UserQuery>\n\n"
            f"<a>\n{options[0]}\n</a>\n\n"
            f"<b>\n{options[1]}\n</b>\n\n"
            f"<c>\n{options[2]}\n</c>\n\n"
            f"<d>\n{options[3]}\n</d>\n\n"
            "Now, ONLY output your selection as the JSON object specified "
            "above; no other text outside of this structure."
        )

    def build_recorded_sampling_evaluation(
        self,
        example: SupportExample,
    ) -> Optional[EvaluationResult]:
        """Persona-disjoint sampling support.

        When `metadata.user_split` is pre-set by the dataset builder, return a
        synthetic EvaluationResult so the optimizer's positive/negative bucket
        machinery routes support_personas → support, query_personas → query.
        Used only when the optimizer is run with
        --debug-use-recorded-pairwise-sampling (which must be unblocked for
        non-pairwise tasks).
        """
        split = (example.metadata or {}).get("user_split")
        if split in {"support", "train_support", "selection"}:
            return EvaluationResult(
                primary_metric_name=self.primary_metric_name,
                primary_score=1.0,
                metrics={self.primary_metric_name: 1.0},
                passed=True,
                summary={"sampling_source": f"persona_disjoint_{split}"},
            )
        if split == "query":
            return EvaluationResult(
                primary_metric_name=self.primary_metric_name,
                primary_score=0.0,
                metrics={self.primary_metric_name: 0.0},
                passed=False,
                summary={"sampling_source": "persona_disjoint_query"},
            )
        return None

    def load_example(
        self,
        item: Dict[str, Any],
        row_idx: int,
        rng: random.Random,
    ) -> SupportExample:
        del rng  # we use deterministic per-question seeding instead
        if "profile" not in item:
            raise ValueError(f"Missing 'profile' at row {row_idx}")
        target = item.get("target")
        if not isinstance(target, dict):
            raise ValueError(f"Missing 'target' object at row {row_idx}")
        query = str(target.get("query") or target.get("user_query") or "")
        correct = str(target["correct_answer"])
        incorrects = [str(x) for x in target.get("incorrect_answers", [])][:3]
        if len(incorrects) < 3:
            raise ValueError(
                f"row {row_idx}: need ≥3 incorrect_answers, got {len(incorrects)}"
            )

        seed_str = str(
            item.get("question_id") or f"{item.get('persona_id', 0)}_{row_idx}"
        )
        local_rng = random.Random(seed_str)
        all_opts = [correct] + incorrects
        order = list(range(4))
        local_rng.shuffle(order)
        shuffled = [all_opts[i] for i in order]
        gold_position = order.index(0)
        gold_letter = _LETTER_CHOICES[gold_position]

        prompt_template = self.format_prompt(
            user_query=query,
            options=shuffled,
            persona=PROFILE_PLACEHOLDER,
            scenario=target.get("scenario") or item.get("conversation_scenario"),
        )
        item_metadata = item.get("metadata") or {}
        return SupportExample(
            profile=item["profile"],
            prompt=prompt_template,
            reference=gold_letter,
            idx=int(item.get("idx", row_idx)),
            metadata={
                "task": self.name,
                # Forwarded from build_personamem_v2_dataset.py so that
                # build_recorded_sampling_evaluation can route this row to
                # the correct support/query bucket.
                "user_split": item_metadata.get("user_split"),
                "profile_source": item_metadata.get("profile_source"),
                "persona_id": item.get("persona_id"),
                "question_id": item.get("question_id"),
                "shuffled_options": shuffled,
                "gold_letter": gold_letter,
                "correct_answer": correct,
                "incorrect_answers": incorrects,
                "topic_query": item.get("topic_query"),
                "topic_preference": item.get("topic_preference"),
                "preference": item.get("preference")
                or item_metadata.get("preference"),
                "conversation_scenario": item.get("conversation_scenario"),
                "pref_type": item.get("pref_type"),
            },
        )

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
            else 64
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
        choice = self.parse_prediction(result)
        return PredictionResult(output=choice, prompt=prompt)

    def parse_prediction(self, result_dict: Dict[str, Any]) -> str:
        raw = str(result_dict.get("choice", "")).strip().lower()
        match = re.search(r"[abcd]", raw)
        return match.group(0) if match else ""

    # NOTE: `_extract_choice` is intentionally disabled. The pipeline relies
    # on the prompt (schema-as-template + explicit "ONLY output ...") to force
    # the model to return {"choice": "<a|b|c|d>"} strictly. Keeping the dead
    # code commented for reference; do NOT silently re-enable as a fallback.
    #
    # @staticmethod
    # def _extract_choice(raw_text: str) -> str:
    #     if not raw_text:
    #         return ""
    #     text = str(raw_text).split("</think>")[-1].strip()
    #     try:
    #         from profile_meta_core import parse_json_object
    #         parsed = parse_json_object(text)
    #     except Exception:
    #         parsed = None
    #     if isinstance(parsed, dict):
    #         for key in ("choice", "answer", "selection", "option", "label", "response"):
    #             value = parsed.get(key)
    #             if value is None:
    #                 continue
    #             m = re.search(r"[abcd]", str(value).lower())
    #             if m:
    #                 return m.group(0)
    #     text_lower = text.lower()
    #     m = re.search(r"(?:^|[\s\(\[\"'>])([abcd])[\)\.\:\,\s]", text_lower)
    #     if m:
    #         return m.group(1)
    #     m = re.search(r"\b([abcd])\b", text_lower)
    #     if m:
    #         return m.group(1)
    #     return ""

    def evaluate(
        self,
        example: SupportExample,
        prediction: PredictionResult,
    ) -> EvaluationResult:
        pred = normalize_label(prediction.output).lower()
        ref = normalize_label(example.reference).lower()
        passed = pred == ref
        score = 1.0 if passed else 0.0
        return EvaluationResult(
            primary_metric_name=self.primary_metric_name,
            primary_score=score,
            metrics={self.primary_metric_name: score},
            passed=passed,
            summary={
                "reference_letter": ref,
                "predicted_letter": pred,
                "outcome": "correct" if passed else "incorrect",
            },
        )

    @staticmethod
    def _common_suffix(strings: Sequence[str]) -> str:
        if not strings:
            return ""
        reversed_strings = [text[::-1] for text in strings]
        suffix = reversed_strings[0]
        for text in reversed_strings[1:]:
            while suffix and not text.startswith(suffix):
                suffix = suffix[:-1]
        return suffix[::-1]

    @staticmethod
    def _short_signal(text: str, max_chars: int = 180) -> str:
        signal = re.sub(r"\s+", " ", text).strip(" \t\n\r,.;:-")
        signal = re.sub(
            r"\b(?:you might|you may|you could|you would|it may|it might)$",
            "",
            signal,
            flags=re.IGNORECASE,
        ).strip(" \t\n\r,.;:-")
        if not signal:
            return ""
        if len(signal) <= max_chars:
            return signal
        truncated = signal[:max_chars].rsplit(" ", 1)[0].strip(" \t\n\r,.;:-")
        return f"{truncated}..."

    @classmethod
    def _option_signal(
        cls,
        letter: Any,
        options: Sequence[str],
    ) -> str:
        normalized = str(letter or "").strip().lower()
        if normalized not in _LETTER_CHOICES:
            return ""
        idx = _LETTER_CHOICES.index(normalized)
        if idx >= len(options):
            return ""

        text = re.sub(r"\s+", " ", str(options[idx])).strip()
        normalized_options = [
            re.sub(r"\s+", " ", str(option)).strip()
            for option in options
            if str(option).strip()
        ]
        suffix = cls._common_suffix(normalized_options)
        signal = text
        if suffix and len(suffix) >= 40 and len(signal) > len(suffix):
            signal = signal[: -len(suffix)]
        signal = cls._short_signal(signal)
        return signal or "generic/no explicit personalized signal"

    def build_meta_summary(
        self,
        example: SupportExample,
        original_prediction: PredictionResult,
        original_evaluation: EvaluationResult,
        optimized_prediction: PredictionResult,
        optimized_evaluation: EvaluationResult,
    ) -> Dict[str, Any]:
        orig_ok = bool(original_evaluation.passed)
        opt_ok = bool(optimized_evaluation.passed)
        if not orig_ok and opt_ok:
            transition = "improved"
        elif orig_ok and not opt_ok:
            transition = "regressed"
        elif not orig_ok and not opt_ok:
            transition = "persistent_failure"
        else:
            transition = "stable_success"

        options = example.metadata.get("shuffled_options") or []
        reference_letter = str(example.reference or "").strip().lower()
        original_letter = str(original_prediction.output or "").strip().lower()
        optimized_letter = str(optimized_prediction.output or "").strip().lower()

        return {
            "transition": transition,
            "scenario": example.metadata.get("conversation_scenario"),
            "pref_type": example.metadata.get("pref_type"),
            "topic_query": example.metadata.get("topic_query"),
            "reference_letter": reference_letter,
            "gold_signal": self._option_signal(reference_letter, options),
            "original_prediction": original_letter,
            "original_signal": self._option_signal(original_letter, options),
            "optimized_prediction": optimized_letter,
            "predicted_signal": self._option_signal(optimized_letter, options),
            "prediction_changed": original_letter != optimized_letter,
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
