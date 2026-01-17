"""LLM-based answer evaluator using DSPy."""

from __future__ import annotations

import json
import re
from typing import List, Sequence, Union, Any

import dspy

from came_bench.utils.lm import init_lm
from came_bench.pipeline.evaluation.base import AnswerEvaluator, register_answer_evaluator
from came_bench.proto import (
    DatasetAnswerEvaluationRequest,
    QuestionAnswerEvaluationResult,
    QuestionAnswerGenerationResult,
    AnswerEvaluationStrategyType,
    Answer,
)


class TravelPlanSynthesisSignature(dspy.Signature):
    """
    Evaluate the generated travel plan by comparing each atomic component
    (accommodation, breakfast, lunch, dinner, individual attractions) with the
    gold answer. Count how many components match exactly.

    You MUST:
    - Parse both gold and generated answers into components.
    - Compare each gold component with the generated output.
    - Return number_of_correct_candidates.
    - Provide a short reasoning summary of which parts matched.
    """

    question: str = dspy.InputField(
        description="The evaluation question being asked"
    )
    gold_answer: str = dspy.InputField(
        description="Gold-standard travel plan in the required format"
    )
    generated_answer: str = dspy.InputField(
        description="Model-generated travel plan in the required format"
    )

    number_of_generated_candidates: int = dspy.OutputField(
        description="How many components the generated answer has"
    )
    number_of_correct_candidates: int = dspy.OutputField(
        description="How many gold components the generated answer got correct"
    )
    reasoning: str = dspy.OutputField(
        description="A short explanation describing which components matched and which did not"
    )



class LLMAnswerEvaluatorNumberSignature(dspy.Signature):
    "Your task is to give the number of correct candidates to a question appearing in the generated answer.\n"
    "You will be given the following data: (1) a question, (2) a gold answer containing several correct candidates, and (3) a generated answer.\n"
    "The gold answer and contains the ground-truth candidates. Go through each candidate in the generated answer carefully and check if any of them appear in the gold answer list. \n"
    "Respond with the number of correct candidates covered by the generated answer, and provide a brief reasoning."

    question: str = dspy.InputField(description="The question being evaluated")
    gold_answer: List[str] = dspy.InputField(description="Gold answer candidates (list)")
    generated_answer: List[str] = dspy.InputField(description="Generated answer candidates (list)")
    number_of_correct_candidates: int = dspy.OutputField(description="Number of correct candidates")
    reasoning: str = dspy.OutputField(description="Short justification")


def _strip_prefix(segment: str, prefix: str) -> str:
    lowered = segment.lower()
    prefix_lower = prefix.lower()
    if lowered.startswith(prefix_lower):
        return segment[len(prefix) :].strip(" :")
    return segment.strip()


def extract_plan_candidates(plan_text: str) -> List[str]:
    """Extract individual plan components to compute total candidate count."""
    if plan_text is None:
        return []

    text = str(plan_text).strip()
    if not text:
        return []

    segments = [seg.strip() for seg in text.split(";") if seg.strip()]
    candidates: List[str] = []
    for segment in segments:
        lower = segment.lower()
        if lower.startswith("stay at "):
            candidates.append(_strip_prefix(segment, "stay at "))
        elif lower.startswith("breakfast at "):
            candidates.append(_strip_prefix(segment, "breakfast at "))
        elif lower.startswith("lunch at "):
            candidates.append(_strip_prefix(segment, "lunch at "))
        elif lower.startswith("dinner at "):
            candidates.append(_strip_prefix(segment, "dinner at "))
        elif lower.startswith("visit"):
            visits = _strip_prefix(segment, "visit")
            if visits:
                for item in visits.replace(" and ", ",").split(","):
                    name = item.strip()
                    if name:
                        candidates.append(name)
            else:
                candidates.append(segment)
        else:
            candidates.append(segment)
    return candidates


@register_answer_evaluator("answer_evaluation_v1")
class LLMAnswerEvaluator(AnswerEvaluator):
    """DSPy-based evaluator that reports precision, recall, and F1 based on correct candidates."""

    def __init__(self, dataset_answer_evaluation_request: DatasetAnswerEvaluationRequest):
        super().__init__(dataset_answer_evaluation_request)
        assert dataset_answer_evaluation_request.answer_evaluation_strategy_type == AnswerEvaluationStrategyType.ANSWER_EVALUATION_STRATEGY_TYPE_answer_evaluation_v1
        config = dataset_answer_evaluation_request.answer_evaluation_v1_config
        self.answer_evaluator_config = dataset_answer_evaluation_request.answer_evaluator_config
        self.answer_evaluator_lm = init_lm(config.language_model_provider_config)
        self._predictor = dspy.Predict(LLMAnswerEvaluatorNumberSignature)
        self._plan_predictor = dspy.Predict(TravelPlanSynthesisSignature)

    async def init(self):
        pass

    @staticmethod
    def _extract_candidates(answer: Union[Answer, str, Sequence[str], None]) -> List[str]:
        """Normalize the answer into a list of string candidates.

        Handles multiple formats:
        - Semicolon-separated strings: "A; B; C"
        - JSON arrays: '["A; B; C"]' or '["A", "B", "C"]'
        - JSON arrays with semicolon-separated elements: '["A; B", "C; D"]'
        """

        if answer is None:
            return []

        if isinstance(answer, Answer):
            raw_value: object = answer.free_form_answer
        elif isinstance(answer, str):
            raw_value = answer
        elif isinstance(answer, Sequence):
            # Already a sequence, but may contain semicolon-separated strings
            candidates = []
            for item in answer:
                item_str = str(item).strip()
                if item_str:
                    # Check if this item contains semicolons
                    if ";" in item_str:
                        # Split by semicolons and add each part
                        parts = [part.strip() for part in item_str.split(";") if part.strip()]
                        candidates.extend(parts)
                    else:
                        candidates.append(item_str)
            return candidates
        else:  # pragma: no cover - defensive branch
            raw_value = str(answer)

        if raw_value is None:
            return []

        text = str(raw_value).strip()
        if not text:
            return []

        # Try parsing as JSON first
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Not JSON, treat as semicolon-separated string
            separators = [segment.strip() for segment in text.replace("\n", ";").split(";")]
            candidates = [segment for segment in separators if segment]
            return candidates if candidates else [text]

        # If parsed successfully, handle the result
        if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes)):
            # It's a list/array - flatten and split semicolon-separated items
            candidates = []
            for item in parsed:
                item_str = str(item).strip()
                if item_str:
                    # Check if this item contains semicolons
                    if ";" in item_str:
                        # Split by semicolons and add each part
                        parts = [part.strip() for part in item_str.split(";") if part.strip()]
                        candidates.extend(parts)
                    else:
                        candidates.append(item_str)
            return candidates

        # Single value (not a sequence)
        single_str = str(parsed).strip()
        if not single_str:
            return []
        # Check if it contains semicolons
        if ";" in single_str:
            separators = [segment.strip() for segment in single_str.split(";")]
            return [segment for segment in separators if segment]
        return [single_str]

    @staticmethod
    def _parse_num_correct(raw_value: object) -> int:
        """Coerce the LLM output into a non-negative integer."""

        if isinstance(raw_value, int):
            return max(raw_value, 0)
        if raw_value is None:
            # raise ValueError("LLM did not provide number_of_correct_candidates")
            return 0
        text_value = str(raw_value).strip()
        if not text_value:
            # raise ValueError("LLM returned an empty count of correct candidates")
            return 0
        try:
            # Accept integer-like floats (e.g. "2" or "2.0").
            return max(int(float(text_value)), 0)
        except ValueError as exc:  # pragma: no cover - defensive branch
            # raise ValueError(f"Unable to parse number_of_correct_candidates from '{text_value}'") from exc
            return 0

    async def _evaluate_answer(self, generation_result: QuestionAnswerGenerationResult) -> QuestionAnswerEvaluationResult:
        # Determine strategy based on dataset and question type
        dataset_name = (self.answer_evaluator_config.dataset_name or "").lower()
        question_type = getattr(generation_result.question, "type", "")
        
        if "travel_planning" in dataset_name and question_type == "type_4":
            return await self._evaluate_plan(generation_result)
        
        return await self._evaluate_number(generation_result)

    async def _evaluate_number(self, generation_result: QuestionAnswerGenerationResult) -> QuestionAnswerEvaluationResult:
        result = QuestionAnswerEvaluationResult()
        result.question_answer_generation_result.CopyFrom(generation_result)

        gold_answer_proto = generation_result.question.answer
        gold_candidates = self._extract_candidates(gold_answer_proto)
        generated_candidates = self._extract_candidates(generation_result.answer.free_form_answer)
        total_gold_candidates = len(gold_candidates)
        total_generated_candidates = len(generated_candidates)

        if total_gold_candidates == 0:
            result.success = False
            result.error_message = "Ground truth answer has no candidates to evaluate."
            return result

        with dspy.context(lm=self.answer_evaluator_lm, adapter=None):
            evaluation = await self._predictor.aforward(
                question=generation_result.question.content,
                gold_answer=gold_candidates,
                generated_answer=generated_candidates,
            )

        num_correct = self._parse_num_correct(getattr(evaluation, "number_of_correct_candidates", None))

        # Ensure num_correct doesn't exceed bounds
        num_correct = min(num_correct, total_gold_candidates, total_generated_candidates)

        # Calculate metrics
        precision = num_correct / total_generated_candidates if total_generated_candidates > 0 else 0.0
        recall = num_correct / total_gold_candidates if total_gold_candidates > 0 else 0.0

        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        result.success = True
        result.is_correct_frq = (num_correct == total_gold_candidates)

        result.precision = precision
        result.recall = recall
        result.f1 = f1

        reasoning = (evaluation.reasoning or "").strip()
        result.evaluation_message = (
            f"Correct: {num_correct}/{total_gold_candidates}. "
            f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}."
            + (f" Reasoning: {reasoning}" if reasoning else "")
        )
        return result

    async def _evaluate_plan(self, generation_result: QuestionAnswerGenerationResult) -> QuestionAnswerEvaluationResult:
        result = QuestionAnswerEvaluationResult()
        result.question_answer_generation_result.CopyFrom(generation_result)

        gold_answer_text = generation_result.question.answer.free_form_answer
        gold_candidates = extract_plan_candidates(gold_answer_text)
        total_candidates = len(gold_candidates)
        
        generated_answer_text = generation_result.answer.free_form_answer
        generated_candidates = extract_plan_candidates(generated_answer_text)
        parsed_generated_total = len(generated_candidates)

        if total_candidates == 0:
            result.success = False
            result.error_message = "Ground truth plan has no components to evaluate."
            return result

        with dspy.context(lm=self.answer_evaluator_lm, adapter=None):
            evaluation = await self._plan_predictor.aforward(
                question=generation_result.question.content,
                gold_answer=gold_answer_text,
                generated_answer=generated_answer_text,
            )

        num_correct = self._parse_num_correct(
            getattr(evaluation, "number_of_correct_candidates", None)
        )
        num_generated_field = getattr(evaluation, "number_of_generated_candidates", None)
        try:
            num_generated = self._parse_num_correct(num_generated_field)
        except Exception:
            num_generated = parsed_generated_total

        if num_generated <= 0:
            num_generated = parsed_generated_total

        num_correct = min(num_correct, total_candidates, num_generated)
        
        # Calculate metrics
        precision = num_correct / num_generated if num_generated > 0 else 0.0
        recall = num_correct / total_candidates if total_candidates > 0 else 0.0
        
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        result.success = True
        result.is_correct_frq = (num_correct == total_candidates)
        
        result.precision = precision
        result.recall = recall
        result.f1 = f1

        reasoning = (evaluation.reasoning or "").strip()
        result.evaluation_message = (
            f"Correct candidates: {num_correct}/{total_candidates}. "
            f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}."
            + (f" Reasoning: {reasoning}" if reasoning else "")
        )
        
        return result
