"""Answer evaluator registry and base classes."""

from __future__ import annotations

import json
import os
import pkgutil
from importlib import import_module
from typing import Dict, List, Type
from google.protobuf.json_format import ParseDict, MessageToDict
from came_bench.proto import (
    DatasetAnswerEvaluationRequest,
    AnswerEvaluationStrategyType,
    QuestionAnswerGenerationResult,
    QuestionAnswerEvaluationResult,
    DatasetAnswerEvaluationResult,
    CostEntry,
    DatasetAnswerGenerationResult
)
from came_bench.utils.encoder import get_class_instance_all_encoder_cost
from came_bench.utils.lm import get_class_instance_all_lm_cost
from tqdm import tqdm
import asyncio
import logging


class AnswerEvaluatorError(Exception):
    """Generic evaluation error."""


class UnknownEvaluatorError(AnswerEvaluatorError):
    """Raised when an unknown evaluator strategy is requested."""


ANSWER_EVALUATOR_REGISTRY: Dict[str, Type["AnswerEvaluator"]] = {}


def register_answer_evaluator(name: str):
    """Class decorator to register answer evaluator implementations."""

    def decorator(cls: Type["AnswerEvaluator"]) -> Type["AnswerEvaluator"]:
        ANSWER_EVALUATOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_answer_evaluator_class(name: str) -> Type["AnswerEvaluator"]:
    try:
        return ANSWER_EVALUATOR_REGISTRY[name]
    except KeyError as exc:
        raise UnknownEvaluatorError(f"No answer evaluator registered under '{name}'") from exc


def discover_answer_evaluators() -> None:
    """Import evaluator implementation modules so their decorators execute."""
    # try:
    #     import src.baseline.evaluator as evaluator_pkg  # type: ignore
    # except Exception:
    #     return
    #
    # prefix = evaluator_pkg.__name__ + "."
    # for module_info in pkgutil.iter_modules(evaluator_pkg.__path__, prefix):  # type: ignore[attr-defined]
    #     import_module(module_info.name)
    pass


class AnswerEvaluator:
    """Base class for answer evaluation strategies."""

    def __init__(self, dataset_answer_evaluation_request: DatasetAnswerEvaluationRequest):
        self.answer_evaluator_config = dataset_answer_evaluation_request.answer_evaluator_config
        self.answer_evaluation_strategy_type: AnswerEvaluationStrategyType = (
            dataset_answer_evaluation_request.answer_evaluation_strategy_type
        )
        self.question_results: List[QuestionAnswerGenerationResult] = []
        self.question_id_to_result: Dict[str, QuestionAnswerGenerationResult] = {}
        self.answer_generation_strategy_name: str = "unknown"
        self._load_inputs()

    def _load_inputs(self) -> None:
        generated_answers_path = self.answer_evaluator_config.generated_answer_jsonl_path

        # Sanitize the generated answers payload before parsing into proto objects:
        # older/alternate generators may include fields (e.g., multiple-choice "choices", "tags")
        # that are not present in the current Question proto. Strip them to avoid ParseError.
        with open(generated_answers_path, "r", encoding="utf-8") as f:
            raw_payload = json.load(f)

        question_results = raw_payload.get("questionAnswerGenerationResults", [])
        # Drop unexpected fields on the top-level question_answer_generation_results entries
        allowed_result_fields = {"question", "answer", "success", "error_message", "errorMessage"}
        allowed_question_fields = {
            "id",
            "type",
            "content",
            "answer",
            "date",
            "question_turn_ids",
            "answer_turn_ids",
            "answer_type",
            # Also accept camelCase variants in case upstream payloads differ
            "questionTurnIds",
            "answerTurnIds",
            "answerType",
        }
        for entry in question_results:
            # Strip extra fields on the result itself (e.g., is_correct_mcq)
            if isinstance(entry, dict):
                result_keys_to_drop = [k for k in entry.keys() if k not in allowed_result_fields]
                for key in result_keys_to_drop:
                    entry.pop(key, None)

            question = entry.get("question") if isinstance(entry, dict) else None
            if isinstance(question, dict):
                keys_to_drop = [k for k in question.keys() if k not in allowed_question_fields]
                for key in keys_to_drop:
                    question.pop(key, None)

        dataset_answer_generation_result = DatasetAnswerGenerationResult()
        ParseDict(raw_payload, dataset_answer_generation_result)
        self.answer_generation_strategy_name = dataset_answer_generation_result.answer_generation_strategy

        raw_results = dataset_answer_generation_result.question_answer_generation_results
        filtered_results: List[QuestionAnswerGenerationResult] = []
        missing_question_id = 0
        for item in raw_results:
            question_id = item.question.id
            if not question_id:
                missing_question_id += 1
                continue
            filtered_results.append(item)
            self.question_id_to_result[question_id] = item

        if missing_question_id:
            logging.getLogger(__name__).warning(
                "Skipping %d answer generation result(s) that do not include a question id. "
                "Re-run answer generation to ensure retrieval and question files are aligned.",
                missing_question_id,
            )
        self.question_results = filtered_results

    async def _evaluate_answer(self, generation_result: QuestionAnswerGenerationResult) -> QuestionAnswerEvaluationResult:
        """Run the evaluation and return a payload with results/metrics."""
        raise NotImplementedError("This method is not implemented")

    def get_cost(self) -> List[CostEntry]:
        cost_entries = []
        cost_entries.extend(get_class_instance_all_lm_cost(self))
        cost_entries.extend(get_class_instance_all_encoder_cost(self))
        return cost_entries

    def _save_answer_evaluation_results(self, generated_answer_file_name: str, dataset_answer_evaluation_result: DatasetAnswerEvaluationResult):
        if not os.path.exists(self.answer_evaluator_config.output_dir):
            os.makedirs(self.answer_evaluator_config.output_dir)

        # Compute macro metrics
        precisions = []
        recalls = []
        f1s = []

        for qa_result in dataset_answer_evaluation_result.question_answer_evaluation_results:
            if qa_result.success:
                precisions.append(qa_result.precision)
                recalls.append(qa_result.recall)
                f1s.append(qa_result.f1)

        if precisions:
            macro_precision = sum(precisions) / len(precisions)
            macro_recall = sum(recalls) / len(recalls)
            macro_f1 = sum(f1s) / len(f1s)
            print(
                f"[Macro] precision={macro_precision:.4f}, recall={macro_recall:.4f}, "
                f"f1={macro_f1:.4f} over {len(precisions)} questions"
            )

        file_name = f"{dataset_answer_evaluation_result.answer_evaluation_strategy}.json"
        with open(os.path.join(self.answer_evaluator_config.output_dir, file_name), "w") as f:
            print(
                f"Saving answer evaluation results to {os.path.join(self.answer_evaluator_config.output_dir, file_name)}")
            json.dump(MessageToDict(dataset_answer_evaluation_result, preserving_proto_field_name=True,
                      always_print_fields_with_no_presence=True), f, indent=2)

    async def async_execute_answer_evaluation(self, max_retries: int = 3):
        semaphore = asyncio.Semaphore(self.answer_evaluator_config.concurrent)

        async def evaluate_answer_with_semaphore(result: QuestionAnswerGenerationResult) -> QuestionAnswerEvaluationResult:
            async with semaphore:
                try:
                    evaluation_result = await self._evaluate_answer(result)
                except Exception as exc:
                    evaluation_result = QuestionAnswerEvaluationResult()
                    evaluation_result.success = False
                    evaluation_result.error_message = str(exc)
                if not evaluation_result.HasField("question_answer_generation_result"):
                    evaluation_result.question_answer_generation_result.CopyFrom(result)
                return evaluation_result

        current_question_results: List[QuestionAnswerGenerationResult] = list(self.question_results)
        output_question_answer_evaluation_results: List[QuestionAnswerEvaluationResult] = []
        attempt_index = 0
        while True:
            tasks = [
                asyncio.create_task(
                    evaluate_answer_with_semaphore(result)
                )
                for result in current_question_results
            ]

            failed_next_round: List[QuestionAnswerGenerationResult] = []

            with tqdm(total=len(current_question_results), desc=f"Attempt {attempt_index + 1}", unit="q") as pbar:
                for task in asyncio.as_completed(tasks):
                    question_answer_evaluation_result: QuestionAnswerEvaluationResult = await task
                    question_id = question_answer_evaluation_result.question_answer_generation_result.question.id
                    if not question_answer_evaluation_result.success:
                        failed_next_round.append(self.question_id_to_result[question_id])
                    if attempt_index >= max_retries or question_answer_evaluation_result.success:
                        output_question_answer_evaluation_results.append(question_answer_evaluation_result)
                    pbar.update(1)

            if not failed_next_round or attempt_index >= max_retries:
                break

            # Prepare next attempt with only failed questions
            current_question_results = list(failed_next_round)
            attempt_index += 1

        dataset_answer_evaluation_result = DatasetAnswerEvaluationResult(
            answer_evaluation_strategy=self.answer_evaluator_config.answer_evaluation_strategy,
            dataset_name=self.answer_evaluator_config.dataset_name,
        )
        dataset_answer_evaluation_result.question_answer_evaluation_results.extend(
            output_question_answer_evaluation_results)
        generated_answer_file_path = self.answer_evaluator_config.generated_answer_jsonl_path
        generated_answer_file_name = os.path.basename(generated_answer_file_path).split(".")[0]
        self._save_answer_evaluation_results(generated_answer_file_name, dataset_answer_evaluation_result)
