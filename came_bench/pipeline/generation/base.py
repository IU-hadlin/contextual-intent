import logging
import os
import json
import asyncio
import pkgutil
from importlib import import_module
from typing import Dict, List, Optional, Sequence, Tuple, Type

from tqdm import tqdm
from google.protobuf.json_format import MessageToDict

from came_bench.proto import (
    Question,
    Turn,
    QuestionRetrievalResult,
    DatasetRetrievalResult,
    DatasetAnswerGenerationRequest,
    CostEntry,
    QuestionAnswerGenerationResult,
    DatasetAnswerGenerationResult,
    AnswerGenerationStrategyType,
    AnswerType,
)
from came_bench.utils.io import load_questions, load_turns, load_retrieval_result
from came_bench.utils.encoder import Encoder, get_class_instance_all_encoder_cost
from came_bench.utils.lm import get_class_instance_all_lm_cost, init_lm
import dspy

from came_bench.utils.notes import (
    ContextNoteRecord,
    load_segment_level_note_records,
)
from came_bench.utils.common import (
    _conversation_from_turn_id,
    _extract_turn_index_from_id,
    load_structured_turn_notes,
)

# Registry for answer generation strategy type -> {strategy_name -> AnswerGenerator subclass}
ANSWER_GENERATOR_REGISTRY: Dict[AnswerGenerationStrategyType, Dict[Optional[str], Type["AnswerGenerator"]]] = {}

logger = logging.getLogger(__name__)


def _normalise_strategy_name(name: str | None) -> Optional[str]:
    if name is None:
        return None
    lowered = name.strip().lower()
    return lowered if lowered else None


def register_answer_generator(
    strategy_type: AnswerGenerationStrategyType, *, strategy_name: str | None = None
):
    """
    Class decorator to register a AnswerGenerator implementation for a given strategy type enum value.
    """

    key = _normalise_strategy_name(strategy_name)

    def decorator(cls: Type["AnswerGenerator"]) -> Type["AnswerGenerator"]:
        registry = ANSWER_GENERATOR_REGISTRY.setdefault(strategy_type, {})
        registry[key] = cls
        return cls

    return decorator


def get_answer_generator_class(
    strategy_type: AnswerGenerationStrategyType, strategy_name: str | None
) -> Type["AnswerGenerator"]:
    registry = ANSWER_GENERATOR_REGISTRY.get(strategy_type)
    if not registry:
        raise ValueError(f"No answer generator registered for strategy type: {strategy_type}")

    lookup_key = _normalise_strategy_name(strategy_name)
    if lookup_key in registry:
        return registry[lookup_key]

    if lookup_key is not None:
        for key, cls in registry.items():
            if key is None:
                continue
            if key in lookup_key:
                return cls

    if None in registry:
        return registry[None]

    # Fallback to the first available implementation
    return next(iter(registry.values()))


def discover_answer_generators() -> None:
    """
    Import answer generator implementation modules so their registration decorators run.
    """
    pass


class AnswerGenerator:
    def __init__(self, dataset_answer_generation_request: DatasetAnswerGenerationRequest):
        # load raw data
        self.questions: List[Question] = load_questions(
            dataset_answer_generation_request.answer_generation_config.questions_jsonl_path)
        self.turns: List[Turn] = load_turns(dataset_answer_generation_request.answer_generation_config.turns_jsonl_path)
        self.retrieval_result: DatasetRetrievalResult = load_retrieval_result(
            dataset_answer_generation_request.answer_generation_config.retrieval_result_jsonl_path)
        # load config
        self.answer_generation_config = dataset_answer_generation_request.answer_generation_config
        # question id to retrieval result mapping
        self.question_id_to_retrieval_result: Dict[str, QuestionRetrievalResult] = {}
        for question_retrieval_result in self.retrieval_result.question_retrieval_results:
            self.question_id_to_retrieval_result[question_retrieval_result.question.id] = question_retrieval_result
        # turn id to turn mapping
        self.turn_id_to_turn: Dict[str, Turn] = {turn.id: turn for turn in self.turns}
        adapter_lm = init_lm(self.answer_generation_config.adapter_language_model_provider_config)
        self.adapter = dspy.TwoStepAdapter(adapter_lm)
        self.segment_level_note_records: Dict[str, List[ContextNoteRecord]] = {}
        self._segment_level_note_text_by_conversation: Dict[str, str] = {}
        self._segment_level_note_cache: Dict[Tuple[str, ...], str] = {}
        self._load_segment_level_notes()
        self.structured_notes_lookup: Dict[int, Dict[str, any]] = {}
        self._load_structured_turn_notes()
        self._filter_questions_without_retrieval()
        self.sanity_check()

    def _filter_questions_without_retrieval(self) -> None:
        """Ensure we only attempt answer generation for questions that have retrieval results."""
        available_question_ids = set(self.question_id_to_retrieval_result.keys())
        if not available_question_ids:
            logger.warning(
                "No retrieval results were loaded; all %d question(s) will be skipped.",
                len(self.questions),
            )
            self.questions = []
            return

        original_count = len(self.questions)
        if original_count == len(available_question_ids):
            return

        filtered_questions = [q for q in self.questions if q.id in available_question_ids]
        missing = original_count - len(filtered_questions)
        if missing > 0:
            logger.warning(
                "Skipping %d question(s) that do not have retrieval results. "
                "Ensure the retrieval summary covers the same conversations as the questions JSONL.",
                missing,
            )
        extra_retrieval_ids = available_question_ids - {q.id for q in self.questions}
        if extra_retrieval_ids:
            logger.warning(
                "Found %d retrieval result(s) whose question ids are not present in the questions JSONL. "
                "They will be ignored.",
                len(extra_retrieval_ids),
            )
        self.questions = filtered_questions

    async def init(self):
        pass

    def sanity_check(self):
        # Placeholder for future validations; currently no-op to allow freeform questions.
        return

    def _load_segment_level_notes(self) -> None:
        segment_level_notes_path = getattr(self.answer_generation_config, "segment_level_notes_jsonl_path", "")
        normalized_path = segment_level_notes_path.strip()
        if not normalized_path:
            return
        try:
            self.segment_level_note_records = load_segment_level_note_records(normalized_path)
        except FileNotFoundError:
            logger.warning(
                "Context notes file %s not found; continuing without context notes.",
                normalized_path,
            )
        else:
            logger.info(
                "Loaded context notes for %d conversations from %s",
                len(self.segment_level_note_records),
                normalized_path,
            )
            for conversation_id, records in self.segment_level_note_records.items():
                lines = [record.formatted_line for record in records if record.formatted_line]
                self._segment_level_note_text_by_conversation[conversation_id] = "\n".join(lines)

    def _load_structured_turn_notes(self) -> None:
        structured_notes_path = getattr(self.answer_generation_config, "structured_notes_jsonl_path", "")
        normalized_path = structured_notes_path.strip()
        if not normalized_path:
            return
        try:
            self.structured_notes_lookup = load_structured_turn_notes(normalized_path)
        except Exception as e:
            logger.warning(
                "Failed to load structured turn notes from %s: %s",
                normalized_path,
                str(e),
            )

    def enrich_turn_content_with_notes(
        self,
        turn: Turn,
        turn_id: str
    ) -> str:
        """Enrich turn content with structured notes metadata."""

        turn_index = _extract_turn_index_from_id(turn_id)
        note_record = {}
        if turn_index is not None:
            note_record = self.structured_notes_lookup.get(turn_index, {})

        def _clean(value: object) -> str:
            if not value:
                return ""
            if isinstance(value, str):
                return value.strip()
            return str(value).strip()

        role_text = _clean(note_record.get("role") or getattr(turn, "role", ""))
        act_text = _clean(
            note_record.get("act")
            or getattr(turn, "act", "")
            or getattr(turn, "action", "")
        )
        target_text = _clean(
            note_record.get("target")
            or getattr(turn, "target", "")
            or getattr(turn, "action_object", "")
        )
        scope_text = _clean(
            note_record.get("context_scope") or getattr(turn, "context_scope", "")
        )
        note_clean = _clean(
            note_record.get("note_text") or getattr(turn, "note_text", "")
        )
        turn_clean = _clean(getattr(turn, "content", ""))

        event_types = note_record.get("event_types") or getattr(turn, "event_types", [])
        if isinstance(event_types, str):
            event_types = [event_types]
        elif not event_types:
            event_types = []
        else:
            event_types = [et for et in event_types if et]

        context_parts: List[str] = []
        if event_types:
            event_types_str = ", ".join(event_types)
            if event_types_str:
                context_parts.append(
                    f"The event types of this turn are: {event_types_str}"
                )
        if scope_text:
            context_parts.append(f"under the context scope '{scope_text}'")
        if role_text and act_text:
            context_parts.append(f"{role_text} {act_text}")
        elif role_text:
            context_parts.append(f"{role_text} acts")
        elif act_text:
            context_parts.append(f"the speaker {act_text}")
        if target_text:
            context_parts.append(f"on the target '{target_text}'")

        if context_parts or note_clean:
            context_description = ", ".join(context_parts) if context_parts else "unspecified context"
            if note_clean:
                context_line = (
                    f"Useful information about this turn: {context_description}. "
                    f"Specific note: {note_clean}"
                )
            else:
                context_line = f"Useful information about this turn: {context_description}"
        else:
            context_line = f"Specific note about this turn: {note_clean}" if note_clean else ""

        enriched_parts: List[str] = []
        if context_line:
            enriched_parts.append(context_line)
        if turn_clean:
            enriched_parts.append(f"Original content: {turn_clean}")

        return " | ".join(enriched_parts) if enriched_parts else turn_clean

    def _segment_level_notes_for_turn_ids(self, turn_ids: Sequence[str]) -> str:
        if not self._segment_level_note_text_by_conversation or not turn_ids:
            return ""

        conversation_ids = sorted({_conversation_from_turn_id(turn_id) for turn_id in turn_ids})
        cache_key = tuple(conversation_ids)
        if cache_key in self._segment_level_note_cache:
            return self._segment_level_note_cache[cache_key]

        multiple_conversations = len(conversation_ids) > 1
        sections: List[str] = []
        for conversation_id in conversation_ids:
            text = self._segment_level_note_text_by_conversation.get(conversation_id, "").strip()
            if not text:
                continue
            if multiple_conversations:
                sections.append(f"Conversation {conversation_id}\n{text}")
            else:
                sections.append(text)

        joined = "\n\n".join(sections)
        self._segment_level_note_cache[cache_key] = joined
        return joined

    async def _generate_answer(self, question: Question, retrieval_result: QuestionRetrievalResult) -> QuestionAnswerGenerationResult:
        raise NotImplementedError("This method is not implemented")

    async def async_execute_answer_generation(self, max_retries: int = 3):
        semaphore = asyncio.Semaphore(self.answer_generation_config.concurrent)

        async def generate_answer_with_semaphore(question: Question) -> QuestionAnswerGenerationResult:
            async with semaphore:
                if question.id not in self.question_id_to_retrieval_result:
                    logger.warning(f"Question {question.id} not found in retrieval result")
                    return QuestionAnswerGenerationResult()
                return await self._generate_answer(question, self.question_id_to_retrieval_result[question.id])

        current_questions: List[Question] = list(self.questions)
        attempt_index = 0
        output_question_answer_generation_results: List[QuestionAnswerGenerationResult] = []

        while True:
            tasks = [asyncio.create_task(generate_answer_with_semaphore(q)) for q in current_questions]

            failed_next_round: List[Question] = []

            with tqdm(total=len(current_questions), desc=f"Attempt {attempt_index + 1}", unit="q") as pbar:
                for task in asyncio.as_completed(tasks):
                    question_answer_generation_result: QuestionAnswerGenerationResult = await task
                    if not question_answer_generation_result.success:
                        failed_next_round.append(question_answer_generation_result.question)
                    if attempt_index >= max_retries or question_answer_generation_result.success:
                        output_question_answer_generation_results.append(question_answer_generation_result)
                    pbar.update(1)

            if not failed_next_round or attempt_index >= max_retries:
                break

            # Prepare next attempt with only failed questions
            current_questions = failed_next_round
            attempt_index += 1

        # save the retrieval results
        dataset_answer_generation_result = DatasetAnswerGenerationResult(
            answer_generation_strategy=self.answer_generation_config.answer_generation_strategy,
            dataset_name=self.answer_generation_config.dataset_name,
        )
        dataset_answer_generation_result.question_answer_generation_results.extend(
            output_question_answer_generation_results)
        self._save_answer_generation_results(dataset_answer_generation_result)

    def _save_answer_generation_results(self, dataset_answer_generation_result: DatasetAnswerGenerationResult):
        if not os.path.exists(self.answer_generation_config.output_dir):
            os.makedirs(self.answer_generation_config.output_dir)
        # retrieval_strategy = self.retrieval_result.retrieval_strategy
        file_name = f"{self.answer_generation_config.answer_generation_strategy}.json"
        with open(os.path.join(self.answer_generation_config.output_dir, file_name), "w") as f:
            json.dump(MessageToDict(dataset_answer_generation_result, preserving_proto_field_name=True,
                      always_print_fields_with_no_presence=True), f, indent=2)

    def get_cost(self) -> List[CostEntry]:
        cost_entries = []
        cost_entries.extend(get_class_instance_all_lm_cost(self))
        cost_entries.extend(get_class_instance_all_encoder_cost(self))
        return cost_entries
