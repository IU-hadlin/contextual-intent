"""Experimental turn-level note generator using precomputed scopes and segment notes."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterator, List, Optional, Sequence, TextIO, Tuple

import dspy
from google.protobuf.json_format import ParseDict
from tqdm import tqdm

from came_bench.proto import (
    LanguageModelProvider,
    LanguageModelProviderConfig,
    Turn,
    TurnLevelNoteGeneratorConfig,
)
from came_bench.utils.lm import init_lm
from .common_utils import (
    extract_abbreviated_utterance,
    _extract_turn_index_from_id,
    _conversation_from_turn_id,
)
from .event_type_labeler import (
    EventTypeLabeler,
)
from .segment_level_note_maintainer import (
    ContextNoteRecord,
    load_segment_level_note_records,
)
from came_bench.utils.io import get_lm_cost, load_config, load_turns

logger = logging.getLogger(__name__)

_env_level = os.environ.get("STITCH_STRUCTURED_NOTE_LOG_LEVEL", "INFO").upper()
_log_level = getattr(logging, _env_level, logging.INFO)
logger.setLevel(_log_level)

if not logger.handlers:
    class _TqdmLoggingHandler(logging.Handler):
        """Logging handler that plays nicely with tqdm progress bars."""

        def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - logging side effect
            try:
                msg = self.format(record)
                tqdm.write(msg)
            except Exception:  # pragma: no cover - defensive
                self.handleError(record)

    _handler = _TqdmLoggingHandler()
    _handler.setLevel(_log_level)
    _formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)

logger.propagate = False


@dataclass
class StructuredTurnNote:
    """Structured representation of a dialogue turn summary."""

    turn_id: str
    role: str
    act: str
    target: Optional[str]
    context_scope: Optional[str]
    note_text: str
    event_types: Optional[List[str]] = None
    functional_type_seeds: Optional[List[str]] = None
    is_blank: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "role": self.role,
            "act": self.act,
            "target": self.target,
            "context_scope": self.context_scope,
            "note_text": self.note_text,
            "event_types": self.event_types,
            "functional_type_seeds": self.functional_type_seeds,
            "is_blank": self.is_blank,
        }

    def to_prompt_block(self) -> str:
        if self.is_blank:
            return f"[Turn {self.turn_id}] (blank - grouped with previous turn)"

        lines = [f"[Turn {self.turn_id}]", f"role: {self.role}", f"act: {self.act or ''}"]
        if self.target:
            lines.append(f"target: {self.target}")
        if self.context_scope:
            lines.append(f"context_scope: {self.context_scope}")
        if self.event_types:
            lines.append(f"event_types: {', '.join(self.event_types)}")
        lines.append(f"note: {self.note_text}")
        return "\n".join(lines)


class TurnNoteContentSignature(dspy.Signature):
    """
    Generate structured note content (act, target, and note_text) for a turn, conditioned on the known context_scope and event types.

    Objective
    Express the speaker’s communicative intent in a concise, structured way,
    identifying their pragmatic action (act), the entity/topic it concerns (target), and a
    short natural-language summary (note_text).

    Guidelines
    1. Act identification:  
       Based on the dataset type and the role of the speaker, determine the speaker’s pragmatic act
    2. Target identification:  
       Identify the specific entity, topic, claim, or object that drives the discussion.  
       - If a concrete object or entity name is explicitly mentioned and drives the discussion, select that as the target.  
       - If not explicitly mentioned, infer the implicit object from the semantic meaning of the utterance.  
       - When ambiguous, refer to prior structured_notes within the same
         context scope or event types to infer or resolve the referent.  
       - Resolve pronouns or elliptical expressions (e.g., “it,” “this one,” “there”) 
         by tracing to previously mentioned entities or items.
    3. Note text composition:  
       Write one short, functional sentence summarizing what the speaker is doing
       and about what.  
       - If a concrete object or entity name is explicitly mentioned and drives the discussion, include the entity name described accurately in the note text.  
       - Focus on communicative intent and salient target attributes.  
       - Do not include irrelevant details or paraphrase entire utterances.
    4. Functional types selection: 
       - The provided functional type candidates are a list of pragmatic and task-driven high-level types aggregating the functions of meaningful details in the dataset.
       - Read the utterance carefully and select 0 to any number of functional types that cover the details that drive the utterance.
    5. Context-awareness:  
       - Ensure the generated act and target align with the given context_scope.
       - The scope defines what sub-topic or thread this turn contributes to.
       Use segment_level_notes to recall prior developments within this scope.
    6. Event-type conditioning:  
       - Use event_types to refine your interpretation
    7. Consistency check:  
       - If multiple prior turns have similar acts or targets under the same scope, maintain consistent terminology and phrasing.

    """

    turn_id: str = dspy.InputField()
    dataset_type: str = dspy.InputField()
    role: str = dspy.InputField()
    utterance: str = dspy.InputField()
    context_scope: str = dspy.InputField()
    event_types: str = dspy.InputField(
        description="Comma-separated event type labels for the current turn"
    )
    prior_structured_notes: str = dspy.InputField(
        description="Prior notes sharing the same context scope or event types, used for disambiguation."
    )
    segment_level_notes: str = dspy.InputField(
        description="JSON list of segment-level summaries for this scope observed up to the current turn."
    )
    functional_type_seeds_candidates: list[str] = dspy.InputField(
        description="List of functional type candidates to choose from for the turn."
    )
    act: str = dspy.OutputField()
    target: str = dspy.OutputField()
    note_text: str = dspy.OutputField()
    functional_type_seeds: list[str] = dspy.OutputField()


class TurnLevelNoteGeneratorExp:
    """Generate turn-level structured notes leveraging segment-level summaries."""

    def __init__(
        self,
        lm_config: LanguageModelProviderConfig,
        *,
        context_scope_assignments: Optional[Dict[str, Dict[str, str]]] = None,
        event_type_assignments: Optional[Dict[str, Dict[str, List[str]]]] = None,
        segment_level_note_records: Optional[Dict[str, Sequence[ContextNoteRecord]]] = None,
        functional_type_seeds: Optional[Sequence[str]] = None,
        prior_window: int = 50,
        group_consecutive_turns: bool = True,
        dataset_type: Optional[str] = None,
    ) -> None:
        if lm_config.provider == LanguageModelProvider.LANGUAGE_MODEL_PROVIDER_UNSPECIFIED:
            raise ValueError("Language model provider config must be specified for note taking")

        self._lm_config = lm_config
        self.note_taking_lm = init_lm(lm_config)
        self._content_predictor = dspy.Predict(TurnNoteContentSignature)
        self._context_scope_assignments = self._normalise_context_scope_assignments(
            context_scope_assignments or {}
        )
        self._event_type_assignments = event_type_assignments or {}
        self._segment_level_note_records = self._normalise_segment_level_notes(segment_level_note_records or {})
        self._prior_window = max(1, prior_window)
        self._group_consecutive_turns = group_consecutive_turns
        self._dataset_type = dataset_type or ""
        self._functional_type_seeds = [seed.strip() for seed in (functional_type_seeds or []) if seed.strip()]

    def generate_structured_notes(
        self,
        turns: Sequence[Turn],
        *,
        jsonl_path: Optional[str | Path] = None,
        turn_mapping_path: Optional[str | Path] = None,
        max_conversations: Optional[int] = None,
    ) -> Tuple[List[StructuredTurnNote], Dict[str, List[str]]]:
        results: List[StructuredTurnNote] = []
        turn_mapping: Dict[str, List[str]] = {}

        conversation_groups = list(self._group_turns_by_conversation(turns))
        if max_conversations is not None:
            limit = max(0, int(max_conversations))
            conversation_groups = conversation_groups[:limit]
            logger.info(
                "Limiting structured note generation to the first %d conversation(s)",
                len(conversation_groups),
            )
        total_turns = sum(len(dialogue_turns) for _, dialogue_turns in conversation_groups)
        if total_turns == 0:
            return results, turn_mapping

        jsonl_stream: Optional[TextIO] = None
        if jsonl_path is not None:
            path = Path(jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            jsonl_stream = path.open("w", encoding="utf-8")

        try:
            total_leading_turns = sum(
                len(self._build_turn_groups(dialogue_turns))
                for _, dialogue_turns in conversation_groups
            )
            total_steps = max(1, total_leading_turns)

            unit_label = "group" if self._group_consecutive_turns else "turn"

            logger.info(
                "Turn grouping by role is %s",
                "enabled" if self._group_consecutive_turns else "disabled",
            )

            with tqdm(total=total_steps, desc="Generating structured notes", unit=unit_label) as progress:
                for conversation_id, dialogue_turns in conversation_groups:
                    logger.info(
                        "Starting experimental note generation for conversation %s (%d turns)",
                        conversation_id,
                        len(dialogue_turns),
                    )
                    dialogue_notes, dialogue_mapping = self._generate_for_dialogue(
                        conversation_id,
                        dialogue_turns,
                        progress,
                        jsonl_stream=jsonl_stream,
                        dataset_type=self._dataset_type,
                        functional_type_seeds=self._functional_type_seeds,
                    )
                    results.extend(dialogue_notes)
                    turn_mapping.update(dialogue_mapping)
        finally:
            if jsonl_stream is not None:
                jsonl_stream.close()

        if turn_mapping_path is not None:
            mapping_path = Path(turn_mapping_path)
            mapping_path.parent.mkdir(parents=True, exist_ok=True)
            with mapping_path.open("w", encoding="utf-8") as f:
                json.dump(turn_mapping, f, ensure_ascii=False, indent=2)
            logger.info("Saved turn mapping to %s", turn_mapping_path)

        self._report_statistics(results, turn_mapping)

        total_cost = get_lm_cost(self.note_taking_lm)
        logger.info("Turn-level note generation (exp) LM cost: $%.4f", total_cost)
        return results, turn_mapping

    def _generate_for_dialogue(
        self,
        conversation_id: str,
        turns: Sequence[Turn],
        progress: Optional[tqdm] = None,
        *,
        jsonl_stream: Optional[TextIO] = None,
        dataset_type: Optional[str] = None,
        functional_type_seeds: Optional[Sequence[str]] = None,
    ) -> Tuple[List[StructuredTurnNote], Dict[str, List[str]]]:
        dialogue_records: List[StructuredTurnNote] = []
        turn_mapping: Dict[str, List[str]] = {}
        conversation_start = perf_counter()
        functional_type_candidates: List[str] = list(functional_type_seeds or [])

        turn_groups = self._build_turn_groups(turns)
        with dspy.context(lm=self.note_taking_lm):
            logger.info("=" * 60)
            logger.info(
                "Turn grouping by role is %s",
                "enabled" if self._group_consecutive_turns else "disabled",
            )
            logger.info(
                "Generating content (act, target, note) for %d %s using precomputed scopes",
                len(turn_groups),
                "turn groups" if self._group_consecutive_turns else "turns",
            )
            logger.info("=" * 60)

            for group in turn_groups:
                leading_turn_index, leading_turn = group[0]
                leading_turn_id = str(leading_turn.id)
                aggregated_utterance = extract_abbreviated_utterance(
                    " ".join(str(t.content) for _, t in group)
                )
                utterance_truncated = self._truncate_to_token_limit(aggregated_utterance)

                context_scope = self._lookup_context_scope(conversation_id, leading_turn_id)
                event_types = self._lookup_event_types(conversation_id, leading_turn_id)

                if progress is not None:
                    progress.set_description(f"Turn {leading_turn_index}: Generating content")

                event_type_list = [event_type for event_type in (event_types or []) if event_type]
                event_type_payload = event_type_list if event_type_list else None
                event_types_prompt = ", ".join(event_type_list)

                prior_notes_content = self._format_prior_notes_for_content(
                    dialogue_records,
                    context_scope,
                    event_type_list,
                )
                segment_notes_payload = self._format_segment_notes_for_turn(
                    conversation_id,
                    leading_turn_id,
                    leading_turn_index,
                    context_scope,
                )
                content_prediction = self._content_predictor(
                    turn_id=leading_turn_id,
                    dataset_type=dataset_type,
                    role=str(leading_turn.role),
                    utterance=utterance_truncated,
                    context_scope=context_scope or "",
                    event_types=event_types_prompt,
                    prior_structured_notes=prior_notes_content,
                    segment_level_notes=segment_notes_payload,
                    functional_type_seeds_candidates=functional_type_candidates,
                )

                if progress is not None:
                    progress.update(1)

                leading_note = StructuredTurnNote(
                    turn_id=leading_turn_id,
                    role=str(leading_turn.role),
                    act=self._normalise_text(content_prediction.act),
                    target=self._optional_text(content_prediction.target),
                    context_scope=context_scope,
                    note_text=self._normalise_text(content_prediction.note_text),
                    event_types=event_type_payload,
                    functional_type_seeds=[seed.strip() for seed in (content_prediction.functional_type_seeds or []) if seed],
                    is_blank=False,
                )
                dialogue_records.append(leading_note)

                if jsonl_stream is not None:
                    jsonl_stream.write(json.dumps(leading_note.to_dict(), ensure_ascii=False))
                    jsonl_stream.write("\n")
                    jsonl_stream.flush()

                if len(group) > 1:
                    consecutive_turn_ids = [str(turn.id) for _, turn in group[1:]]
                    turn_mapping[leading_turn_id] = consecutive_turn_ids
                    logger.info(
                        "Turn %d leads %d consecutive turns: %s",
                        leading_turn_index,
                        len(consecutive_turn_ids),
                        consecutive_turn_ids,
                    )

                    for turn_index, turn in group[1:]:
                        blank_note = StructuredTurnNote(
                            turn_id=str(turn.id),
                            role=str(turn.role),
                            act="",
                            target=None,
                            context_scope=None,
                            note_text="",
                            event_types=event_type_payload,
                            is_blank=True,
                        )
                        dialogue_records.append(blank_note)

                        if jsonl_stream is not None:
                            jsonl_stream.write(json.dumps(blank_note.to_dict(), ensure_ascii=False))
                            jsonl_stream.write("\n")
                            jsonl_stream.flush()

            if progress is not None:
                progress.set_description("Generating structured notes")

            logger.info("=" * 60)
            logger.info("PHASE 2 COMPLETE: Generated %d structured notes", len(dialogue_records))
            logger.info("=" * 60)

        total_conversation_time = perf_counter() - conversation_start
        total_turns = len(dialogue_records)
        blank_turns = sum(1 for note in dialogue_records if note.is_blank)
        real_notes = total_turns - blank_turns
        logger.info(
            "Completed experimental note generation for conversation %s in %.4fs | "
            "Total turns: %d, Real notes: %d, Blank notes: %d (%.1f%% skipped)",
            conversation_id,
            total_conversation_time,
            total_turns,
            real_notes,
            blank_turns,
            (blank_turns / total_turns * 100) if total_turns > 0 else 0,
        )
        return dialogue_records, turn_mapping

    def _lookup_event_types(
        self,
        conversation_id: str,
        turn_id: str,
    ) -> Optional[List[str]]:
        mapping = self._event_type_assignments.get(conversation_id)
        if not mapping:
            return None
        if turn_id in mapping:
            return mapping.get(turn_id)
        fallback_idx = _extract_turn_index_from_id(turn_id)
        if fallback_idx is not None:
            # Handle legacy numeric keys
            legacy_key = str(fallback_idx)
            return mapping.get(legacy_key) or mapping.get(fallback_idx)  # type: ignore[index]
        return None

    def _lookup_context_scope(
        self,
        conversation_id: str,
        turn_id: str,
    ) -> Optional[str]:
        mapping = self._context_scope_assignments.get(conversation_id)
        if not mapping:
            return None
        if turn_id in mapping:
            scope = mapping.get(turn_id)
        else:
            fallback_idx = _extract_turn_index_from_id(turn_id)
            scope = None
            if fallback_idx is not None:
                legacy_key = str(fallback_idx)
                scope = mapping.get(legacy_key) or mapping.get(fallback_idx)  # type: ignore[index]
        if scope is None:
            return None
        text = str(scope).strip()
        return text if text else None

    def _format_prior_notes_for_content(
        self,
        records: Sequence[StructuredTurnNote],
        context_scope: Optional[str],
        event_types: list[str],
    ) -> str:
        """Format prior notes with priority-based selection.
        
        Priority tiers:
        1. Notes matching BOTH context_scope AND event_type (highest priority)
        2. Notes matching EITHER context_scope OR event_type (medium priority)
        
        Selects up to 50 notes total, prioritizing higher-relevance matches.
        """
        if not records:
            return "[]"

        # Filter out blank notes
        valid_notes = [note for note in records if not note.is_blank and note.note_text]
        if not valid_notes:
            return "[]"

        # Separate notes by match priority
        both_match: List[StructuredTurnNote] = []
        either_match: List[StructuredTurnNote] = []
        
        for note in valid_notes:
            scope_match = bool(context_scope and note.context_scope and note.context_scope == context_scope)
            event_match = bool(
                event_types
                and note.event_types
                and any(et in event_types for et in note.event_types)
            )
            
            if scope_match and event_match:
                both_match.append(note)
            elif scope_match or event_match:
                either_match.append(note)
        
        # Target window size (50 non-blank leading turns)
        max_window = 50
        
        # Build final selection prioritizing both_match, then either_match
        selected_notes: List[StructuredTurnNote] = []
        
        # Priority 1: Take most recent notes with both matches
        if both_match:
            # Take most recent from both_match (up to max_window)
            recent_both = both_match[-max_window:]
            selected_notes.extend(recent_both)
        
        # Priority 2: Fill remaining slots with either matches
        remaining_slots = max_window - len(selected_notes)
        if remaining_slots > 0 and either_match:
            # Take most recent from either_match to fill remaining slots
            recent_either = either_match[-remaining_slots:]
            selected_notes.extend(recent_either)
        
        # If no matches at all, return empty
        if not selected_notes:
            return "[]"
        
        # Log selection statistics for debugging
        logger.debug(
            "Prior note selection for content: total_valid=%d, both_match=%d, either_match=%d, selected=%d (both=%d, either=%d)",
            len(valid_notes),
            len(both_match),
            len(either_match),
            len(selected_notes),
            min(len(both_match), max_window),
            min(len(selected_notes) - min(len(both_match), max_window), remaining_slots) if remaining_slots > 0 else 0,
        )
        
        # Sort by extracted turn index (fallback to lexicographic turn_id) to maintain chronological order
        selected_notes.sort(
            key=lambda x: (
                _extract_turn_index_from_id(str(x.turn_id)) or float("inf"),
                str(x.turn_id),
            )
        )
        
        payload = [
            {
                "turn_id": note.turn_id,
                "role": note.role,
                "context_scope": note.context_scope,
                "event_types": note.event_types,
                "act": note.act,
                "target": note.target,
                "note_text": note.note_text,
            }
            for note in selected_notes
        ]
        return json.dumps(payload, ensure_ascii=False)

    def _report_statistics(
        self,
        records: List[StructuredTurnNote],
        turn_mapping: Dict[str, List[str]],
    ) -> None:
        total_turns = len(records)
        blank_turns = sum(1 for note in records if note.is_blank)
        real_notes = total_turns - blank_turns
        blank_percentage = (blank_turns / total_turns * 100) if total_turns > 0 else 0

        num_multi_turn_groups = len(turn_mapping)
        num_single_turn_groups = real_notes - num_multi_turn_groups

        logger.info("=" * 60)
        logger.info("Experimental Note Generation Statistics:")
        logger.info("  Total turns processed: %d", total_turns)
        unit_desc = "turn groups" if self._group_consecutive_turns else "turns"
        logger.info("  Turns with real notes (LLM calls / %s): %d", unit_desc, real_notes)
        if self._group_consecutive_turns:
            logger.info("    - Single-turn groups: %d", num_single_turn_groups)
            logger.info("    - Multi-turn groups (leading turns): %d", num_multi_turn_groups)
        logger.info("  Turns with blank notes (grouped/skipped): %d", blank_turns)
        logger.info("  Percentage of turns skipped: %.2f%%", blank_percentage)
        if self._group_consecutive_turns and turn_mapping:
            total_consecutive = sum(len(consecutive) for consecutive in turn_mapping.values())
            avg_group_size = total_consecutive / len(turn_mapping) if turn_mapping else 0
            logger.info("  Average multi-turn group size: %.2f turns", avg_group_size + 1)
        logger.info("=" * 60)

    @staticmethod
    def save_structured_notes_as_jsonl(
        records: Sequence[StructuredTurnNote],
        output_path: str | Path,
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False))
                f.write("\n")

    @staticmethod
    def save_prompt_view(
        records: Sequence[StructuredTurnNote],
        output_path: str | Path,
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(record.to_prompt_block())
                f.write("\n\n")

    def _build_turn_groups(
        self,
        turns: Sequence[Turn],
    ) -> List[List[Tuple[int, Turn]]]:
        if not self._group_consecutive_turns:
            # Use 1-indexed turn_ids for consistency
            return [[(idx, turn)] for idx, turn in enumerate(turns, start=1)]
        return self._group_consecutive_turns_by_role(turns)

    @staticmethod
    def _group_consecutive_turns_by_role(turns: Sequence[Turn]) -> List[List[Tuple[int, Turn]]]:
        if not turns:
            return []

        groups: List[List[Tuple[int, Turn]]] = []
        current_group: List[Tuple[int, Turn]] = []
        current_role: Optional[str] = None

        for turn_index, turn in enumerate(turns, start=1):
            role = str(turn.role)
            if current_role is None or role != current_role:
                if current_group:
                    groups.append(current_group)
                current_group = [(turn_index, turn)]
                current_role = role
            else:
                current_group.append((turn_index, turn))

        if current_group:
            groups.append(current_group)

        return groups

    def _group_turns_by_conversation(
        self,
        turns: Sequence[Turn],
    ) -> Iterator[Tuple[str, List[Turn]]]:
        conversations: Dict[str, List[Turn]] = {}
        for turn in turns:
            turn_id = str(turn.id)
            conversation_id = self._conversation_from_turn_id(turn_id)
            conversations.setdefault(conversation_id, []).append(turn)
        for conversation_id, dialogue_turns in conversations.items():
            yield conversation_id, dialogue_turns

    def _conversation_from_turn_id(self, turn_id: str) -> str:
        return _conversation_from_turn_id(turn_id)

    def _truncate_to_token_limit(self, text: str) -> str:
        max_tokens = getattr(self._lm_config, "max_tokens", 0)
        if not max_tokens or max_tokens <= 0:
            return text

        tokens = text.split()
        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        truncated_text = " ".join(truncated_tokens)
        if not truncated_text.endswith("…"):
            truncated_text += " …"
        return truncated_text

    @staticmethod
    def _normalise_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _optional_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _normalise_segment_level_notes(
        records_by_conversation: Dict[str, Sequence[ContextNoteRecord]]
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        normalised: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for conversation_id, records in records_by_conversation.items():
            scope_map: Dict[str, List[Dict[str, Any]]] = {}
            for idx, record in enumerate(records):
                if record is None:
                    continue

                if isinstance(record, ContextNoteRecord):
                    start_turn_id = str(record.start_turn_index).strip()
                    end_turn_id = str(record.end_turn_index).strip() or start_turn_id
                    context_scope = (record.context_scope or "").strip()
                    formatted_line = record.formatted_line
                    start_order = record.start_turn_order
                    end_order = record.end_turn_order
                elif isinstance(record, dict):
                    start_turn_id = str(
                        record.get("start_turn_id")
                        or record.get("start_turn_index")
                        or ""
                    ).strip()
                    end_turn_id = str(
                        record.get("end_turn_id")
                        or record.get("end_turn_index")
                        or start_turn_id
                        or ""
                    ).strip()
                    context_scope = str(record.get("context_scope") or "").strip()
                    formatted_line = str(
                        record.get("formatted_line")
                        or record.get("note")
                        or ""
                    )
                    start_order = _extract_turn_index_from_id(str(record.get("start_turn_order", "")).strip())
                    end_order = _extract_turn_index_from_id(str(record.get("end_turn_order", "")).strip())
                else:
                    continue

                if not context_scope:
                    continue

                start_turn_index = start_order or _extract_turn_index_from_id(start_turn_id)
                end_turn_index = end_order or _extract_turn_index_from_id(end_turn_id)
                if start_turn_index is None:
                    start_turn_index = idx + 1
                if end_turn_index is None:
                    end_turn_index = start_turn_index

                scope_entry = {
                    "start_turn_id": start_turn_id,
                    "end_turn_id": end_turn_id,
                    "start_turn_index": start_turn_index,
                    "end_turn_index": end_turn_index,
                    "formatted_line": formatted_line,
                }
                scope_map.setdefault(context_scope, []).append(scope_entry)

            if scope_map:
                for scope, entries in scope_map.items():
                    entries.sort(key=lambda item: (item["start_turn_index"], item["end_turn_index"]))
                normalised[conversation_id] = scope_map
        return normalised

    def _format_segment_notes_for_turn(
        self,
        conversation_id: str,
        turn_id: str,
        turn_position: int,
        context_scope: Optional[str],
    ) -> str:
        notes = self._collect_segment_notes_for_turn(conversation_id, turn_id, turn_position, context_scope)
        if not notes:
            return "[]"
        logger.debug(
            "Segment note context | conversation=%s turn=%d scope='%s' segments=%d",
            conversation_id,
            turn_position,
            context_scope,
            len(notes),
        )
        payload = [
            {
                "start_turn_index": note.get("start_turn_id") or note.get("start_turn_index"),
                "end_turn_index": note.get("end_turn_id") or note.get("end_turn_index"),
                "note": note["formatted_line"],
            }
            for note in notes
        ]
        return json.dumps(payload, ensure_ascii=False)

    def _collect_segment_notes_for_turn(
        self,
        conversation_id: str,
        turn_id: str,
        turn_position: int,
        context_scope: Optional[str],
    ) -> List[Dict[str, Any]]:
        if not context_scope:
            return []
        conversation_segments = self._segment_level_note_records.get(conversation_id)
        if not conversation_segments:
            return []
        scope_segments = conversation_segments.get(context_scope)
        if not scope_segments:
            return []

        relevant: List[Dict[str, Any]] = []
        for segment in scope_segments:
            start_turn = int(segment.get("start_turn_index", 0))
            if start_turn > turn_position:
                break
            end_turn = int(segment.get("end_turn_index", start_turn))
            relevant.append(
                {
                    "start_turn_index": start_turn,
                    "end_turn_index": end_turn,
                    "start_turn_id": str(segment.get("start_turn_id") or ""),
                    "end_turn_id": str(segment.get("end_turn_id") or segment.get("start_turn_id") or ""),
                    "formatted_line": str(segment.get("formatted_line", "")),
                }
            )

        if not relevant:
            return []

        max_segments = max(1, self._prior_window)
        if len(relevant) > max_segments:
            relevant = relevant[-max_segments:]
        return relevant


    @staticmethod
    def _normalise_context_scope_assignments(
        assignments: Dict[str, Dict[Any, Any]],
    ) -> Dict[str, Dict[str, str]]:
        normalised: Dict[str, Dict[str, str]] = {}
        for conversation_id, mapping in assignments.items():
            if not isinstance(mapping, dict):
                continue
            converted: Dict[str, str] = {}
            for turn_key, scope_value in mapping.items():
                turn_id = str(turn_key).strip()
                scope_text = str(scope_value).strip()
                if turn_id and scope_text:
                    converted[turn_id] = scope_text
            if converted:
                normalised[conversation_id] = converted
        return normalised


def _load_config(
    config_path: str | Path,
    output_override: Optional[str],
) -> Tuple[
    str,
    str,
    LanguageModelProviderConfig,
    Optional[Path],
    Optional[Path],
    Optional[Path],
    Path,
    Optional[Path],
    Optional[Path],
]:
    config_path = Path(config_path)
    config_data = load_config(config_path)

    raw_cfg = config_data.get("turn_level_note_generator_config")
    if not isinstance(raw_cfg, dict):
        raise ValueError("turn_level_note_generator_config section is required in the config")

    cfg_payload = dict(raw_cfg)
    functional_type_seeds_path = cfg_payload.pop("functional_type_seeds_path", None)

    cfg = ParseDict(cfg_payload, TurnLevelNoteGeneratorConfig())

    if not cfg.turns_jsonl_path:
        raise ValueError("turns_jsonl_path must be provided in turn_level_note_generator_config")
    turns_path_str = cfg.turns_jsonl_path

    dataset_type = cfg.dataset_name

    lm_config = cfg.language_model_provider_config
    if lm_config.provider == LanguageModelProvider.LANGUAGE_MODEL_PROVIDER_UNSPECIFIED:
        raise ValueError("language_model_provider_config must specify a provider")

    segment_notes_path = Path(cfg.segment_level_notes_jsonl_path) if cfg.segment_level_notes_jsonl_path else None
    context_scope_path_str = getattr(cfg, "context_scope_assignments_path", None) or ""
    context_scope_path = Path(context_scope_path_str) if context_scope_path_str.strip() else None
    event_assignments_path = Path(cfg.event_type_assignments_path) if cfg.event_type_assignments_path else None

    if output_override is not None:
        output_path = Path(output_override)
    elif cfg.structured_notes_output_path:
        output_path = Path(cfg.structured_notes_output_path)
    else:
        output_path = config_path.with_name(f"{config_path.stem}_structured_notes.jsonl")

    turn_mapping_path = Path(cfg.turn_mapping_output_path) if cfg.turn_mapping_output_path else None

    return (
        dataset_type,
        turns_path_str,
        lm_config,
        segment_notes_path,
        context_scope_path,
        event_assignments_path,
        output_path,
        turn_mapping_path,
        Path(functional_type_seeds_path) if functional_type_seeds_path else None,
    )
def load_functional_type_seeds(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Functional type seeds file {path} must contain description line and JSON list.")
    try:
        seeds = json.loads(lines[1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unable to parse functional type seeds JSON from {path}") from exc
    if not isinstance(seeds, list):
        raise ValueError(f"Functional type seeds JSON must be a list, got {type(seeds).__name__}")
    return [str(seed).strip() for seed in seeds if str(seed).strip()]


def load_context_scope_assignments(path: Path) -> Dict[str, Dict[Any, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Context scope assignments JSON must be an object mapping conversation_id -> turn scopes")
    
    # Handle nested format: {conversation_id: {turn_id: scope}}
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate structured notes (experimental) for dialogue turns.",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to turn-level note generator JSON config file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output JSONL path; overrides structured_notes_output_path in the config.",
    )
    parser.add_argument(
        "--prompt-view-output",
        help="Optional path to write an LLM-friendly prompt rendering of the structured notes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file(s) if they already exist.",
    )
    parser.add_argument(
        "--prior-window",
        type=int,
        default=50,
        help="Maximum number of prior structured notes to supply for content generation (prioritizes matching scope+event, then either). Default: 50.",
    )
    parser.add_argument(
        "--group-consecutive-turns",
        dest="group_consecutive_turns",
        action="store_true",
        help="Group consecutive turns by role before processing. If not specified, each turn is processed individually (default).",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        help="Limit processing to the first N conversations.",
    )
    args = parser.parse_args()

    (
        dataset_type,
        turns_path,
        lm_config,
        segment_notes_path,
        context_scope_path,
        event_assignments_path,
        output_path,
        turn_mapping_path,
        config_functional_type_path,
    ) = _load_config(
        args.config,
        args.output,
    )
    functional_type_path: Optional[Path] = config_functional_type_path
    functional_type_seeds: Optional[List[str]] = None
    if functional_type_path is not None:
        functional_type_seeds = load_functional_type_seeds(functional_type_path)
        logger.info(
            "Loaded %d functional type seeds from %s",
            len(functional_type_seeds),
            functional_type_path,
        )


    prompt_view_path: Optional[Path] = Path(args.prompt_view_output) if args.prompt_view_output else None

    collisions: List[Path] = [output_path]
    if prompt_view_path is not None:
        collisions.append(prompt_view_path)
    if turn_mapping_path is not None:
        collisions.append(turn_mapping_path)

    existing_conflicts = [path for path in collisions if path.exists()]
    if existing_conflicts and not args.overwrite:
        raise FileExistsError(
            "; ".join(str(path) for path in existing_conflicts)
            + " already exists. Re-run with --overwrite to replace it."
        )

    segment_level_note_records: Dict[str, Sequence[ContextNoteRecord]] = {}
    if segment_notes_path is not None:
        try:
            segment_level_note_records = load_segment_level_note_records(segment_notes_path)
            logger.info(
                "Loaded segment-level notes for %d conversations from %s",
                len(segment_level_note_records),
                segment_notes_path,
            )
        except FileNotFoundError:
            logger.warning(
                "Segment-level notes file %s not found; proceeding without segment notes.",
                segment_notes_path,
            )

    context_scope_assignments: Dict[str, Dict[Any, Any]] = {}
    if context_scope_path is not None:
        try:
            context_scope_assignments = load_context_scope_assignments(context_scope_path)
            logger.info(
                "Loaded context scope assignments for %d conversations from %s",
                len(context_scope_assignments),
                context_scope_path,
            )
        except FileNotFoundError:
            logger.warning(
                "Context scope assignments file %s not found; proceeding without context scope assignments.",
                context_scope_path,
            )
    else:
        logger.info("Not using context scope assignments (context_scope_assignments_path not provided)")

    event_type_assignments: Optional[Dict[str, Dict[str, List[str]]]] = None
    if event_assignments_path is not None:
        try:
            event_type_assignments = EventTypeLabeler.load_event_type_assignments(event_assignments_path)
            logger.info(
                "Loaded event type assignments for %d conversations from %s",
                len(event_type_assignments),
                event_assignments_path,
            )
        except FileNotFoundError:
            logger.warning(
                "Event type assignments file %s not found; proceeding without event filtering.",
                event_assignments_path,
            )

    turns = load_turns(dataset_type, turns_path)
    logger.info(
        "Loaded %d turns for dataset '%s' from %s",
        len(turns),
        dataset_type or "<unspecified>",
        turns_path,
    )

    generator = TurnLevelNoteGeneratorExp(
        lm_config=lm_config,
        context_scope_assignments=context_scope_assignments,
        event_type_assignments=event_type_assignments,
        segment_level_note_records=segment_level_note_records,
        functional_type_seeds=functional_type_seeds,
        prior_window=args.prior_window,
        group_consecutive_turns=args.group_consecutive_turns,
        dataset_type=dataset_type,
    )

    records, turn_mapping = generator.generate_structured_notes(
        turns=turns,
        jsonl_path=output_path,
        turn_mapping_path=turn_mapping_path,
        max_conversations=args.max_conversations,
    )

    logger.info("Generated %d structured notes and wrote them to %s", len(records), output_path)
    if turn_mapping_path:
        logger.info("Generated turn mapping with %d leading turns", len(turn_mapping))

    total_turns = len(records)
    blank_turns = sum(1 for note in records if note.is_blank)
    real_notes = total_turns - blank_turns
    num_multi_turn_groups = len(turn_mapping)
    num_single_turn_groups = real_notes - num_multi_turn_groups

    print("\n" + "=" * 70)
    print("EXPERIMENTAL NOTE GENERATION SUMMARY:")
    print(f"  Total turns: {total_turns}")
    group_label = "Turn groups" if args.group_consecutive_turns else "Turns"
    print(f"  {group_label} (LLM calls made): {real_notes}")
    if args.group_consecutive_turns:
        print(f"    ├─ Single-turn groups: {num_single_turn_groups}")
        print(f"    └─ Multi-turn groups (with consecutive turns): {num_multi_turn_groups}")
    print(f"  Blank notes (grouped/skipped turns): {blank_turns}")
    if total_turns:
        efficiency_pct = blank_turns / total_turns * 100
        print(f"  Efficiency gain: {blank_turns}/{total_turns} = {efficiency_pct:.2f}% turns skipped")
    else:
        print("  Efficiency gain: n/a (no turns processed)")
    print(f"  Output saved to: {output_path}")
    if functional_type_path:
        print(f"  Functional type seeds source: {functional_type_path}")
    if turn_mapping_path:
        print(f"  Turn mapping saved to: {turn_mapping_path}")
    print("=" * 70 + "\n")

    if prompt_view_path is not None:
        generator.save_prompt_view(records, prompt_view_path)
        logger.info("Wrote prompt-view structured notes to %s", prompt_view_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    main()
