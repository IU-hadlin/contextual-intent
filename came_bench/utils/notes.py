"""Utilities for maintaining coarse-grained conversation context notes."""

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

from came_bench.proto import LanguageModelProvider, LanguageModelProviderConfig, Turn
from came_bench.utils.lm import init_lm, get_lm_cost
from came_bench.utils.io import load_config, load_turns
from .common import (
    _conversation_from_turn_id,
    _extract_turn_index_from_id,
    extract_abbreviated_utterance,
)

logger = logging.getLogger(__name__)

_env_level = os.environ.get("SIVAKO_segment_level_note_TIMINGS_LOG_LEVEL", "INFO").upper()
_log_level = getattr(logging, _env_level, logging.INFO)
logger.setLevel(_log_level)

if not logger.handlers:
    class _TqdmLoggingHandler(logging.Handler):
        """Logging handler that plays nicely with tqdm progress bars."""

        def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - logging side effect
            try:
                msg = self.format(record)
                tqdm.write(msg)
            except Exception:
                self.handleError(record)

    _handler = _TqdmLoggingHandler()
    _handler.setLevel(_log_level)
    _formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)

logger.propagate = False


@dataclass
class ContextNoteRecord:
    """Note entry summarising a contiguous segment of turns within a dialogue."""

    conversation_id: str
    segment_index: int
    start_turn_index: str  # stores the raw turn_id for the segment start
    end_turn_index: str    # stores the raw turn_id for the segment end
    turn_ids: List[str]
    context_scope: Optional[str]
    note: str
    start_turn_order: Optional[int] = None  # numeric order in conversation (for sorting)
    end_turn_order: Optional[int] = None

    @property
    def formatted_line(self) -> str:
        scope_prefix = f"[{self.context_scope}] " if self.context_scope else ""
        return f"{scope_prefix}Turns {self.start_turn_index}-{self.end_turn_index}: {self.note}".strip()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of this context note record."""

        return {
            "conversation_id": self.conversation_id,
            "segment_index": self.segment_index,
            "start_turn_index": self.start_turn_index,
            "end_turn_index": self.end_turn_index,
            "start_turn_order": self.start_turn_order,
            "end_turn_order": self.end_turn_order,
            "turn_ids": list(self.turn_ids),
            "context_scope": self.context_scope,
            "note": self.note,
            "formatted_line": self.formatted_line,
        }


@dataclass
class ScopeSegment:
    """Contiguous run of turns sharing the same non-empty context scope."""

    context_scope: str
    turns: List[Tuple[int, Turn]]

    @property
    def start_turn_index(self) -> int:
        return self.turns[0][0] if self.turns else 0

    @property
    def end_turn_index(self) -> int:
        return self.turns[-1][0] if self.turns else 0

    @property
    def start_turn_id(self) -> str:
        return str(self.turns[0][1].id) if self.turns else ""

    @property
    def end_turn_id(self) -> str:
        return str(self.turns[-1][1].id) if self.turns else ""

    @property
    def turn_ids(self) -> List[str]:
        return [str(turn.id) for _, turn in self.turns]

@dataclass
class ContextualSegmentNote:
    """
    A note summarising a contiguous segment of turns within a dialogue.
    """
    context_scope: str
    segment_level_notes: List[str]

class ContextNoteKeepingSignature(dspy.Signature):
    """
    Your goal is to maintain a clear, scope-explicit summaries.

    Rules:
        1. Resolve ambiguity or confusion
        - The prior context notes are from the same scope as the current segment
        - The prior context notes provide all the information mentioned in the same scope
        - Use prior context notes safely to resolve all vague references to disambiguate the current segment
        2. Content
        - Capture only new, semantically meaningful developments
        - List important new targets within each scope as numbered or bulleted items for clarity
        - You should not list any vague references or pronouns that are not resolved by the prior context notes
    """

    context_scope: str = dspy.InputField(
        description="Name of the context scope shared by this segment."
    )
    prior_segment_level_notes: str = dspy.InputField(
        description="JSON list of previous notes for this same scope, ordered chronologically."
    )
    segment_turns: str = dspy.InputField(
        description="JSON list of dictionaries with keys (turn_index, role, utterance) for each turn in this scope segment."
    )
    segment_level_note: str = dspy.OutputField(
        description="1–3 sentence update capturing new developments for this scope."
    )


class TurnScopeSignature(dspy.Signature):
    """
    Determine which context scope this utterance belongs to.

    Guidelines
    1. The granularity of the context scope should be similar to as previous context scopes. 
    2. Combined with the utterance, you must consider the best context scope to quickly partition the conversation into different scopes.
    3. Observe the current utterance carefully to identify whether it signals a context scope transition or continuation within the same context scope.
    4. Compare with prior_structured_notes to check if the utterance refers back to or continues a prior context scope.
    5. Default to continuity:  
       Read full utterance carefully, if the utterance does not explicitly introduce a new context scope, assign the same context_scope as the most recent relevant prior note.
    6. Detect transitions:  
       When the speaker introduces a transition to a new context, assign a new context scope label to reflect that shift.
    7. Maintain consistency:  
       ALWAYS check the existing_context_scopes list first. If the utterance refers to a topic that matches an existing scope (even if semantically similar), reuse that exact string form. Only create a new scope if no existing scope matches.
    """

    turn_id: str = dspy.InputField()
    role: str = dspy.InputField()
    utterance: str = dspy.InputField()
    conversation_type: str = dspy.InputField()
    prior_structured_notes: str = dspy.InputField(
        description="Previously predicted scopes with turn_id, role, and context_scope (last 20 turns)."
    )
    existing_context_scopes: str = dspy.InputField(
        description="JSON list of all unique context scope labels that have been used in this conversation so far. Check this list first to reuse existing scopes before creating new ones."
    )
    context_scope: str = dspy.OutputField()

class ContextNoteMaintainer:
    """Maintain context notes anchored on LLM-predicted scopes."""

    def __init__(
        self,
        lm_config: LanguageModelProviderConfig,
        *,
        scope_history_window: int = 20,
        prior_notes_limit: int = 8,
        dataset_name: Optional[str] = None,
        group_consecutive_turns: bool = True,
    ) -> None:
        if lm_config.provider == LanguageModelProvider.LANGUAGE_MODEL_PROVIDER_UNSPECIFIED:
            raise ValueError("Language model provider config must be specified for context note taking")
        if scope_history_window <= 0:
            raise ValueError("scope_history_window must be a positive integer")
        if prior_notes_limit <= 0:
            raise ValueError("prior_notes_limit must be a positive integer")
        self._lm_config = lm_config
        self._scope_history_window = scope_history_window
        self._prior_notes_limit = prior_notes_limit
        self._dataset_name = dataset_name
        self._group_consecutive_turns = group_consecutive_turns
        self.segment_level_note_lm = init_lm(lm_config)
        self._scope_predictor = dspy.Predict(TurnScopeSignature)
        self._segment_predictor = dspy.Predict(ContextNoteKeepingSignature)

    def generate_turn_scopes(
        self,
        turns: Sequence[Turn],
        *,
        output_path: Optional[str | Path] = None,
        checkpoint_frequency: int = 50,
        max_conversations: Optional[int] = None,
    ) -> Dict[str, Dict[str, str]]:
        """Predict context scope assignments for each conversation.
        
        Args:
            turns: Sequence of turns to predict scopes for.
            output_path: Optional path to write scope assignments incrementally.
                        The file is updated after each conversation is processed.
            checkpoint_frequency: Write checkpoint after every N turn groups within a conversation.
        
        Returns:
            Dict mapping conversation_id to turn_id (raw id string) to scope assignments.
        """

        scope_assignments: Dict[str, Dict[str, str]] = {}
        conversation_groups = list(self._group_turns_by_conversation(turns))
        if max_conversations is not None:
            limit = max(0, int(max_conversations))
            conversation_groups = conversation_groups[:limit]
            logger.info(
                "Limiting turn scope generation to the first %d conversation(s)",
                len(conversation_groups),
            )
        if not conversation_groups:
            return scope_assignments

        output_file_path: Optional[Path] = None
        if output_path is not None:
            output_file_path = Path(output_path)
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

        initial_cost = get_lm_cost(self.segment_level_note_lm)

        try:
            with dspy.context(lm=self.segment_level_note_lm):
                for conversation_id, dialogue_turns in conversation_groups:
                    logger.info(
                        "Starting turn scope prediction for conversation %s (%d turns)",
                        conversation_id,
                        len(dialogue_turns),
                    )
                    conversation_start = perf_counter()
                    turn_groups = self._build_turn_groups(dialogue_turns)
                    conversation_scopes = self._predict_scopes_for_dialogue(
                        conversation_id,
                        turn_groups,
                        output_file_path=output_file_path,
                        all_scope_assignments=scope_assignments,
                        checkpoint_frequency=checkpoint_frequency,
                    )
                    if conversation_scopes:
                        scope_assignments[conversation_id] = conversation_scopes
                    
                    # Write incrementally after each conversation
                    if output_file_path is not None:
                        write_scope_assignments(scope_assignments, output_file_path)
                    
                    total_conversation_time = perf_counter() - conversation_start
                    logger.info(
                        "Completed turn scope prediction for conversation %s in %.4fs",
                        conversation_id,
                        total_conversation_time,
                    )
        except Exception:
            # If there's an error, still write what we have
            if output_file_path is not None and scope_assignments:
                write_scope_assignments(scope_assignments, output_file_path)
                logger.info("Wrote partial scope assignments to %s before error", output_file_path)
            raise

        final_cost = get_lm_cost(self.segment_level_note_lm)
        logger.info("Turn scope LM cost: $%.4f", final_cost - initial_cost)
        return scope_assignments

    def generate_segment_level_notes(
        self,
        turns: Sequence[Turn],
        *,
        jsonl_path: Optional[str | Path] = None,
        scope_assignments: Optional[Dict[str, Dict[str, str]]] = None,
        max_conversations: Optional[int] = None,
    ) -> Tuple[List[ContextNoteRecord], Dict[str, Dict[str, str]]]:
        """Generate context notes for the provided turns, grouped by conversation."""

        results: List[ContextNoteRecord] = []
        provided_scopes = scope_assignments or {}
        all_scope_assignments: Dict[str, Dict[str, str]] = {}

        conversation_groups = list(self._group_turns_by_conversation(turns))
        if max_conversations is not None:
            limit = max(0, int(max_conversations))
            conversation_groups = conversation_groups[:limit]
            logger.info(
                "Limiting segment note generation to the first %d conversation(s)",
                len(conversation_groups),
            )
        if not conversation_groups:
            return results, all_scope_assignments

        jsonl_stream: Optional[TextIO] = None
        if jsonl_path is not None:
            path = Path(jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            jsonl_stream = path.open("w", encoding="utf-8")

        initial_cost = get_lm_cost(self.segment_level_note_lm)

        try:
            with dspy.context(lm=self.segment_level_note_lm):
                for conversation_id, dialogue_turns in conversation_groups:
                    logger.info(
                        "Starting segment note generation for conversation %s (%d turns)",
                        conversation_id,
                        len(dialogue_turns),
                    )
                    conversation_start = perf_counter()
                    turn_groups = self._build_turn_groups(dialogue_turns)
                    conversation_scopes = provided_scopes.get(conversation_id)
                    if conversation_scopes is None:
                        conversation_scopes = self._predict_scopes_for_dialogue(
                            conversation_id,
                            turn_groups,
                        )
                    else:
                        # Normalise provided scopes to string turn_ids
                        normalised_scopes: Dict[str, str] = {}
                        for turn_key, scope_value in conversation_scopes.items():
                            turn_id = str(turn_key).strip()
                            scope_text = str(scope_value).strip()
                            if turn_id and scope_text:
                                normalised_scopes[turn_id] = scope_text
                        conversation_scopes = normalised_scopes

                    if conversation_scopes:
                        all_scope_assignments[conversation_id] = conversation_scopes

                    dialogue_records = self._generate_notes_for_dialogue(
                        conversation_id,
                        turn_groups,
                        conversation_scopes,
                        jsonl_stream=jsonl_stream,
                    )
                    if dialogue_records:
                        results.extend(dialogue_records)

                    total_conversation_time = perf_counter() - conversation_start
                    logger.info(
                        "Completed segment note generation for conversation %s in %.4fs",
                        conversation_id,
                        total_conversation_time,
                    )
        finally:
            if jsonl_stream is not None:
                jsonl_stream.close()

        final_cost = get_lm_cost(self.segment_level_note_lm)
        logger.info("Context note LM cost: $%.4f", final_cost - initial_cost)
        return results, all_scope_assignments

    @staticmethod
    def save_segment_level_notes_as_jsonl(records: Sequence[ContextNoteRecord], output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False))
                f.write("\n")

    @staticmethod
    def save_segment_level_notes_as_json(records: Sequence[ContextNoteRecord], output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialised = [record.to_dict() for record in records]
        with path.open("w", encoding="utf-8") as f:
            json.dump(serialised, f, ensure_ascii=False, indent=2)

    def _predict_scopes_for_dialogue(
        self,
        conversation_id: str,
        turn_groups: Sequence[List[Tuple[int, Turn]]],
        output_file_path: Optional[Path] = None,
        all_scope_assignments: Optional[Dict[str, Dict[str, str]]] = None,
        checkpoint_frequency: int = 50,
    ) -> Dict[str, str]:
        scope_assignments: Dict[str, str] = {}
        scope_history: List[Dict[str, Any]] = []
        unique_scopes: set[str] = set()  # Track all unique scopes seen so far

        logger.info("=" * 60)
        unit_desc = "role-based turn groups" if self._group_consecutive_turns else "individual turns"
        logger.info("Turn grouping by role is %s", "enabled" if self._group_consecutive_turns else "disabled")
        
        logger.info(
            "PHASE 1: Predicting context scopes for %d %s",
                len(turn_groups),
                unit_desc,
            )
        logger.info("=" * 60)

        unit_label = "group" if self._group_consecutive_turns else "turn"

        with tqdm(
            total=len(turn_groups),
            desc=f"[{conversation_id}] Phase 1: Predicting scopes",
            unit=unit_label
        ) as phase1_progress:
            for idx, group in enumerate(turn_groups, start=1):
                leading_turn_index, leading_turn = group[0]
                leading_turn_id = str(leading_turn.id)
                utterance_joined = " ".join(str(t.content) for _, t in group)
                utterance = extract_abbreviated_utterance(utterance_joined)

                prior_scopes_payload = self._format_scope_history(scope_history)
                existing_scopes_payload = json.dumps(sorted(unique_scopes), ensure_ascii=False)

                scope_prediction = self._scope_predictor(
                    turn_id=leading_turn_id,
                    role=str(leading_turn.role),
                    utterance=utterance,
                    conversation_type=self._dataset_name,
                    prior_structured_notes=prior_scopes_payload,
                    existing_context_scopes=existing_scopes_payload,
                )
                context_scope = self._normalise_scope(scope_prediction.context_scope)
                scope_history.append(
                    {
                        "turn_id": leading_turn_id,
                        "role": str(leading_turn.role),
                        "context_scope": context_scope,
                    }
                )
                if context_scope:
                    scope_assignments[leading_turn_id] = context_scope
                    unique_scopes.add(context_scope)  # Track unique scopes

                phase1_progress.set_postfix_str(f"Turn {leading_turn_index}")
                phase1_progress.update(1)
                
                # Checkpoint every N turn groups
                if output_file_path is not None and all_scope_assignments is not None and idx % checkpoint_frequency == 0:
                    # Update the main dict with current progress
                    checkpoint_assignments = dict(all_scope_assignments)
                    checkpoint_assignments[conversation_id] = scope_assignments
                    write_scope_assignments(checkpoint_assignments, output_file_path)
                    logger.debug(
                        "Checkpoint: Wrote %d scope assignments to %s (conversation %s at group %d/%d)",
                        len(scope_assignments),
                        output_file_path,
                        conversation_id,
                        idx,
                        len(turn_groups),
                    )

        logger.info("=" * 60)
        logger.info(
            "PHASE 1 COMPLETE: Predicted scopes for %d/%d turn groups",
            len(scope_assignments),
            len(turn_groups),
        )
        logger.info("=" * 60)

        return scope_assignments

    def _generate_notes_for_dialogue(
        self,
        conversation_id: str,
        turn_groups: Sequence[List[Tuple[int, Turn]]],
        scope_assignments: Optional[Dict[str, str]],
        *,
        jsonl_stream: Optional[TextIO] = None,
    ) -> List[ContextNoteRecord]:
        if not scope_assignments:
            logger.info(
                "Skipping segment note generation for conversation %s due to missing scopes",
                conversation_id,
            )
            return []

        segments = self._build_scope_segments(turn_groups, scope_assignments)

        if not segments:
            logger.info(
                "No non-empty context scope segments produced for conversation %s",
                conversation_id,
            )
            return []

        logger.info("=" * 60)
        logger.info(
            "PHASE 2: Generating notes for %d context scope segments",
            len(segments),
        )
        logger.info("=" * 60)

        dialogue_records: List[ContextNoteRecord] = []

        with tqdm(
            total=len(segments),
            desc=f"[{conversation_id}] Phase 2: Generating notes",
            unit="segment"
        ) as phase2_progress:
            for idx, segment in enumerate(segments, start=1):
                segment_start_time = perf_counter()

                prior_notes_payload = self._format_prior_notes(
                    dialogue_records,
                    context_scope=segment.context_scope,
                )
                segment_turns_payload = self._format_segment_turns(segment)

                prediction = self._segment_predictor(
                    context_scope=segment.context_scope,
                    prior_segment_level_notes=prior_notes_payload,
                    segment_turns=segment_turns_payload,
                )

                note_text = prediction.segment_level_note.strip()
                record = ContextNoteRecord(
                    conversation_id=conversation_id,
                    segment_index=len(dialogue_records) + 1,
                    start_turn_index=segment.start_turn_id,
                    end_turn_index=segment.end_turn_id,
                    start_turn_order=segment.start_turn_index,
                    end_turn_order=segment.end_turn_index,
                    turn_ids=segment.turn_ids,
                    context_scope=segment.context_scope,
                    note=note_text,
                )
                dialogue_records.append(record)

                if jsonl_stream is not None:
                    jsonl_stream.write(json.dumps(record.to_dict(), ensure_ascii=False))
                    jsonl_stream.write("\n")
                    jsonl_stream.flush()

                phase2_progress.set_postfix_str(f"Scope: {segment.context_scope}")
                phase2_progress.update(1)

                total_segment_time = perf_counter() - segment_start_time
                logger.info(
                    "Context note timings | conversation=%s segment=%d scope='%s' range=%s total=%.4fs",
                    conversation_id,
                    record.segment_index,
                    segment.context_scope,
                    f"Turns {record.start_turn_index}-{record.end_turn_index}",
                    total_segment_time,
                )

        logger.info("=" * 60)
        logger.info(
            "PHASE 2 COMPLETE: Generated %d scope-specific notes",
            len(dialogue_records),
        )
        logger.info("=" * 60)

        return dialogue_records

    def _format_segment_turns(self, segment: ScopeSegment) -> str:
        formatted_segment: List[Dict[str, Any]] = []
        for turn_index, turn in segment.turns:
            role = str(turn.role)
            utterance_raw = str(turn.content)
            utterance = extract_abbreviated_utterance(utterance_raw)
            formatted_segment.append(
                {
                    "turn_index": turn_index,
                    "role": role,
                    "utterance": utterance,
                }
            )
        return json.dumps(formatted_segment, ensure_ascii=False)


    def _format_prior_notes(
        self,
        records: Sequence[ContextNoteRecord],
        *,
        context_scope: Optional[str],
    ) -> str:
        if not context_scope:
            return "[]"
        filtered = [
            record.formatted_line
            for record in records
            if record.context_scope and record.context_scope == context_scope
        ]
        if not filtered:
            return "[]"
        window = filtered[-self._prior_notes_limit :]
        return json.dumps(window, ensure_ascii=False)

    def _format_scope_history(
        self,
        scope_records: Sequence[Dict[str, Any]],
    ) -> str:
        if not scope_records:
            return "[]"
        filtered = [
            {
                "turn_id": record["turn_id"],
                "role": record["role"],
                "context_scope": record["context_scope"],
            }
            for record in scope_records
            if record.get("context_scope")
        ]
        if not filtered:
            return "[]"
        window = filtered[-self._scope_history_window :]
        return json.dumps(window, ensure_ascii=False)

    def _build_scope_segments(
        self,
        turn_groups: Sequence[List[Tuple[int, Turn]]],
        scope_assignments: Dict[str, str],
    ) -> List[ScopeSegment]:
        segments: List[ScopeSegment] = []
        current_segment: Optional[ScopeSegment] = None

        # Walk turn-by-turn (even when groups contain multiple turns) so we
        # respect per-turn scope assignments and don't collapse mixed-scope
        # groups under the leading turn's scope.
        for group in turn_groups:
            for turn_index, turn in group:
                turn_id = str(turn.id)
                context_scope = scope_assignments.get(turn_id)
                if not context_scope:
                    # Skip turns without a scope (no anchor contribution)
                    continue

                if current_segment is None or current_segment.context_scope != context_scope:
                    if current_segment is not None and current_segment.turns:
                        segments.append(current_segment)
                    current_segment = ScopeSegment(context_scope=context_scope, turns=[])

                # current_segment guaranteed not None here
                current_segment.turns.append((turn_index, turn))

        if current_segment is not None and current_segment.turns:
            segments.append(current_segment)

        return segments

    def _build_turn_groups(
        self,
        turns: Sequence[Turn],
    ) -> List[List[Tuple[int, Turn]]]:
        if not self._group_consecutive_turns:
            # Use 1-indexed turn_ids for consistency
            return [[(idx, turn)] for idx, turn in enumerate(turns, start=1)]
        return self._group_consecutive_turns_by_role(turns)

    @staticmethod
    def _group_consecutive_turns_by_role(
        turns: Sequence[Turn],
    ) -> List[List[Tuple[int, Turn]]]:
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

    @staticmethod
    def _normalise_scope(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text if text else None

    def _group_turns_by_conversation(self, turns: Sequence[Turn]) -> Iterator[Tuple[str, List[Turn]]]:
        conversations: Dict[str, List[Turn]] = {}
        for turn in turns:
            turn_id = str(turn.id)
            conversation_id = _conversation_from_turn_id(turn_id)
            conversations.setdefault(conversation_id, []).append(turn)
        for conversation_id, dialogue_turns in conversations.items():
            yield conversation_id, dialogue_turns

    def _sort_single_trajectory_turns(self, turns: Sequence[Turn]) -> List[Turn]:
        """Sort turns by (conversation_id, turn_index, original_position)."""

        def _sort_key(item: Tuple[int, Turn]) -> Tuple[str, int, int]:
            original_idx, turn = item
            turn_id = str(turn.id)
            conversation_id = _conversation_from_turn_id(turn_id) or ""
            turn_index = _extract_turn_index_from_id(turn_id)
            # Fall back to original index to keep ordering stable when turn_index missing
            return (
                conversation_id,
                turn_index if turn_index is not None else original_idx,
                original_idx,
            )

        enumerated = list(enumerate(turns))
        enumerated.sort(key=_sort_key)
        return [turn for _, turn in enumerated]


def load_segment_level_note_records(path: str | Path) -> Dict[str, List[ContextNoteRecord]]:
    """Load context note records from a JSONL file and group by conversation."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Context note file {path} does not exist")
    records: Dict[str, List[ContextNoteRecord]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            conversation_id = str(payload.get("conversation_id"))
            start_turn_id = str(payload.get("start_turn_index", "")).strip()
            end_turn_id = str(payload.get("end_turn_index", "")) or start_turn_id
            start_turn_order = (
                _extract_turn_index_from_id(str(payload.get("start_turn_order", "")).strip())
                if payload.get("start_turn_order") is not None
                else None
            )
            end_turn_order = (
                _extract_turn_index_from_id(str(payload.get("end_turn_order", "")).strip())
                if payload.get("end_turn_order") is not None
                else None
            )
            record = ContextNoteRecord(
                conversation_id=conversation_id,
                segment_index=int(payload.get("segment_index", 0)),
                start_turn_index=start_turn_id,
                end_turn_index=end_turn_id,
                start_turn_order=start_turn_order,
                end_turn_order=end_turn_order,
                turn_ids=[str(item) for item in payload.get("turn_ids", [])],
                context_scope=payload.get("context_scope"),
                note=str(payload.get("note", "")),
            )
            records.setdefault(conversation_id, []).append(record)

    def _segment_sort_key(record: ContextNoteRecord, fallback: int) -> Tuple[int, int, int]:
        start_idx = record.start_turn_order or _extract_turn_index_from_id(record.start_turn_index)
        end_idx = record.end_turn_order or _extract_turn_index_from_id(record.end_turn_index) or start_idx
        start_val = start_idx if start_idx is not None else fallback
        end_val = end_idx if end_idx is not None else fallback
        return (start_val, end_val, record.segment_index)

    for conversation_id, conversation_records in list(records.items()):
        with_order = list(enumerate(conversation_records))
        with_order.sort(key=lambda item: _segment_sort_key(item[1], item[0]))
        records[conversation_id] = [rec for _, rec in with_order]
    return records


def _derive_default_segment_level_notes_path(config_path: Path, retrieval_cfg: Dict[str, Any]) -> Path:
    output_dir = retrieval_cfg.get("output_dir")
    strategy = retrieval_cfg.get("retrieval_strategy") or "context_reduction_note_taking"
    if output_dir:
        return Path(output_dir) / "note_logs" / f"{strategy}_segment_level_notes.jsonl"
    config_stem = config_path.stem
    return config_path.with_name(f"{config_stem}_segment_level_notes.jsonl")


def write_scope_assignments(
    assignments: Dict[str, Dict[str, str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # For single_trajectory, flatten the structure to remove conversation-level keys
    if len(assignments) == 1 and "single_trajectory" in assignments:
        # Flatten: just turn_id -> scope, no conversation-level nesting
        serialised: Dict[str, str] = {
            str(turn_id): str(scope)
            for turn_id, scope in assignments["single_trajectory"].items()
            if str(scope).strip()
        }
    else:
        # Keep nested structure for other datasets
        serialised: Dict[str, Dict[str, str]] = {}
        for conversation_id, mapping in assignments.items():
            serialised[conversation_id] = {
                str(turn_id): str(scope)
                for turn_id, scope in mapping.items()
                if str(scope).strip()
            }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serialised, f, ensure_ascii=False, indent=2)
    logger.info("Wrote context scope assignments to %s", output_path)


def load_scope_assignments(path: str | Path) -> Dict[str, Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scope assignment file {path} does not exist")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    
    # Handle flattened format (single_trajectory): {turn_id: scope}
    if payload and not any(isinstance(v, dict) for v in payload.values()):
        # Flattened format - wrap in single_trajectory
        assignments: Dict[str, Dict[str, str]] = {
            "single_trajectory": {
                str(turn_id): str(scope)
                for turn_id, scope in payload.items()
                if str(scope).strip() and str(turn_id).strip()
            }
        }
        return assignments
    
    # Handle nested format: {conversation_id: {turn_id: scope}}
    assignments: Dict[str, Dict[str, str]] = {}
    for conversation_id, mapping in payload.items():
        normalised: Dict[str, str] = {}
        for turn_id, scope in mapping.items():
            scope_text = str(scope).strip()
            if not scope_text:
                continue
            turn_id_text = str(turn_id).strip()
            if not turn_id_text:
                continue
            normalised[turn_id_text] = scope_text
        if normalised:
            assignments[str(conversation_id)] = normalised
    return assignments


def load_generation_config(
    config_path: str | Path,
    output_override: Optional[str],
    scope_output_override: Optional[str] = None,
) -> Tuple[
    str,
    str,
    LanguageModelProviderConfig,
    Path,
    int,
    int,
    Optional[Path],
]:
    config_path = Path(config_path)
    config_data = load_config(config_path)

    retrieval_cfg_raw = config_data.get("retrieval_config")
    retrieval_cfg: Dict[str, Any] = retrieval_cfg_raw if isinstance(retrieval_cfg_raw, dict) else {}

    context_cfg = config_data.get("context_reduction_retrieval_config")
    if not isinstance(context_cfg, dict):
        raise ValueError("context_reduction_retrieval_config section is required in the config")

    turns_path_value = context_cfg.get("turns_jsonl_path")
    if not turns_path_value:
        raise ValueError("turns_jsonl_path must be provided in context_reduction_retrieval_config")
    turns_path_str = str(turns_path_value)

    # Get dataset_name from context_reduction_retrieval_config (preferred) or retrieval_config (fallback)
    dataset_name = str(context_cfg.get("dataset_name") or retrieval_cfg.get("dataset_name") or "")

    lm_cfg_dict = context_cfg.get("language_model_provider_config")
    if not isinstance(lm_cfg_dict, dict):
        raise ValueError("language_model_provider_config must be provided in context_reduction_retrieval_config")
    lm_config = ParseDict(lm_cfg_dict, LanguageModelProviderConfig())

    scope_history_window_value = context_cfg.get("scope_history_window", 20)
    try:
        scope_history_window = int(scope_history_window_value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - config validation
        raise ValueError("scope_history_window must be an integer") from exc
    if scope_history_window <= 0:
        raise ValueError("scope_history_window must be positive")

    prior_notes_limit_value = context_cfg.get("prior_notes_limit", 8)
    try:
        prior_notes_limit = int(prior_notes_limit_value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - config validation
        raise ValueError("prior_notes_limit must be an integer") from exc
    if prior_notes_limit <= 0:
        raise ValueError("prior_notes_limit must be positive")

    if output_override is not None:
        output_path = Path(output_override)
    else:
        context_output_value = context_cfg.get("segment_level_notes_output_path")
        if context_output_value:
            output_path = Path(str(context_output_value))
        else:
            logger.warning("segment_level_notes_output_path not found in config, using default path")
            output_path = _derive_default_segment_level_notes_path(config_path, retrieval_cfg)

    if scope_output_override is not None:
        scope_output_path = Path(scope_output_override)
    else:
        context_scope_assignments_value = context_cfg.get("context_scope_assignments_output_path")
        scope_output_path = Path(str(context_scope_assignments_value)) if context_scope_assignments_value else None

    return (
        dataset_name,
        turns_path_str,
        lm_config,
        output_path,
        scope_history_window,
        prior_notes_limit,
        scope_output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and persist context summaries for dialogue segments."
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to context reduction retrieval JSON config file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output JSONL path; overrides segment_level_notes_output_path in the config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--group-consecutive-turns",
        dest="group_consecutive_turns",
        action="store_true",
        help="Group consecutive turns by role before processing (default).",
    )
    parser.add_argument(
        "--no-group-consecutive-turns",
        dest="group_consecutive_turns",
        action="store_false",
        help="Disable role-based grouping so every turn is processed individually.",
    )
    parser.set_defaults(group_consecutive_turns=True)
    parser.add_argument(
        "--max-conversations",
        type=int,
        help="Limit processing to the first N conversations.",
    )

    args = parser.parse_args()

    (
        dataset_name,
        turns_path,
        lm_config,
        output_path,
        scope_history_window,
        prior_notes_limit,
        scope_output_path,
    ) = load_generation_config(
        args.config,
        args.output,
    )

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file {output_path} already exists. Re-run with --overwrite to replace it."
        )

    if scope_output_path and scope_output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Scope assignment file {scope_output_path} already exists. Re-run with --overwrite to replace it."
        )

    maintainer = ContextNoteMaintainer(
        lm_config,
        scope_history_window=scope_history_window,
        prior_notes_limit=prior_notes_limit,
        dataset_name=dataset_name,
        group_consecutive_turns=args.group_consecutive_turns,
    )
    turns = load_turns(dataset_name, turns_path)
    logger.info(
        "Loaded %d turns for dataset '%s' from %s",
        len(turns),
        dataset_name or "<unspecified>",
        turns_path,
    )
    records, scope_assignments = maintainer.generate_segment_level_notes(
        turns,
        jsonl_path=output_path,
        max_conversations=args.max_conversations,
    )
    logger.info("Generated %d context notes and wrote them to %s", len(records), output_path)

    if scope_output_path:
        write_scope_assignments(scope_assignments, scope_output_path)


if __name__ == "__main__":
    main()
