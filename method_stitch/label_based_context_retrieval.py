"""Label-based retrieval pipeline with LLM-selected field filtering.

Replaces hierarchy construction with direct filtering on structured note fields:
- context_scope
- event_types  
- target


For each question:
1. LLM selects 0 to any number of values per field from all unique values in the dataset
2. Filter turns matching selected labels
3. Expand to include all consecutive turns for matched leading turns
4. Rank by representativeness (# of matching fields: 3 > 2 > 1)
5. Cap at max_label_selected_turns
6. Continue with embedding retrieval
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import dspy
import litellm
from google.protobuf.json_format import MessageToDict, ParseDict, ParseError
from qdrant_client import AsyncQdrantClient
from tqdm import tqdm
import threading

from came_bench.proto import EmbeddingModelProvider, LabelBasedContextRetrievalConfig
from came_bench.utils.lm import init_lm
from came_bench.utils.io import load_config, load_questions, get_lm_cost, load_turns
from .common_utils import (
    load_structured_turn_notes,
    _extract_turn_index_from_id,
    _conversation_from_turn_id,
)

logger = logging.getLogger(__name__)


@dataclass
class LabelSelectionCandidate:
    """Candidate value for a field."""
    field_name: str
    value: str


@dataclass(frozen=True)
class TurnIndexEntry:
    point_id: Union[str, int]
    turn_id: str
    note_id: Optional[str]


@dataclass
class QuestionRetrievalSummary:
    question_id: str
    question_content: str
    label_selected_turn_ids: List[str]
    label_selected_turn_details: List[Dict[str, object]]
    label_selected_turn_count: int
    embedding_candidate_turn_ids: List[str]
    embedding_candidate_turn_details: List[Dict[str, object]]
    embedding_candidate_turn_count: int
    selected_context_scopes: List[str]
    selected_event_types: List[str]
    selected_targets: List[str]
    selected_functional_type_seeds: List[str]
    context_scope_reasoning: str
    event_type_reasoning: str
    target_reasoning: str
    functional_type_reasoning: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "question_id": self.question_id,
            "question": self.question_content,
            "node_selected_turn_ids": self.label_selected_turn_ids,  # Keep same key for compatibility
            "node_selected_turn_details": self.label_selected_turn_details,
            "node_selected_turn_count": self.label_selected_turn_count,
            "embedding_candidate_turn_ids": self.embedding_candidate_turn_ids,
            "embedding_candidate_turn_details": self.embedding_candidate_turn_details,
            "embedding_candidate_turn_count": self.embedding_candidate_turn_count,
            "selected_context_scopes": self.selected_context_scopes,
            "selected_event_types": self.selected_event_types,
            "selected_targets": self.selected_targets,
            "selected_functional_type_seeds": self.selected_functional_type_seeds,
            "context_scope_reasoning": self.context_scope_reasoning,
            "event_type_reasoning": self.event_type_reasoning,
            "target_reasoning": self.target_reasoning,
            "functional_type_reasoning": self.functional_type_reasoning,
        }


DEFAULT_MAX_LABEL_SELECTED_TURNS = 20
DEFAULT_EMBEDDING_TOPK = 20
DEFAULT_MAX_USEFUL_LLM_TURNS = 12
DEFAULT_MAX_CONCURRENT_USEFULNESS_CHECKS = 10
DEFAULT_MAX_LLM_CHECKS = 39  # Maximum number of LLM usefulness checks (less than 40)


class ContextScopeLabelSelectionSignature(dspy.Signature):
    """
    Select appropriate number of labels for context scope field to filter conversation turns that can help answer the question.
    
    Your task is, read the label names to judge if the label is useful to answer the question. Don't overthink. The useful label names will implicitly or explicitly tell relevant information to the question. Then select an appropriate number(can be 0) of useful labels.
    Select the label names that are likely to be useful to answer the question.
    """

    question: str = dspy.InputField()
    context_scope_candidates: List[str] = dspy.InputField(
        description="All unique context_scope values in the dataset"
    )
    
    selected_context_scopes: List[str] = dspy.OutputField(
        description="An appropriate number of selected context_scope values (empty list if none relevant)"
    )
    reasoning: str = dspy.OutputField(
        description="Brief reasoning for selecting (or not selecting) the context_scope labels."
    )


class EventTypeLabelSelectionSignature(dspy.Signature):
    """
    Select appropriate number of labels for event type field to filter conversation turns that can help answer the question.

    Your task is, read the label names to judge if the label is useful to answer the question. Don't overthink. The useful label names will implicitly or explicitly tell relevant information to the question. Then select an appropriate number(can be 0) of useful labels.
    Select the label names that are likely to be useful to answer the question.
    """

    question: str = dspy.InputField()
    event_type_candidates: List[str] = dspy.InputField(
        description="All unique event type values in the dataset"
    )
    selected_event_types: List[str] = dspy.OutputField(
        description="An appropriate number of selected event type values (empty list if none relevant)"
    )
    reasoning: str = dspy.OutputField(
        description="Brief reasoning for selecting (or not selecting) the event type labels."
    )
    reasoning: str = dspy.OutputField(
        description="Reasoning for the selected event type values"
    )
class TargetLabelSelectionSignature(dspy.Signature):
    """
    Select appropriate number of labels for target field to filter conversation turns that can help answer the question.

    Your task is, read the label names to judge if the label is useful to answer the question. Don't overthink. The useful label names will implicitly or explicitly tell relevant information to the question. Then select an appropriate number(can be 0) of useful labels.
    Select the label names that are likely to be useful to answer the question.
    """

    question: str = dspy.InputField()
    target_candidates: List[str] = dspy.InputField(
        description="All unique target values in the dataset"
    )
    selected_targets: List[str] = dspy.OutputField(
        description="An appropriate number of selected target values (empty list if none relevant)"
    )
    reasoning: str = dspy.OutputField(
        description="Brief reasoning for selecting (or not selecting) the target labels."
    )
    reasoning: str = dspy.OutputField(
        description="Reasoning for the selected target values"
    )

class FunctionalTypeSeedsLabelSelectionSignature(dspy.Signature):
    """
    Select appropriate number of labels for functional type seeds field to filter conversation turns that can help answer the question.

    Your task is, read the label names to judge if the label is useful to answer the question. Don't overthink. The useful label names will implicitly or explicitly tell relevant information to the question. Then select an appropriate number(can be 0) of useful labels.
    Select the label names that are likely to be useful to answer the question.
    """

    question: str = dspy.InputField()
    functional_type_seeds_candidates: List[str] = dspy.InputField(
        description="All unique functional type seeds values in the dataset"
    )
    
    selected_functional_type_seeds: List[str] = dspy.OutputField(
        description="An appropriate number of selected functional type seeds values (empty list if none relevant)"
    )
    reasoning: str = dspy.OutputField(
        description="Brief reasoning for selecting (or not selecting) the functional type seed labels."
    )
    reasoning: str = dspy.OutputField(
        description="Reasoning for the selected functional type seeds values"
    )   


class TurnUsefulnessSignature(dspy.Signature):
    """
    Based on the turn content and the question, determine if the turn contains answer to the question.
    """

    question: str = dspy.InputField()
    turn_content: str = dspy.InputField()

    is_useful: bool = dspy.OutputField()


class RoleSensitivitySignature(dspy.Signature):
    """To answer the given question, determine if role information is explicitly needed to answer the question. If yes, output the role(s) to focus on. If not, output an empty list."""

    question: str = dspy.InputField(
        description="The evaluation question whose answer may depend on specific speaker roles."
    )
    available_roles: List[str] = dspy.InputField(
        description="List of unique speaker roles present in the candidate turn pool."
    )

    roles_to_focus: List[str] = dspy.OutputField(
        description="Optional list of roles to prioritize when role filtering is required."
    )


def extract_turn_index_from_turn_id(turn_id: str) -> Optional[int]:
    """Extract numeric turn index from turn_id string."""
    return _extract_turn_index_from_id(turn_id)


def load_turn_mapping(mapping_path: str) -> Dict[str, List[str]]:
    """Load turn mapping from JSON file.
    
    Returns dict mapping leading_turn_id -> [consecutive_turn_ids]
    """
    mapping_file = Path(mapping_path)
    if not mapping_file.exists():
        raise FileNotFoundError(f"Turn mapping file not found: {mapping_path}")
    
    with mapping_file.open("r", encoding="utf-8") as f:
        raw_mapping = json.load(f)
    
    if not isinstance(raw_mapping, dict):
        raise ValueError("Turn mapping file must contain an object mapping turn_id -> list of consecutive ids")

    turn_mapping: Dict[str, List[str]] = {}
    for key, values in raw_mapping.items():
        leader_id = str(key).strip()
        if not leader_id:
            continue
        if not isinstance(values, list):
            logger.warning("Skipping leader %s with non-list consecutive turns", leader_id)
            continue
        consecutive_turns = [str(v).strip() for v in values if str(v).strip()]
        if consecutive_turns:
            turn_mapping[leader_id] = consecutive_turns
    
    logger.info("Loaded turn mapping with %d leading turns", len(turn_mapping))
    return turn_mapping


def collect_label_candidates(
    structured_notes: Dict[str, Dict[str, Any]]
) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    """Collect all unique values for the label fields."""
    context_scopes: Set[str] = set()
    event_types: Set[str] = set()
    targets: Set[str] = set()
    functional_type_seeds: Set[str] = set()
    
    for note in structured_notes.values():
        if note.get("is_blank", False):
            continue
            
        # Collect context_scope
        scope = note.get("context_scope")
        if scope and isinstance(scope, str) and scope.strip():
            context_scopes.add(scope.strip())
        
        # Collect event_types (can be list or single value)
        event_types_val = note.get("event_types")
        if event_types_val:
            if isinstance(event_types_val, list):
                for et in event_types_val:
                    if et and isinstance(et, str) and et.strip():
                        event_types.add(et.strip())
            elif isinstance(event_types_val, str) and event_types_val.strip():
                event_types.add(event_types_val.strip())
        
        # Collect target
        target = note.get("target")
        if target and isinstance(target, str) and target.strip():
            targets.add(target.strip())

        # Collect functional type seeds
        func_types = note.get("functional_type_seeds")
        if func_types:
            if isinstance(func_types, list):
                for ft in func_types:
                    if ft and isinstance(ft, str) and ft.strip():
                        functional_type_seeds.add(ft.strip())
            elif isinstance(func_types, str) and func_types.strip():
                functional_type_seeds.add(func_types.strip())
    
    return context_scopes, event_types, targets, functional_type_seeds


def normalize_list_output(raw_output: Any) -> List[str]:
    """Normalize LLM output to list of strings."""
    if raw_output is None:
        return []
    if isinstance(raw_output, list):
        result = []
        for item in raw_output:
            if isinstance(item, str) and item.strip():
                result.append(item.strip())
        return result
    if isinstance(raw_output, str):
        text = raw_output.strip()
        if not text or text.lower() in ("none", "n/a", "[]"):
            return []
        try:
            parsed = json.loads(text)
            return normalize_list_output(parsed)
        except json.JSONDecodeError:
            # Split by comma if it looks like a comma-separated list
            if "," in text:
                return [item.strip() for item in text.split(",") if item.strip()]
            return [text]
    return []


def normalize_bool_output(raw_output: Any) -> bool:
    if isinstance(raw_output, bool):
        return raw_output
    if isinstance(raw_output, (int, float)):
        return bool(raw_output)
    if isinstance(raw_output, str):
        text = raw_output.strip().lower()
        if text in {"true", "yes", "y", "1"}:
            return True
        if text in {"false", "no", "n", "0"}:
            return False
    return False


def truncate_tokens(text: str, max_tokens: int) -> str:
    if not text:
        return ""
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens]) + " ..."


def build_turn_id_index_mapping(questions: List[Any]) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    """Build mapping from turn id strings to sequential indices per conversation.

    Returns:
        - global_turn_id_to_index: maps turn_id string -> extracted numeric turn index
        - conversation_turn_map: maps conversation_id -> {turn_id -> extracted turn index}
    """

    conversation_turn_map: Dict[str, Dict[str, int]] = {}
    global_turn_id_to_index: Dict[str, int] = {}

    for question in questions:
        question_conv = _conversation_from_turn_id(question.id)
        for turn_id in getattr(question, "question_turn_ids", []):
            turn_index = _extract_turn_index_from_id(turn_id)
            if turn_index is None:
                continue

            conv = _conversation_from_turn_id(turn_id) or question_conv
            if conv:
                conversation_turn_map.setdefault(conv, {})[turn_id] = turn_index

            # Use the numeric turn index so lookups align with structured note keys
            global_turn_id_to_index.setdefault(turn_id, turn_index)

    logger.info(
        "Constructed turn-id mapping for %d conversations covering %d unique turns",
        len(conversation_turn_map),
        len(global_turn_id_to_index),
    )
    return global_turn_id_to_index, conversation_turn_map


async def select_labels_with_llm(
    *,
    question_text: str,
    context_scope_candidates: Set[str],
    event_type_candidates: Set[str],
    target_candidates: Set[str],
    functional_type_candidates: Set[str],
    context_scope_selector: Optional[dspy.Predict],
    event_type_selector: Optional[dspy.Predict],
    target_selector: Optional[dspy.Predict],
    functional_type_selector: Optional[dspy.Predict],
    selection_lm: dspy.LM,
) -> Tuple[List[str], List[str], List[str], List[str], str, str, str, str]:
    """Use LLM to select 0 to any number of values per field (separate prompts per label type) and return reasoning.
    
    Parallelizes the 4 label selection calls for better performance.
    """

    async def _select_scope():
        if context_scope_selector is not None:
            def _call():
                with dspy.context(lm=selection_lm):
                    return context_scope_selector(
                        question=question_text,
                        context_scope_candidates=sorted(context_scope_candidates),
                    )
            return await asyncio.to_thread(_call)
        else:
            class EmptyScopeResult:
                selected_context_scopes = []
                reasoning = ""
            return EmptyScopeResult()
    
    async def _select_event():
        if event_type_selector is not None:
            def _call():
                with dspy.context(lm=selection_lm):
                    return event_type_selector(
                        question=question_text,
                        event_type_candidates=sorted(event_type_candidates),
                    )
            return await asyncio.to_thread(_call)
        else:
            class EmptyEventResult:
                selected_event_types = []
                reasoning = ""
            return EmptyEventResult()
    
    async def _select_target():
        if target_selector is not None:
            def _call():
                with dspy.context(lm=selection_lm):
                    return target_selector(
                        question=question_text,
                        target_candidates=sorted(target_candidates),
                    )
            return await asyncio.to_thread(_call)
        else:
            class EmptyTargetResult:
                selected_targets = []
                reasoning = ""
            return EmptyTargetResult()
    
    async def _select_functional():
        if functional_type_selector is not None:
            def _call():
                with dspy.context(lm=selection_lm):
                    return functional_type_selector(
                        question=question_text,
                        functional_type_seeds_candidates=sorted(functional_type_candidates),
                    )
            return await asyncio.to_thread(_call)
        else:
            # Temporarily disabled: return empty selection
            class EmptyFunctionalResult:
                selected_functional_type_seeds = []
                reasoning = ""
            return EmptyFunctionalResult()
    
    # Run all label selections in parallel
    scope_res, event_res, target_res, func_res = await asyncio.gather(
        _select_scope(),
        _select_event(),
        _select_target(),
        _select_functional(),
    )

    selected_scopes = normalize_list_output(scope_res.selected_context_scopes)
    selected_events = normalize_list_output(event_res.selected_event_types)
    selected_targets = normalize_list_output(target_res.selected_targets)
    selected_functional_types = normalize_list_output(func_res.selected_functional_type_seeds)

    scope_reason = str(getattr(scope_res, "reasoning", "") or "").strip()
    event_reason = str(getattr(event_res, "reasoning", "") or "").strip()
    target_reason = str(getattr(target_res, "reasoning", "") or "").strip()
    func_reason = str(getattr(func_res, "reasoning", "") or "").strip()

    logger.info(
        "Selected labels - scopes: %s, events: %s, targets: %s, functional types: %s",
        selected_scopes,
        selected_events,
        selected_targets,
        selected_functional_types,
    )
    
    return (
        selected_scopes,
        selected_events,
        selected_targets,
        selected_functional_types,
        scope_reason,
        event_reason,
        target_reason,
        func_reason,
    )


def rank_turns_by_representativeness(
    turn_ids: List[str],
    structured_notes: Dict[str, Dict[str, Any]],
    selected_scopes: List[str],
    selected_events: List[str],
    selected_targets: List[str],
    selected_functional_types: List[str],
) -> List[Tuple[str, int]]:
    """Rank turns by number of matching fields (3 > 2 > 1).
    
    Returns list of (turn_index, num_matches) tuples, sorted by num_matches descending.
    """
    scored_turns: List[Tuple[str, int, Tuple[int | float, str]]] = []  # (turn_id, num_matches, sort_key)
    
    for turn_id in turn_ids:
        note = structured_notes.get(turn_id)
        if not note or note.get("is_blank", False):
            continue
        
        num_matches = 0
        
        # Check context_scope
        scope_raw = note.get("context_scope")
        if scope_raw in (None, ""):
            logger.debug("Turn %s missing context_scope", turn_id)
        scope_raw = scope_raw if scope_raw is not None else ""
        scope = (
            scope_raw.strip()
            if isinstance(scope_raw, str)
            else str(scope_raw).strip() if scope_raw is not None
            else ""
        )
        if scope and scope in selected_scopes:
            num_matches += 1
        
        # Check event_types
        event_types_val = note.get("event_types")
        matched_event = False
        if event_types_val:
            if isinstance(event_types_val, list):
                for et in event_types_val:
                    if et and isinstance(et, str) and et.strip() in selected_events:
                        matched_event = True
                        break
            elif isinstance(event_types_val, str) and event_types_val.strip() in selected_events:
                matched_event = True
        if matched_event:
            num_matches += 1
        
        # Check target
        target_raw = note.get("target")
        if target_raw in (None, ""):
            logger.debug("Turn %s missing target", turn_id)
        target_raw = target_raw if target_raw is not None else ""
        target = (
            target_raw.strip()
            if isinstance(target_raw, str)
            else str(target_raw).strip() if target_raw is not None
            else ""
        )
        if target and target in selected_targets:
            num_matches += 1

        # Check functional type seeds
        func_values = note.get("functional_type_seeds")
        functional_match = False
        if func_values:
            if isinstance(func_values, list):
                for ft in func_values:
                    if ft and isinstance(ft, str) and ft.strip() in selected_functional_types:
                        functional_match = True
                        break
            elif isinstance(func_values, str) and func_values.strip() in selected_functional_types:
                functional_match = True
        if functional_match:
            num_matches += 1
        
        if num_matches > 0:
            sort_key = (_extract_turn_index_from_id(turn_id) or float("inf"), turn_id)
            scored_turns.append((turn_id, num_matches, sort_key))
    
    # Sort by num_matches (desc), then by turn order (asc) for stable ordering
    scored_turns.sort(key=lambda x: (-x[1], x[2]))
    
    return [(turn_id, num_matches) for turn_id, num_matches, _ in scored_turns]


async def build_note_lookup(
    client: AsyncQdrantClient,
    collection_name: str,
    *,
    turn_id_index_lookup: Dict[str, int],
) -> Dict[str, List[TurnIndexEntry]]:
    """Scan Qdrant collection to map note identifiers to point ids."""
    
    note_index: Dict[str, List[TurnIndexEntry]] = {}
    total_points = 0
    offset = None

    while True:
        points, next_offset = await client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=offset,
            with_payload=["id"],
        )
        if not points:
            break
        for point in points:
            payload = point.payload or {}
            turn_id = payload.get("id")
            if not isinstance(turn_id, str):
                continue
            entry = TurnIndexEntry(point_id=point.id, turn_id=turn_id, note_id=turn_id)
            total_points += 1
            note_index.setdefault(turn_id, []).append(entry)
        if next_offset is None:
            break
        offset = next_offset

    logger.info(
        "Indexed %d turns across %d note ids from collection '%s'",
        total_points,
        len(note_index),
        collection_name,
    )
    return note_index


def load_turn_content_from_jsonl(turns_jsonl_path: str) -> Dict[str, str]:
    """Load turn content from turns.jsonl file.
    
    Returns a dictionary mapping turn_id -> turn content.
    """
    turn_content_map: Dict[str, str] = {}
    turns_path = Path(turns_jsonl_path)
    
    if not turns_path.exists():
        logger.warning("Turns JSONL file not found: %s", turns_jsonl_path)
        return turn_content_map
    
    with turns_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                turn_data = json.loads(line)
                turn_id = str(turn_data.get("id", "")).strip()
                content = str(turn_data.get("content", "")).strip()
                if turn_id and content:
                    turn_content_map[turn_id] = content
            except (json.JSONDecodeError, KeyError) as exc:
                logger.debug("Failed to parse turn line: %s", exc)
                continue
    
    logger.info("Loaded turn content for %d turns from %s", len(turn_content_map), turns_jsonl_path)
    return turn_content_map


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


async def fetch_turn_payloads(
    client: AsyncQdrantClient,
    collection_name: str,
    point_ids: List[Union[str, int]],
    *,
    question_id: Optional[str] = None,
) -> Dict[Union[str, int], Dict[str, object]]:
    """Retrieve payloads for the specified point identifiers."""
    
    payloads: Dict[Union[str, int], Dict[str, object]] = {}
    if not point_ids:
        return payloads

    batch_size = 128
    for start in range(0, len(point_ids), batch_size):
        chunk = point_ids[start : start + batch_size]
        try:
            records = await client.retrieve(
                collection_name=collection_name,
                ids=list(chunk),
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            if question_id:
                logger.error(
                    "Failed to retrieve payloads for question %s (chunk size %d): %s",
                    question_id,
                    len(chunk),
                    exc,
                )
            else:
                logger.error("Failed to retrieve payloads for chunk: %s", exc)
            continue
        for record in records:
            payloads[record.id] = record.payload or {}
    return payloads


def build_embedding_text(note_text: str, turn_text: str) -> str:
    """Concatenate note text in front of turn content for embedding."""
    
    note_clean = note_text.strip()
    turn_clean = turn_text.strip()
    if not note_clean:
        return turn_clean
    if not turn_clean:
        return note_clean
    if note_clean.endswith((".", "!", "?")):
        return f"{note_clean} {turn_clean}"
    return f"{note_clean}. {turn_clean}"


def build_usefulness_turn_content(note: Dict[str, Any], *, turn_content: str = "") -> str:
    """Build JSON string for LLM usefulness filtering using structured note fields."""

    action = str(note.get("act") or "").strip() or None
    target = str(note.get("target") or "").strip() or None
    context_scope = str(note.get("context_scope") or "").strip() or None
    summary = str(note.get("note_text") or "").strip() or None
    utterance = truncate_tokens(turn_content.strip(), 200) if turn_content else None
    event_types_val = note.get("event_types") or []
    if isinstance(event_types_val, str):
        event_types_val = [event_types_val]
    event_types: List[str] = [
        str(et).strip() for et in event_types_val if str(et).strip()
    ]

    content = {
        "action": action,
        "target": target,
        "context_scope": context_scope,
        "event_types": event_types,
        "summary": summary,
        "utterance": utterance,
    }
    return json.dumps(content, ensure_ascii=True)


def build_enriched_turn_content(
    *,
    turn_content: str,
    note_text: str,
    role: str = "",
    act: str = "",
    target: str = "",
    context_scope: str = "",
    event_types: Optional[List[str]] = None,
) -> str:
    """Build enriched turn content with structured metadata."""
    
    role_text = role.strip()
    act_text = act.strip()
    target_text = target.strip()
    scope_text = context_scope.strip()
    note_clean = note_text.strip()
    turn_clean = turn_content.strip()
    
    context_parts = []
    if event_types:
        event_types_str = ", ".join(et for et in event_types if et)
        if event_types_str:
            context_parts.append(f"The event types of this turn are: {event_types_str}")
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
            context_line = f"Useful information about this turn: {context_description}. Specific note: {note_clean}"
        else:
            context_line = f"Useful information about this turn: {context_description}"
    else:
        context_line = f"Specific note about this turn: {note_clean}" if note_clean else ""
    
    enriched_parts = []
    if context_line:
        enriched_parts.append(context_line)
    
    return " | ".join(enriched_parts) if enriched_parts else turn_clean


def extract_embedding_kwargs(provider_config) -> Dict[str, object]:
    provider_name = EmbeddingModelProvider.Name(provider_config.provider).lower().replace(
        "embedding_model_provider_",
        "",
    )
    config_attr = f"{provider_name}_config"
    config = getattr(provider_config, config_attr, None)
    if config is None:
        return {}
    return MessageToDict(config, preserving_proto_field_name=True)


async def embed_texts(
    texts: List[str], *, model_name: str, provider: EmbeddingModelProvider, kwargs: Dict[str, object]
) -> List[List[float]]:
    if not texts:
        return []

    # Normalise inputs to non-empty strings; Azure rejects blank inputs.
    cleaned_texts: List[str] = []
    for idx, text in enumerate(texts):
        if text is None:
            continue
        text_str = str(text).strip()
        if not text_str:
            logger.debug("Skipping empty embedding input at index %d", idx)
            continue
        cleaned_texts.append(text_str)

    if not cleaned_texts:
        raise ValueError("No non-empty texts provided for embedding")

    adjusted_model_name = model_name
    params = dict(kwargs or {})
    if provider == EmbeddingModelProvider.EMBEDDING_MODEL_PROVIDER_AZURE_OPENAI:
        if not adjusted_model_name.startswith("azure/"):
            adjusted_model_name = f"azure/{model_name}"
        deployment_name = adjusted_model_name.split("/", 1)[1]
        params.setdefault("azure_deployment", deployment_name)
        params.setdefault("deployment_id", deployment_name)
        params.setdefault("deployment_name", deployment_name)
        params.setdefault("api_type", "azure")

    max_batch_size = 10  # DashScope embedding API limit is 10 per batch

    async def _run_async_batch(batch: List[str]) -> List[List[float]]:
        response = await litellm.aembedding(
            model=adjusted_model_name,
            input=list(batch),
            **params,
        )
        return [item["embedding"] for item in response.data]

    def _run_sync_batch(batch: List[str]) -> List[List[float]]:
        response = litellm.embedding(
            model=adjusted_model_name,
            input=list(batch),
            **params,
        )
        return [item["embedding"] for item in response.data]

    async def _run_batches_async() -> List[List[float]]:
        embeddings: List[List[float]] = []
        for start in range(0, len(cleaned_texts), max_batch_size):
            batch = cleaned_texts[start : start + max_batch_size]
            embeddings.extend(await _run_async_batch(batch))
        return embeddings

    def _run_batches_sync() -> List[List[float]]:
        embeddings: List[List[float]] = []
        for start in range(0, len(cleaned_texts), max_batch_size):
            batch = cleaned_texts[start : start + max_batch_size]
            embeddings.extend(_run_sync_batch(batch))
        return embeddings

    async def _run_async_with_retries(max_attempts: int = 3, base_delay: float = 1.0) -> List[List[float]]:
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await _run_batches_async()
            except (litellm.exceptions.APIConnectionError, json.JSONDecodeError) as exc:
                last_exc = exc
                if attempt == max_attempts:
                    logger.error(
                        "Async embedding failed after %d attempts (%s): %s",
                        attempt,
                        type(exc).__name__,
                        exc,
                    )
                    break
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Async embedding attempt %d/%d failed (%s). Retrying in %.1fs",
                    attempt,
                    max_attempts,
                    type(exc).__name__,
                    delay,
                )
                await asyncio.sleep(delay)
            except AttributeError:
                # Older litellm versions may lack aembedding; fall back to sync path
                break

        logger.warning("Falling back to sync embedding call after async failures")
        return await asyncio.to_thread(_run_batches_sync)

    try:
        return await _run_async_with_retries()
    except AttributeError:
        # If async isn't available at all, go straight to sync
        return await asyncio.to_thread(_run_batches_sync)


def _require_non_empty(value: str, field_name: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must be provided in the config")
    return value.strip()


def _require_existing_file(value: str, field_name: str) -> Path:
    path = Path(_require_non_empty(value, field_name))
    if not path.exists():
        raise FileNotFoundError(f"{field_name} file not found: {path}")
    return path


def load_label_based_config(config_path: str) -> Tuple[LabelBasedContextRetrievalConfig, Optional[Path], bool]:
    raw_config = load_config(config_path)
    payload = dict(raw_config)
    label_selection_output_path = payload.pop("label_selection_output_path", None)
    skip_label_filtering = payload.pop("skip_label_filtering", False)  # Extract skip_label_filtering parameter
    config = LabelBasedContextRetrievalConfig()
    try:
        ParseDict(payload, config)
    except ParseError as exc:
        raise ValueError(
            f"Failed to parse LabelBasedContextRetrievalConfig at {config_path}: {exc}"
        ) from exc
    final_path = Path(label_selection_output_path) if label_selection_output_path else None
    return config, final_path, skip_label_filtering


def load_existing_results(
    output_path: Path,
    label_selection_output_path: Optional[Path] = None,
) -> Tuple[Set[str], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Load existing results from output files.
    
    Returns:
        - Set of processed question IDs
        - Dictionary mapping question_id -> question result dict (from main output)
        - Dictionary mapping question_id -> label selection dict (from label selection output)
    """
    processed_question_ids: Set[str] = set()
    existing_results: Dict[str, Dict[str, Any]] = {}
    existing_label_selections: Dict[str, Dict[str, Any]] = {}
    temp_path = output_path.with_suffix(".jsonl.tmp")
    
    # Load main output file
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as f:
                output_data = json.load(f)
                question_results = output_data.get("question_results", [])
                for result in question_results:
                    question_id = result.get("question_id")
                    if question_id:
                        processed_question_ids.add(question_id)
                        existing_results[question_id] = result
            logger.info(
                "Loaded %d existing results from %s",
                len(processed_question_ids),
                output_path,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load existing results from %s: %s. Starting fresh.",
                output_path,
                exc,
            )
    # If main output missing, fall back to temp JSONL (from interrupted run)
    elif temp_path.exists():
        try:
            with temp_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    summary = entry.get("summary", {})
                    question_id = summary.get("question_id")
                    if question_id:
                        processed_question_ids.add(question_id)
                        existing_results[question_id] = summary
            logger.info(
                "Loaded %d partial results from temp file %s",
                len(processed_question_ids),
                temp_path,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load partial results from %s: %s. Ignoring temp file.",
                temp_path,
                exc,
            )
    
    # Load label selection output file
    if label_selection_output_path and label_selection_output_path.exists():
        try:
            with label_selection_output_path.open("r", encoding="utf-8") as f:
                label_data = json.load(f)
                if isinstance(label_data, list):
                    for record in label_data:
                        question_id = record.get("question_id")
                        if question_id:
                            existing_label_selections[question_id] = record
                elif isinstance(label_data, dict):
                    # Handle dict format if needed
                    pass
            logger.info(
                "Loaded %d existing label selections from %s",
                len(existing_label_selections),
                label_selection_output_path,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load existing label selections from %s: %s",
                label_selection_output_path,
                exc,
            )
    
    return processed_question_ids, existing_results, existing_label_selections


def save_results_incremental(
    output_path: Path,
    summaries: List[QuestionRetrievalSummary],
    max_label_selected_turns: int,
    embedding_topk: int,
    lock: threading.Lock,
) -> None:
    """Save results incrementally to output file (thread-safe)."""
    with lock:
        # Load existing results if file exists
        existing_question_ids, existing_results, _ = load_existing_results(output_path)
        
        # Build result map from summaries
        new_results: Dict[str, Dict[str, Any]] = {
            summary.question_id: summary.to_dict() for summary in summaries
        }
        
        # Merge with existing results (new results override old ones)
        all_results = {**existing_results, **new_results}
        
        # Write merged results
        output_payload = {
            "retrieval_strategy": "label_based_filtering",
            "max_label_selected_turns": max_label_selected_turns,
            "embedding_topk": embedding_topk,
            "question_results": list(all_results.values()),
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as outfile:
            json.dump(output_payload, outfile, indent=2)


def save_label_selections_incremental(
    label_selection_output_path: Path,
    summaries: List[QuestionRetrievalSummary],
    lock: threading.Lock,
) -> None:
    """Save label selections incrementally to output file (thread-safe)."""
    with lock:
        # Load existing label selections if file exists
        _, _, existing_label_selections = load_existing_results(
            Path("/dev/null"),  # Dummy path, we only need label selections
            label_selection_output_path,
        )
        
        # Build new label selection records
        new_label_records: Dict[str, Dict[str, Any]] = {}
        for summary in summaries:
            new_label_records[summary.question_id] = {
                "question_id": summary.question_id,
                "question": summary.question_content,
                "selected_context_scopes": summary.selected_context_scopes,
                "context_scope_reasoning": summary.context_scope_reasoning,
                "selected_event_types": summary.selected_event_types,
                "event_type_reasoning": summary.event_type_reasoning,
                "selected_targets": summary.selected_targets,
                "target_reasoning": summary.target_reasoning,
                "selected_functional_type_seeds": summary.selected_functional_type_seeds,
                "functional_type_reasoning": summary.functional_type_reasoning,
            }
        
        # Merge with existing records (new records override old ones)
        all_label_records = {**existing_label_selections, **new_label_records}
        
        # Write merged records
        label_selection_output_path.parent.mkdir(parents=True, exist_ok=True)
        with label_selection_output_path.open("w", encoding="utf-8") as label_file:
            json.dump(list(all_label_records.values()), label_file, indent=2)


async def process_single_question(
    question: Any,
    index: int,
    total_questions: int,
    *,
    global_turn_id_index: Dict[str, int],
    structured_notes: Dict[str, Dict[str, Any]],
    turn_mapping: Dict[str, List[str]],
    note_lookup: Optional[Dict[str, List[TurnIndexEntry]]],
    qdrant_client: Optional[AsyncQdrantClient],
    collection_name: Optional[str],
    turn_content_map: Optional[Dict[str, str]],
    selection_lm: dspy.LM,
    context_scope_selector: Optional[dspy.Predict],
    event_type_selector: Optional[dspy.Predict],
    target_selector: Optional[dspy.Predict],
    functional_type_selector: Optional[dspy.Predict],
    role_selector: dspy.Predict,
    turn_usefulness_selector: dspy.Predict,
    embedding_provider_config: Any,
    embedding_model_name: str,
    embedding_kwargs: Dict[str, object],
    max_label_selected_turns: int,
    embedding_topk: int,
    skip_label_filtering: bool = False,
) -> QuestionRetrievalSummary:
    """Process a single question and return its retrieval summary."""
    logger.info("Processing question %d/%d: %s", index, total_questions, question.id)
    _ = global_turn_id_index  # retained for compatibility with legacy callers

    question_conv = _conversation_from_turn_id(question.id)
    allowed_turn_ids: Set[str] = set()
    for turn_id in getattr(question, "question_turn_ids", []):
        if turn_id in structured_notes:
            allowed_turn_ids.add(turn_id)

    if not allowed_turn_ids:
        logger.warning(
            "Question %s has no mapped turn ids in structured notes; using all turns",
            question.id,
        )
        allowed_turn_ids = set(structured_notes.keys())

    # Collect question-specific label candidates
    question_scopes: Set[str] = set()
    question_events: Set[str] = set()
    question_targets: Set[str] = set()
    question_roles: Set[str] = set()
    question_functional_types: Set[str] = set()

    for turn_id in allowed_turn_ids:
        note = structured_notes.get(turn_id)
        if not note or note.get("is_blank", False):
            continue
        scope = str(note.get("context_scope", "")).strip()
        if scope:
            question_scopes.add(scope)
        event_val = note.get("event_types")
        if isinstance(event_val, list):
            for et in event_val:
                if isinstance(et, str) and et.strip():
                    question_events.add(et.strip())
        elif isinstance(event_val, str) and event_val.strip():
            question_events.add(event_val.strip())
        target_val = str(note.get("target", "")).strip()
        if target_val:
            question_targets.add(target_val)
        role_val = str(note.get("role", "")).strip()
        if role_val:
            question_roles.add(role_val)
        func_values = note.get("functional_type_seeds")
        if func_values:
            if isinstance(func_values, list):
                for ft in func_values:
                    if isinstance(ft, str) and ft.strip():
                        question_functional_types.add(ft.strip())
            elif isinstance(func_values, str) and func_values.strip():
                question_functional_types.add(func_values.strip())

    if not question_scopes and not question_events and not question_targets and not question_functional_types:
        (
            global_scopes,
            global_events,
            global_targets,
            global_functional_types,
        ) = collect_label_candidates(
            structured_notes
        )
        question_scopes = global_scopes
        question_events = global_events
        question_targets = global_targets
        question_functional_types = global_functional_types

    # Vanilla RAG shortcut: skip label filtering if configured
    if skip_label_filtering:
        logger.info(
            "skip_label_filtering=True for %s: Bypassing Steps 1-4, directly using all %d allowed turns for semantic similarity ranking",
            question.id,
            len(allowed_turn_ids),
        )
        turns_for_semantic_ranking = list(allowed_turn_ids)

        # Set empty values for label selection results (for output compatibility)
        selected_scopes = []
        selected_events = []
        selected_targets = []
        selected_functional_types = []
        scope_reason = "Skipped (vanilla RAG mode)"
        event_reason = "Skipped (vanilla RAG mode)"
        target_reason = "Skipped (vanilla RAG mode)"
        func_reason = "Skipped (vanilla RAG mode)"
        ranked_turns = []
        role_filter_set = None  # No role filtering in vanilla RAG mode
    else:
        # Original STITCH pipeline: Steps 1-4

        # Role filtering (execute before label selection)
        role_filter_set: Optional[Set[str]] = None
        if question_roles:
            with dspy.context(lm=selection_lm):
                role_result = role_selector(
                    question=question.content,
                    available_roles=sorted(question_roles),
                )
            roles_focus = normalize_list_output(role_result.roles_to_focus)

            if roles_focus:
                normalized_roles = {
                    role.lower(): role for role in question_roles
                }
                selected_roles = {
                    normalized_roles.get(role.lower())
                    for role in roles_focus
                    if isinstance(role, str) and role.lower() in normalized_roles
                }
                selected_roles.discard(None)
                if selected_roles:
                    role_filter_set = set(selected_roles)
                    logger.info(
                        "Role filtering enabled for %s with roles: %s",
                        question.id,
                        sorted(role_filter_set),
                    )
            else:
                logger.info("Role filtering not required for %s", question.id)

        # Step 1: Label selection with LLM (parallelized)
        (
            selected_scopes,
            selected_events,
            selected_targets,
            selected_functional_types,
            scope_reason,
            event_reason,
            target_reason,
            func_reason,
        ) = await select_labels_with_llm(
            question_text=question.content,
            context_scope_candidates=question_scopes,
            event_type_candidates=question_events,
            target_candidates=question_targets,
            functional_type_candidates=question_functional_types,
            context_scope_selector=context_scope_selector,
            event_type_selector=event_type_selector,
            target_selector=target_selector,
            functional_type_selector=functional_type_selector,
            selection_lm=selection_lm,
        )
    
        # Step 2: Filter turns by selected labels
        # Build a reverse mapping to identify which turns are consecutive (not leaders)
        consecutive_turn_set: Set[str] = set()
        for leader, consecutive_list in turn_mapping.items():
            if allowed_turn_ids and leader not in allowed_turn_ids:
                continue
            for follower in consecutive_list:
                if not allowed_turn_ids or follower in allowed_turn_ids:
                    consecutive_turn_set.add(follower)
        
        # Identify which turns to evaluate for label matching
        # - Leading turns (keys in turn_mapping)
        # - Standalone turns (not in mapping at all)
        matched_leading_turns: Set[str] = set()
        matched_standalone_turns: Set[str] = set()
        
        for turn_id, note in structured_notes.items():
            if allowed_turn_ids and turn_id not in allowed_turn_ids:
                continue
            if note.get("is_blank", False):
                continue
    
            # Skip consecutive turns - they'll be added when their leader matches
            if turn_id in consecutive_turn_set:
                continue
    
            if role_filter_set:
                note_role = str(note.get("role", "")).strip()
                if note_role not in role_filter_set:
                    continue
    
            matches = False
            
            # Check context_scope
            scope_raw = note.get("context_scope", "")
            scope = (
                scope_raw.strip()
                if isinstance(scope_raw, str)
                else str(scope_raw).strip() if scope_raw is not None
                else ""
            )
            if scope and scope in selected_scopes:
                matches = True
            
            # Check event_types
            event_types_val = note.get("event_types")
            if event_types_val:
                if isinstance(event_types_val, list):
                    for et in event_types_val:
                        if et and isinstance(et, str) and et.strip() in selected_events:
                            matches = True
                            break
                elif isinstance(event_types_val, str) and event_types_val.strip() in selected_events:
                    matches = True
            
            # Check target
            target_raw = note.get("target", "")
            target = (
                target_raw.strip()
                if isinstance(target_raw, str)
                else str(target_raw).strip() if target_raw is not None
                else ""
            )
            if target and target in selected_targets:
                matches = True
    
            # Check functional type seeds
            func_values = note.get("functional_type_seeds")
            if func_values and selected_functional_types:
                if isinstance(func_values, list):
                    for ft in func_values:
                        if ft and isinstance(ft, str) and ft.strip() in selected_functional_types:
                            matches = True
                            break
                elif isinstance(func_values, str) and func_values.strip() in selected_functional_types:
                    matches = True
            
            if matches:
                # Check if this is a leading turn or standalone turn
                if turn_id in turn_mapping:
                    matched_leading_turns.add(turn_id)
                else:
                    matched_standalone_turns.add(turn_id)
        
        # Expand to include all consecutive turns for matched leading turns
        candidate_turn_ids: Set[str] = set()
        
        # Add standalone turns (no expansion needed)
        candidate_turn_ids.update(matched_standalone_turns)
        
        # Add leading turns and their consecutive turns
        for leading_turn in matched_leading_turns:
            if allowed_turn_ids and leading_turn not in allowed_turn_ids:
                continue
            candidate_turn_ids.add(leading_turn)
            consecutive_turns = turn_mapping.get(leading_turn, [])
            for follower in consecutive_turns:
                if not allowed_turn_ids or follower in allowed_turn_ids:
                    candidate_turn_ids.add(follower)
        
        logger.info(
            "Label filtering: %d leading + %d standalone turns matched -> %d total turns (with consecutive)",
            len(matched_leading_turns),
            len(matched_standalone_turns),
            len(candidate_turn_ids),
        )
    
        # Step 3: Rank by representativeness
        ranked_turns = rank_turns_by_representativeness(
            list(candidate_turn_ids),
            structured_notes,
            selected_scopes,
            selected_events,
            selected_targets,
            selected_functional_types,
        )
    
        logger.info("Ranked %d turns by label density", len(ranked_turns))
        
        # Check if we need fallback to semantic similarity
        # If 0 turns selected, fall back to semantic similarity on all allowed turns
        use_semantic_fallback = len(candidate_turn_ids) == 0
    
        if use_semantic_fallback:
            logger.info(
                "Label selection yielded %d turns. Falling back to semantic similarity ranking on all %d allowed turns",
                len(candidate_turn_ids),
                len(allowed_turn_ids),
            )
            # Use all allowed turns for semantic similarity ranking (skip usefulness filtering)
            turns_for_semantic_ranking = list(allowed_turn_ids)
            ranked_turns = []  # Will skip usefulness filtering and go straight to semantic similarity
        else:
            # Step 4: LLM usefulness filtering in rank order (only if we have label-selected turns)
            # Stop when: (1) 20 useful turns found, OR (2) checked max_checks (capped at DEFAULT_MAX_LLM_CHECKS)
            initial_ranked_count = len(ranked_turns)
            max_checks = min(DEFAULT_MAX_LLM_CHECKS, max(1, initial_ranked_count // 2))  # Cap at DEFAULT_MAX_LLM_CHECKS, but at least 1
    
            # Fetch turn content for usefulness filtering (all ranked turns)
            turn_content_lookup: Dict[str, str] = {}
            if ranked_turns:
                turns_for_payload = [turn_idx for turn_idx, _ in ranked_turns]
                
                # Try to get turn content from turn_content_map first (from turns.jsonl)
                if turn_content_map:
                    for turn_idx in turns_for_payload:
                        content = turn_content_map.get(turn_idx, "").strip()
                        if content:
                            turn_content_lookup[turn_idx] = content
                
                # Fallback to Qdrant if turn_content_map doesn't have all turns
                missing_turns = [turn_idx for turn_idx in turns_for_payload if turn_idx not in turn_content_lookup]
                if missing_turns and note_lookup and qdrant_client and collection_name:
                    point_ids: List[Union[str, int]] = []
                    turn_entries: Dict[str, List[TurnIndexEntry]] = {}
                    for turn_idx in missing_turns:
                        entries = note_lookup.get(turn_idx)
                        if entries:
                            turn_entries[turn_idx] = entries
                            point_ids.extend(entry.point_id for entry in entries)
                    if point_ids:
                        payloads = await fetch_turn_payloads(
                            qdrant_client,
                            collection_name,
                            list(set(point_ids)),
                        )
                        for turn_idx, entries in turn_entries.items():
                            for entry in entries:
                                payload = payloads.get(entry.point_id, {})
                                content = str(payload.get("content", "")).strip()
                                if content:
                                    turn_content_lookup[turn_idx] = content
                                    break
    
            filtered_ranked_turns: List[Tuple[str, int]] = []
            checks_performed = 0
    
            for turn_idx, num_matches in ranked_turns:
                # Stop if we've found enough useful turns
                if len(filtered_ranked_turns) >= DEFAULT_MAX_USEFUL_LLM_TURNS:
                    logger.info(
                        "Reached usefulness cap of %d accepted turns; stopping checks",
                        DEFAULT_MAX_USEFUL_LLM_TURNS,
                    )
                    break
                
                # Stop if we've reached the max check limit
                if checks_performed >= max_checks:
                    logger.info(
                        "Reached check limit of %d (capped from DEFAULT_MAX_LLM_CHECKS of %d ranked turns); stopping checks",
                        DEFAULT_MAX_LLM_CHECKS,
                        initial_ranked_count,
                    )
                    break
    
                note = structured_notes.get(turn_idx)
                if not note:
                    continue
    
                turn_content = build_usefulness_turn_content(
                    note,
                    turn_content=turn_content_lookup.get(turn_idx, ""),
                )
    
                with dspy.context(lm=selection_lm):
                    usefulness_res = turn_usefulness_selector(
                        question=question.content,
                        turn_content=turn_content,
                    )
                is_useful = normalize_bool_output(getattr(usefulness_res, "is_useful", None))
                checks_performed += 1
    
                logger.info(
                    "Usefulness LLM response for turn %s: is_useful=%s raw=%s (check %d/%d)",
                    turn_idx,
                    is_useful,
                    usefulness_res,
                    checks_performed,
                    max_checks,
                )
    
                if is_useful:
                    filtered_ranked_turns.append((turn_idx, num_matches))
                else:
                    logger.debug("Turn %s filtered out by usefulness check", turn_idx)
    
            ranked_turns = filtered_ranked_turns
            logger.info(
                "Usefulness filtering kept %d/%d ranked turns",
                len(ranked_turns),
                initial_ranked_count,
            )
            # Use label-selected turns for semantic similarity ranking
            # (will be ranked by semantic similarity and top 20 selected)
            if ranked_turns:
                turns_for_semantic_ranking = [turn_idx for turn_idx, _ in ranked_turns]
            else:
                # All label-selected turns were rejected; fall back to full allowed pool
                logger.info(
                    "Usefulness filtering removed all label-selected turns; falling back to semantic similarity on all %d allowed turns",
                    len(allowed_turn_ids),
                )
                turns_for_semantic_ranking = list(allowed_turn_ids)

    # Step 5: Always rank by semantic similarity and select top 20
    # Convert to turn entries for semantic similarity ranking
    candidate_entries: List[TurnIndexEntry] = []
    for turn_idx in turns_for_semantic_ranking:
        if note_lookup:
            entries = note_lookup.get(turn_idx)
            if entries:
                candidate_entries.extend(entries)
        else:
            # If no note_lookup (Qdrant not available), create entry directly from turn_id
            entry = TurnIndexEntry(point_id=turn_idx, turn_id=turn_idx, note_id=turn_idx)
            candidate_entries.append(entry)

    # Deduplicate by turn_id
    unique_entries: Dict[str, TurnIndexEntry] = {}
    for entry in candidate_entries:
        unique_entries[entry.turn_id] = entry
    candidate_entries = [unique_entries[tid] for tid in unique_entries]

    # Fetch turn content for semantic similarity ranking
    turn_content_lookup_semantic: Dict[str, str] = {}
    if turn_content_map:
        for entry in candidate_entries:
            content = turn_content_map.get(entry.turn_id, "").strip()
            if content:
                turn_content_lookup_semantic[entry.turn_id] = content
    
    # Fill missing from Qdrant if available
    missing_turns_semantic = [entry.turn_id for entry in candidate_entries if entry.turn_id not in turn_content_lookup_semantic]
    if missing_turns_semantic and note_lookup and qdrant_client and collection_name:
        point_ids: List[Union[str, int]] = []
        turn_entries_semantic: Dict[str, List[TurnIndexEntry]] = {}
        for entry in candidate_entries:
            if entry.turn_id in missing_turns_semantic:
                if entry.turn_id not in turn_entries_semantic:
                    turn_entries_semantic[entry.turn_id] = []
                turn_entries_semantic[entry.turn_id].append(entry)
                point_ids.append(entry.point_id)
        
        if point_ids:
            payloads = await fetch_turn_payloads(
                qdrant_client,
                collection_name,
                list(set(point_ids)),
                question_id=question.id,
            )
            for entry in candidate_entries:
                if entry.turn_id in missing_turns_semantic:
                    payload = payloads.get(entry.point_id, {})
                    content = str(payload.get("content", "")).strip()
                    if content:
                        turn_content_lookup_semantic[entry.turn_id] = content
                        break

    # Embedding retrieval - always rank by semantic similarity and select top 20
    embedding_candidate_ids_snapshot: List[str] = []
    embedding_candidate_details_snapshot: List[Dict[str, object]] = []
    label_selected_turn_ids: List[str] = []
    label_selected_turn_details: List[Dict[str, object]] = []

    if candidate_entries:
        # Embed question
        question_embeddings = await embed_texts(
            [question.content],
            model_name=embedding_model_name,
            provider=embedding_provider_config.provider,
            kwargs=embedding_kwargs,
        )
        if not question_embeddings:
            raise RuntimeError(f"Failed to embed question {question.id}")
        question_embedding = question_embeddings[0]

        # Build payloads from turn_content_lookup_semantic
        payloads: Dict[Union[str, int], Dict[str, object]] = {}
        for entry in candidate_entries:
            content = turn_content_lookup_semantic.get(entry.turn_id, "").strip()
            if content:
                payloads[entry.point_id] = {"content": content}

        # Build embedding inputs and metadata
        embedding_inputs: List[str] = []
        entry_embeddings_meta: List[Tuple[TurnIndexEntry, Dict[str, object]]] = []
        entry_metadata: Dict[str, Dict[str, object]] = {}
        
        for entry in candidate_entries:
            payload = payloads.get(entry.point_id, {})
            turn_content = str(payload.get("content", ""))
            
            # Get note metadata
            note = structured_notes.get(entry.note_id, {})
            note_text = str(note.get("note_text", "")).strip()
            role = str(note.get("role", "")).strip()
            act = str(note.get("act", "")).strip()
            target = str(note.get("target", "")).strip()
            context_scope = str(note.get("context_scope", "")).strip()
            event_types_val = note.get("event_types", [])
            if not isinstance(event_types_val, list):
                event_types_val = [event_types_val] if event_types_val else []
            
            metadata = {
                "turn_id": entry.turn_id,
                "note_id": entry.note_id,
                "note_text": note_text,
                "turn_content": turn_content,
                "context_scope": context_scope,
                "event_types": event_types_val,
                "target": target,
                "role": role,
                "act": act,
            }
            entry_metadata[entry.turn_id] = metadata
            
            enriched_turn_content = build_enriched_turn_content(
                turn_content=turn_content,
                note_text=note_text,
                role=role,
                act=act,
                target=target,
                context_scope=context_scope,
                event_types=event_types_val,
            )
            metadata["enriched_turn_content"] = enriched_turn_content
            
            # For embedding, use note + turn
            embedding_text = build_embedding_text(note_text, turn_content)
            if not embedding_text.strip():
                # Skip entries that have neither note nor content to avoid invalid embedding input
                logger.debug(
                    "Skipping embedding for turn %s (empty note/content)",
                    entry.turn_id,
                )
                continue

            embedding_inputs.append(embedding_text)
            entry_embeddings_meta.append((entry, metadata))

        # Embed candidates
        candidate_embeddings = await embed_texts(
            embedding_inputs,
            model_name=embedding_model_name,
            provider=embedding_provider_config.provider,
            kwargs=embedding_kwargs,
        )

        # Score and rank
        scored_candidates: List[Tuple[float, TurnIndexEntry, Dict[str, object]]] = []
        for embedding, (entry, metadata) in zip(
            candidate_embeddings, entry_embeddings_meta
        ):
            similarity = cosine_similarity(question_embedding, embedding)
            scored_candidates.append((similarity, entry, metadata))

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        
        # Always take top 20 based on semantic similarity (or embedding_topk, whichever is larger)
        top_k = max(20, embedding_topk)
        top_candidates = scored_candidates[:top_k]
        
        # Set label_selected_turn_ids to top 20 from semantic similarity
        label_selected_turn_ids = [entry.turn_id for _, entry, _ in top_candidates]
        
        # Build label_selected_turn_details from top candidates
        for rank, (similarity, entry, metadata) in enumerate(top_candidates, start=1):
            record = {
                "turn_id": entry.turn_id,
                "note_id": entry.note_id,
                "note_text": metadata.get("note_text", ""),
                "context_scope": metadata.get("context_scope", ""),
                "event_types": metadata.get("event_types", []),
                "target": metadata.get("target", ""),
                "role": metadata.get("role", ""),
                "act": metadata.get("act", ""),
            }
            label_selected_turn_details.append(record)
        
        # Take top-k for embedding candidates (embedding_topk)
        for rank, (similarity, entry, metadata) in enumerate(
            scored_candidates[:embedding_topk],
            start=1,
        ):
            metadata["embedding_rank"] = rank
            metadata["embedding_similarity"] = similarity
            
            embedding_candidate_ids_snapshot.append(entry.turn_id)
            embedding_candidate_details_snapshot.append(
                {
                    "turn_id": entry.turn_id,
                    "embedding_rank": rank,
                    "embedding_score": similarity,
                    "note_id": entry.note_id,
                    "note_text": metadata.get("note_text", ""),
                    "turn_content": metadata.get("turn_content", ""),
                    "enriched_turn_content": metadata.get("enriched_turn_content", ""),
                    "context_scope": metadata.get("context_scope", ""),
                    "event_types": metadata.get("event_types", []),
                    "target": metadata.get("target", ""),
                    "role": metadata.get("role", ""),
                    "act": metadata.get("act", ""),
                }
            )

    return QuestionRetrievalSummary(
        question_id=question.id,
        question_content=question.content,
        label_selected_turn_ids=label_selected_turn_ids,
        label_selected_turn_details=label_selected_turn_details,
        label_selected_turn_count=len(label_selected_turn_ids),
        embedding_candidate_turn_ids=embedding_candidate_ids_snapshot,
        embedding_candidate_turn_details=embedding_candidate_details_snapshot,
        embedding_candidate_turn_count=len(embedding_candidate_ids_snapshot),
        selected_context_scopes=selected_scopes,
        selected_event_types=selected_events,
        selected_targets=selected_targets,
        selected_functional_type_seeds=selected_functional_types,
        context_scope_reasoning=scope_reason,
        event_type_reasoning=event_reason,
        target_reasoning=target_reason,
        functional_type_reasoning=func_reason,
    )


async def async_main(
    config: LabelBasedContextRetrievalConfig,
    *,
    config_path: str,
    label_selection_output_path: Optional[Path] = None,
    skip_label_filtering: bool = False,
    max_conversations: Optional[int] = None,
    max_concurrent_questions: int = 5,
    overwrite: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info("Loaded label-based retrieval config from %s", config_path)

    # Only validate label selector config if not skipping label filtering
    if not skip_label_filtering:
        if not config.HasField("label_selector_language_model_provider_config"):
            raise ValueError(
                "label_selector_language_model_provider_config must be set for label selection"
            )

    if not config.HasField("query_embedding_model_provider_config"):
        raise ValueError("query_embedding_model_provider_config must be set for embeddings")

    # Qdrant is optional - we can load turn content from turns.jsonl instead
    use_qdrant = config.HasField("qdrant_config_turn_collection")

    # Only initialize label selection LLM if not skipping label filtering
    if skip_label_filtering:
        selection_lm = None
        context_scope_selector = None
        event_type_selector = None
        target_selector = None
        functional_type_selector = None
        role_selector = None
        turn_usefulness_selector = None
        logger.info("skip_label_filtering=True: Skipping label selector LLM initialization")
    else:
        selection_lm = init_lm(config.label_selector_language_model_provider_config)
        context_scope_selector = dspy.Predict(ContextScopeLabelSelectionSignature)
        event_type_selector = dspy.Predict(EventTypeLabelSelectionSignature)
        target_selector = dspy.Predict(TargetLabelSelectionSignature)
        functional_type_selector = dspy.Predict(FunctionalTypeSeedsLabelSelectionSignature)
        role_selector = dspy.Predict(RoleSensitivitySignature)
        turn_usefulness_selector = dspy.Predict(TurnUsefulnessSignature)


    structured_notes_path = _require_existing_file(
        config.structured_notes_jsonl_path,
        "structured_notes_jsonl_path",
    )
    turn_mapping_path = _require_existing_file(
        config.turn_mapping_json_path,
        "turn_mapping_json_path",
    )
    questions_path = _require_existing_file(
        config.questions_jsonl_path,
        "questions_jsonl_path",
    )
    output_path = Path(_require_non_empty(config.output_json_path, "output_json_path"))

    # Modify output path for vanilla RAG mode
    if skip_label_filtering:
        original_stem = output_path.stem
        output_path = output_path.with_stem(f"{original_stem}_vanilla_rag")
        logger.info("skip_label_filtering=True: Output path modified to %s", output_path)

    # Handle overwrite flag
    if overwrite:
        if output_path.exists():
            logger.info("Overwrite flag set: removing existing results file %s", output_path)
            output_path.unlink()
        if label_selection_output_path and label_selection_output_path.exists():
            logger.info("Overwrite flag set: removing existing label selection file %s", label_selection_output_path)
            label_selection_output_path.unlink()
        processed_question_ids: Set[str] = set()
        existing_results: Dict[str, Dict[str, Any]] = {}
        existing_label_selections: Dict[str, Dict[str, Any]] = {}
    else:
        # Load existing results to resume from previous run
        processed_question_ids, existing_results, existing_label_selections = load_existing_results(
            output_path,
            label_selection_output_path,
        )
        
        if processed_question_ids:
            logger.info(
                "Found %d already-processed questions. Will skip them and continue.",
                len(processed_question_ids),
            )

    if config.max_label_selected_turns > 0:
        max_label_selected_turns = config.max_label_selected_turns
    else:
        max_label_selected_turns = DEFAULT_MAX_LABEL_SELECTED_TURNS
        logger.info(
            "max_label_selected_turns not set or <=0; defaulting to %d",
            max_label_selected_turns,
        )

    if config.embedding_topk > 0:
        embedding_topk = config.embedding_topk
    else:
        embedding_topk = DEFAULT_EMBEDDING_TOPK
        logger.info(
            "embedding_topk not set or <=0; defaulting to %d",
            embedding_topk,
        )

    # Load structured notes
    raw_structured_notes = load_structured_turn_notes(str(structured_notes_path))
    structured_notes = {
        str(turn_id): record
        for turn_id, record in raw_structured_notes.items()
        if isinstance(turn_id, str)
    }
    logger.info("Loaded %d structured turn notes", len(structured_notes))
    # Drop blank notes up front to avoid passing empty content into embeddings/filters
    filtered_structured_notes = {
        tid: rec for tid, rec in structured_notes.items() if not rec.get("is_blank", False)
    }
    if len(filtered_structured_notes) != len(structured_notes):
        logger.info(
            "Skipping %d blank turn notes (remaining %d)",
            len(structured_notes) - len(filtered_structured_notes),
            len(filtered_structured_notes),
        )
    structured_notes = filtered_structured_notes

    # Load turn mapping
    turn_mapping = load_turn_mapping(str(turn_mapping_path))

    questions = load_questions(str(questions_path))
    if max_conversations is not None and max_conversations > 0:
        allowed_conversations = {
            f"conv-{idx}" for idx in range(1, int(max_conversations) + 1)
        }
        filtered_questions = [
            q for q in questions if _conversation_from_turn_id(q.id) in allowed_conversations
        ]
        logger.info(
            "Restricting to first %d conversation(s): %d -> %d questions",
            len(allowed_conversations),
            len(questions),
            len(filtered_questions),
        )
        questions = filtered_questions

    # Filter out already-processed questions
    if processed_question_ids:
        original_count = len(questions)
        questions = [q for q in questions if q.id not in processed_question_ids]
        skipped_count = original_count - len(questions)
        logger.info(
            "Skipping %d already-processed questions. Remaining: %d questions",
            skipped_count,
            len(questions),
        )

    total_questions = len(questions)
    logger.info(
        "Loaded %d questions from %s (after filtering already-processed)",
        total_questions,
        questions_path,
    )

    global_turn_id_index, conversation_turn_map = build_turn_id_index_mapping(questions)

    embedding_provider_config = config.query_embedding_model_provider_config
    embedding_model_name = embedding_provider_config.model_name
    if not embedding_model_name:
        raise ValueError(
            "query_embedding_model_provider_config.model_name must be provided"
        )
    embedding_kwargs = extract_embedding_kwargs(embedding_provider_config)

    # Initialize Qdrant client and note lookup if Qdrant is configured
    qdrant_client: Optional[AsyncQdrantClient] = None
    collection_name: Optional[str] = None
    note_lookup: Optional[Dict[str, List[TurnIndexEntry]]] = None
    
    if use_qdrant:
        qdrant_config = config.qdrant_config_turn_collection
        collection_name = _require_non_empty(qdrant_config.collection, "qdrant_config_turn_collection.collection")
        if not qdrant_config.url:
            raise ValueError("qdrant_config_turn_collection.url must be provided")

        qdrant_client = AsyncQdrantClient(
            url=qdrant_config.url,
            api_key=qdrant_config.api_key or None,
        )
    else:
        logger.info("Qdrant not configured - will load turn content from turns.jsonl")

    # Load turn content from turns.jsonl (derive path from structured_notes path)
    turn_content_map: Optional[Dict[str, str]] = None
    if not use_qdrant or True:  # Always try to load from turns.jsonl as fallback
        # Derive turns.jsonl path from structured_notes path
        structured_notes_parent = structured_notes_path.parent.parent
        turns_jsonl_path = structured_notes_parent / "turns.jsonl"
        
        if turns_jsonl_path.exists():
            turn_content_map = load_turn_content_from_jsonl(str(turns_jsonl_path))
            logger.info("Loaded turn content from %s", turns_jsonl_path)
        else:
            logger.warning("Turns JSONL not found at %s - will try Qdrant if available", turns_jsonl_path)

    try:
        # Build note lookup from Qdrant if available
        if use_qdrant and qdrant_client:
            try:
                note_lookup = await build_note_lookup(
                    qdrant_client,
                    collection_name,
                    turn_id_index_lookup=global_turn_id_index,
                )
            except Exception as exc:
                logger.warning("Failed to build note lookup from Qdrant: %s. Will use turns.jsonl instead.", exc)
                note_lookup = None
                # Create dummy entries for candidate_entries if needed
                if not turn_content_map:
                    logger.error("Neither Qdrant nor turns.jsonl available. Cannot proceed.")
                    raise

        # Set up incremental writing with temp JSONL files
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_jsonl_path = output_path.with_suffix(".jsonl.tmp")
        # If we already have processed questions (resume), keep temp for crash recovery; otherwise start clean
        if temp_jsonl_path.exists() and not processed_question_ids:
            temp_jsonl_path.unlink()
        
        label_temp_jsonl_path = None
        if label_selection_output_path:
            label_selection_output_path.parent.mkdir(parents=True, exist_ok=True)
            label_temp_jsonl_path = label_selection_output_path.with_suffix(".jsonl.tmp")
            if label_temp_jsonl_path.exists() and not processed_question_ids:
                label_temp_jsonl_path.unlink()
        
        # File writing lock for thread-safe incremental writes
        file_write_lock = asyncio.Lock()
        
        logger.info("Incremental writing enabled: results will be saved to %s as questions complete", temp_jsonl_path)

        # Process questions in parallel with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent_questions)
        logger.info("Processing %d questions with max concurrency of %d", total_questions, max_concurrent_questions)

        def _finalize_outputs(cleanup_temp: bool = True) -> None:
            """Merge partial results (including temp files) and write outputs.

            Called on both success and crash so we don't lose progress.
            """
            merged_results: Dict[str, Dict[str, Any]] = dict(existing_results)

            def _merge_result_dict(summary_dict: Dict[str, Any]) -> None:
                qid = summary_dict.get("question_id")
                if qid:
                    merged_results[qid] = summary_dict

            # Include progress from temp JSONL if present
            if temp_jsonl_path.exists():
                temp_results: List[Tuple[int, Dict[str, Any]]] = []
                with temp_jsonl_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            temp_results.append((entry["index"], entry["summary"]))
                temp_results.sort(key=lambda x: x[0])
                for _, summary in temp_results:
                    _merge_result_dict(summary)

            # Include in-memory results collected this run
            for _, summary_obj in results:
                _merge_result_dict(summary_obj.to_dict())

            if merged_results:
                output_payload = {
                    "retrieval_strategy": "label_based_filtering",
                    "max_label_selected_turns": max_label_selected_turns,
                    "embedding_topk": embedding_topk,
                    "question_results": list(merged_results.values()),
                }
                with output_path.open("w", encoding="utf-8") as outfile:
                    json.dump(output_payload, outfile, indent=2)
                logger.info("Wrote retrieval summary to %s", output_path)

            # Label selection output (if configured)
            if label_selection_output_path:
                merged_label_records: Dict[str, Dict[str, Any]] = dict(existing_label_selections)

                def _merge_label_record(record: Dict[str, Any]) -> None:
                    qid = record.get("question_id")
                    if qid:
                        merged_label_records[qid] = record

                if label_temp_jsonl_path and label_temp_jsonl_path.exists():
                    with label_temp_jsonl_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                record = json.loads(line)
                                _merge_label_record(record)

                for _, summary_obj in results:
                    _merge_label_record(
                        {
                            "question_id": summary_obj.question_id,
                            "question": summary_obj.question_content,
                            "selected_context_scopes": summary_obj.selected_context_scopes,
                            "context_scope_reasoning": summary_obj.context_scope_reasoning,
                            "selected_event_types": summary_obj.selected_event_types,
                            "event_type_reasoning": summary_obj.event_type_reasoning,
                            "selected_targets": summary_obj.selected_targets,
                            "target_reasoning": summary_obj.target_reasoning,
                            "selected_functional_type_seeds": summary_obj.selected_functional_type_seeds,
                            "functional_type_reasoning": summary_obj.functional_type_reasoning,
                        }
                    )

                if merged_label_records:
                    with label_selection_output_path.open("w", encoding="utf-8") as label_file:
                        json.dump(list(merged_label_records.values()), label_file, indent=2)
                    logger.info("Wrote label selection summary to %s", label_selection_output_path)

            # Only clean temp files on success; keep on crash for debugging/resume
            if cleanup_temp:
                if temp_jsonl_path.exists():
                    temp_jsonl_path.unlink()
                if label_temp_jsonl_path and label_temp_jsonl_path.exists():
                    label_temp_jsonl_path.unlink()

        async def process_with_semaphore(index: int, question: Any) -> Tuple[int, QuestionRetrievalSummary]:
            async with semaphore:
                summary = await process_single_question(
                    question=question,
                    index=index,
                    total_questions=total_questions,
                    global_turn_id_index=global_turn_id_index,
                    structured_notes=structured_notes,
                    turn_mapping=turn_mapping,
                    note_lookup=note_lookup,
                    qdrant_client=qdrant_client,
                    collection_name=collection_name,
                    turn_content_map=turn_content_map,
                    selection_lm=selection_lm,
                    context_scope_selector=context_scope_selector,
                    event_type_selector=event_type_selector,
                    target_selector=target_selector,
                    functional_type_selector=functional_type_selector,
                    role_selector=role_selector,
                    turn_usefulness_selector=turn_usefulness_selector,
                    embedding_provider_config=embedding_provider_config,
                    embedding_model_name=embedding_model_name,
                    embedding_kwargs=embedding_kwargs,
                    max_label_selected_turns=max_label_selected_turns,
                    embedding_topk=embedding_topk,
                    skip_label_filtering=skip_label_filtering,
                )
                
                # Write result incrementally to temp JSONL
                async with file_write_lock:
                    # Write main result to temp JSONL
                    def _write_main_result():
                        with temp_jsonl_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "index": index - 1,
                                "summary": summary.to_dict()
                            }, ensure_ascii=False) + "\n")
                            f.flush()
                    await asyncio.to_thread(_write_main_result)
                    
                    # Write label selection result if needed
                    if label_temp_jsonl_path:
                        label_record = {
                            "question_id": summary.question_id,
                            "question": summary.question_content,
                            "selected_context_scopes": summary.selected_context_scopes,
                            "context_scope_reasoning": summary.context_scope_reasoning,
                            "selected_event_types": summary.selected_event_types,
                            "event_type_reasoning": summary.event_type_reasoning,
                            "selected_targets": summary.selected_targets,
                            "target_reasoning": summary.target_reasoning,
                            "selected_functional_type_seeds": summary.selected_functional_type_seeds,
                            "functional_type_reasoning": summary.functional_type_reasoning,
                        }
                        def _write_label_result():
                            with label_temp_jsonl_path.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(label_record, ensure_ascii=False) + "\n")
                                f.flush()
                        await asyncio.to_thread(_write_label_result)
                
                return (index - 1, summary)  # Return with original index for ordering

        # Create tasks for all questions
        tasks = [
            process_with_semaphore(index, question)
            for index, question in enumerate(questions, start=1)
        ]

        # Process with progress tracking and preserve order
        results: List[Tuple[int, QuestionRetrievalSummary]] = []
        completed = 0
        try:
            with tqdm(total=total_questions, desc="Processing questions", unit="question") as pbar:
                for coro in asyncio.as_completed(tasks):
                    try:
                        result = await coro
                        results.append(result)
                        completed += 1
                        pbar.update(1)
                        if completed % 10 == 0 or completed == total_questions:
                            logger.info("Completed %d/%d questions", completed, total_questions)
                    except Exception as exc:
                        logger.error("Error processing question: %s", exc, exc_info=True)
                        raise
        except Exception as exc:
            logger.error("Error during processing, writing partial results before exiting")
            _finalize_outputs(cleanup_temp=False)
            sys.exit(1)
        
        # Sort by original index to preserve question order
        results.sort(key=lambda x: x[0])
        summaries = [summary for _, summary in results]

        # Read from temp JSONL and write final JSON files
        _finalize_outputs(cleanup_temp=True)
        
        logger.info("Total LLM cost: %s", get_lm_cost(selection_lm))

    finally:
        if qdrant_client:
            await qdrant_client.close()
            logger.info("Closed Qdrant client")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run label-based retrieval with LLM-selected field filtering",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to LabelBasedContextRetrievalConfig JSON (env vars resolved via load_config)",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="If provided, limit processing to the first N conversations (conv-1..conv-N)",
    )
    parser.add_argument(
        "--max-concurrent-questions",
        type=int,
        default=5,
        help="Maximum number of questions to process concurrently (default: 5)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results file instead of resuming",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    config_path = cli_args.config
    config, label_selection_output_path, skip_label_filtering = load_label_based_config(config_path)
    asyncio.run(
        async_main(
            config,
            config_path=config_path,
            label_selection_output_path=label_selection_output_path,
            skip_label_filtering=skip_label_filtering,
            max_conversations=cli_args.max_conversations,
            max_concurrent_questions=cli_args.max_concurrent_questions,
            overwrite=cli_args.overwrite,
        )
    )
