"""Convert retrieval summaries into DatasetRetrievalResult JSON with enriched memory snippets."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from google.protobuf.json_format import MessageToDict

from came_bench.proto import DatasetRetrievalResult, QuestionRetrievalResult
from came_bench.utils.io import load_questions
from .common_utils import load_structured_turn_notes, _conversation_from_turn_id

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def _load_summary(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as infile:
        return json.load(infile)


def _collect_reasoning(detail: Dict[str, object]) -> str:
    parts: List[str] = []
    initial_rank = detail.get("initial_rank")
    if initial_rank is not None:
        parts.append(f"initial_rank={initial_rank}")
    similarity = detail.get("initial_score") or detail.get("similarity")
    if similarity is not None:
        if isinstance(similarity, (int, float)):
            parts.append(f"similarity={similarity:.4f}")
        else:
            parts.append(f"similarity={similarity}")
    rerank_position = detail.get("rerank_position")
    if rerank_position is not None:
        parts.append(f"rerank_position={rerank_position}")
    hierarchy_descriptions = detail.get("hierarchy_descriptions", [])
    if isinstance(hierarchy_descriptions, list) and hierarchy_descriptions:
        description_text = "; ".join(
            f"{item.get('node_id', '')}: {item.get('description', '').strip()}"
            for item in hierarchy_descriptions
            if isinstance(item, dict)
        )
        if description_text.strip():
            parts.append(f"nodes={description_text.strip()}")
    note_text = str(detail.get("note_text", "")).strip()
    if note_text:
        parts.append(f"note={note_text}")
    aggregated = str(detail.get("aggregated_note_text", "")).strip()
    if aggregated and aggregated != note_text:
        parts.append(f"hierarchy_notes={aggregated}")
    hierarchy_context = str(detail.get("hierarchy_context_text", "")).strip()
    if hierarchy_context and hierarchy_context not in (note_text, aggregated):
        parts.append(f"hierarchy_context={hierarchy_context}")
    turn_content = str(detail.get("turn_content", "")).strip()
    if turn_content:
        parts.append(f"turn={turn_content}")
    enriched_turn_content = str(detail.get("enriched_turn_content", "")).strip()
    if enriched_turn_content:
        parts.append(f"enriched_turn_content={enriched_turn_content}")
    return " | ".join(parts)


def _normalize_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if isinstance(item, (str, int, float)) and str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(str(text).split())


def build_turn_id_index_mapping(questions) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    conversation_turn_map: Dict[str, Dict[str, int]] = {}
    global_turn_id_to_index: Dict[str, int] = {}
    global_counter = 0

    for question in questions:
        conv_id = _conversation_from_turn_id(question.id)
        for turn_id in getattr(question, "question_turn_ids", []):
            turn_id_str = str(turn_id)
            if not turn_id_str:
                continue
            normalized_conv = _conversation_from_turn_id(turn_id_str) or conv_id
            conv_map = conversation_turn_map.setdefault(normalized_conv, {})
            if turn_id_str not in conv_map:
                conv_map[turn_id_str] = len(conv_map)
            if turn_id_str not in global_turn_id_to_index:
                global_turn_id_to_index[turn_id_str] = global_counter
                global_counter += 1

    logger.info(
        "Constructed turn-id mapping for %d conversations (%d unique turn ids)",
        len(conversation_turn_map),
        len(global_turn_id_to_index),
    )
    return global_turn_id_to_index, conversation_turn_map


def load_segment_notes(path: Path) -> Dict[str, List[Dict[str, object]]]:
    segments_by_conv: Dict[str, List[Dict[str, object]]] = {}
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            conv_id = str(record.get("conversation_id", "")).strip()
            if not conv_id:
                continue
            segments_by_conv.setdefault(conv_id, []).append(record)
    logger.info(
        "Loaded %d segment-level notes across %d conversations",
        sum(len(v) for v in segments_by_conv.values()),
        len(segments_by_conv),
    )
    return segments_by_conv


def _truncate_with_ellipsis(text: str, max_tokens: int) -> str:
    """
    Truncate a string to max_tokens, append "..." always.
    Does NOT attempt smart boundary detection (per user requirement).
    """
    if not text or max_tokens <= 0:
        return "..."
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text

    truncated = " ".join(tokens[:max_tokens])
    return truncated + " ..."


def build_memory_snippet(
    turn_id: str,
    detail: Dict[str, object],
    structured_notes: Dict[int, Dict[str, object]],
    note_lookup: Dict[str, int],
    *,
    max_tokens_per_utterance: Optional[int] = None,
) -> Dict[str, object]:
    """
    Build structured memory snippet as JSON/dict.
    Speaker field is omitted (speaker appears inside utterance).
    Only 'utterance' is capped when exceeding token limit.
    """

    # -------------------------
    # 1. Resolve note metadata
    # -------------------------
    note_id = detail.get("note_id")
    note_data = None
    if isinstance(note_id, int):
        note_data = structured_notes.get(note_id)

    if note_data is None:
        # fallback via turn lookup
        idx = note_lookup.get(turn_id)
        if idx is not None:
            note_data = structured_notes.get(idx)

    note_data = note_data or {}

    # -------------------------
    # 2. Extract fields
    # -------------------------
    # Speaker intentionally NOT included per user request
    act = str(detail.get("act") or note_data.get("act") or "").strip() or None
    target = str(detail.get("target") or note_data.get("target") or "").strip() or None
    context_scope = (
        str(detail.get("context_scope") or note_data.get("context_scope") or "").strip() or None
    )
    summary = str(detail.get("note_text") or note_data.get("note_text") or "").strip() or None

    # Event types normalized to list[str]
    event_types = detail.get("event_types") or note_data.get("event_types") or []
    if isinstance(event_types, str):
        event_types = [event_types]
    event_types = [str(et).strip() for et in event_types if str(et).strip()]

    # Utterance
    utterance = (
        str(detail.get("turn_content") or detail.get("content") or "").strip()
    )
    if not utterance:
        utterance = summary or ""

    # -------------------------
    # 3. Apply capping (only utterance)
    # -------------------------
    if utterance and max_tokens_per_utterance:
        utterance = _truncate_with_ellipsis(utterance, max_tokens_per_utterance)

    # -------------------------
    # 4. Optional turn index
    # -------------------------
    turn_index = None
    if turn_id in note_lookup:
        turn_index = note_lookup[turn_id]

    # -------------------------
    # 5. Return LLM-friendly dict
    # -------------------------
    return {
        "turn_id": turn_id,
        "turn_index": turn_index,
        "utterance": utterance,
        "action": act,
        "target": target,
        "context_scope": context_scope,
        "event_types": event_types,
        "summary": summary,
    }


def build_segment_summary_snippet(
    matched_segments: Iterable[Dict[str, object]],
) -> List[str]:
    """
    Wrap segment-level summaries into a list of note strings.

    Context scope is excluded as it's redundant:
    - The note text already contains scope information (e.g., "[Personal update] ...")
    - All turns already have context_scope fields
    - Saves tokens by avoiding repetition
    """
    summaries: List[str] = []
    for segment in matched_segments:
        note = str(segment.get("note", "")).strip()
        if not note:
            continue
        summaries.append(note)
    return summaries


def _build_question_context(
    *,
    retrieved_turn_ids: List[str],
    detail_lookup: Dict[str, Dict[str, object]],
    structured_notes: Dict[int, Dict[str, object]],
    note_lookup: Dict[str, int],
    segment_notes_by_conv: Dict[str, List[Dict[str, object]]],
    question_id: str,
    include_segment_summary: bool,
    max_turn_tokens: Optional[int],
) -> Dict[str, object]:
    """
    Build a nested conversation structure for one question:

    {
      "turns": [ {turn snippet dict}, ... ],
      "segment_summaries": [ "note string", ... ]   # optional list of note strings
    }
    """
    # 1. Turn snippets (respect retrieved order)
    turns: List[Dict[str, object]] = []
    retrieved_set = {str(tid) for tid in retrieved_turn_ids if str(tid)}
    for turn_id in retrieved_turn_ids:
        turn_id_str = str(turn_id)
        if not turn_id_str:
            continue
        detail = detail_lookup.get(turn_id_str, {})
        snippet = build_memory_snippet(
            turn_id_str,
            detail,
            structured_notes,
            note_lookup,
            max_tokens_per_utterance=max_turn_tokens,
        )
        turns.append(snippet)

    # 2. Conversation id candidates (from retrieved turns + question id)
    conv_candidates: List[str] = []
    for turn in retrieved_turn_ids:
        conv_id = _conversation_from_turn_id(str(turn))
        if conv_id:
            conv_candidates.append(conv_id)
    question_conv = _conversation_from_turn_id(question_id)
    if question_conv:
        conv_candidates.append(question_conv)
    # Preserve order while deduplicating.
    conv_candidates = list(dict.fromkeys(conv_candidates))

    # 3. Segment summaries (wrapped as dicts)
    segment_summaries: List[Dict[str, object]] = []
    if include_segment_summary and segment_notes_by_conv and conv_candidates:
        matched_segments: List[Dict[str, object]] = []
        for conv_id in conv_candidates:
            for segment in segment_notes_by_conv.get(conv_id, []):
                turn_ids = _normalize_list(segment.get("turn_ids"))
                if not turn_ids:
                    continue
                if retrieved_set.intersection(turn_ids):
                    matched_segments.append(segment)
        segment_summaries = build_segment_summary_snippet(matched_segments)

    # 4. Build final context object (exclude metadata IDs - they're redundant and waste tokens)
    # Only include actual content: turns and segment_summaries
    context: Dict[str, object] = {
        "turns": turns,
    }
    if segment_summaries:
        context["segment_summaries"] = segment_summaries

    return context


def _enforce_global_token_cap_on_context(
    context: Dict[str, object],
    token_cap: int,
) -> Tuple[Dict[str, object], int]:
    """
    Enforce a global token cap over the JSON-serialized context.

    Strategy:
      1. Serialize context and compute token count.
      2. If within cap, keep as is.
      3. If above cap and 'segment_summaries' present -> remove that key and recompute.
      4. If still above cap, keep as-is (we do not further truncate, per user spec).

    Returns:
      (possibly_modified_context, final_token_count)
    """
    json_str = json.dumps(context, ensure_ascii=False)
    tokens = _count_tokens(json_str)
    if tokens <= token_cap:
        return context, tokens

    if "segment_summaries" in context:
        logger.info(
            "Global token cap exceeded (%d > %d); removing segment_summaries",
            tokens,
            token_cap,
        )
        context = dict(context)  # shallow copy
        context.pop("segment_summaries", None)
        json_str = json.dumps(context, ensure_ascii=False)
        tokens = _count_tokens(json_str)
        # If still above cap, we keep as-is (no further truncation beyond per-utterance cap)
        return context, tokens

    # Already no segment_summaries; just return original
    return context, tokens


def transform_summary(
    *,
    summary_path: Path,
    questions_path: Path,
    output_path: Path,
    dataset_name: str,
    retrieval_strategy: Optional[str] = None,
    topk_override: Optional[int] = None,
    structured_notes_path: Optional[Path] = None,
    segment_notes_path: Optional[Path] = None,
    enforce_snippet_cap: bool = True,
) -> None:
    summary_payload = _load_summary(summary_path)
    questions = load_questions(str(questions_path))
    question_by_id: Dict[str, object] = {question.id: question for question in questions}

    turn_id_index_map, _ = build_turn_id_index_mapping(questions)

    structured_notes: Dict[int, Dict[str, object]] = {}
    if structured_notes_path is not None:
        structured_notes = load_structured_turn_notes(str(structured_notes_path))
    else:
        logger.warning("No structured notes supplied; snippets will use retrieval metadata only")

    segment_notes_by_conv: Dict[str, List[Dict[str, object]]] = {}
    if segment_notes_path is not None:
        segment_notes_by_conv = load_segment_notes(segment_notes_path)
    else:
        logger.warning("No segment-level notes supplied; aggregated snippet will be empty")

    topk_value = topk_override
    if topk_value is None:
        candidate = summary_payload.get("final_topk")
        if isinstance(candidate, int) and candidate > 0:
            topk_value = candidate
    if topk_value is None:
        topk_value = 0

    strategy = retrieval_strategy or str(summary_payload.get("retrieval_strategy") or "hierarchy_llm_rank")

    dataset_result = DatasetRetrievalResult(
        topk=topk_value,
        retrieval_strategy=strategy,
        dataset_name=dataset_name,
    )

    question_results = summary_payload.get("question_results", [])
    for entry in question_results:
        if not isinstance(entry, dict):
            continue
        question_id = entry.get("question_id")
        if not isinstance(question_id, str):
            continue
        question_proto = question_by_id.get(question_id)
        if question_proto is None:
            raise KeyError(f"Question id '{question_id}' not found in {questions_path}")
        retrieved_turn_ids = (
            entry.get("retrieved_turn_ids")
            or entry.get("embedding_candidate_turn_ids")
            or entry.get("node_selected_turn_ids")
            or []
        )
        if not isinstance(retrieved_turn_ids, list):
            retrieved_turn_ids = []

        result = QuestionRetrievalResult()
        result.question.CopyFrom(question_proto)
        result.turn_ids.extend([str(tid) for tid in retrieved_turn_ids if isinstance(tid, str)])
        result.success = True

        details = entry.get("retrieved_turn_details", [])
        if isinstance(details, list):
            for detail in details:
                if not isinstance(detail, dict):
                    continue
                turn_id = detail.get("turn_id")
                if not isinstance(turn_id, str):
                    continue
                reasoning = _collect_reasoning(detail)
                if reasoning:
                    result.turn_retrieval_reasoning[turn_id] = reasoning

        detail_lookup: Dict[str, Dict[str, object]] = {}
        details_candidate = (
            entry.get("retrieved_turn_details")
            or entry.get("embedding_candidate_turn_details")
            or []
        )
        if isinstance(details_candidate, list):
            for detail in details_candidate:
                if isinstance(detail, dict) and isinstance(detail.get("turn_id"), str):
                    detail_lookup[detail["turn_id"]] = detail

        per_utterance_cap = None
        if enforce_snippet_cap:
            turn_count = max(len(retrieved_turn_ids), 1)
            # same budget logic as before, but applied per utterance
            per_utterance_cap = max(1, 4096 // turn_count)
            logger.info(
                "Question %s: %d retrieved turns -> per-utterance token cap=%d (budget=4096)",
                question_id,
                len(retrieved_turn_ids),
                per_utterance_cap,
            )

        # Build nested conversation context (dict)
        context_obj = _build_question_context(
            retrieved_turn_ids=retrieved_turn_ids,
            detail_lookup=detail_lookup,
            structured_notes=structured_notes,
            note_lookup=turn_id_index_map,
            segment_notes_by_conv=segment_notes_by_conv,
            question_id=question_id,
            include_segment_summary=True,
            max_turn_tokens=per_utterance_cap if enforce_snippet_cap else None,
        )

        if enforce_snippet_cap:
            capped_context, used_tokens = _enforce_global_token_cap_on_context(
                context_obj,
                token_cap=4096,
            )
            logger.info(
                "Question %s memory context tokens: %d (after global cap enforcement)",
                question_id,
                used_tokens,
            )
            context_json = json.dumps(capped_context, ensure_ascii=False)
        else:
            context_json = json.dumps(context_obj, ensure_ascii=False)
            used_tokens = _count_tokens(context_json)
            logger.info(
                "Question %s: snippet capping disabled; tokens=%d",
                question_id,
                used_tokens,
            )

        # Store as a single JSON-string memory snippet
        if context_json:
            result.memory_snippets.append(context_json)

        dataset_result.question_retrieval_results.append(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump(
            MessageToDict(dataset_result, preserving_proto_field_name=True),
            outfile,
            indent=2,
            ensure_ascii=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform retrieval output into DatasetRetrievalResult JSON.",
    )
    parser.add_argument("--config", required=True, help="JSON config containing transform parameters.")
    parser.add_argument("--summary", help="Path to retrieval summary JSON (overrides config)")
    parser.add_argument("--questions-jsonl", help="Path to questions JSONL used for retrieval (overrides config)")
    parser.add_argument("--output-json", help="Destination for DatasetRetrievalResult JSON (overrides config)")
    parser.add_argument("--dataset-name", help="Dataset name to store in the output proto (overrides config)")
    parser.add_argument("--retrieval-strategy", default=None, help="Optional override for retrieval_strategy field")
    parser.add_argument("--topk", type=int, default=None, help="Optional override for topk field")
    parser.add_argument("--structured-notes-jsonl", help="Structured turn notes JSONL path for enrichment")
    parser.add_argument("--segment-notes-jsonl", help="Segment-level notes JSONL path for aggregated context")
    parser.add_argument(
        "--disable-snippet-cap",
        action="store_true",
        help="If set, skip all snippet token capping and include full text (except per-utterance cap).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as infile:
        config_data = json.load(infile)

    def _get(key: str, fallback=None):
        if key in config_data:
            return config_data.get(key, fallback)
        return fallback

    # Read from config, with command-line args as override
    summary = args.summary or _get("summary_json_path")
    questions = args.questions_jsonl or _get("questions_jsonl_path")
    output = args.output_json or _get("output_json_path")
    dataset_name = args.dataset_name or _get("dataset_name")

    # Validate required fields
    if not summary:
        raise ValueError("summary_json_path must be provided in config file or via --summary")
    if not questions:
        raise ValueError("questions_jsonl_path must be provided in config file or via --questions-jsonl")
    if not output:
        raise ValueError("output_json_path must be provided in config file or via --output-json")
    if not dataset_name:
        raise ValueError("dataset_name must be provided in config file or via --dataset-name")

    retrieval_strategy = args.retrieval_strategy or _get("retrieval_strategy")
    topk = args.topk if args.topk is not None else _get("topk")
    structured_notes = args.structured_notes_jsonl or _get("structured_notes_jsonl_path")
    segment_notes = args.segment_notes_jsonl or _get("segment_notes_jsonl_path")
    cap_snippets_config = _get("cap_snippets", None)

    if cap_snippets_config is None:
        enforce_snippet_cap = not args.disable_snippet_cap
    else:
        enforce_snippet_cap = bool(cap_snippets_config)

    transform_summary(
        summary_path=Path(summary),
        questions_path=Path(questions),
        output_path=Path(output),
        dataset_name=str(dataset_name),
        retrieval_strategy=retrieval_strategy,
        topk_override=int(topk) if topk is not None else None,
        structured_notes_path=Path(structured_notes) if structured_notes else None,
        segment_notes_path=Path(segment_notes) if segment_notes else None,
        enforce_snippet_cap=enforce_snippet_cap,
    )


if __name__ == "__main__":
    main()
