import argparse
import json
import re
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from src import Dataset, Turn, Question, AnswerType, Choice, MultipleChoiceAnswer
from dataset_construction.debate.datatypes import Storyboard, EvalQA, AnswerFormat, DebateTurn, PriorTurn
from came_bench.utils.io import write_dataset
from came_bench.utils.common import _split_into_sentences

DATASET_NAME = "debate"

_FREEFORM_TYPE_OVERRIDES = {
    "ANSWER_TYPE_FREEFORM": AnswerType.ANSWER_TYPE_FREEFORM,
    "ANSWER_TYPE_FREEFORM_DEBATE": AnswerType.ANSWER_TYPE_FREEFORM_DEBATE,
    "ANSWER_TYPE_FREEFORM_LOCOMO": AnswerType.ANSWER_TYPE_FREEFORM_LOCOMO,
}


def _resolve_freeform_answer_type(qa: EvalQA) -> AnswerType:
    meta_value = None
    if isinstance(qa.meta, dict):
        meta_value = qa.meta.get("answer_type")
    if isinstance(meta_value, str):
        candidate = meta_value.strip().upper()
        if candidate in _FREEFORM_TYPE_OVERRIDES:
            return _FREEFORM_TYPE_OVERRIDES[candidate]
    if isinstance(qa.qtype, str) and qa.qtype.startswith("role_evidence"):
        return AnswerType.ANSWER_TYPE_FREEFORM_DEBATE
    return AnswerType.ANSWER_TYPE_FREEFORM


def _is_contention_object(action_object: Any) -> bool:
    if not isinstance(action_object, str):
        return False
    return action_object.startswith("pro_contention_") or action_object.startswith("con_contention_")


def _extract_timestamp_suffix(text: str) -> Tuple[str, str]:
    marker = "----TIMESTAMP:"
    if not text or marker not in text:
        return text, ""
    body, suffix = text.split(marker, 1)
    return body.strip(), f" ----TIMESTAMP:{suffix.strip()}"


def _chunk_sentences(text: str, chunk_size: int = 3) -> List[str]:
    sentences = _split_into_sentences(text)
    if not sentences:
        return [text.strip()]
    chunks: List[str] = []
    for idx in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentence.strip() for sentence in sentences[idx : idx + chunk_size] if sentence.strip())
        if chunk:
            chunks.append(chunk)
    return chunks if chunks else [text.strip()]


def _normalize_action_object_value(action_object: Any) -> str:
    if isinstance(action_object, list):
        return "|".join(str(item) for item in action_object)
    return str(action_object) if action_object is not None else ""


def _split_turn_if_needed(turn: DebateTurn) -> List[DebateTurn]:
    action_object = turn.debate_meta.action_object
    utterance_text = turn.utterance or ""
    if turn.action not in {"attack", "defend"}:
        clone = deepcopy(turn)
        clone.evidence_used = clone.evidence_used or []
        return [clone]
    if not (_is_contention_object(action_object) and utterance_text.strip()):
        clone = deepcopy(turn)
        clone.evidence_used = clone.evidence_used or []
        return [clone]

    body, timestamp_suffix = _extract_timestamp_suffix(utterance_text)
    sentence_chunks = _chunk_sentences(body)
    split_turns: List[DebateTurn] = []
    for idx, chunk in enumerate(sentence_chunks, start=1):
        new_turn = deepcopy(turn)
        new_turn.debate_meta.prior_relevant_turns = []
        new_turn.evidence_used = []
        base_argument_id = turn.debate_meta.argument_id or f"argument_{turn.turn_id}"
        new_turn.debate_meta.argument_id = f"{base_argument_id}"
        chunk_text = chunk.strip()
        if timestamp_suffix:
            chunk_text = f"{chunk_text}{timestamp_suffix}"
        new_turn.utterance = chunk_text
        split_turns.append(new_turn)
    return split_turns


def _rebuild_prior_turns(turns: List[DebateTurn]) -> None:
    history: Dict[Any, List[PriorTurn]] = {}
    for turn in turns:
        action_object = turn.debate_meta.action_object
        key = tuple(action_object) if isinstance(action_object, list) else action_object
        existing = history.get(key, [])
        rebuilt_prior = [deepcopy(ev) for ev in existing]
        current_event = PriorTurn(
            turn_id=turn.turn_id,
            role=turn.role,
            argument_id=turn.debate_meta.argument_id,
            action=turn.action,
            action_object=_normalize_action_object_value(action_object),
        )
        rebuilt_prior.append(current_event)
        turn.debate_meta.prior_relevant_turns = rebuilt_prior
        history[key] = rebuilt_prior


def split_storyboard_by_sentences(storyboard: Storyboard) -> Storyboard:
    new_turns: List[DebateTurn] = []
    for turn in storyboard.turns:
        new_turns.extend(_split_turn_if_needed(turn))

    for idx, turn in enumerate(new_turns, start=1):
        turn.turn_id = idx

    _rebuild_prior_turns(new_turns)

    updated_storyboard = deepcopy(storyboard)
    updated_storyboard.turns = new_turns
    updated_storyboard.num_turns = len(new_turns)
    return updated_storyboard


def load_shared_evidence_map(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not path.exists():
        return mapping

    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            for block in entry.get("serper_retrieved_documents", []):
                for doc in block.get("retrieved_documents", []):
                    uuid_key = doc.get("uuid")
                    neutral_name = doc.get("neutral_name")
                    if not uuid_key or not neutral_name:
                        continue
                    mapping.setdefault(uuid_key, neutral_name)
    return mapping


def resolve_shared_evidence_path(input_dir: str) -> Optional[Path]:
    base = Path(input_dir)
    preferred_names = [
        "shared_evidence_with_names.jsonl",
        "shared_evidence.jsonl",
    ]

    def candidate_paths(root: Path) -> List[Path]:
        paths: List[Path] = []
        for name in preferred_names:
            paths.append(root / name)
            paths.append(root / "shared_data" / name)
        return paths

    candidates = candidate_paths(base)
    for parent in base.parents:
        candidates.extend(candidate_paths(parent))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def add_quotes_to_evidence_mentions(
    text: Optional[str],
    evidence_ids: Set[str],
    uuid_to_name: Dict[str, str],
) -> Optional[str]:
    if not text or not evidence_ids or not uuid_to_name:
        return text

    updated = text
    for evidence_id in evidence_ids:
        neutral_name = uuid_to_name.get(evidence_id)
        if not neutral_name:
            continue
        pattern = re.escape(neutral_name)

        def repl(match: re.Match) -> str:
            source = match.string
            start, end = match.start(), match.end()
            before = source[start - 1] if start > 0 else ""
            after = source[end] if end < len(source) else ""
            if before == '"' and after == '"':
                return match.group(0)
            return f'"{match.group(0)}"'

        updated = re.sub(pattern, repl, updated)

    return updated


def process_debate(input_dir: str, output_dir: str, split_contention_turns: bool = False):
    dataset = Dataset()
    dataset.name = DATASET_NAME
    numeric_turn_id_to_proto_id: Dict[int, str] = {}
    storyboard_turns_cache: List[Any] = []
    all_proto_turn_ids: List[str] = []
    shared_evidence_path = resolve_shared_evidence_path(input_dir)
    uuid_to_neutral_name: Dict[str, str] = {}
    if shared_evidence_path:
        uuid_to_neutral_name = load_shared_evidence_map(shared_evidence_path)

    storyboard_with_utterances_path = Path(input_dir) / "storyboard_with_utterances.json"
    with storyboard_with_utterances_path.open("r", encoding="utf-8") as f:
        storyboard_with_utterances = Storyboard.from_dict(json.load(f))

    if split_contention_turns:
        split_storyboard_with_utterances = split_storyboard_by_sentences(storyboard_with_utterances)
        split_with_path = storyboard_with_utterances_path.with_name("storyboard_with_utterances_split.json")
        split_with_path.parent.mkdir(parents=True, exist_ok=True)
        with split_with_path.open("w", encoding="utf-8") as handle:
            json.dump(split_storyboard_with_utterances.to_dict(), handle, indent=4)

        split_storyboard_base = deepcopy(split_storyboard_with_utterances)
        for turn in split_storyboard_base.turns:
            turn.utterance = None
        base_split_path = storyboard_with_utterances_path.with_name("storyboard_split.json")
        with base_split_path.open("w", encoding="utf-8") as handle:
            json.dump(split_storyboard_base.to_dict(), handle, indent=4)

        storyboard = split_storyboard_with_utterances
    else:
        storyboard = storyboard_with_utterances
    topic = storyboard.topic
    topic_uuid = uuid.uuid5(uuid.NAMESPACE_URL, topic)
    turns = storyboard.turns
        
    for turn in turns:
            evidence_ids: Set[str] = set()
            if getattr(turn, "evidence_used", None):
                for evidence_uuid in turn.evidence_used:
                    if evidence_uuid in uuid_to_neutral_name:
                        evidence_ids.add(evidence_uuid)

            action_object_raw = turn.debate_meta.action_object
            if isinstance(action_object_raw, list):
                action_object = ", ".join(str(item) for item in action_object_raw)
                for item in action_object_raw:
                    if isinstance(item, str) and item in uuid_to_neutral_name:
                        evidence_ids.add(item)
            else:
                action_object = str(action_object_raw) if action_object_raw is not None else ""
                if isinstance(action_object_raw, str) and action_object_raw in uuid_to_neutral_name:
                    evidence_ids.add(action_object_raw)

            utterance_with_quotes = add_quotes_to_evidence_mentions(
                turn.utterance, evidence_ids, uuid_to_neutral_name
            )

            proto_turn = Turn()
            proto_turn.id = f"topic-{topic_uuid}-turn-{turn.turn_id}"
            # Ensure content is always a string (protobuf fields don't accept None)
            content_value = utterance_with_quotes if utterance_with_quotes is not None else turn.utterance
            proto_turn.content = content_value if content_value is not None else ""
            proto_turn.role = f"{turn.role}-side-debater"
            proto_turn.partition.append(f"topic-{topic_uuid}")
            proto_turn.action = turn.action
            proto_turn.action_object = action_object
            dataset.turns.append(proto_turn)
            numeric_turn_id_to_proto_id[turn.turn_id] = proto_turn.id
            all_proto_turn_ids.append(proto_turn.id)
            storyboard_turns_cache.append({
                "turn_id": turn.turn_id,
                "role": turn.role,
                "action": turn.action,
                "action_object": turn.debate_meta.action_object,
            })

    # ----------------------------
    # Build indexes from action_traces (preferred) or storyboard fallback
    # ----------------------------

    def read_action_traces(path: str) -> Dict[str, List[dict]]:
        if not path:
            return {}
        try:
            with open(path, "r") as f:
                content = f.read().strip()
            if not content:
                return {}
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
            merged: Dict[str, List[dict]] = {}
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        continue
                    for k, v in obj.items():
                        merged.setdefault(k, [])
                        merged[k].extend(v)
            return merged
        except Exception:
            return {}

    traces: Dict[str, List[dict]] = read_action_traces(input_dir + "/action_traces.jsonl")

    # Indexes we will populate:
    # (contention_id, action) -> set(turn_id)
    contention_action_to_turns: Dict[Tuple[str, str], Set[int]] = {}
    # (evidence_uuid, action) -> set(turn_id)
    evidence_action_to_turns: Dict[Tuple[str, str], Set[int]] = {}
    # (role, action, contention_id) -> set(turn_id)
    role_action_contention_to_turns: Dict[Tuple[str, str, str], Set[int]] = {}
    # (role, action, evidence_uuid) -> set(turn_id)
    role_action_evidence_to_turns: Dict[Tuple[str, str, str], Set[int]] = {}

    def add_to_set(map_obj: Dict[Tuple, Set[int]], key: Tuple, turn_id: int):
        s = map_obj.get(key)
        if s is None:
            s = set()
            map_obj[key] = s
        s.add(turn_id)

    if traces:
        # Preferred: index from action_traces
        for k_str, events in traces.items():
            try:
                obj_key = json.loads(k_str)
            except Exception:
                continue
            typ = obj_key.get("type")
            contention_id = obj_key.get("contention_id")
            evidence_uuid = obj_key.get("evidence_uuid")
            for ev in events:
                try:
                    turn_id = int(ev.get("turn_id"))
                except Exception:
                    continue
                action = ev.get("action")
                role = ev.get("role")
                if typ == "contention" and contention_id and action:
                    add_to_set(contention_action_to_turns, (contention_id, action), turn_id)
                    if role:
                        add_to_set(role_action_contention_to_turns, (role, action, contention_id), turn_id)
                elif typ == "evidence" and evidence_uuid and action:
                    add_to_set(evidence_action_to_turns, (evidence_uuid, action), turn_id)
                    if role:
                        add_to_set(role_action_evidence_to_turns, (role, action, evidence_uuid), turn_id)
    else:
        # Fallback: infer from storyboard turns
        for t in storyboard_turns_cache:
            action = t.get("action")
            role = t.get("role")
            turn_id = t.get("turn_id")
            obj = t.get("action_object")
            if not isinstance(obj, (str, int)):
                # Skip unsupported action objects (lists, None)
                continue
            if isinstance(obj, int):
                obj = str(obj)
            if isinstance(obj, str):
                if obj.startswith("pro_contention_") or obj.startswith("con_contention_"):
                    add_to_set(contention_action_to_turns, (obj, action), turn_id)
                    if role:
                        add_to_set(role_action_contention_to_turns, (role, action, obj), turn_id)
                else:
                    # Treat as evidence UUID
                    add_to_set(evidence_action_to_turns, (obj, action), turn_id)
                    if role:
                        add_to_set(role_action_evidence_to_turns, (role, action, obj), turn_id)

    with open(input_dir + "/eval_questions.jsonl", "r") as f:
        eval_qas = [EvalQA.from_dict(json.loads(line)) for line in f]
        for qa in eval_qas:
            # Build structured answer fields
            # qa.answer is either a single choice id (str) or a list of choice ids (List[str])
            # Set answer_type and populate the oneof in Answer accordingly
            # We do not convert to choice content strings here; we preserve ids

            # Compute relevant numeric turn_ids ONLY from meta.evidence_turn_ids
            relevant_numeric_turn_ids: Set[int] = set()

            evidence_turn_ids_field = []
            try:
                evidence_turn_ids_field = qa.meta.get("evidence_turn_ids", []) if isinstance(qa.meta, dict) else []
            except Exception:
                evidence_turn_ids_field = []
            if not isinstance(evidence_turn_ids_field, list):
                evidence_turn_ids_field = []
            for token in evidence_turn_ids_field:
                try:
                    relevant_numeric_turn_ids.add(int(token))
                except Exception:
                    continue

            # Map to proto turn ids and assign
            proto_turn_ids: List[str] = []
            for tid in sorted(relevant_numeric_turn_ids):
                proto_id = numeric_turn_id_to_proto_id.get(int(tid))
                if proto_id:
                    proto_turn_ids.append(proto_id)

            proto_question = Question()
            proto_question.id = qa.qid
            proto_question.content = qa.prompt
            proto_question.type = qa.qtype
            # Include the full conversation context for retrieval
            proto_question.question_turn_ids.extend(all_proto_turn_ids)
            # By construction, proto_turn_ids are the ground-truth turns that support the answer
            # Populate answer_turn_ids (GT for recall)
            try:
                proto_question.answer_turn_ids.extend(proto_turn_ids)
            except Exception:
                # In case the generated bindings differ, fail gracefully
                pass
            answer_format = getattr(qa, "answer_format", AnswerFormat.MULTIPLE_CHOICE.value)
            if answer_format == AnswerFormat.FREE_RESPONSE.value:
                proto_question.answer_type = _resolve_freeform_answer_type(qa)
                if isinstance(qa.answer, (list, tuple)):
                    candidates = [str(item) for item in qa.answer if item is not None]
                elif qa.answer is None:
                    candidates = []
                else:
                    candidates = [str(qa.answer)]
                proto_question.answer.free_form_answer = json.dumps(candidates, ensure_ascii=False)
            else:
                if isinstance(qa.answer, list):
                    proto_question.answer_type = AnswerType.ANSWER_TYPE_MULTIPLE_CHOICE_MULTIPLE
                    proto_question.answer.multiple_choice_ids.CopyFrom(
                        MultipleChoiceAnswer(choice_ids=[str(choice_id) for choice_id in qa.answer])
                    )
                else:
                    proto_question.answer_type = AnswerType.ANSWER_TYPE_MULTIPLE_CHOICE_SINGLE
                    proto_question.answer.single_choice_id = str(qa.answer)

            # answer_turn_ids left empty
            dataset.questions.append(proto_question)

    write_dataset(dataset, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert JSONL aligned with Dataset proto"
        )
    )
    parser.add_argument("-i", "--input_dir", required=True, help="Path to input directory")
    parser.add_argument(
        "--split_contention_turns",
        action="store_true",
        help="If set, split contention-level attack/defend turns into 3-sentence chunks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_debate(args.input_dir, args.input_dir, split_contention_turns=args.split_contention_turns)

if __name__ == "__main__":
    main()
