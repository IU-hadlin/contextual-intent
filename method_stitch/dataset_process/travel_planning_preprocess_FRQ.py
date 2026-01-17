import argparse
import json
import random
import uuid
from typing import List, Dict, Any, Tuple, Optional, Set
from src import Dataset, Turn, Question, AnswerType, Choice, Answer
from dataset_construction.travel_plan.datatypes import Turn as TravelTurn, EvalQA, AnswerFormat
import os
from src.utils import write_dataset

DATASET_NAME = "travel_planning"


def _extract_action_object(object_info: dict) -> str:
    if not isinstance(object_info, dict):
        return ""
    names: List[str] = []
    if "object_names" in object_info and isinstance(object_info["object_names"], list):
        names = [n for n in object_info["object_names"] if isinstance(n, str)]
    elif "object_name_1" in object_info or "object_name_2" in object_info:
        first = object_info.get("object_name_1")
        second = object_info.get("object_name_2")
        names = [n for n in [first, second] if isinstance(n, str) and n]
    elif "name" in object_info and isinstance(object_info["name"], str):
        names = [object_info["name"]]
    return ", ".join(names)


def process_travel_planning(utterance_input_path: str, qa_input_path: str, output_dir: str):
    dataset = Dataset()
    dataset.name = DATASET_NAME
    question_turn_ids: List[str] = []

    # Derive a deterministic trip UUID from the utterance file path
    trip_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"trip://{utterance_input_path}")

    # Read utterances (JSONL of TravelPlan Turn dicts)
    # Index structures for per-question selection
    # turn_id -> proto Turn id
    source_turn_id_to_proto_id: Dict[int, str] = {}
    # proto Turn id -> content
    proto_id_to_content: Dict[str, str] = {}
    # ordered list of (source_turn_id, proto_id, role, action, object_names, option_type, day_index)
    ordered_turns: List[Tuple[int, str, str, str, List[str], Optional[str], Optional[int]]] = []
    # object name -> first PROPOSE turn proto_id
    first_propose_by_name: Dict[str, str] = {}
    # object name -> first USER PROPOSE turn proto_id
    first_user_propose_by_name: Dict[str, str] = {}
    # (day_index, slot_key, object name) -> first PROPOSE turn proto_id
    first_propose_by_day_slot_name: Dict[Tuple[int, str, str], str] = {}
    # (day_index, slot_key, object name) -> first USER PROPOSE turn proto_id
    first_user_propose_by_day_slot_name: Dict[Tuple[int, str, str], str] = {}
    # object name -> list of DECIDE turn proto_ids (keep last for final)
    decide_turns_by_name: Dict[str, List[str]] = {}
    # object name -> list of USER DECIDE turn proto_ids (keep last for final)
    user_decide_turns_by_name: Dict[str, List[str]] = {}
    # (day_index, slot_key, object name) -> list of DECIDE turn proto_ids
    decide_turns_by_day_slot_name: Dict[Tuple[int, str, str], List[str]] = {}
    # (day_index, slot_key, object name) -> list of USER DECIDE turn proto_ids
    user_decide_turns_by_day_slot_name: Dict[Tuple[int, str, str], List[str]] = {}
    # object name -> all INQUIRE turn proto_ids (both roles)
    inquire_turns_by_name: Dict[str, List[str]] = {}
    # (day_index, slot_key, object name) -> all INQUIRE turn proto_ids
    inquire_turns_by_day_slot_name: Dict[Tuple[int, str, str], List[str]] = {}
    # (day_index, slot_key, object name, feature) -> all INQUIRE turn proto_ids
    inquire_turns_by_day_slot_name_feature: Dict[Tuple[int, str, str, str], List[str]] = {}
    # (object name, feature) -> all INQUIRE turn proto_ids
    inquire_turns_by_name_feature: Dict[Tuple[str, str], List[str]] = {}
    # (day_index, object name) -> set of slot_keys observed for that name on the day
    slots_by_day_name: Dict[Tuple[int, str], Set[str]] = {}

    def _normalize_day(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _candidate_slots(day: Optional[int], name: str, preferred_slot: Optional[str]) -> List[str]:
        slots: List[str] = []
        if isinstance(day, int):
            available = sorted(slots_by_day_name.get((day, name), set()))
            if isinstance(preferred_slot, str) and preferred_slot in available:
                slots.append(preferred_slot)
            for slot_name in available:
                if slot_name not in slots:
                    slots.append(slot_name)
        return slots

    def _first_propose_for(name: str, day: Optional[int], slot: Optional[str], prefer_user: bool = False) -> Optional[str]:
        if isinstance(day, int):
            for slot_name in _candidate_slots(day, name, slot):
                key = (day, slot_name, name)
                if prefer_user:
                    pid = first_user_propose_by_day_slot_name.get(key)
                    if pid:
                        return pid
                pid = first_propose_by_day_slot_name.get(key)
                if pid:
                    return pid
        if prefer_user:
            pid = first_user_propose_by_name.get(name)
            if pid:
                return pid
        return first_propose_by_name.get(name)

    def _decide_turns_for(name: str, day: Optional[int], slot: Optional[str], prefer_user: bool = False) -> List[str]:
        if isinstance(day, int):
            collected: List[str] = []
            for slot_name in _candidate_slots(day, name, slot):
                key = (day, slot_name, name)
                if prefer_user:
                    collected.extend(user_decide_turns_by_day_slot_name.get(key, []))
                if not prefer_user or not collected:
                    collected.extend(decide_turns_by_day_slot_name.get(key, []))
            if collected:
                return collected
        if prefer_user:
            collected = user_decide_turns_by_name.get(name, [])
            if collected:
                return list(collected)
        return list(decide_turns_by_name.get(name, []))

    def _inquire_turns_for(name: str, day: Optional[int], slot: Optional[str], feature: Optional[str] = None) -> List[str]:
        if isinstance(day, int) and isinstance(feature, str):
            collected: List[str] = []
            for slot_name in _candidate_slots(day, name, slot):
                key = (day, slot_name, name, feature)
                collected.extend(inquire_turns_by_day_slot_name_feature.get(key, []))
            if collected:
                return collected
        if isinstance(feature, str):
            return list(inquire_turns_by_name_feature.get((name, feature), []))
        if isinstance(day, int):
            collected: List[str] = []
            for slot_name in _candidate_slots(day, name, slot):
                key = (day, slot_name, name)
                collected.extend(inquire_turns_by_day_slot_name.get(key, []))
            if collected:
                return collected
        return list(inquire_turns_by_name.get(name, []))
    # (day, slot_key) -> all turn proto_ids that discuss this slot
    discussed_turns_by_day_slot: Dict[Tuple[int, str], List[str]] = {}
    # (day) -> mapping of GT objects to later collect for Q6
    # We'll compute Q6 from GT names derived from Q6 prompt later

    with open(utterance_input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            turn_dict = json.loads(line)
            turn = TravelTurn.from_dict(turn_dict)

            proto_turn = Turn()
            proto_turn.id = f"trip-{trip_uuid}-turn-{turn.turn_id}"
            proto_turn.content = turn.utterance
            proto_turn.role = turn.role
            proto_turn.partition.append(f"trip-{trip_uuid}")
            # Also partition by day for convenience
            try:
                proto_turn.partition.append(f"day-{int(turn.day_index)}")
            except Exception:
                pass
            proto_turn.action = turn.action
            proto_turn.action_object = _extract_action_object(turn.object_info)

            dataset.turns.append(proto_turn)
            question_turn_ids.append(proto_turn.id)
            proto_id_to_content[proto_turn.id] = proto_turn.content

            # Build indices
            source_turn_id_to_proto_id[turn.turn_id] = proto_turn.id
            option_type = None
            names: List[str] = []
            obj = turn.object_info if isinstance(turn.object_info, dict) else {}
            if isinstance(obj, dict):
                option_type = obj.get("object_type")
                if turn.action == "COMPARE":
                    for key in ["object_name_1", "object_name_2"]:
                        if isinstance(obj.get(key), str):
                            names.append(obj[key])
                else:
                    if isinstance(obj.get("object_names"), list):
                        names.extend([n for n in obj["object_names"] if isinstance(n, str)])
                    elif isinstance(obj.get("name"), str):
                        names.append(obj["name"])

            try:
                di = int(turn.day_index)
            except Exception:
                di = None

            # Determine slot for this turn to support day-aware tracking
            slot_key = None
            if option_type == "ACCOMMODATION":
                slot_key = "accommodation"
            elif option_type == "ATTRACTION":
                slot_key = "attraction"
            elif option_type == "DINING":
                # try to get dining type
                dt = obj.get("object_dining_type") or obj.get("Object Dining Type") or obj.get("dining_type")
                if isinstance(dt, str) and dt.upper() in ("BREAKFAST", "LUNCH", "DINNER"):
                    slot_key = dt.lower()
                else:
                    slot_key = "dining"

            ordered_turns.append((turn.turn_id, proto_turn.id, turn.role, turn.action, names, option_type, di))

            # Track first PROPOSE
            if turn.action == "PROPOSE":
                for n in names:
                    if n not in first_propose_by_name:
                        first_propose_by_name[n] = proto_turn.id
                    if turn.role == "user" and n not in first_user_propose_by_name:
                        first_user_propose_by_name[n] = proto_turn.id
                    if di is not None and slot_key is not None:
                        key = (di, slot_key, n)
                        if key not in first_propose_by_day_slot_name:
                            first_propose_by_day_slot_name[key] = proto_turn.id
                        if turn.role == "user" and key not in first_user_propose_by_day_slot_name:
                            first_user_propose_by_day_slot_name[key] = proto_turn.id
                        slots_by_day_name.setdefault((di, n), set()).add(slot_key)

            # Track DECIDE turns
            if turn.action == "DECIDE":
                for n in names:
                    decide_turns_by_name.setdefault(n, []).append(proto_turn.id)
                    if di is not None and slot_key is not None:
                        key = (di, slot_key, n)
                        decide_turns_by_day_slot_name.setdefault(key, []).append(proto_turn.id)
                        slots_by_day_name.setdefault((di, n), set()).add(slot_key)
                    if turn.role == "user":
                        user_decide_turns_by_name.setdefault(n, []).append(proto_turn.id)
                        if di is not None and slot_key is not None:
                            key = (di, slot_key, n)
                            user_decide_turns_by_day_slot_name.setdefault(key, []).append(proto_turn.id)
                            slots_by_day_name.setdefault((di, n), set()).add(slot_key)

            # Track INQUIRE turns (both roles)
            if turn.action == "INQUIRE":
                # Extract features from object_info
                features: List[str] = []
                if isinstance(obj, dict) and "object_features" in obj:
                    obj_features = obj["object_features"]
                    if isinstance(obj_features, list):
                        features = [f for f in obj_features if isinstance(f, str)]
                
                for n in names:
                    inquire_turns_by_name.setdefault(n, []).append(proto_turn.id)
                    if di is not None and slot_key is not None:
                        key = (di, slot_key, n)
                        inquire_turns_by_day_slot_name.setdefault(key, []).append(proto_turn.id)
                        slots_by_day_name.setdefault((di, n), set()).add(slot_key)
                    
                    # Track by feature
                    for feature in features:
                        inquire_turns_by_name_feature.setdefault((n, feature), []).append(proto_turn.id)
                        if di is not None and slot_key is not None:
                            key_with_feature = (di, slot_key, n, feature)
                            inquire_turns_by_day_slot_name_feature.setdefault(key_with_feature, []).append(proto_turn.id)

            # Track discussed per day-slot
            if di is not None and slot_key is not None:
                discussed_turns_by_day_slot.setdefault((di, slot_key), []).append(proto_turn.id)
                for n in names:
                    if n:
                        slots_by_day_name.setdefault((di, n), set()).add(slot_key)

    # Read evaluation questions (JSONL of EvalQA)
    rng = random.Random(str(trip_uuid))

    with open(qa_input_path, "r") as f:
        indexed_qas: List[Tuple[int, EvalQA]] = []
        for idx, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue
            qa = EvalQA.from_dict(json.loads(line))
            indexed_qas.append((idx, qa))

        grouped_qas: Dict[str, List[Tuple[int, EvalQA]]] = {}
        for idx, qa in indexed_qas:
            qtype_key = (qa.qtype or "").upper()
            grouped_qas.setdefault(qtype_key, []).append((idx, qa))

        filtered_indexed_qas: List[Tuple[int, EvalQA]] = []
        for qtype_key, qa_list in grouped_qas.items():
            if qtype_key in {"Q2", "Q3", "Q4"} and len(qa_list) > 15:
                selected = rng.sample(qa_list, 15)
                selected.sort(key=lambda item: item[0])
                filtered_indexed_qas.extend(selected)
            else:
                filtered_indexed_qas.extend(qa_list)

        filtered_indexed_qas.sort(key=lambda item: item[0])

        for _, qa in filtered_indexed_qas:

            proto_question = Question()
            proto_question.id = qa.qid
            proto_question.content = qa.prompt
            proto_question.type = qa.qtype
            answer_format = getattr(qa, "answer_format", AnswerFormat.MULTIPLE_CHOICE.value)
            is_free_response = answer_format == AnswerFormat.FREE_RESPONSE.value
            if is_free_response:
                proto_question.answer_type = AnswerType.ANSWER_TYPE_FREEFORM
                proto_question.answer.free_form_answer = str(qa.answer)
            else:
                # Default to multiple choice single answer
                proto_question.answer_type = AnswerType.ANSWER_TYPE_MULTIPLE_CHOICE_SINGLE
                proto_question.answer.single_choice_id = str(qa.answer)

            # Compute question_turn_ids per qtype
            qtype = (qa.qtype or "").upper()
            selected_ids: List[str] = []


            if qtype == "Q1":
                # decision tracking: first USER PROPOSE and last USER DECIDE for the decided object(s)
                decided_names: List[str] = []
                slot_for_q1 = None
                day_for_q1: Optional[int] = None
                if isinstance(qa.meta, dict):
                    slot_for_q1 = qa.meta.get("slot")
                    day_for_q1 = _normalize_day(qa.meta.get("day_index"))
                    meta_name = qa.meta.get("object_name")
                    if isinstance(meta_name, str):
                        decided_names.append(meta_name)
                    meta_names = qa.meta.get("object_names")
                    if isinstance(meta_names, list):
                        decided_names.extend([n for n in meta_names if isinstance(n, str) and n])
                if not decided_names:
                    if isinstance(qa.answer, str):
                        decided_names = [qa.answer]
                dedup_decided: List[str] = []
                seen_decided: Set[str] = set()
                for nm in decided_names:
                    if isinstance(nm, str) and nm and nm not in seen_decided:
                        dedup_decided.append(nm)
                        seen_decided.add(nm)
                for name in dedup_decided:
                    # First USER PROPOSE for this object
                    pid = _first_propose_for(name, day_for_q1, slot_for_q1, prefer_user=True)
                    if pid:
                        selected_ids.append(pid)
                    # Last USER DECIDE for this object
                    decides = _decide_turns_for(name, day_for_q1, slot_for_q1, prefer_user=True)
                    if decides:
                        selected_ids.append(decides[-1])

            elif qtype == "Q2":
                # attribute recall: all INQUIRE turns (both roles) for that object with the specific feature
                name = None
                day_for_q2: Optional[int] = None
                slot_for_q2: Optional[str] = None
                feature_for_q2: Optional[str] = None
                if isinstance(qa.meta, dict):
                    name = qa.meta.get("object_name")
                    day_for_q2 = _normalize_day(qa.meta.get("day_index"))
                    slot_for_q2 = qa.meta.get("slot")
                    feature_for_q2 = qa.meta.get("attribute")
                if not name:
                    if isinstance(qa.answer, str):
                        # cannot infer reliably from price/rating wording; rely on meta if possible
                        name = None
                if name:
                    selected_ids = _inquire_turns_for(name, day_for_q2, slot_for_q2, feature_for_q2)

            elif qtype == "Q3":
                # multi-hop object tracking: first PROPOSE per object for the day/slot
                di = None
                slot = None
                if isinstance(qa.meta, dict):
                    di = _normalize_day(qa.meta.get("day_index"))
                    slot = qa.meta.get("slot")
                candidate_names: List[str] = []
                if isinstance(qa.meta, dict):
                    meta_names = qa.meta.get("object_names")
                    if isinstance(meta_names, list):
                        candidate_names.extend([n for n in meta_names if isinstance(n, str) and n])
                if not candidate_names and isinstance(qa.answer, str):
                    parts = [p.strip() for p in qa.answer.split(",") if p and p.strip()]
                    candidate_names.extend(parts)
                dedup_names: List[str] = []
                seen_names_for_q3: Set[str] = set()
                for nm in candidate_names:
                    if nm not in seen_names_for_q3:
                        dedup_names.append(nm)
                        seen_names_for_q3.add(nm)

                selected: List[str] = []
                if isinstance(di, int) and isinstance(slot, str):
                    keys_to_try = [(di, slot)]
                    if di - 1 >= 0:
                        keys_to_try.append((di - 1, slot))
                    for name in dedup_names:
                        pid_found: Optional[str] = None
                        for day_key in keys_to_try:
                            key = (day_key[0], day_key[1], name)
                            if key in first_propose_by_day_slot_name:
                                pid_found = first_propose_by_day_slot_name[key]
                                break
                        if pid_found:
                            selected.append(pid_found)
                        else:
                            fallback_pid = first_propose_by_name.get(name)
                            if fallback_pid:
                                selected.append(fallback_pid)
                else:
                    for name in dedup_names:
                        fallback_pid = first_propose_by_name.get(name)
                        if fallback_pid:
                            selected.append(fallback_pid)
                selected_ids = selected

            elif qtype == "Q4":
                # plan synthesis: retrieve all GT objects' related turns of that day
                di = None
                if isinstance(qa.meta, dict):
                    di = _normalize_day(qa.meta.get("day_index"))
                names_in_plan: Set[str] = set()
                if isinstance(qa.meta, dict):
                    meta_names = qa.meta.get("object_names") or []
                    if isinstance(meta_names, list):
                        names_in_plan.update([n for n in meta_names if isinstance(n, str) and n])
                if not names_in_plan:
                    plan_str = None
                    if isinstance(qa.answer, str):
                        plan_str = qa.answer
                    if isinstance(plan_str, str):
                        known_names = set(first_propose_by_name.keys()) | set(decide_turns_by_name.keys())
                        for nm in known_names:
                            if nm and nm in plan_str:
                                names_in_plan.add(nm)
                if not names_in_plan and isinstance(di, int):
                    for (day, slot), pids in discussed_turns_by_day_slot.items():
                        if day == di:
                            for src_id, pid, role, action, names, _ot, _di in ordered_turns:
                                if pid in pids:
                                    for nm in names:
                                        names_in_plan.add(nm)
                collected: List[str] = []
                seen_collected: Set[str] = set()
                ordered_names = sorted(names_in_plan)
                for nm in ordered_names:
                    if not nm:
                        continue
                    slot_candidates = _candidate_slots(di, nm, None)
                    if not slot_candidates:
                        pid = _first_propose_for(nm, di, None)
                        if pid and pid not in seen_collected:
                            collected.append(pid)
                            seen_collected.add(pid)
                        decides = _decide_turns_for(nm, di, None)
                        if decides:
                            final_decide = decides[-1]
                            if final_decide and final_decide not in seen_collected:
                                collected.append(final_decide)
                                seen_collected.add(final_decide)
                        continue
                    for slot_name in slot_candidates:
                        pid = _first_propose_for(nm, di, slot_name)
                        if pid and pid not in seen_collected:
                            collected.append(pid)
                            seen_collected.add(pid)
                        decides = _decide_turns_for(nm, di, slot_name)
                        if decides:
                            final_decide = decides[-1]
                            if final_decide and final_decide not in seen_collected:
                                collected.append(final_decide)
                                seen_collected.add(final_decide)
                selected_ids = collected

            else:
                # Default fallback: use all turns
                selected_ids = list(question_turn_ids)

            # De-duplicate while preserving order
            seen = set()
            deduped = []
            for tid in selected_ids:
                if tid not in seen:
                    deduped.append(tid)
                    seen.add(tid)
            # Build final ids (default to GT turns)
            final_ids: List[str] = list(deduped)

            try:
                proto_question.answer_turn_ids.extend(final_ids)
            except Exception:
                pass
            proto_question.question_turn_ids.extend(question_turn_ids)
            # answer_turn_ids now populated above
            dataset.questions.append(proto_question)
                
    write_dataset(dataset, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert JSONL aligned with Dataset proto"
        )
    )
    parser.add_argument("-o", "--output_dir", required=True, help="Path to output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    utterance_input_path = os.path.join(args.output_dir, "storytelling_plan_with_utterances_post_processed.jsonl")
    qa_input_path = os.path.join(args.output_dir, "eval_questions_FRQ.jsonl")
    process_travel_planning(utterance_input_path, qa_input_path, args.output_dir)

if __name__ == "__main__":
    main()
