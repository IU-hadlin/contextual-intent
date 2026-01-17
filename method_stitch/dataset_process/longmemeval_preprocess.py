"""
Converter: Read JSONL aligned with longmemeval.proto and output Dataset from project_dataset_uniform.proto

Usage:
  python -m src.dataset_preprocess.longmemeval_preprocess \
    --input data/longmemeval-m/longmemeval-m.jsonl \
    --output-dir data/longmemeval-m \
    --name longmemeval-m \
    [--num-questions 50]

When --format jsonl is used, --output-dir should be a directory. The script will write:
  - questions.jsonl: one JSON object per question
  - turns.jsonl: one JSON object per turn
  - meta.jsonl: a single JSON object with dataset metadata and file paths
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from src.utils import write_dataset, load_questions, load_turns
from tqdm import tqdm
from google.protobuf import json_format
from google.protobuf.json_format import MessageToDict


from src import LongMemEvalItem, Dataset, Turn, Question, AnswerType



def read_items_from_jsonl(jsonl_path: str) -> List[LongMemEvalItem]:
    items: List[LongMemEvalItem] = []
    
    # Check if file is JSON array or JSONL format
    with open(jsonl_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            # JSON array format - read entire file
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON array, got {type(data)}")
            total_items = len(data)
            for obj in tqdm(data, total=total_items, desc="Processing items"):
                try:
                    obj["haystack_sessions"] = [
                        {"messages": session} for session in obj["haystack_sessions"]
                    ]
                    obj["answer"] = str(obj["answer"])
                    # Accept unknown fields to be flexible with inputs
                    msg = LongMemEvalItem()
                    json_format.ParseDict(obj, msg, ignore_unknown_fields=True)
                    items.append(msg)
                except Exception as e:
                    print(f"Error parsing item: {obj.get('question_id', 'unknown')}")
                    print(f"Error: {e}")
                    raise
        else:
            # JSONL format - one JSON object per line
            total_num_lines = sum(1 for _ in open(jsonl_path, "r", encoding="utf-8"))
            for line in tqdm(f, total=total_num_lines, desc="Processing lines"):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line) 
                    obj["haystack_sessions"] = [
                        {"messages": session} for session in obj["haystack_sessions"]
                    ]
                    obj["answer"] = str(obj["answer"])
                    # Accept unknown fields to be flexible with inputs
                    msg = LongMemEvalItem()
                    json_format.ParseDict(obj, msg, ignore_unknown_fields=True)
                    items.append(msg)
                except Exception as e:
                    print(f"Error parsing line: {line[:100]}")
                    print(f"Error: {e}")
                    raise
    return items


def convert_to_uniform(
    items: List[LongMemEvalItem], dataset_name: str
) -> Dataset:
    dataset = Dataset()
    dataset.name = dataset_name

    # Aggregate unique turns keyed by turn_id
    turn_id_to_turn: Dict[str, Turn] = {}

    for item in items:
        question = Question()
        question.id = item.question_id
        if item.question_type:
            question.type = item.question_type
        question.content = item.question
        question.answer_type = AnswerType.ANSWER_TYPE_FREEFORM
        question.answer.free_form_answer = item.answer
        if item.question_date:
            question.date = item.question_date

        # Index sessions
        num_sessions = len(item.haystack_sessions)
        for idx in range(num_sessions):
            # Align parallel arrays defensively
            session = item.haystack_sessions[idx]
            session_id_raw = item.haystack_session_ids[idx] if idx < len(item.haystack_session_ids) else str(idx)
            session_id_clean = session_id_raw.replace("answer_", "")
            session_date = item.haystack_dates[idx] if idx < len(item.haystack_dates) else ""

            for turn_idx, msg in enumerate(session.messages):
                if not msg.content:
                    continue
                
                # Get original content (without timestamp) for consistency checking
                original_content = str(msg.content)
                
                # Append session timestamp to content for downstream visibility
                turn_content = original_content
                if session_date:
                    turn_content = f"{turn_content} ---TIMESTAMP: {session_date}"

                turn_id = f"session-{session_id_clean}-turn-{turn_idx}"
                partition_id = f"question:{item.question_id}"
                if turn_id not in turn_id_to_turn:
                    turn = Turn()
                    turn.id = turn_id
                    turn.role = msg.role
                    turn.content = turn_content
                    turn.timestamp_mapping[partition_id] = session_date
                    # Helpful partitions for downstream filtering
                    turn.partition.extend(
                        [
                            partition_id
                        ]
                    )
                    turn_id_to_turn[turn_id] = turn
                else:
                    # Same turn_id encountered again - validate consistency and map timestamp correctly
                    existing = turn_id_to_turn[turn_id]
                    
                    # Extract original content from existing turn (remove timestamp if present)
                    existing_original_content = existing.content
                    if " ---TIMESTAMP:" in existing_original_content:
                        existing_original_content = existing_original_content.split(" ---TIMESTAMP:")[0]
                    
                    # Validate consistency: original content and role must match
                    if existing_original_content != original_content:
                        raise ValueError(
                            f"Turn {turn_id} has inconsistent content: "
                            f"existing='{existing_original_content[:50]}...' vs new='{original_content[:50]}...'"
                        )
                    if existing.role != msg.role:
                        raise ValueError(
                            f"Turn {turn_id} has inconsistent role: existing='{existing.role}' vs new='{msg.role}'"
                        )
                    
                    # Same turn appearing in different partition/question - map timestamp correctly
                    # Each partition can have its own timestamp, store it in timestamp_mapping
                    existing.timestamp_mapping[partition_id] = session_date
                    
                    # Add partition if not already present
                    if partition_id not in existing.partition:
                        existing.partition.append(partition_id)
                    
                    # Note: Content field keeps the first timestamp encountered for downstream visibility
                    # All partition-specific timestamps are correctly stored in timestamp_mapping[partition_id] 

                # Associate with question
                question.question_turn_ids.append(turn_id)
                if getattr(msg, "has_answer", False):
                    question.answer_turn_ids.append(turn_id)

        dataset.questions.append(question)

    # Attach aggregated turns
    dataset.turns.extend(turn_id_to_turn.values())
    return dataset


def write_simple_dataset(dataset: Dataset, output_dir: Path):
    """Write dataset with simple filenames: questions.jsonl, turns.jsonl, meta.json"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    questions_path = output_dir / "questions.jsonl"
    turns_path = output_dir / "turns.jsonl"
    meta_path = output_dir / "meta.json"
    
    # Write questions
    with open(questions_path, "w", encoding="utf-8") as f:
        for q in dataset.questions:
            obj = MessageToDict(q, preserving_proto_field_name=True)
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")
    
    # Write turns
    with open(turns_path, "w", encoding="utf-8") as f:
        for t in dataset.turns:
            obj = MessageToDict(t, preserving_proto_field_name=True)
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")
    
    # Write meta
    meta_obj = {
        "dataset_name": dataset.name,
        "questions_file": str(questions_path),
        "turns_file": str(turns_path),
        "num_questions": len(dataset.questions),
        "num_turns": len(dataset.turns),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_obj, f, ensure_ascii=False)

def _load_turn_order_map(turns_path: Path) -> Dict[str, int]:
    """Return turn_id -> position map from a reference turns.jsonl."""
    order: Dict[str, int] = {}
    if not turns_path.exists():
        return order
    with turns_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                obj = json.loads(line)
                turn_id = str(obj.get("id", "")).strip()
                if turn_id and turn_id not in order:
                    order[turn_id] = idx
            except json.JSONDecodeError:
                continue
    return order


def _sort_turns_by_reference(
    turns: List[Dict],
    order_map: Dict[str, int],
) -> List[Dict]:
    """Sort turns to match a reference ordering; unknown ids keep relative order at the end."""

    fallback_start = len(order_map)
    sortable: List[Tuple[int, int, Dict]] = []
    for idx, obj in enumerate(turns):
        turn_id = str(obj.get("id", "")).strip()
        order_val = order_map.get(turn_id, fallback_start + idx)
        sortable.append((order_val, idx, obj))
    sortable.sort(key=lambda item: (item[0], item[1]))
    return [obj for _, _, obj in sortable]


def _extract_first_timestamp(obj: Dict) -> Optional[str]:
    """Fetch the first non-empty timestamp value from a turn JSON object."""
    mapping = obj.get("timestamp_mapping")
    if not isinstance(mapping, dict):
        return None
    for value in mapping.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _ensure_timestamp_on_content(obj: Dict) -> bool:
    """Append timestamp to content if available and not already present."""
    ts = _extract_first_timestamp(obj)
    if not ts:
        return False
    content = str(obj.get("content", ""))
    marker = " ---TIMESTAMP:"
    if marker in content:
        return False
    obj["content"] = f"{content} {marker} {ts}"
    return True


def reorder_turn_file(turns_path: Path, order_map: Dict[str, int]) -> bool:
    """Reorder a turns.jsonl file in-place to follow the reference order_map."""
    if not turns_path.exists():
        return False
    with turns_path.open("r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]
    turns: List[Dict] = []
    modified = False
    for line in raw_lines:
        try:
            obj = json.loads(line)
            if _ensure_timestamp_on_content(obj):
                modified = True
            turns.append(obj)
        except json.JSONDecodeError:
            continue
    if not turns:
        return False
    sorted_turns = _sort_turns_by_reference(turns, order_map)
    # Only rewrite if order changed
    if turns == sorted_turns and not modified:
        return False
    with turns_path.open("w", encoding="utf-8") as f:
        for obj in sorted_turns:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")
    return True


def create_individual_question_datasets(
    dataset: Dataset,
    num_questions: int = 50,
    seed: int = 42,
    base_output_dir: Path = None,
    base_turn_order: Optional[Dict[str, int]] = None,
):
    """Create individual subdirectories for each randomly selected question.
    
    Each subdirectory contains:
    - questions.jsonl: one question
    - turns.jsonl: only turns from that question's question_turn_ids
    - meta.json: dataset metadata
    """
    if len(dataset.questions) <= num_questions:
        num_questions = len(dataset.questions)
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Randomly select questions
    selected_questions = random.sample(dataset.questions, num_questions)
    
    # Create a mapping of turn_id to Turn for quick lookup
    turn_id_to_turn: Dict[str, Turn] = {turn.id: turn for turn in dataset.turns}
    
    # Create a subdirectory for each question
    for idx, question in enumerate(selected_questions, 1):
        # Create subdirectory with numeric name (1-indexed from 1-50)
        question_dir = base_output_dir / str(idx)
        question_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect turns from this question's question_turn_ids preserving declared order
        ordered_turn_ids: List[str] = []
        seen: Set[str] = set()
        for tid in question.question_turn_ids:
            if tid in seen:
                continue
            if tid in turn_id_to_turn:
                ordered_turn_ids.append(tid)
                seen.add(tid)
        question_turns = [turn_id_to_turn[tid] for tid in ordered_turn_ids]

        # If a reference order is available, align to it for deterministic, global ordering
        if base_turn_order:
            question_turns.sort(
                key=lambda t: (
                    base_turn_order.get(t.id, len(base_turn_order)),
                    ordered_turn_ids.index(t.id) if t.id in ordered_turn_ids else 0,
                )
            )
        
        # Create dataset with single question and its turns
        question_dataset = Dataset()
        question_dataset.name = f"{dataset.name}_{idx}"
        question_dataset.questions.append(question)
        question_dataset.turns.extend(question_turns)
        
        # Write dataset to question's subdirectory with simple filenames
        write_simple_dataset(question_dataset, question_dir)
        print(f"Created dataset for question {idx}/{num_questions}: {question.id} -> subdir '{idx}' ({len(question_turns)} turns)")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert JSONL aligned with longmemeval.proto to Dataset"
        )
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input JSONL file")
    parser.add_argument("-o", "--output-dir", required=True, help="Path to output directory")
    parser.add_argument(
        "--name",
        required=True,
        choices=["longmemeval-m", "longmemeval-o", "longmemeval-s"],
        help="Dataset name to embed in the output (must be one of: longmemeval-m, longmemeval-o, longmemeval-s)"
    )
    parser.add_argument(
        "-nq",
        "--num-questions",
        type=int,
        default=50,
        help="Number of questions to sample into individual datasets (default: 50)",
    )
    return parser.parse_args()


def check_existing_files(output_dir: Path, dataset_name: str) -> Tuple[bool, Path, Path, Path]:
    """Check if processed files exist. Returns (exists, questions_path, turns_path, meta_path)."""
    questions_path = output_dir / f"{dataset_name}_questions.jsonl"
    turns_path = output_dir / f"{dataset_name}_turns.jsonl"
    meta_path = output_dir / f"{dataset_name}_meta.json"
    
    exists = questions_path.exists() and turns_path.exists() and meta_path.exists()
    return exists, questions_path, turns_path, meta_path


def load_dataset_from_files(questions_path: Path, turns_path: Path, dataset_name: str) -> Dataset:
    """Load a Dataset from existing processed files."""
    print(f"Loading dataset from existing files...")
    print(f"  Questions: {questions_path}")
    print(f"  Turns: {turns_path}")
    
    questions = load_questions(str(questions_path))
    turns = load_turns(dataset_name, str(turns_path))
    
    dataset = Dataset()
    dataset.name = dataset_name
    dataset.questions.extend(questions)
    dataset.turns.extend(turns)
    
    print(f"Loaded {len(questions)} questions and {len(turns)} turns")
    return dataset


def prompt_user_reprocess() -> bool:
    """Ask user if they want to reprocess. Returns True if yes, False if no."""
    while True:
        response = input("Processed files detected. Do you want to process again? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'")


def rename_existing_files(base_output_dir: Path):
    """Rename existing files in subdirectories to use simple names."""
    base_dir = Path(base_output_dir)
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist, skipping rename")
        return
    
    renamed_count = 0
    for subdir in base_dir.iterdir():
        if not subdir.is_dir():
            continue
        
        # Look for files with dataset name prefix
        for old_file in subdir.glob("*_questions.jsonl"):
            new_file = subdir / "questions.jsonl"
            if old_file != new_file:
                old_file.rename(new_file)
                renamed_count += 1
                print(f"Renamed {old_file.name} -> {new_file.name}")
        
        for old_file in subdir.glob("*_turns.jsonl"):
            new_file = subdir / "turns.jsonl"
            if old_file != new_file:
                old_file.rename(new_file)
                renamed_count += 1
                print(f"Renamed {old_file.name} -> {new_file.name}")
        
        for old_file in subdir.glob("*_meta.json"):
            new_file = subdir / "meta.json"
            if old_file != new_file:
                old_file.rename(new_file)
                renamed_count += 1
                print(f"Renamed {old_file.name} -> {new_file.name}")
    
    if renamed_count > 0:
        print(f"\nRenamed {renamed_count} files total")
    else:
        print("No files needed renaming")


def main() -> None:
    args = parse_args()
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if processed files already exist
    files_exist, questions_path, turns_path, meta_path = check_existing_files(
        base_output_dir, args.name
    )
    
    if files_exist:
        should_reprocess = prompt_user_reprocess()
        if not should_reprocess:
            # Load dataset from existing files
            dataset = load_dataset_from_files(questions_path, turns_path, args.name)
        else:
            # Process from input file
            items = read_items_from_jsonl(args.input)
            print(f"Read {len(items)} items")
            dataset = convert_to_uniform(items, args.name)
            write_dataset(dataset, base_output_dir)
            print(f"Wrote dataset to {base_output_dir}")
    else:
        # Process from input file
        items = read_items_from_jsonl(args.input)
        print(f"Read {len(items)} items")
        dataset = convert_to_uniform(items, args.name)
        write_dataset(dataset, base_output_dir)
        print(f"Wrote dataset to {base_output_dir}")
    
    # Rename existing files in subdirectories if they exist
    print(f"\nRenaming existing files in subdirectories...")
    rename_existing_files(base_output_dir)
    
    # Compute reference order from the full turns file
    reference_turns_path = base_output_dir / f"{args.name}_turns.jsonl"
    reference_order = _load_turn_order_map(reference_turns_path)

    # Validate and clamp requested question count
    if args.num_questions <= 0:
        raise ValueError("--num-questions must be a positive integer")
    num_questions = min(args.num_questions, len(dataset.questions))

    # Create individual question datasets
    print(f"\nCreating {num_questions} individual question datasets...")
    create_individual_question_datasets(
        dataset,
        num_questions=num_questions,
        base_output_dir=base_output_dir,
        base_turn_order=reference_order,
    )
    # Reorder any existing per-question turns.jsonl files to match the reference
    if reference_order:
        print("\nReordering per-question turns.jsonl files to match reference order...")
        reordered = 0
        for subdir in base_output_dir.iterdir():
            if not subdir.is_dir():
                continue
            turns_path = subdir / "turns.jsonl"
            if reorder_turn_file(turns_path, reference_order):
                reordered += 1
        print(f"Reordered turns in {reordered} subdirectories")
    print(f"\nFinished creating individual question datasets in {base_output_dir}")

if __name__ == "__main__":
    main()
