"""Compute recall of hierarchy-based retrieval against ground-truth answer turns."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


@dataclass
class QuestionRetrievalStages:
    node_selected: List[str]
    node_selected_count: int
    embedding_candidates: List[str]
    embedding_candidate_count: int
    retrieved: List[str]
    retrieved_count: int
    excluded_turns: List[Tuple[str, List[str]]]


@dataclass
class QuestionGroundTruth:
    answer_turn_ids: Set[str]
    question_type: str


@dataclass
class QuestionRecallResult:
    question_id: str
    ground_truth: Set[str]
    question_type: str
    node_selected: List[str]
    node_selected_count: int
    embedding_candidates: List[str]
    embedding_candidate_count: int
    retrieved: List[str]
    retrieved_count: int
    excluded_turns: List[Tuple[str, List[str]]]
    node_selected_hits: Set[str]
    embedding_hits: Set[str]
    retrieved_hits: Set[str]
    node_selected_hits_at_k: Set[str]
    embedding_hits_at_k: Set[str]
    retrieved_hits_at_k: Set[str]
    k: int

    @property
    def recall_node_selected(self) -> float:
        if not self.ground_truth or self.is_type_five:
            return 1.0
        return len(self.node_selected_hits) / len(self.ground_truth)

    @property
    def recall_embedding_candidates(self) -> float:
        if not self.ground_truth or self.is_type_five:
            return 1.0
        return len(self.embedding_hits) / len(self.ground_truth)

    @property
    def recall_retrieved(self) -> float:
        if not self.ground_truth or self.is_type_five:
            return 1.0
        return len(self.retrieved_hits) / len(self.ground_truth)

    @property
    def recall_node_selected_at_k(self) -> float:
        if not self.ground_truth or self.is_type_five:
            return 1.0
        return len(self.node_selected_hits_at_k) / len(self.ground_truth)

    @property
    def recall_retrieved_at_k(self) -> float:
        if not self.ground_truth or self.is_type_five:
            return 1.0
        return len(self.retrieved_hits_at_k) / len(self.ground_truth)

    @property
    def missing_after_node_selection(self) -> Set[str]:
        if self.is_type_five:
            return set()
        return self.ground_truth - self.node_selected_hits

    @property
    def missing_after_embedding(self) -> Set[str]:
        if self.is_type_five:
            return set()
        return self.ground_truth - self.embedding_hits

    @property
    def missing_after_retrieval(self) -> Set[str]:
        if self.is_type_five:
            return set()
        return self.ground_truth - self.retrieved_hits

    @property
    def is_type_five(self) -> bool:
        return str(self.question_type) == "5"


def load_ground_truth(question_path: Path) -> Dict[str, QuestionGroundTruth]:
    mapping: Dict[str, QuestionGroundTruth] = {}
    with question_path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            question_id = payload.get("id")
            if not isinstance(question_id, str):
                continue
            answer_turn_ids = payload.get("answer_turn_ids", [])
            question_type = payload.get("type")
            if question_type is None:
                question_type = payload.get("question_type")
            if question_type is None:
                question_type = payload.get("category")
            question_type = "" if question_type is None else str(question_type)
            if isinstance(answer_turn_ids, (list, tuple, set)):
                turns = {
                    str(turn_id) for turn_id in answer_turn_ids if turn_id is not None
                }
            else:
                turns = set()
            mapping[question_id] = QuestionGroundTruth(
                answer_turn_ids=turns,
                question_type=question_type,
            )
    return mapping


def load_retrieved_turns(summary_path: Path) -> tuple[Dict[str, QuestionRetrievalStages], int]:
    def _dedupe_preserve_order(items: List[str]) -> List[str]:
        seen: Set[str] = set()
        ordered: List[str] = []
        for value in items:
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    with summary_path.open("r", encoding="utf-8") as infile:
        payload = json.load(infile)
    question_results = payload.get("question_results", [])
    retrieved: Dict[str, QuestionRetrievalStages] = {}
    for entry in question_results:
        if not isinstance(entry, dict):
            continue
        question_id = entry.get("question_id")
        node_selected_turn_ids = entry.get("node_selected_turn_ids", [])
        embedding_candidate_turn_ids = entry.get("embedding_candidate_turn_ids", [])
        retrieved_turn_ids = entry.get("retrieved_turn_ids", [])
        node_selected_turn_count = entry.get("node_selected_turn_count")
        embedding_candidate_turn_count = entry.get("embedding_candidate_turn_count")
        retrieved_turn_count = entry.get("retrieved_turn_count")
        excluded_turns_raw = entry.get("excluded_turns", [])
        if not isinstance(question_id, str):
            continue
        if not isinstance(node_selected_turn_ids, list):
            node_selected_turn_ids = []
        if not isinstance(retrieved_turn_ids, list):
            retrieved_turn_ids = []
        node_selected_list = _dedupe_preserve_order(
            [str(tid) for tid in node_selected_turn_ids if isinstance(tid, str)]
        )
        embedding_candidate_list = _dedupe_preserve_order(
            [str(tid) for tid in embedding_candidate_turn_ids if isinstance(tid, str)]
        )
        retrieved_list = _dedupe_preserve_order(
            [str(tid) for tid in retrieved_turn_ids if isinstance(tid, str)]
        )
        if not isinstance(node_selected_turn_count, int):
            node_selected_turn_count = len(node_selected_list)
        if not isinstance(embedding_candidate_turn_count, int):
            embedding_candidate_turn_count = len(embedding_candidate_list)
        if not isinstance(retrieved_turn_count, int):
            retrieved_turn_count = len(retrieved_list)
        excluded_turns: List[Tuple[str, List[str]]] = []
        if isinstance(excluded_turns_raw, list):
            for item in excluded_turns_raw:
                if not isinstance(item, dict):
                    continue
                turn_id = item.get("turn_id")
                node_ids = item.get("node_ids", [])
                if not isinstance(turn_id, str):
                    continue
                if not isinstance(node_ids, list):
                    node_ids = []
                node_ids_clean = [str(node_id) for node_id in node_ids if isinstance(node_id, str)]
                excluded_turns.append((turn_id, node_ids_clean))
        retrieved[question_id] = QuestionRetrievalStages(
            node_selected=node_selected_list,
            node_selected_count=node_selected_turn_count,
            embedding_candidates=embedding_candidate_list,
            embedding_candidate_count=embedding_candidate_turn_count,
            retrieved=retrieved_list,
            retrieved_count=retrieved_turn_count,
            excluded_turns=excluded_turns,
        )
    final_topk = payload.get("final_topk")
    if not isinstance(final_topk, int) or final_topk <= 0:
        candidate_lengths = [
            len(data.retrieved)
            for data in retrieved.values()
            if data.retrieved
        ]
        final_topk = min(candidate_lengths) if candidate_lengths else 0
    return retrieved, final_topk


def evaluate_recall(
    ground_truth: Dict[str, QuestionGroundTruth],
    retrieved: Dict[str, QuestionRetrievalStages],
    cutoff_k: int,
) -> List[QuestionRecallResult]:
    results: List[QuestionRecallResult] = []
    for question_id, truth_info in ground_truth.items():
        truth = truth_info.answer_turn_ids
        question_type = truth_info.question_type
        stages = retrieved.get(
            question_id,
            QuestionRetrievalStages(
                node_selected=[],
                node_selected_count=0,
                embedding_candidates=[],
                embedding_candidate_count=0,
                retrieved=[],
                retrieved_count=0,
                excluded_turns=[],
            ),
        )
        node_hits = truth & set(stages.node_selected)
        embedding_hits = truth & set(stages.embedding_candidates)
        retrieved_hits = truth & set(stages.retrieved)
        k = cutoff_k if cutoff_k > 0 else len(stages.retrieved)
        node_topk_hits = truth & set(stages.node_selected[:k])
        embedding_topk_hits = truth & set(stages.embedding_candidates[:k])
        retrieved_topk_hits = truth & set(stages.retrieved[:k])

        if str(question_type) == "5":
            node_hits = set(truth)
            embedding_hits = set(truth)
            retrieved_hits = set(truth)
            node_topk_hits = set(truth)
            embedding_topk_hits = set(truth)
            retrieved_topk_hits = set(truth)
        results.append(
            QuestionRecallResult(
                question_id=question_id,
                ground_truth=truth,
                question_type=question_type,
                node_selected=stages.node_selected,
                node_selected_count=stages.node_selected_count,
                embedding_candidates=stages.embedding_candidates,
                embedding_candidate_count=stages.embedding_candidate_count,
                retrieved=stages.retrieved,
                retrieved_count=stages.retrieved_count,
                excluded_turns=stages.excluded_turns,
                node_selected_hits=node_hits,
                embedding_hits=embedding_hits,
                retrieved_hits=retrieved_hits,
                node_selected_hits_at_k=node_topk_hits,
                embedding_hits_at_k=embedding_topk_hits,
                retrieved_hits_at_k=retrieved_topk_hits,
                k=k,
            )
        )
    return results


def summarise(results: Iterable[QuestionRecallResult]) -> None:
    total_truth = 0
    total_node_hits = 0
    total_embedding_hits = 0
    total_retrieved_hits = 0
    total_node_selected_turns = 0
    total_embedding_turns = 0
    total_retrieved_turns = 0
    total_missing_after_nodes = 0
    total_missing_after_embedding = 0
    total_missing_after_retrieval = 0

    columns = [
        ("question_id", "<30", "str"),
        ("gt_N", ">6", "int"),
        ("node_recall@all", ">18.3f", "float"),
        ("node_turns", ">12", "int"),
        ("embed_recall@all", ">18.3f", "float"),
        ("excluded_after_nodes", ">20", "int"),
        ("excluded_after_embedding", ">24", "int"),
    ]

    header_line = "\t".join(f"{name:{fmt.split('.')[0]}}" for name, fmt, _ in columns)
    print(header_line)

    for result in results:
        gt_size = len(result.ground_truth)
        hits_node = len(result.node_selected_hits)
        hits_embedding = len(result.embedding_hits)
        hits_retrieved = len(result.retrieved_hits)
        total_truth += gt_size
        total_node_hits += hits_node
        total_embedding_hits += hits_embedding
        total_retrieved_hits += hits_retrieved
        total_node_selected_turns += result.node_selected_count
        total_embedding_turns += result.embedding_candidate_count
        total_retrieved_turns += result.retrieved_count
        excluded_after_nodes = len(result.missing_after_node_selection)
        excluded_after_embedding = len(result.missing_after_embedding)
        excluded_after_retrieval = len(result.missing_after_retrieval)
        total_missing_after_nodes += excluded_after_nodes
        total_missing_after_embedding += excluded_after_embedding
        total_missing_after_retrieval += excluded_after_retrieval

        row_data = {
            "question_id": result.question_id,
            "gt_N": gt_size,
            "node_recall@all": result.recall_node_selected,
            "node_turns": result.node_selected_count,
            "embed_recall@all": result.recall_embedding_candidates,
            "excluded_after_nodes": excluded_after_nodes,
            "excluded_after_embedding": excluded_after_embedding,
        }

        row_values = []
        for name, fmt, value_type in columns:
            value = row_data[name]
            if value_type == "float":
                row_values.append(f"{value:{fmt}}")
            else:
                row_values.append(f"{str(value):{fmt}}")
        print("\t".join(row_values))

    if total_truth:
        overall_node_recall = total_node_hits / total_truth
        overall_embedding_recall = total_embedding_hits / total_truth
    else:
        overall_node_recall = 1.0 if results else 0.0
        overall_embedding_recall = 1.0 if results else 0.0

    if results:
        print("\nOverall node-selected recall:", f"{overall_node_recall:.3f}")
        print("Overall embedding-candidate recall:", f"{overall_embedding_recall:.3f}")
        print("Total excluded after node selection:", total_missing_after_nodes)
        print("Total excluded after embedding candidates:", total_missing_after_embedding)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate recall of hierarchy retrieval results",
    )
    parser.add_argument(
        "--questions-jsonl",
        required=True,
        help="Path to questions JSONL containing answer_turn_ids",
    )
    parser.add_argument(
        "--retrieval-summary",
        required=True,
        help="Path to hierarchy_context_retrieval output JSON",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help=(
            "If set, only include question IDs whose conv-<n> prefix has n less "
            "than or equal to this value."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions_path = Path(args.questions_jsonl)
    retrieval_path = Path(args.retrieval_summary)

    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    if not retrieval_path.exists():
        raise FileNotFoundError(f"Retrieval summary not found: {retrieval_path}")

    ground_truth = load_ground_truth(questions_path)
    retrieved, final_topk = load_retrieved_turns(retrieval_path)

    max_conversations = args.max_conversations
    if max_conversations is not None:
        if max_conversations <= 0:
            raise ValueError("--max-conversations must be a positive integer")

        conv_prefix_pattern = re.compile(r"^conv-(\d+)")

        def within_limit(question_id: str) -> bool:
            match = conv_prefix_pattern.match(question_id)
            if not match:
                return False
            return int(match.group(1)) <= max_conversations

        ground_truth = {
            qid: truth for qid, truth in ground_truth.items() if within_limit(qid)
        }
        retrieved = {
            qid: stages for qid, stages in retrieved.items() if within_limit(qid)
        }

    results = evaluate_recall(ground_truth, retrieved, cutoff_k=final_topk)
    summarise(results)


if __name__ == "__main__":  # pragma: no cover
    main()
