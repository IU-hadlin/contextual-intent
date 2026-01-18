from __future__ import annotations
from tqdm import tqdm
from google.protobuf.json_format import ParseDict, MessageToDict
import dspy
from came_bench.proto import Question, Turn, Dataset, DatasetRetrievalResult

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Union, Literal
from typing import List, Any, Optional
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DSPY_CACHE = Path(__file__).resolve().parents[1] / "data" / "dspy_cache"
cache_env = os.environ.get("DSPY_CACHEDIR")
if not cache_env:
    cache_env = str(DEFAULT_DSPY_CACHE)
    os.environ["DSPY_CACHEDIR"] = cache_env

try:
    Path(cache_env).expanduser().mkdir(parents=True, exist_ok=True)
except OSError:
    pass


EnvMapping = Mapping[str, str]
JSONLike = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


def ensure_subject_tag_objects(node: JSONLike) -> None:
    """Ensure any tag arrays contain dict entries with a subject field."""

    if isinstance(node, dict):
        tags = node.get("tags")
        if isinstance(tags, list):
            normalized: List[JSONLike] = []
            for item in tags:
                if isinstance(item, str):
                    normalized.append({"subject": item})
                else:
                    normalized.append(item)
            node["tags"] = normalized
        for value in node.values():
            ensure_subject_tag_objects(value)
    elif isinstance(node, list):
        for element in node:
            ensure_subject_tag_objects(element)


class MissingEnvironmentVariablesError(ValueError):
    """Raised when one or more ${VAR_NAME} placeholders cannot be resolved."""

    def __init__(self, missing: List[tuple[str, str]]):
        # missing: list of (var_name, json_path)
        self.missing = missing
        details = ", ".join(
            f"{name} at {path if path else '<root>'}" for name, path in missing
        )
        super().__init__(
            "Missing environment variables for config placeholders: " + details
        )


_PLACEHOLDER_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _substitute_string_placeholders(
    value: str, env: EnvMapping, json_path: str, missing: List[tuple[str, str]]
) -> str:
    """Replace ${VAR} placeholders in a string, collecting any missing vars."""

    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        if var_name in env and env[var_name] is not None:
            return env[var_name]
        # Record missing and keep the original placeholder to allow aggregated errors
        missing.append((var_name, json_path))
        return match.group(0)

    return _PLACEHOLDER_PATTERN.sub(replacer, value)


def _substitute_placeholders_recursive(
    node: JSONLike,
    env: EnvMapping,
    json_path: str,
    missing: List[tuple[str, str]],
) -> JSONLike:
    if isinstance(node, dict):
        result: MutableMapping[str, Any] = {}
        for key, val in node.items():
            child_path = f"{json_path}.{key}" if json_path else str(key)
            result[key] = _substitute_placeholders_recursive(val, env, child_path, missing)
        return dict(result)
    if isinstance(node, list):
        return [
            _substitute_placeholders_recursive(val, env, f"{json_path}[{idx}]", missing)
            for idx, val in enumerate(node)
        ]
    if isinstance(node, str):
        return _substitute_string_placeholders(node, env, json_path, missing)
    # primitives
    return node


def load_config(config_path: Union[str, Path], dotenv_path: Union[str, Path, None] = None) -> Dict[str, Any]:
    """
    Load a JSON config, substitute ${VAR} placeholders from environment/.env, and
    return the resolved dictionary. If any placeholders cannot be resolved, raise
    MissingEnvironmentVariablesError listing the variables and their JSON paths.

    - If `dotenv_path` is provided and python-dotenv is available, that file is loaded first.
    - If `dotenv_path` is None, we attempt to load a default .env from the current working
      directory and its parents (python-dotenv behavior). If python-dotenv is not available,
      only os.environ is used.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load .env if available
    if load_dotenv is not None:
        if dotenv_path is not None:
            load_dotenv(dotenv_path=str(dotenv_path))
        else:
            load_dotenv()

    with config_path.open("r", encoding="utf-8") as f:
        raw: JSONLike = json.load(f)

    missing: List[tuple[str, str]] = []
    resolved = _substitute_placeholders_recursive(raw, os.environ, json_path="", missing=missing)

    # If any placeholder could not be resolved, raise a helpful error
    # Deduplicate by (name, path) while preserving order
    if missing:
        seen: set[tuple[str, str]] = set()
        unique_missing: List[tuple[str, str]] = []
        for item in missing:
            if item not in seen:
                unique_missing.append(item)
                seen.add(item)
        raise MissingEnvironmentVariablesError(unique_missing)

    if not isinstance(resolved, dict):
        raise ValueError("Top-level JSON must be an object (dict)")

    return resolved


def load_questions(questions_jsonl_path: str) -> List[Question]:
    assert os.path.exists(questions_jsonl_path), f"Questions JSONL file not found: {questions_jsonl_path}"
    total_num_lines = sum(1 for _ in open(questions_jsonl_path, "r"))
    questions: List[Question] = []
    with open(questions_jsonl_path, "r") as f:
        for line in tqdm(f, total=total_num_lines, desc="Loading questions from JSONL"):
            question = Question()
            ParseDict(json.loads(line), question)
            questions.append(question)
    return questions


def load_turns(turns_jsonl_path: str) -> List[Turn]:
    assert os.path.exists(turns_jsonl_path), f"Turns JSONL file not found: {turns_jsonl_path}"
    total_num_lines = sum(1 for _ in open(turns_jsonl_path, "r"))
    turns: List[Turn] = []
    with open(turns_jsonl_path, "r") as f:
        for line in tqdm(f, total=total_num_lines, desc="Loading turns from JSONL"):
            turn = Turn()
            ParseDict(json.loads(line), turn)
            turns.append(turn)
    return turns


def load_retrieval_result(retrieval_result_jsonl_path: str) -> DatasetRetrievalResult:
    assert os.path.exists(
        retrieval_result_jsonl_path), f"Retrieval result JSONL file not found: {retrieval_result_jsonl_path}"
    dataset_retrieval_result = DatasetRetrievalResult()
    with open(retrieval_result_jsonl_path, "r", encoding="utf-8") as infile:
        payload: JSONLike = json.load(infile)
    ensure_subject_tag_objects(payload)
    ParseDict(payload, dataset_retrieval_result)
    return dataset_retrieval_result


def write_dataset(dataset: Dataset, output_dir: str) -> None:
    """Write dataset into three JSONL files (questions, turns, meta) under output_dir.

    - questions.jsonl: each line is a JSON object of Question
    - turns.jsonl: each line is a JSON object of Turn
    - meta.jsonl: a single-line JSON object with dataset metadata and file paths
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    questions_path = out_dir / f"{dataset.name}_questions.jsonl"
    turns_path = out_dir / f"{dataset.name}_turns.jsonl"
    meta_path = out_dir / f"{dataset.name}_meta.json"

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

    # Write meta (single JSONL line)
    meta_obj = {
        "dataset_name": dataset.name,
        "questions_file": str(questions_path),
        "turns_file": str(turns_path),
        "num_questions": len(dataset.questions),
        "num_turns": len(dataset.turns),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_obj, f, ensure_ascii=False)


def get_lm_cost(lm):
    total_cost = 0.0
    for entry in getattr(lm, "history", []):
        if isinstance(entry, dict) and "cost" in entry and isinstance(entry["cost"], float):
            total_cost += entry["cost"]
    return total_cost


def parse_turn_reasoning_metadata(reasoning: Optional[str]) -> Dict[str, str]:
    """Extract key/value metadata pairs from a turn retrieval reasoning string."""

    if not reasoning:
        return {}

    metadata: Dict[str, str] = {}
    for segment in reasoning.split(" | "):
        if "=" not in segment:
            continue
        key, value = segment.split("=", 1)
        key = key.strip()
        if not key:
            continue
        metadata[key] = value.strip()
    return metadata


def format_retrieved_turn(dataset_name: str, retrieved_turn: Turn, question_id: str) -> str:
    if dataset_name == "debate" or dataset_name == "travel_planning":
        formatted_turn = f"{retrieved_turn.role}: {retrieved_turn.content}"
    elif hasattr(retrieved_turn, "timestamp_mapping") and f"question:{question_id}" in retrieved_turn.timestamp_mapping:
        formatted_turn = f"{retrieved_turn.role}: {retrieved_turn.content} [Turn Timestamp: {retrieved_turn.timestamp_mapping[question_id]}]"
    else:
        formatted_turn = f"{retrieved_turn.role}: {retrieved_turn.content}"
    return formatted_turn


def format_question_choices(question: Question) -> str:
    choices = [f"({choice.id}) {choice.content}" for choice in question.choices]
    choices.append(f"(idk) I don't know")
    return "\n".join(choices)
