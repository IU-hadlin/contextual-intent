#!/usr/bin/env python3
"""
Streaming-style dataset description & functional-type seed generation.

New streaming behavior:

1. Use the FIRST full block of 50 raw turns (fewer if dataset <50)
   to generate:
       - initial dataset description
       - initial functional-type seeds

2. For EACH ADDITIONAL full block of 50 raw turns:
       - Add entire new block to turn pool (all previous turns + new block)
       - Sample from candidate pool to stay within token limit (200k):
         * Build candidate pool: ALL_PREVIOUS_TURNS + entire new block
         * Calculate avg tokens per turn
         * Determine safe number: 200k / avg_tokens_per_turn
         * If pool size <= safe number, use entire pool
         * Otherwise, randomly sample safe number of turns
       - Update dataset description using sampled pool:
             previous_description + sampled_pool
       - Update functional-type seeds by RECOMPUTING the entire seed list
         using the same sampled pool
         (NOT additive; LLM generates the whole new list each time)


This avoids seeing future turns and aligns with streaming constraints.
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import dspy
from google.protobuf.json_format import ParseDict
from tqdm import tqdm

from came_bench.proto import DatasetDescriptionConfig
from came_bench.utils.lm import init_lm
from came_bench.utils.io import get_lm_cost, load_config


BLOCK_SIZE = 50
SAMPLE_RATE = 0.2
# MAX_SEEDS = 15
MAX_CONTEXT_TOKENS = 200_000
_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            import tiktoken

            _TOKENIZER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TOKENIZER = None
    return _TOKENIZER


def truncate_text_to_token_limit(
    text: str,
    max_tokens: int,
    keep_tail: bool = True,
) -> Tuple[str, int, bool]:
    """
    Truncate text to at most max_tokens tokens.

    Returns (possibly-truncated-text, token_count_after_truncation, was_truncated).
    If no tokenizer is available, falls back to a 4 chars/token heuristic.
    """
    assert max_tokens > 0, "max_tokens must be greater than 0"

    tokenizer = _get_tokenizer()
    if tokenizer:
        token_ids = tokenizer.encode(text)
        if len(token_ids) <= max_tokens:
            return text, len(token_ids), False
        slice_tokens = token_ids[-max_tokens:] if keep_tail else token_ids[:max_tokens]
        return tokenizer.decode(slice_tokens), len(slice_tokens), True

    approx_tokens = max(1, len(text) // 4)
    if approx_tokens <= max_tokens:
        return text, approx_tokens, False

    approx_chars = max_tokens * 4
    truncated = text[-approx_chars:] if keep_tail else text[:approx_chars]
    return truncated, max_tokens, True


def enforce_token_limit(text: str, label: str) -> str:
    """Clamp text to MAX_CONTEXT_TOKENS and log when truncation happens."""
    truncated, token_count, was_truncated = truncate_text_to_token_limit(
        text, MAX_CONTEXT_TOKENS
    )
    if was_truncated:
        print(
            f"[token-guard] {label} truncated to ~{token_count} tokens "
            f"(limit {MAX_CONTEXT_TOKENS})."
        )
    return truncated


class DatasetDescriptionSignature(dspy.Signature):
    """You are given a sample of turns from a dataset and the dataset type. Derive a concise one-sentence description decribing the nature of the dataset, avoid including details."""
    dataset_type: str = dspy.InputField()
    sample_turns: str = dspy.InputField()
    description: str = dspy.OutputField()


class DatasetDescriptionUpdateSignature(dspy.Signature):
    """Update the description into another concise one-sentence description to reflect the new block of turns and the nature of the dataset."""
    dataset_type: str = dspy.InputField(description="The type of the dataset.")
    previous_description: str = dspy.InputField(description="The previous one-sentence description of the dataset.")
    new_block_turns: str = dspy.InputField(description="The new block of turns from the dataset.")
    updated_description: str = dspy.OutputField(description="The updated one-sentence description of the dataset.")


class FineGrainedFunctionalSeedSignature(dspy.Signature):
    """You are given (1) a dataset description and (2) a sample of turns. Your task is to derive a list of fine-grained functional type that describe how specific details, facts, references, options, or objects are being used to advance the task in this dataset. IMPORTANT: - You ARE generating a list of pragmatic, task-driven functional type names that that represent how fine-grained details participate in argumentation, reasoning, decision-making, planning, critique, or exploration. - Think in terms of what work each detail is doing for the speaker or agent. Method: 1. Understand the dataset's task type from the description. 2. From the sample turns, identify recurring uses of specific details: - What are these details achieving pragmatically? - How do they drive the conversation or task forward? 3. Abstract the observed uses into general *functional types*. - Names should be concise noun or noun-phrase labels (2–4 words). - Different names should distinguish from each other. - The number of functional type names is limited, so you should use the most general and neutral names that can cover meaningful details in the dataset. 4. Order the list from most representative to least representative.
    """
    dataset_description: str = dspy.InputField()
    sample_turns: str = dspy.InputField()
    functional_type_seeds: list[str] = dspy.OutputField()


def load_turns_jsonl(path: Path) -> List[dict]:
    turns = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                turns.append(json.loads(line))
    return turns


def concatenate_turn_content(turns: Sequence[dict]) -> str:
    parts = []
    for t in turns:
        content = t.get("content") or ""
        if content:
            parts.append(content.strip())
    return "\n".join(parts)


def count_tokens(text: str) -> int:
    """Count tokens in text, using tokenizer if available, else heuristic."""
    tokenizer = _get_tokenizer()
    if tokenizer:
        return len(tokenizer.encode(text))
    return max(1, len(text) // 4)


def sample_turns_within_token_limit(turns: Sequence[dict], max_tokens: int) -> List[dict]:
    """
    Sample turns from the pool to stay within token limit.

    Strategy:
    1. Calculate average tokens per turn
    2. Determine safe number of turns: max_tokens / avg_tokens_per_turn
    3. If pool size <= safe number, return entire pool
    4. Otherwise, randomly sample safe number of turns
    """
    if not turns:
        return []

    # Calculate total tokens and average
    total_tokens = 0
    for t in turns:
        content = t.get("content") or ""
        total_tokens += count_tokens(content)

    if total_tokens == 0:
        return list(turns)

    avg_tokens_per_turn = total_tokens / len(turns)
    safe_num_turns = max(1, int(max_tokens / avg_tokens_per_turn))

    # If pool fits within limit, return all
    if len(turns) <= safe_num_turns:
        return list(turns)

    # Otherwise, sample safe number of turns
    return random.sample(list(turns), safe_num_turns)


def sample_20pct(turns: Sequence[dict]) -> List[dict]:
    if not turns:
        return []
    sample_size = max(1, int(len(turns) * SAMPLE_RATE))
    return random.sample(turns, sample_size)


def load_job_config(path: Path) -> DatasetDescriptionConfig:
    cfg_data = load_config(path)
    raw_cfg = cfg_data.get("dataset_description_config")
    if not isinstance(raw_cfg, dict):
        raise ValueError("dataset_description_config section missing")
    cfg = DatasetDescriptionConfig()
    ParseDict(raw_cfg, cfg)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streaming-style block-based dataset description & functional seeds."
    )
    parser.add_argument("-c", "--config", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_job_config(args.config)

    turns_path = Path(cfg.turns_jsonl_path)
    if not turns_path.exists():
        raise FileNotFoundError(turns_path)

    turns = load_turns_jsonl(turns_path)
    num_turns = len(turns)
    if num_turns == 0:
        raise ValueError("Turns empty")

    dataset_type = cfg.dataset_type

    # Init LM + predictors
    lm = init_lm(cfg.language_model_provider_config)
    description_predictor = dspy.Predict(DatasetDescriptionSignature)
    description_update_predictor = dspy.Predict(DatasetDescriptionUpdateSignature)
    seed_predictor = dspy.Predict(FineGrainedFunctionalSeedSignature)

    with dspy.context(lm=lm):
        # Dedup guards to avoid reusing identical truncated prompts
        last_desc_input = None  # tuple(dataset_type, previous_description, sample_text)
        last_seed_input = None  # tuple(description, pool_text)

        # -------------------------
        # INITIAL BLOCK (first ≤50)
        # -------------------------
        init_end = min(BLOCK_SIZE, num_turns)
        initial_block = turns[:init_end]
        initial_text = concatenate_turn_content(initial_block)
        initial_text = enforce_token_limit(initial_text, "initial block turns")

        # First description
        desc_res = description_predictor(
            dataset_type=dataset_type,
            sample_turns=initial_text,
        )
        description = (desc_res.description or "").strip()
        if not description:
            description = f"A dataset of {dataset_type} conversations."

        # First functional seeds from initial block
        seeds_res = seed_predictor(
            dataset_description=description,
            sample_turns=initial_text,
        )
        functional_types = seeds_res.functional_type_seeds or []
        functional_types = [s.strip() for s in functional_types if s.strip()]
        # functional_types = functional_types[:MAX_SEEDS]
        last_seed_input = (description, initial_text)

        # Add initial block to pool
        turn_pool = list(initial_block)

        # -------------------------
        # SUBSEQUENT FULL BLOCKS
        # -------------------------
        num_blocks = (num_turns + BLOCK_SIZE - 1) // BLOCK_SIZE

        block_iter = range(1, num_blocks)
        iterator = (
            tqdm(block_iter, desc="Processing blocks", unit="block")
            if num_blocks > 1
            else block_iter
        )

        for b in iterator:
            start = b * BLOCK_SIZE
            end = min((b + 1) * BLOCK_SIZE, num_turns)
            block = turns[start:end]
            if len(block) < BLOCK_SIZE:
                break  # no update for incomplete block (streaming-consistent)

            # (1) Add entire new block to pool first
            turn_pool.extend(block)

            # (2) Sample from the entire new pool based on token limits
            sampled_pool = sample_turns_within_token_limit(turn_pool, MAX_CONTEXT_TOKENS)
            pool_text = concatenate_turn_content(sampled_pool)

            # (3) Update description using sampled pool (same approach as functional seeds)
            desc_input = (dataset_type, description, pool_text)
            if desc_input != last_desc_input:
                upd = description_update_predictor(
                    dataset_type=dataset_type,
                    previous_description=description,
                    new_block_turns=pool_text,
                )
                new_desc = (upd.updated_description or "").strip()
                if new_desc:
                    description = new_desc
                last_desc_input = desc_input
            else:
                print(f"[token-guard] Skipping description update for block {b} (input unchanged).")

            # (4) Update functional type seeds:
            #     FULL RECOMPUTATION using sampled pool (already within token limits)
            seed_input = (description, pool_text)
            if seed_input != last_seed_input:
                seed_res = seed_predictor(
                    dataset_description=description,
                    sample_turns=pool_text,
                )
                new_seeds = seed_res.functional_type_seeds or []
                new_seeds = [s.strip() for s in new_seeds if s.strip()]

                # functional_types = new_seeds[:MAX_SEEDS]
                functional_types = new_seeds
                last_seed_input = seed_input
            else:
                print(f"[token-guard] Skipping seed update for block {b} (input unchanged).")

    total_cost = get_lm_cost(lm)

    # -------------------------
    # SAVE OUTPUT
    # -------------------------
    out = Path(cfg.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        f.write(description + "\n")
        json.dump(functional_types, f, ensure_ascii=False)
        f.write("\n")

    print(f"Processed {num_turns} turns in blocks of 50.")
    print(f"Final description:\n{description}")
    print("\nFinal functional type seeds:")
    for s in functional_types:
        print(f"- {s}")
    print(f"\nEstimated LM cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
