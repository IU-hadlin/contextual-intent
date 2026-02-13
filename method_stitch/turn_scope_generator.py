"""CLI tool for predicting and assigning context scopes to dialogue turns.

This module provides a standalone command-line interface for generating turn scope
assignments, which determine which segments of context are relevant for each turn
in a dialogue. The tool:

1. Loads dialogue turns from a specified dataset
2. Uses ContextNoteMaintainer to predict appropriate context scopes for each turn
3. Persists the scope assignments to a JSON file for later use in context reduction

Turn scopes help identify which historical segments should be considered when
processing a given turn, enabling more efficient context management in long
conversations. This tool can be run independently to pre-compute scope assignments
before running the full context reduction pipeline.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from came_bench.utils.io import load_turns

from .segment_level_note_maintainer import (
    ContextNoteMaintainer,
    load_generation_config,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict context scopes for dialogue turns and persist the assignments.",
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
        help="Optional scope assignment output path; overrides config value.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the scope assignment file if it already exists.",
    )
    parser.add_argument(
        "--group-consecutive-turns",
        dest="group_consecutive_turns",
        action="store_true",
        help="Group consecutive turns by role before processing. If not specified, each turn is processed individually (default).",
    )

    parser.set_defaults(group_consecutive_turns=False)
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
        _notes_output_path,
        scope_history_window,
        prior_notes_limit,
        scope_output_path,
    ) = load_generation_config(
        args.config,
        output_override=None,
        scope_output_override=args.output,
    )

    if scope_output_path is None:
        raise ValueError(
            "Scope assignment output path is not configured. Provide one in the config or via --output."
        )

    if scope_output_path.exists() and not args.overwrite:
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

    turns = load_turns(turns_path)
    logger.info(
        "Loaded %d turns for dataset '%s' from %s",
        len(turns),
        dataset_name or "<unspecified>",
        turns_path,
    )

    # Generate scopes with incremental writing
    scope_assignments = maintainer.generate_turn_scopes(
        turns,
        output_path=scope_output_path,
        max_conversations=args.max_conversations,
    )

    logger.info(
        "Generated scope assignments for %d conversations and wrote them to %s",
        len(scope_assignments),
        scope_output_path,
    )


if __name__ == "__main__":
    main()
