"""CLI entry point for generating segment-level notes from precomputed scopes."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from came_bench.utils.io import load_turns

from .segment_level_note_maintainer import (
    ContextNoteMaintainer,
    load_generation_config,
    load_scope_assignments,
    write_scope_assignments,
)

logger = logging.getLogger(__name__)


def _determine_scope_output_path(
    *,
    explicit_scope_output: Optional[str],
    default_scope_output: Optional[Path],
    used_scope_input: bool,
) -> Optional[Path]:
    if explicit_scope_output is not None:
        return Path(explicit_scope_output)
    if not used_scope_input:
        return default_scope_output
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate segment-level notes, optionally consuming precomputed scope assignments.",
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
        help="Optional segment note output path; overrides config value.",
    )
    parser.add_argument(
        "--scope-input",
        help="Path to scope assignments produced by turn_scope_generator.py.",
    )
    parser.add_argument(
        "--scope-output",
        help="Optional path to write scope assignments used during note generation.",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        help="Limit processing to the first N conversations.",
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
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs when writing files.",
    )

    args = parser.parse_args()

    (
        dataset_name,
        turns_path,
        lm_config,
        notes_output_path,
        scope_history_window,
        prior_notes_limit,
        default_scope_output_path,
    ) = load_generation_config(
        args.config,
        output_override=args.output,
    )

    if notes_output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Segment note file {notes_output_path} already exists. Re-run with --overwrite to replace it."
        )

    scope_input_path: Optional[Path]
    if args.scope_input:
        scope_input_path = Path(args.scope_input)
    else:
        scope_input_path = default_scope_output_path

    scope_assignments = None
    used_scope_input = False
    if scope_input_path is not None:
        if not scope_input_path.exists():
            raise FileNotFoundError(
                f"Scope assignment file {scope_input_path} does not exist. Run turn_scope_generator.py first."
            )
        scope_assignments = load_scope_assignments(scope_input_path)
        used_scope_input = True

    scope_output_path = _determine_scope_output_path(
        explicit_scope_output=args.scope_output,
        default_scope_output=default_scope_output_path,
        used_scope_input=used_scope_input,
    )

    if scope_output_path and scope_output_path.exists() and not args.overwrite and not used_scope_input:
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

    records, generated_scope_assignments = maintainer.generate_segment_level_notes(
        turns,
        jsonl_path=notes_output_path,
        scope_assignments=scope_assignments,
        max_conversations=args.max_conversations,
    )
    logger.info("Generated %d context notes and wrote them to %s", len(records), notes_output_path)

    if scope_output_path is not None:
        if scope_output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Scope assignment file {scope_output_path} already exists. Re-run with --overwrite to replace it."
            )
        write_scope_assignments(generated_scope_assignments, scope_output_path)


if __name__ == "__main__":
    main()
