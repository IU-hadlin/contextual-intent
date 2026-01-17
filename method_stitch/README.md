# Context reduction pipeline

This directory contains scripts for building structured notes, labels, and retrieval outputs used in context-reduction experiments. Each step is driven by a JSON config that defines dataset paths, model settings, and output locations.

## Configuration files

Sample configuration files are provided in `sample_config_files/` directory. These files contain example configurations with all file paths filled in as templates. **You should copy these files and update the paths to match your own data locations and settings.** The sample configs demonstrate the expected format and structure for each pipeline step.

## Script overview
- `dataset_description.py`: streaming dataset description plus functional-type seed generation.
- `turn_scope_generator.py`: predicts context scopes for turns.
- `segment_note_generator.py`: generates segment-level notes (can consume scope assignments).
- `segment_level_note_maintainer.py`: alternative segment-level summarization workflow with the same config schema.
- `event_type_labeler.py`: discovers and assigns event type labels.
- `turn_level_note_generator.py`: builds structured notes per turn using scopes, events, and segment notes.
- `label_based_context_retrieval.py`: label-driven retrieval (LLM selects filters, then embeddings rank).
- `recall_evaluator.py`: evaluates recall against question ground truth.
- `transform_retrieval_output.py`: converts retrieval output into the format consumed by answer generation.

## How to Run

Instead of running each step manually, we provide a master script that executes the full pipeline end-to-end.

**Prerequisite**: Ensure you have generated the Protocol Buffers. [See Setup Guide](../doc/PROTO_SETUP.md).

Run (from this directory):
```bash
bash ../scripts/sample_run.sh
```

Ensure you have updated the paths in `scripts/sample_run.sh` and your config files to point to your actual data locations.

## Notes
- Most scripts support `--max-conversations` for quick test runs.
