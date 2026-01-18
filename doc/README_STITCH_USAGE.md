# Adapting STITCH to Any Benchmark

This guide explains how to run the STITCH memory system on your own dataset or a new benchmark.

To use STITCH, you simply need to convert your raw data into our **Standard Input Format**. Once converted, the entire pipeline (intent tracking, retrieval, answer generation) runs out of the box.

## 1. Prerequisites

Before starting, ensure you have generated the necessary Protocol Buffer definitions (which define the data schema):

```bash
python3 generate_proto_universal.py
```

*If you are unsure what this does, see [PROTO_SETUP.md](PROTO_SETUP.md).*

## 2. Prepare Your Data

You need to create two JSONL files in your data directory: `turns.jsonl` and `questions.jsonl`.

You can refer to `method_stitch/dataset_process` for example preprocessing scripts (e.g., `longmemeval_preprocess.py`) that map raw logs to this schema.

### Standard Format: `turns.jsonl`
Each line is a JSON object mapping to the `Turn` message in `proto/project_dataset_uniform.proto`.

```json
{
  "id": "turn_1",
  "role": "user",  // or "agent"
  "content": "I'm looking for a hotel in Tokyo.",
  "partition": ["conv_1"],  // Used to group turns (e.g., conversation ID)
  "timestamp_mapping": {    // Optional: context-specific timestamps
    "conv_1": "2023-10-01T10:00:00Z"
  }
}
```

### Standard Format: `questions.jsonl`
Each line is a JSON object mapping to the `Question` message in `proto/project_dataset_uniform.proto`.

```json
{
  "id": "q_1",
  "content": "What was the user's budget for the hotel?",
  "type": "FREE_FORM", // Question category
  "answer_type": "ANSWER_TYPE_FREEFORM", // Enum value (see proto)
  "answer": {
    "free_form_answer": "Under $200 per night"
  },
  "question_turn_ids": ["turn_5"],  // The turn where the question is asked
  "answer_turn_ids": ["turn_2", "turn_3"],  // Evidence turns (for recall eval)
}
```

### Tips for Conversion
- **Ensure unique IDs**: Every turn must have a unique `id`.
- **Partitioning**: Use the `partition` field (list of strings) to group turns. The retriever uses this to know which turns belong to which conversation/session.
- **Chronological Order**: Ensure `turns.jsonl` is sorted by time/sequence within each partition.

## 3. Configure the Pipeline

Once your data is in `turns.jsonl` and `questions.jsonl`:

1. **Copy Sample Configs**: Copy the files from `sample_config_files/` to your local config folder.
2. **Update Paths**: Edit the JSON configs to point to your new `data/` directory (where you saved `turns.jsonl` and `questions.jsonl`).

## 4. Run STITCH

We provide a master script to run the full pipeline end-to-end.

```bash
# Verify configs are updated, then run:
bash scripts/sample_run.sh
```

**Note:** If you are using STITCH to run CAME-Bench, we recommend using the provided answer generator and evaluator scripts. However, for other benchmarks or custom datasets, please follow their specific answer generation and evaluation procedures and prompts.


This script will sequentially execute:
1. **Proto Generation**: Ensures message types are ready.
2. **Encoding**: Embeds your `turns.jsonl` for retrieval.
3. **Intent Tracking (STITCH Core)**:
   - Describes the dataset structure.
   - Generates segment-level summaries.
   - Labels event types and intents.
   - Builds turn-level structured notes.
4. **Retrieval**: Retrieves context for your questions using the structured notes.
5. **Answer Generation**: Uses the retrieved context to answer the questions.
6. **Evaluation**: Compares generated answers against your ground truth.

### Customizing the Run
If you only need specific parts (e.g., only retrieval, no answer generation), you can comment out sections in `scripts/sample_run.sh`.
