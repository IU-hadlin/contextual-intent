# generate proto 
python3 scripts/generate_proto_universal.py 

# Encode the turns for semantic retrieval
python3 -m src.dataset_process.encode_turns \
  -c path/to/encode_config.json

# Generate dataset description and functional-type seeds
python3 -m method_stitch.dataset_description \
  -c path/to/dataset_description_config.json

# Turn scope generation
python3 -m method_stitch.turn_scope_generator \
  --config path/to/segment_level_note_maintainer_config.json \
  --overwrite

# Segment-level note generation (consumes scope assignments)
python3 -m method_stitch.segment_note_generator \
  --config path/to/segment_level_note_maintainer_config.json \
  --output path/to/segment_level_notes.jsonl \
  --overwrite

# Event type label discovery and assignment
python3 -m method_stitch.event_type_labeler \
  -c path/to/event_type_labeler_config.json \
  --overwrite

# Turn-level note generation with event filtering
python3 -m method_stitch.turn_level_note_generator \
  -c path/to/turn_level_note_generator_config.json \
  --overwrite

###############################################################################
# LABEL-BASED RETRIEVAL PIPELINE 
###############################################################################

# Step 1: Label-based context retrieval
python3 -m method_stitch.label_based_context_retrieval \
  --config path/to/label_based_context_retrieval_config.json \
  --overwrite

# Step 2: Evaluate recall
python3 -m method_stitch.recall_evaluator \
  --path_to_question path/to/questions.jsonl \
  --path_to_retrieval_result path/to/label_based_retrieval.json

# Note: This transformation format is optimized for CAME-Bench. 
# If running other benchmarks, please adapt this step accordingly.
# Step 3: Transform for answer generation
python3 -m method_stitch.transform_retrieval_output \
  --config path/to/transform_retrieval_output_config.json

###############################################################################
# ANSWER GENERATION & EVALUATION
###############################################################################

# Note: If you are running CAME-Bench, use the provided method_stitch scripts below.
# For other benchmarks, please refer to their specific generation and evaluation guidelines.

# Feed the new config to the answer generator:
# python3 -m method_stitch.run_answer_generator -c path/to/answer_gen.json

# For debate, run the evaluator with:
# python3 -m method_stitch.run_answer_evaluator -c path/to/answer_eval.json
