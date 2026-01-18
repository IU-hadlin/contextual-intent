"""Run an answer evaluation strategy for a given dataset."""

from __future__ import annotations

import argparse
import logging
import os
import asyncio
from typing import Any, Dict

from google.protobuf.json_format import MessageToDict, ParseDict

from came_bench.proto import DatasetAnswerEvaluationRequest
from came_bench.utils.io import load_config, get_lm_cost
from came_bench.pipeline.evaluation.llm_eval import LLMAnswerEvaluator


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _formatter = logging.Formatter("%(levelname)s:%(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.propagate = False


async def main(args: argparse.Namespace) -> None:
    answer_evaluation_request = DatasetAnswerEvaluationRequest()

    ParseDict(load_config(args.config), answer_evaluation_request)

    generated_answer_file_name = os.path.basename(answer_evaluation_request.answer_evaluator_config.generated_answer_jsonl_path).split(".")[0]
    output_dir = answer_evaluation_request.answer_evaluator_config.output_dir
    strategy_name = answer_evaluation_request.answer_evaluator_config.answer_evaluation_strategy
    
    # check output directory
    output_filename = f"{strategy_name}-{generated_answer_file_name}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    if os.path.exists(output_path):
        logger.info(f"Output file {output_path} already exists. Skipping execution.")
        user_input = input("Do you want to override the output file? (y/N): ")
        if user_input.lower() != "y":
            return

    # Directly use LLMAnswerEvaluator as requested
    answer_evaluator = LLMAnswerEvaluator(answer_evaluation_request)

    await answer_evaluator.init()
    await answer_evaluator.async_execute_answer_evaluation(args.retries)

    print(f"Total LM Cost: ${get_lm_cost(answer_evaluator.answer_evaluator_lm):.4f}")
    
    # Force exit to prevent hanging on open async resources
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to evaluator config JSON")
    parser.add_argument("-r", "--retries", type=int, default=3, help="Number of retries")
    args = parser.parse_args()
    asyncio.run(main(args))
