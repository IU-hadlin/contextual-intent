"""
Run an answer generation strategy for a given dataset.

Usage:
python -m method_stitch.run_answer_generator_strategy -c {path_to_config_json} -r {number_of_retries}
"""

import argparse
import logging
import os
import asyncio
from google.protobuf.json_format import ParseDict

from came_bench.proto import DatasetAnswerGenerationRequest
from came_bench.utils.io import load_config
from came_bench.pipeline.generation.direct import DirectAnswerGeneration
import warnings

# Suppress leaked semaphore warning likely caused by os._exit(0)
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _formatter = logging.Formatter("%(levelname)s:%(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.propagate = False


async def main(args):
    # Build retrieval request from resolved JSON
    answer_generation_request = DatasetAnswerGenerationRequest()
    ParseDict(load_config(args.config), answer_generation_request)

    # check output directory
    output_dir = answer_generation_request.answer_generation_config.output_dir
    strategy_name = answer_generation_request.answer_generation_config.answer_generation_strategy
    output_file = os.path.join(output_dir, f"{strategy_name}.json")
    
    if os.path.exists(output_file):
        logger.info(f"Output file {output_file} already exists. Skipping execution.")
        user_input = input("Do you want to override the output file? (y/N): ")
        if user_input.lower() != "y":
            return
        
    # Directly use DirectAnswerGeneration as requested
    answer_generator = DirectAnswerGeneration(answer_generation_request)

    # Initialize resources and execute retrieval concurrently with retries
    await answer_generator.init()
    await answer_generator.async_execute_answer_generation(args.retries)

    # print the cost
    cost_entries = answer_generator.get_cost()
    for cost_entry in cost_entries:
        logger.info(f"{cost_entry.type}: {cost_entry.description} - ${cost_entry.cost:.4f}")
    
    # Force exit to prevent hanging on open async resources
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to answer generation config JSON")
    parser.add_argument("-r", "--retries", type=int, default=3, help="Number of retries")
    args = parser.parse_args()
    asyncio.run(main(args))
