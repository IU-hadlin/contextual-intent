from came_bench.proto import AnswerGenerationConfig, AnswerEvaluationConfig, MCQAnswerGenerationResult, DatasetRetrievalResult
import json
from came_bench.utils.lm import init_lm
import dspy
from google.protobuf.json_format import ParseDict
from typing import List
from came_bench.utils.io import load_config
# from came_bench.utils.io import normalize_llm_output, is_correct # These functions were not found in utils.io
import argparse


class MCQAnswerGenerator(dspy.Signature):
    """
    Generate a comprehensive, well-supported answer to the provided query using the given set of retrieved debate turns.
    Each line in the retrieved turns is a turn index: "turn_<id> | <role> side argues: ...".
    If the question says "(Select all that apply)", output a python list-formatted string of letters among: a-z that are the answers to the question.
    Otherwise, output a python list-formatted string of one single letter choice among: a-z that is the answer to the question.
    """
    retrieved_turns: List[str] = dspy.InputField(description="A list of debate turn contents relevant to the query.")
    question: str = dspy.InputField(description="The question to be answered")
    choices: str = dspy.InputField(description="Formatted options (a-f), one per line")
    output: str = dspy.OutputField(description="A python list-formatted string of letter(s) among: a-z")


class FreeformAnswerGenerator(dspy.Signature):
    """
    Generate a comprehensive, well-supported answer to the provided query using the given set of retrieved debate turns.
    """
    retrieved_turns: List[str] = dspy.InputField(description="A list of debate turn contents relevant to the query.")
    question: str = dspy.InputField(description="The question to be answered")
    output: str = dspy.OutputField(description="A string answer to the question")


def generate_answer(answer_generation_config: AnswerGenerationConfig) -> List[MCQAnswerGenerationResult]:
    """
    Generate answers for all questions in the retrieval result using the specified config.
    Returns a list of MCQAnswerGenerationResult objects.
    """
    results = []
    lm = init_lm(answer_generation_config.language_model_provider_config)
    with dspy.context(lm=lm):
        dataset_name = answer_generation_config.dataset_name
        if dataset_name == "debate":
            answer_generator = dspy.Predict(MCQAnswerGenerator)
        elif dataset_name in ("locomo10", "longmemeval-m"):
            answer_generator = dspy.Predict(FreeformAnswerGenerator)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

        # Load the retrieval result
        with open(answer_generation_config.retrieval_result_path, "r") as f:
            retrieval_result_dict = json.load(f)
        retrieval_result = DatasetRetrievalResult()
        ParseDict(retrieval_result_dict, retrieval_result)

        for question_retrieval_result in retrieval_result.question_retrieval_results:
            question = question_retrieval_result.question
            result = MCQAnswerGenerationResult()
            result.question_id = question.id
            result.question_type = question.type
            result.question_text = question.content
            turns = question_retrieval_result.turn_ids
            ground_truth_answer = question.answer
            if ', ' in ground_truth_answer:
                ground_truth_answer = ground_truth_answer.split(', ')
            else:
                ground_truth_answer = [ground_truth_answer]
            result.ground_truth_answer.extend(ground_truth_answer)

            output = answer_generator(
                retrieved_turns=turns,
                question=question.content
            )
            result.LLM_answer = output.output

            results.append(result)
    return results


def evaluate_answer(answer_evaluation_config: AnswerEvaluationConfig):
    """
    Placeholder for answer evaluation logic.
    """
    pass


def main(args):
    """
    Main entry point: loads config, runs answer generation.
    """
    answer_generation_config = AnswerGenerationConfig()
    ParseDict(load_config(args.config), answer_generation_config)
    generate_answer(answer_generation_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to config JSON")
    args = parser.parse_args()
    main(args)
