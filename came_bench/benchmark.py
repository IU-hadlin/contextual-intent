import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import asyncio
from google.protobuf.json_format import MessageToDict

from came_bench.data_process import load_dataset
from came_bench.utils.io import load_questions, load_turns
from came_bench.proto import (
    Question,
    Turn,
    QuestionRetrievalResult,
    DatasetRetrievalResult,
    LanguageModelProviderConfig,
    LanguageModelProvider,
    AnswerGenerationConfig,
    DatasetAnswerGenerationRequest,
    AnswerGenerationStrategyType,
    DirectAnswerGenerationConfig,
    AnswerEvaluatorConfig,
    DatasetAnswerEvaluationRequest,
    AnswerEvaluationStrategyType,
    DirectAnswerEvaluationConfig,
    DatasetAnswerEvaluationResult
)
from came_bench.pipeline.generation.direct import DirectAnswerGeneration
from came_bench.pipeline.evaluation.llm_eval import LLMAnswerEvaluator


class Benchmark:
    def __init__(self, data_dir: Optional[str] = None, token_limit: int = 4096):
        """
        Initialize the CAME-Bench interface.

        Args:
            data_dir: Optional path to the data directory. If None, uses default.
                      Will download/decode data if missing.
            token_limit: Maximum number of tokens allowed for retrieved context (default: 4096).
        """
        self.data_path = load_dataset(data_dir)
        self.meta_path = self.data_path / "benchmark_meta.json"
        self.token_limit = token_limit

        if not self.meta_path.exists():
            raise FileNotFoundError(f"Benchmark meta file not found at {self.meta_path}")

        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta_data = json.load(f)

        self.traj_map = {item["id"]: item for item in self.meta_data}

    def list_trajectories(self) -> List[str]:
        """List all trajectory IDs in the benchmark."""
        return [item["id"] for item in self.meta_data]

    def get_trajectory_meta(self, traj_id: str) -> Dict[str, Any]:
        """Get metadata for a specific trajectory."""
        if traj_id not in self.traj_map:
            raise ValueError(f"Trajectory {traj_id} not found.")
        return self.traj_map[traj_id]

    def get_turns(self, traj_id: str) -> List[Turn]:
        """Get the list of turns for a specific trajectory."""
        if traj_id not in self.traj_map:
            raise ValueError(f"Trajectory {traj_id} not found.")

        turns_path = self.data_path / traj_id / "turns.jsonl"
        return load_turns(str(turns_path))

    def get_questions(self, traj_id: str) -> List[Question]:
        """Get the list of questions for a specific trajectory."""
        if traj_id not in self.traj_map:
            raise ValueError(f"Trajectory {traj_id} not found.")

        questions_path = self.data_path / traj_id / "questions.jsonl"
        return load_questions(str(questions_path))

    async def evaluate(
        self,
        traj_id: str,
        retrieval_results: List[Union[QuestionRetrievalResult, Dict[str, Any]]],
        lm_gen_config: LanguageModelProviderConfig,
        lm_jud_config: LanguageModelProviderConfig,
        output_dir: str = "results",
        questions_map: Optional[Dict[str, Question]] = None
    ) -> DatasetAnswerEvaluationResult:
        """
        Run answer generation and evaluation for a trajectory given retrieval results.

        Args:
            traj_id: The trajectory ID.
            retrieval_results: List of retrieval results. Can be QuestionRetrievalResult objects
                               or dicts with keys (question_id, turn_ids OR memory_snippets).
            lm_gen_config: Configuration for the LLM to use for generation.
            lm_jud_config: Configuration for the LLM to use for judgment.
            output_dir: Directory to save results.
            questions_map: Optional mapping from question ID to Question object.
                          If provided, avoids reloading questions from disk.

        Returns:
            DatasetAnswerEvaluationResult: The final evaluation results.
        """
        if traj_id not in self.traj_map:
            raise ValueError(f"Trajectory {traj_id} not found.")

        task = self.traj_map[traj_id].get("task", "unknown")

        if questions_map is None:
            questions = self.get_questions(traj_id)
            questions_map = {q.id: q for q in questions}

        # Prepare Retrieval Result
        normalized_results = []
        for res in retrieval_results:
            if isinstance(res, QuestionRetrievalResult):
                normalized_results.append(res)
            elif isinstance(res, dict):
                # Helper to create from dict
                q_id = res.get("question_id") or res.get("id")
                if not q_id or q_id not in questions_map:
                    print(f"Warning: Question ID {q_id} not found in trajectory {traj_id}. Skipping.")
                    continue

                q_res = QuestionRetrievalResult()
                q_res.question.CopyFrom(questions_map[q_id])
                q_res.success = True

                if "turn_ids" in res:
                    q_res.turn_ids.extend(res["turn_ids"])
                if "memory_snippets" in res:
                    q_res.memory_snippets.extend(res["memory_snippets"])

                normalized_results.append(q_res)
            else:
                raise TypeError(f"Unsupported retrieval result type: {type(res)}")

        dataset_retrieval_result = DatasetRetrievalResult()
        dataset_retrieval_result.retrieval_strategy = "custom"
        dataset_retrieval_result.dataset_name = task
        dataset_retrieval_result.question_retrieval_results.extend(normalized_results)

        # Paths
        traj_dir_path = self.data_path / traj_id
        questions_path = str(traj_dir_path / "questions.jsonl")
        turns_path = str(traj_dir_path / "turns.jsonl")

        os.makedirs(output_dir, exist_ok=True)
        retrieval_path = os.path.join(output_dir, f"{traj_id}_retrieval.json")
        with open(retrieval_path, "w") as f:
            json.dump(MessageToDict(dataset_retrieval_result, preserving_proto_field_name=True), f, indent=2)

        # 1. Answer Generation
        gen_config = AnswerGenerationConfig(
            questions_jsonl_path=questions_path,
            turns_jsonl_path=turns_path,
            retrieval_result_jsonl_path=retrieval_path,
            answer_generation_strategy="direct_answer_generation",
            concurrent=5,
            dataset_name=task,
            output_dir=output_dir,
            adapter_language_model_provider_config=lm_gen_config,
            max_tokens_for_retrieved_results=self.token_limit
        )

        gen_request = DatasetAnswerGenerationRequest(
            answer_generation_config=gen_config,
            answer_generation_strategy_type=AnswerGenerationStrategyType.ANSWER_GENERATION_STRATEGY_TYPE_DIRECT_ANSWER_GENERATION,
            direct_answer_generation_config=DirectAnswerGenerationConfig(
                language_model_provider_config=lm_gen_config
            )
        )

        generator = DirectAnswerGeneration(gen_request)
        # Note: We don't have a return value from async_execute... it writes to file.
        # But we can look at internal state or reload.
        # Looking at base.py, it populates output_question_answer_generation_results but returns None.
        # It writes to {strategy}.json in output_dir.
        await generator.async_execute_answer_generation()

        generated_answers_path = os.path.join(output_dir, "direct_answer_generation.json")

        # 2. Answer Evaluation
        eval_config = AnswerEvaluatorConfig(
            generated_answer_jsonl_path=generated_answers_path,
            questions_jsonl_path=questions_path,
            answer_evaluation_strategy="answer_evaluation_v1",
            concurrent=5,
            dataset_name=task,
            output_dir=output_dir,
            adapter_language_model_provider_config=lm_jud_config
        )

        eval_request = DatasetAnswerEvaluationRequest(
            answer_evaluator_config=eval_config,
            answer_evaluation_strategy_type=AnswerEvaluationStrategyType.ANSWER_EVALUATION_STRATEGY_TYPE_answer_evaluation_v1,
            answer_evaluation_v1_config=DirectAnswerEvaluationConfig(
                language_model_provider_config=lm_jud_config
            )
        )

        evaluator = LLMAnswerEvaluator(eval_request)
        await evaluator.async_execute_answer_evaluation()

        # Load final results to return
        eval_output_path = os.path.join(output_dir, "answer_evaluation_v1.json")  # Strategy name based
        if os.path.exists(eval_output_path):
            with open(eval_output_path, "r") as f:
                # We can parse it back to proto or just return dict
                # Let's return the proto for consistency if possible, or maybe easier for user if we return dict?
                # User asked for "return the evaluated result", likely the object or dict.
                # Let's return the object.
                from google.protobuf.json_format import Parse
                res = DatasetAnswerEvaluationResult()
                Parse(f.read(), res)
                return res

        return DatasetAnswerEvaluationResult()
