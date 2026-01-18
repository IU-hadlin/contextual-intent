import os
import asyncio
from typing import List, Dict, Any
from came_bench import Benchmark
from came_bench.proto import LanguageModelProvider, LanguageModelProviderConfig, OpenAIConfig, Question, Turn

# --- 1. Define Your Custom Retrieval Logic ---


def retrieve_context(question: Question, full_history: List[Turn]) -> List[str]:
    """
    Retrieve relevant context for a SINGLE question.

    Args:
        question: The Question object (has .content, .id, etc.).
        full_history: List of Turn objects representing the entire conversation.

    Returns:
        List of strings (memory snippets) or List of turn IDs to use as context.
    """
    # --- TODO: YOUR LOGIC HERE ---
    # 1. Analyze the question `question.content`
    # 2. Search `full_history` for relevant information
    # 3. Return relevant content

    return ["LIST OF MEMORY SNIPPETS OR LIST OF TURN IDs"]


# --- 2. Run Benchmark ---


async def main():
    # Initialize Benchmark (downloads data if needed)
    # Default token_limit is 4096 as stated in the paper. You can adjust it here if needed:
    # bench = Benchmark(token_limit=...)
    bench = Benchmark(data_dir="came_bench_data")

    # List available trajectories
    traj_ids = bench.list_trajectories()
    print(f"Found {len(traj_ids)} trajectories: {traj_ids}")

    # Configure LLM for Answer Generation & Evaluation
    # See doc/LLM_CONFIG.md for other providers (Azure, Anthropic, etc.)
    lm_gen_config = LanguageModelProviderConfig(
        provider=LanguageModelProvider.LANGUAGE_MODEL_PROVIDER_OPENAI,
        model_name="gpt-5-mini",
        temperature=1.0,
        max_tokens=20000,
        openai_config=OpenAIConfig(
            api_key=os.environ.get("OPENAI_API_KEY", "")
        )
    )
    lm_jud_config = LanguageModelProviderConfig(
        provider=LanguageModelProvider.LANGUAGE_MODEL_PROVIDER_OPENAI,
        model_name="gpt-4.1-mini",
        temperature=1.0,
        max_tokens=1024,
        top_p=0.9,
        openai_config=OpenAIConfig(
            api_key=os.environ.get("OPENAI_API_KEY", "")
        )
    )

    # Check if API key is set
    if not lm_gen_config.openai_config.api_key:
        print("Warning: OPENAI_API_KEY not set. Answer generation may fail.")
    if not lm_jud_config.openai_config.api_key:
        print("Warning: OPENAI_API_KEY not set. Answer evaluation may fail.")

    # Process each trajectory
    for traj_id in traj_ids[:1]:
        print(f"\nProcessing Trajectory: {traj_id}")

        # Get Data
        meta = bench.get_trajectory_meta(traj_id)
        turns: List[Turn] = bench.get_turns(traj_id)
        questions: List[Question] = bench.get_questions(traj_id)

        print(f"  Task: {meta.get('task', 'unknown')}, Turns: {len(turns)}, Questions: {len(questions)}")

        # Run Custom Retrieval for each question
        retrieval_results = []
        for q in questions:
            # 1. Call your custom retrieval function
            context = retrieve_context(q, turns)

            # 2. Format result for benchmark
            retrieval_results.append({
                "question_id": q.id,
                "memory_snippets": context  # TODO: Or "turn_ids": context if returning IDs
            })

        # Run Generation & Evaluation
        print("  Running Answer Generation and Evaluation...")
        result = await bench.evaluate(
            traj_id=traj_id,
            retrieval_results=retrieval_results,
            lm_gen_config=lm_gen_config,
            lm_jud_config=lm_jud_config,
            output_dir=f"results/{traj_id}"
        )

        # Print Summary
        total = len(result.question_answer_evaluation_results)
        correct = sum(1 for r in result.question_answer_evaluation_results if r.is_correct_frq)

        # Calculate macro metrics
        precisions = [r.precision for r in result.question_answer_evaluation_results if r.success]
        recalls = [r.recall for r in result.question_answer_evaluation_results if r.success]
        f1s = [r.f1 for r in result.question_answer_evaluation_results if r.success]

        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0

        print(f"  Result: {correct}/{total} correct ({correct/total:.1%})")
        print(f"  Macro Metrics: Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, F1={avg_f1:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
