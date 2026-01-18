# Evaluating Your Method on CAME-Bench

This guide explains how to benchmark your own memory or retrieval method using the CAME-Bench Python API.

## 1. Prerequisites

Ensure you have installed the project dependencies:

```bash
pip install -r requirements.txt
```

*(Optional) Troubleshooting Imports:*
If you encounter `ModuleNotFoundError` related to protobufs, run the fix script:
```bash
python scripts/fix_proto_imports.py
```

## 2. The Benchmark API

The `came_bench` library provides a high-level `Benchmark` class that handles:
- **Data Management**: Automatically downloads and decodes the dataset from Hugging Face.
- **Evaluation Pipeline**: Orchestrates answer generation (using your retrieved context) and LLM-based evaluation.

### How Evaluation Works

To ensure a fair comparison between different retrieval methods, CAME-Bench standardizes the answer generation and evaluation process:

1.  **Your Role (Retrieval)**: Your method selects the most relevant information (turn IDs or text snippets) from the history.
2.  **Standardized Generation**: The benchmark takes your retrieved context and feeds it into a fixed DSPy module with a standardized prompt to generate an answer.
    *   *Note:* In our paper, we use `gpt-5-mini` for this stage.
    *   **Constraint**: To ensure fair comparison, we enforce a shared retrieval budget. The inference context is capped at **4,096 tokens** by default. If your retrieved context exceeds this, it will be truncated.
        *   *You can adjust this limit in the `Benchmark` constructor if needed (see below), but 4096 is the standard for fair comparison.*
3.  **Standardized Evaluation**: A "Judge" LLM compares the generated answer against the ground truth to calculate accuracy.
    *   *Note:* In our paper, we use `gpt-4.1-mini` for this stage.

### Basic Workflow

1. **Initialize Benchmark**:
   ```python
   from came_bench import Benchmark
   # Default token limit is 4096. You can adjust it if necessary.
   bench = Benchmark(token_limit=4096) 
   ```

2. **Access Data**:
   Get turns (history) and questions for a specific trajectory:
   ```python
   traj_id = "traj-0" # Trajectory IDs are like 'traj-0', 'traj-1', etc.
   turns = bench.get_turns(traj_id)        # List of Turn objects
   questions = bench.get_questions(traj_id) # List of Question objects
   ```

3. **Implement Retrieval**:
   Write a function that takes a question and history, and returns relevant text snippets or turn IDs.

4. **Run Evaluation**:
   Pass your results to the benchmark for scoring.

## 3. Example Implementation

The easiest way to start is using our example script.

**File:** [`scripts/example_run_benchmark.py`](../scripts/example_run_benchmark.py)

```python
import asyncio
from came_bench import Benchmark
from came_bench.proto import LanguageModelProviderConfig, LanguageModelProvider, OpenAIConfig

# 1. Define your custom retrieval logic
def my_retrieval(question, turns):
    # Your logic here: find relevant info in 'turns' for 'question'
    # Remember: Keep total context < 4096 tokens!
    return ["Retrieved memory snippet 1", "Retrieved memory snippet 2"]

async def main():
    # Initialize benchmark with default token limit (4096)
    # You can change this via: Benchmark(token_limit=8192)
    bench = Benchmark()
    
    # 2. Configure the Judge LLM (OpenAI, Azure, Anthropic, etc.)
    # We recommend using gpt-5-mini for the answer generation and gpt-4.1-mini for the judgment.
    lm_gen_config = LanguageModelProviderConfig(
        provider=LanguageModelProvider.LANGUAGE_MODEL_PROVIDER_OPENAI,
        model_name="gpt-5-mini",
        temperature=1.0,
        max_tokens=20000,
        openai_config=OpenAIConfig(api_key="sk-...")
    )
    
    lm_jud_config = LanguageModelProviderConfig(
        provider=LanguageModelProvider.LANGUAGE_MODEL_PROVIDER_OPENAI,
        model_name="gpt-4.1-mini",
        temperature=1.0,
        max_tokens=1024,
        top_p=0.9,
        openai_config=OpenAIConfig(api_key="sk-...")
    )

    for traj_id in bench.list_trajectories():
        questions = bench.get_questions(traj_id)
        turns = bench.get_turns(traj_id)
        
        # 3. Batch process your retrieval
        results = []
        for q in questions:
            context = my_retrieval(q, turns)
            results.append({
                "question_id": q.id,
                "memory_snippets": context
            })
            
        # 4. Evaluate
        result = await bench.evaluate(
            traj_id=traj_id,
            retrieval_results=results,
            lm_gen_config=lm_gen_config,
            lm_jud_config=lm_jud_config
        )
        
        print(f"Accuracy: {result.accuracy}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 4. Configuration

### LLM Providers
CAME-Bench supports many providers (OpenAI, Azure, Anthropic, Gemini, local Ollama, etc.) for the evaluation/generation step.
👉 **See [LLM_CONFIG.md](LLM_CONFIG.md) for detailed configuration examples.**

### Data Structure

- **Question Object**: Contains `.content` (text) and `.id`.
- **Turn Object**: Contains `.role` (user/assistant/debater), `.content` (message text), and additional metadata like `timestamp` or `action`.

## 5. Advanced Usage

If you need to generate Protocol Buffer definitions manually (e.g., if you modified the `.proto` files):
```bash
python scripts/generate_proto_universal.py
```
