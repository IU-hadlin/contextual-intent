"""
Note that for huggingface, we need to modify the liellm source file.
From llms/huggingface/embedding/handler.py, locate _process_embedding_response() function,
add following code:

elif "embeddings" in embeddings:
    for idx, embedding in enumerate(embeddings["embeddings"]):
        output_data.append(
            {
                "object": "embedding",
                "index": idx,
                "embedding": embedding,  # flatten list returned from hf
            }
        )
"""
from came_bench.proto import (
    EmbeddingModelProviderConfig,
    EmbeddingModelProvider,
)
import asyncio
from pathlib import Path
import sys
from typing import List

import litellm
from google.protobuf.json_format import MessageToDict
import threading
from came_bench.proto import CostEntry, CostType


class Encoder:
    def __init__(self, embedding_model_provider_config: EmbeddingModelProviderConfig):
        self.max_concurrent = embedding_model_provider_config.max_concurrent
        self.semaphore = asyncio.Semaphore(embedding_model_provider_config.max_concurrent)
        provider = embedding_model_provider_config.provider
        self.cost_per_1M_tokens = embedding_model_provider_config.cost_per_1M_tokens
        self.total_tokens = 0
        self._token_lock = asyncio.Lock()  # For async thread safety
        self._sync_token_lock = threading.Lock()  # For sync thread safety

        provider_name = EmbeddingModelProvider.Name(provider).lower().replace("embedding_model_provider_", "")
        config_attr = f"{provider_name}_config"
        config = getattr(embedding_model_provider_config, config_attr, None)
        if config is not None:
            self.kwargs = MessageToDict(config, preserving_proto_field_name=True)
            self.model_name = embedding_model_provider_config.model_name
        else:
            raise ValueError(
                f"Expected {config_attr} in embedding_model_provider_config, but got {embedding_model_provider_config}")

    def encode(self, input: list[str], batch_size: int = 10) -> list[list[float]]:
        # NOTE: This method is not thread-safe for total_tokens. If you need thread safety, use the async version.
        # Split into batches to respect API limits (e.g. DashScope max 10 per request)
        all_embeddings = []
        for i in range(0, len(input), batch_size):
            batch = input[i:i + batch_size]
            try:
                output = litellm.embedding(model=self.model_name, input=batch, **self.kwargs)
                all_embeddings.extend([item["embedding"] for item in output.data])
            except Exception as e:
                print(f"Error encoding input: {e}")
                print("input: ", batch)
                raise e
        return all_embeddings

    async def aencode(self, input: list[str]) -> list[list[float]]:
        """
        Encode each input string in parallel, one per request, maximizing throughput up to max_concurrent.
        """
        async def encode_one(text):
            async with self.semaphore:
                try:
                    timeout_sec = self.kwargs.get("request_timeout") or self.kwargs.get("timeout") or 60
                    output = await asyncio.wait_for(
                        litellm.aembedding(model=self.model_name, input=[text], **self.kwargs),
                        timeout=timeout_sec,
                    )
                    return output.data[0]["embedding"], getattr(output.usage, "total_tokens", 0)
                except asyncio.TimeoutError:
                    print(f"Timeout encoding input after {timeout_sec}s")
                    print("input: ", text)
                    raise
                except Exception as e:
                    print(f"Error encoding input: {e}")
                    print("input: ", text)
                    raise e

        results = await asyncio.gather(*(encode_one(text) for text in input))
        embeddings = [result[0] for result in results]
        total_new_tokens = sum(result[1] for result in results)
        async with self._token_lock:
            self.total_tokens += total_new_tokens
        return embeddings

    def get_cost(self) -> float:
        # Not thread-safe if used with async methods. For async, use aget_cost.
        with self._sync_token_lock:
            return self.total_tokens * self.cost_per_1M_tokens / 1000000


def get_class_instance_all_encoder_cost(class_instance) -> List[CostEntry]:
    cost_entries = []
    for attr_name in dir(class_instance):
        if attr_name.endswith('_encoder') and not attr_name.startswith('_'):
            try:
                encoder_instance = getattr(class_instance, attr_name)
                if hasattr(encoder_instance, 'model_name') and hasattr(encoder_instance, 'get_cost'):
                    cost_entries.append(
                        CostEntry(
                            type=CostType.COST_TYPE_EMBEDDING,
                            description=f"{attr_name} (model: {encoder_instance.model_name})",
                            cost=encoder_instance.get_cost()
                        )
                    )
            except Exception as e:
                # Skip if the attribute doesn't have the expected LM properties
                continue
    return cost_entries
