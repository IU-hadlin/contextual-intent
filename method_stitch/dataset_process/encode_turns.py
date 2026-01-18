"""
Encode turns with schema of Dataset turns proto (defined in proto/project_dataset_uniform.proto) and upload embeddings to Qdrant.

Usage:
python -m method_stitch.dataset_process.encode_turns -c {path_to_config_json}
"""
#!/usr/bin/env python3
from __future__ import annotations

# Standard library imports
import argparse
import asyncio
from typing import Iterable, List, Optional, Set
import uuid

# Third-party imports
from google.protobuf.json_format import MessageToDict, ParseDict
from pydantic_core.core_schema import none_schema
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import litellm
import tiktoken
import dspy

# Local imports
from src import Turn, TurnEncodingResponse, DatasetTurnEncodeRequest, TurnEncodeStrategy, LanguageModelProvider
from ..encoder import Encoder
from ..utils import load_config, load_turns
import logging
from ..lm import init_lm, get_lm_cost

# Configure root logger to ERROR so dependencies only emit errors
logging.basicConfig(level=logging.ERROR)

# Configure this module's logger to INFO with its own handler
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _formatter = logging.Formatter("%(levelname)s:%(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
# Prevent propagation to root to avoid double logging and root-level filtering
logger.propagate = False

MAX_INPUT_TOKENS = 8192


def truncate_text_to_tokens(text: str, max_tokens: int) -> tuple[str, int]:
    """Truncate text to at most max_tokens for the given model.

    Returns (possibly-truncated-text, token_count_after_truncation).
    If a tokenizer isn't available, falls back to a 4 chars/token heuristic.
    """
    assert max_tokens > 0, "max_tokens must be greater than 0"

    tokenizer = tiktoken.get_encoding("cl100k_base")
    token_ids = tokenizer.encode(text)
    if len(token_ids) <= max_tokens:
        return text
    token_ids = token_ids[:max_tokens]
    truncated = tokenizer.decode(token_ids)
    return truncated


async def upsert_batch(client: AsyncQdrantClient, collection_name: str, batch_turn_encoding_responses: List[TurnEncodingResponse]):
    for _ in range(3):
        try:
            points = [
                models.PointStruct(id=turn_encoding_response.id, vector=turn_encoding_response.embedding,
                                   payload=MessageToDict(turn_encoding_response.turn, preserving_proto_field_name=True))
                for turn_encoding_response in batch_turn_encoding_responses
            ]
            await asyncio.wait_for(
                client.upsert(collection_name=collection_name, points=points),
                timeout=60,
            )
            return
        except Exception as e:
            logger.error(f"Error upserting batch: {e}. Retrying...")
            continue
    raise Exception("Failed to upsert batch after 3 attempts")


async def ensure_collection(client: AsyncQdrantClient, collection_name: str, vector_size: int, reset_collection: bool):
    # Check if collection exists
    collections = await client.get_collections()
    exists = any(c.name == collection_name for c in collections.collections)
    if exists and reset_collection:
        logger.info(f"Collection '{collection_name}' already exists.")
        logger.info("Resetting collection per configuration (non-interactive).")
        logger.info(f"Deleting collection '{collection_name}'...")
        await client.delete_collection(collection_name=collection_name)
    if not exists or reset_collection:
        logger.info(f"Creating collection '{collection_name}'...")
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        )

async def get_content_to_encode(turn: Turn, lm: Optional[dspy.LM], dataset_turn_encode_request: DatasetTurnEncodeRequest) -> str:
    if dataset_turn_encode_request.turn_encode_strategy == TurnEncodeStrategy.TURN_ENCODE_STRATEGY_TURN_CONTENT_ONLY:
        if dataset_turn_encode_request.dataset_name == "debate" or dataset_turn_encode_request.dataset_name == "travel_planning":
            turn_id = turn.id.split("-turn-")[-1]
            turn.content = f"turn_{turn_id} | {turn.role}: {turn.content}"
            return turn.content
        else:
            return turn.content

async def get_all_turn_ids(client: AsyncQdrantClient, collection_name: str) -> Set[str]:
    all_ids = set()
    if not await client.collection_exists(collection_name):
        return all_ids
    offset = None
    while True:
        logger.info(f"Scrolling with offset: {offset}")
        result = await client.scroll(
            collection_name=collection_name,
            limit=10000,
            offset=offset,
            with_payload=["id"]
        )
        records, next_page_offset = result
        if len(records) == 0:
            break
        all_ids.update([record.payload["id"] for record in records])
        if next_page_offset is None:
            break
        offset = next_page_offset
    return all_ids

async def async_embed_and_upsert(dataset_turn_encode_request: DatasetTurnEncodeRequest):

    turns = load_turns(dataset_turn_encode_request.dataset_name, dataset_turn_encode_request.turns_jsonl_path)
    encoder = Encoder(dataset_turn_encode_request.encoder_config)
    lm = None
    if dataset_turn_encode_request.language_model_provider_config.provider != LanguageModelProvider.LANGUAGE_MODEL_PROVIDER_UNSPECIFIED:
        lm = init_lm(dataset_turn_encode_request.language_model_provider_config)

    client = AsyncQdrantClient(url=dataset_turn_encode_request.qdrant_config.url, api_key=dataset_turn_encode_request.qdrant_config.api_key)

    # Ensure collection exists or is reset/created as needed BEFORE checking existing IDs
    await ensure_collection(
        client=client,
        collection_name=dataset_turn_encode_request.qdrant_config.collection,
        vector_size=dataset_turn_encode_request.qdrant_config.vector_size,
        reset_collection=dataset_turn_encode_request.reset_collection
    )

    # If we're resetting the collection, there are no existing IDs to skip
    existing_turn_ids: Set[str] = set()
    if not dataset_turn_encode_request.reset_collection:
        existing_turn_ids = await get_all_turn_ids(client=client, collection_name=dataset_turn_encode_request.qdrant_config.collection)
    turns = [turn for turn in turns if turn.id not in existing_turn_ids]
    logger.info(f"Found {len(existing_turn_ids)} existing turn ids. Encoding {len(turns)} turns.")

    async def process_turn(turn: Turn) -> TurnEncodingResponse:
        for partition_entry in turn.partition:
            if not (
                partition_entry.startswith("question:")
                or partition_entry.startswith("topic-")
                or partition_entry.startswith("trip-")
                or partition_entry.startswith("day-")
            ):
                logger.warning(
                    f"Unexpected partition entry '{partition_entry}' for turn_id={turn.id}"
                )
        turn_id = str(uuid.uuid5(uuid.NAMESPACE_URL, turn.id))
        current_max_tokens = MAX_INPUT_TOKENS

        try:
            content_to_encode = await get_content_to_encode(turn, lm, dataset_turn_encode_request)
        except Exception as e:
            logger.error(f"Error getting content to encode for turn_id={turn.id}: {e}")
            return TurnEncodingResponse(id=turn_id, turn=turn, success=False)

        for _ in range(3):
            try:
                embedding = await encoder.aencode([content_to_encode])
                turn_encoding_response = TurnEncodingResponse(id=turn_id, turn=turn, success=True)
                turn_encoding_response.embedding.extend(embedding[0])
                return turn_encoding_response
            except litellm.ContextWindowExceededError as e:
                logger.info(f"Context window exceeded for turn_id={turn.id}: {e}. Truncating content...")
                # Reduce allowance and re-truncate by tokens
                content_to_encode = truncate_text_to_tokens(content_to_encode, current_max_tokens)
                current_max_tokens = max(256, int(current_max_tokens * 0.9))
                continue
            except Exception as e:
                raise e
                logger.error(f"Embedding failed for turn_id={turn.id}: {e}")
                return TurnEncodingResponse(id=turn_id, turn=turn, success=False)
        return TurnEncodingResponse(id=turn_id, turn=turn, success=False)

    async def process_turns(turns: List[Turn]):
        results: List[TurnEncodingResponse] = []
        failed_turns: List[Turn] = []
        semaphore = asyncio.Semaphore(encoder.max_concurrent)
        async def sem_task(turn):
            async with semaphore:
                try:
                    return await asyncio.wait_for(process_turn(turn), timeout=120)
                except asyncio.TimeoutError:
                    logger.error(f"Turn processing timed out for turn_id={turn.id}")
                    turn_id = str(uuid.uuid5(uuid.NAMESPACE_URL, turn.id))
                    return TurnEncodingResponse(id=turn_id, turn=turn, success=False)

        turn_iter = iter(turns)
        running = set()
        try:
            for _ in range(min(encoder.max_concurrent, len(turns))):
                turn = next(turn_iter)
                running.add(asyncio.create_task(sem_task(turn)))
        except StopIteration:
            pass

        pbar = tqdm_asyncio(total=len(turns), desc="Async embedding and upsert")
        while running:
            done, running = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            for fut in done:
                turn_encoding_response = await fut
                if turn_encoding_response.success:
                    results.append(turn_encoding_response)
                else:
                    failed_turns.append(turn_encoding_response.turn)
                pbar.update(1)
                try:
                    turn = next(turn_iter)
                    running.add(asyncio.create_task(sem_task(turn)))
                except StopIteration:
                    pass
                if len(results) >= dataset_turn_encode_request.batch_upload_size:
                    await upsert_batch(client=client, collection_name=dataset_turn_encode_request.qdrant_config.collection, batch_turn_encoding_responses=results)
                    results.clear()
        pbar.close()
        if results:
            await upsert_batch(client=client, collection_name=dataset_turn_encode_request.qdrant_config.collection, batch_turn_encoding_responses=results)
        return failed_turns

    # Retry logic: process all turns, then retry failed ones up to 3 times
    max_retries = 3
    current_failed_turns = turns
    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            logger.info(f"\n--- Retry attempt {attempt} for {len(current_failed_turns)} failed turns ---")
        else:
            logger.info(f"\n--- Embedding attempt {attempt} for {len(current_failed_turns)} turns ---")
        failed_turns = await process_turns(current_failed_turns)
        if not failed_turns:
            logger.info(f"All turns embedded successfully after {attempt} attempt(s).")
            break
        current_failed_turns = failed_turns
    else:
        # After max_retries, still have failed turns
        logger.error(f"Failed to embed {len(current_failed_turns)} turns after {max_retries} attempts:")
        for turn in current_failed_turns:
            logger.error(f"  - turn_id={turn.id}")
    
    if lm is not None:
        logger.info(f"LM cost: {get_lm_cost(lm)}")
    logger.info(f"Encoder cost: {encoder.get_cost()}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode turns and upload embeddings to Qdrant")
    parser.add_argument("-c", "--config", required=True, help="Path to config JSON")
    parser.add_argument(
        "--if_reset",
        choices=["true", "false"],
        default=None,
        help=(
            "Override config.reset_collection. 'true' resets the collection (skip existing ID check). "
            "'false' keeps the collection and skips already-embedded turns."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


async def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    dataset_turn_encode_request = DatasetTurnEncodeRequest()
    ParseDict(load_config(args.config), dataset_turn_encode_request)

    # If user specifies --if_reset, override config value
    if args.if_reset is not None:
        dataset_turn_encode_request.reset_collection = (args.if_reset.lower() == "true")

    await async_embed_and_upsert(dataset_turn_encode_request=dataset_turn_encode_request)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
