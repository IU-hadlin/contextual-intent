"""Event type label discovery and assignment for conversation turns."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import dspy
import numpy as np
from tqdm import tqdm

from came_bench.proto import (
    EmbeddingModelProvider,
    EmbeddingModelProviderConfig,
    EventTypeLabelerConfig,
    LanguageModelProvider,
    LanguageModelProviderConfig,
    Turn,
)
from came_bench.utils.encoder import Encoder
from came_bench.utils.lm import init_lm

from came_bench.utils.io import get_lm_cost, load_config, load_turns
from .common_utils import _conversation_from_turn_id as common_conversation_from_turn_id
from google.protobuf.json_format import ParseDict

logger = logging.getLogger(__name__)


def extract_first_and_last_sentence(text: str) -> str:
    """Extract first and last sentence from text, connected with '...'
    
    Args:
        text: Input text to extract sentences from
        
    Returns:
        First sentence + "..." + last sentence, or just the text if too short
    """
    if not text:
        return ""
    
    # Split into sentences (simple approach using common sentence terminators)
    sentences = []
    current = []
    
    for char in text:
        current.append(char)
        if char in '.!?' and len(current) > 1:
            sentence = ''.join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []
    
    # Add any remaining text as a sentence
    if current:
        sentence = ''.join(current).strip()
        if sentence:
            sentences.append(sentence)
    
    if not sentences:
        return text.strip()
    
    # If only one sentence, return it
    if len(sentences) == 1:
        return sentences[0]
    
    # Return first + "..." + last
    return f"{sentences[0]} ... {sentences[-1]}"


class EventTypeLabelGenerationSignature(dspy.Signature):
    """
    Generate a list of event type labels that describe the different types of 
    discussions or events that occur in a conversation.
    
    Given the first 50 turns of a conversation and the dataset type, identify 
    the distinct event types or discussion themes present. Event types should be 
    concise labels that capture the nature of the events and can be used to describe
    multiple similar events.
    """
    
    first_turns_summary: str = dspy.InputField(
        description="Summary of the first 50 turns (first sentence + '...' + last sentence of each turn)"
    )
    dataset_type: str = dspy.InputField(
        description="Type of dataset"
    )
    
    event_type_labels: list[str] = dspy.OutputField(
        description="List of event type labels for this conversation"
    )


class EventTypeLabelValidationSignature(dspy.Signature):
    """
    Given a batch of consecutive turns and the current set of event type labels,
    determine if new event types are present that are not covered by existing labels. 
    Also, check if the existing labels have appropriate coverage and description to describe the batch of turns.
    Make least number of changes to the existing labels. Only add new labels if necessary.
    Return ONLY the new event type labels that should be added to the existing set. 
    If the existing labels are sufficient, return an empty list.
    Do NOT return the full updated list, only return the new labels to add.
    """
    
    batch_turns_summary: str = dspy.InputField(
        description="Summary of a batch of 50 consecutive turns (first sentence + '...' + last sentence of each turn)"
    )
    existing_event_labels: list[str] = dspy.InputField(
        description="List of existing event type labels"
    )
    dataset_type: str = dspy.InputField(
        description="Type of dataset"
    )
    
    new_event_labels: list[str] = dspy.OutputField(
        description="ONLY the new event type labels to ADD to the existing set. Return empty list if no new labels needed."
    )


class EventTypeLabelSelectionSignature(dspy.Signature):
    """
    The recent event context is the summary of the recent events and their event type labels. 
    Comprehend the turn summary and select the most appropriate event type labels under the recent event context. 
    If the current turn is discussing objects or topics same as previous turns, you should refer to the previous event type labels.
    """

    turn_summary: str = dspy.InputField(
        description="Abbreviated content of the current turn"
    )
    candidate_event_labels: str = dspy.InputField(
        description="Ranked candidate event labels with cosine similarity context"
    )
    dataset_type: str = dspy.InputField(
        description="Type of dataset to anchor label selection"
    )
    recent_event_context: str = dspy.InputField(
        description="Recent event context of the conversation"
    )

    selected_event_labels: list[str] = dspy.OutputField(
        description="List of selected event type labels"
    )


class EventTypeLabeler:
    """Discover and assign event type labels to conversation turns."""

    def __init__(
        self,
        lm_config: LanguageModelProviderConfig,
        encoder_config: EmbeddingModelProviderConfig,
        *,
        dataset_type: str = "unknown",
        max_workers: int = 5,
        group_consecutive_turns: bool = True,
    ) -> None:
        """Initialize the event type labeler.
        
        Args:
            lm_config: Language model configuration for label generation
            encoder_config: Encoder configuration for similarity computation
            dataset_type: Type of dataset for context in label generation
            max_workers: Maximum number of concurrent workers for parallel processing
        """
        if lm_config.provider == LanguageModelProvider.LANGUAGE_MODEL_PROVIDER_UNSPECIFIED:
            raise ValueError("Language model provider config must be specified")
        
        if encoder_config.provider == EmbeddingModelProvider.EMBEDDING_MODEL_PROVIDER_UNSPECIFIED:
            raise ValueError("Encoder provider config must be specified")
        
        self._lm_config = lm_config
        self._dataset_type = dataset_type
        self._max_workers = max_workers
        self._group_consecutive_turns = group_consecutive_turns
        self._block_size = 50  # number of RAW turns per block for streaming-style discovery
        
        self.lm = init_lm(lm_config)
        
        # Configure DSPy to use the initialized LM
        dspy.configure(lm=self.lm)
        
        self._event_label_predictor = dspy.Predict(EventTypeLabelGenerationSignature)
        self._event_label_validator = dspy.Predict(EventTypeLabelValidationSignature)
        self._event_label_selector = dspy.Predict(EventTypeLabelSelectionSignature)
        
        # Initialize encoder
        logger.info(
            "Initializing encoder with config: provider=%s, model=%s",
            EmbeddingModelProvider.Name(encoder_config.provider),
            encoder_config.model_name,
        )
        self._encoder = Encoder(encoder_config)
        
        # Cache for event labels and embeddings per conversation
        # Final label set (after all blocks)
        self._conversation_event_labels: Dict[str, List[str]] = {}
        # Embeddings for final label set (for fallback / convenience)
        self._conversation_event_embeddings: Dict[str, np.ndarray] = {}
        # Block-level label snapshots: conv_id -> {block_index -> labels_at_end_of_block}
        self._conversation_event_label_snapshots: Dict[str, Dict[int, List[str]]] = {}
        # Block-level embeddings: conv_id -> {block_index -> embeddings_for_labels_at_end_of_block}
        self._conversation_event_embeddings_per_block: Dict[str, Dict[int, np.ndarray]] = {}
        
        logger.info("Initialized EventTypeLabeler with max_workers=%d", max_workers)

    def discover_and_assign_event_types(
        self,
        turns: Sequence[Turn],
        *,
        output_path: Optional[str | Path] = None,
        max_conversations: Optional[int] = None,
    ) -> Dict[str, Dict[str, List[str]]]:
        """Discover event types and assign them to turns.
        
        Args:
            turns: Sequence of turns to process
            output_path: Optional path to save event type assignments
            
        Returns:
            Dictionary mapping conversation_id to {turn_index: [event_types]}
        """
        conversation_groups = list(self._group_turns_by_conversation(turns))

        if max_conversations is not None:
            max_conversations = max(0, int(max_conversations))
            conversation_groups = conversation_groups[:max_conversations]
            logger.info(
                "Limiting processing to the first %d conversation(s)",
                len(conversation_groups),
            )
        
        if not conversation_groups:
            return {}
        
        # Load existing results if output path exists
        all_assignments: Dict[str, Dict[str, List[str]]] = {}
        if output_path is not None:
            output_file = Path(output_path)
            if output_file.exists():
                logger.info("Loading existing results from %s", output_path)
                try:
                    with output_file.open("r", encoding="utf-8") as f:
                        loaded_data = json.load(f)
                        all_assignments = {
                            conv_id: {
                                str(turn_key).strip(): labels
                                for turn_key, labels in conv_assignments.items()
                                if str(turn_key).strip()
                            }
                            for conv_id, conv_assignments in loaded_data.items()
                        }
                    logger.info("Loaded %d existing conversations", len(all_assignments))
                except Exception as e:
                    logger.warning("Failed to load existing results: %s", e)
        
        for conversation_id, dialogue_turns in tqdm(conversation_groups, desc="Processing conversations"):
            # Skip if already processed
            if conversation_id in all_assignments:
                logger.info("Skipping conversation %s (already processed)", conversation_id)
                continue
                
            logger.info("Processing conversation %s (%d turns)", conversation_id, len(dialogue_turns))
            
            # Discover event labels for this conversation (streaming-style over raw turns)
            final_labels, labels_by_block = self._generate_event_labels_for_conversation(
                conversation_id,
                dialogue_turns,
            )
            self._conversation_event_labels[conversation_id] = final_labels
            
            # Generate embeddings for final label set (for fallback)
            final_label_embeddings_list = self._encoder.encode(final_labels)
            final_label_embeddings = np.array(final_label_embeddings_list)
            self._conversation_event_embeddings[conversation_id] = final_label_embeddings

            # Generate embeddings for each block-level label snapshot
            block_embeddings: Dict[int, np.ndarray] = {}
            for block_idx, labels in labels_by_block.items():
                label_embeddings_list = self._encoder.encode(labels)
                block_embeddings[block_idx] = np.array(label_embeddings_list)

            self._conversation_event_label_snapshots[conversation_id] = labels_by_block
            self._conversation_event_embeddings_per_block[conversation_id] = block_embeddings
            
            # Assign event types to each turn (using block-level label snapshots)
            assignments = self._assign_event_types_to_turns(conversation_id, dialogue_turns)
            all_assignments[conversation_id] = assignments
            
            # Save incrementally after each conversation
            if output_path is not None:
                self.save_event_type_assignments(all_assignments, output_path)
                logger.info("Saved progress to %s", output_path)
        
        return all_assignments

    def _generate_event_labels_for_conversation(
        self,
        conversation_id: str,
        turns: Sequence[Turn],
    ) -> Tuple[List[str], Dict[int, List[str]]]:
        """Generate event type labels dynamically with streaming-style block updates.
        
        Streaming-style logic (simulated offline over the full trajectory):
        1. Use the first block of up to 50 RAW turns to generate initial seed labels L0.
        2. For each subsequent completed 50-turn block, randomly sample 20% of raw turns
           in that block to validate/expand the label set using EventTypeLabelValidationSignature.
        3. For each block b, we snapshot the label set Lb that is available after processing
           that block.
        
        Returns:
            final_labels: List[str] - label set after processing all blocks
            labels_by_block: Dict[block_index, List[str]] - snapshot at each block
        """
        logger.info("=" * 80)
        logger.info("GENERATING EVENT TYPE LABELS FOR CONVERSATION: %s", conversation_id)
        logger.info("=" * 80)
        
        raw_turns = list(turns)
        num_turns = len(raw_turns)

        if num_turns == 0:
            logger.info("No turns found, returning default label")
            return ["general_discussion"], {0: ["general_discussion"]}

        logger.info("Total raw turns in conversation: %d", num_turns)

        block_size = self._block_size
        num_blocks = math.ceil(num_turns / block_size)

        # Step 1: Generate initial seed labels from first block of up to 50 raw turns.
        logger.info("")
        logger.info("STEP 1: Generating initial seed labels from first block of up to %d turns", block_size)
        logger.info("-" * 80)

        first_block_end = min(block_size, num_turns)
        first_turns_summaries: List[str] = []
        for i in range(first_block_end):
            turn = raw_turns[i]
            first_and_last = extract_first_and_last_sentence(str(turn.content))
            first_turns_summaries.append(first_and_last)

        if not first_turns_summaries:
            logger.info("No turns found, returning default label")
            return ["general_discussion"], {0: ["general_discussion"]}

        # Create summary of first block turns
        first_turns_summary = "\n".join(
            f"{i+1}. {sent}" for i, sent in enumerate(first_turns_summaries)
        )

        logger.info("Calling LLM to generate initial seed labels from first block...")
        with dspy.context(lm=self.lm):
            prediction = self._event_label_predictor(
                first_turns_summary=first_turns_summary,
                dataset_type=self._dataset_type,
            )
        logger.info("LLM call for initial seed labels completed!")

        labels = prediction.event_type_labels if prediction.event_type_labels else []
        logger.info("Raw LLM output for initial labels: %s", labels)

        if not labels:
            labels = ["general_discussion"]
            logger.warning("No valid labels returned for initial block, using default label.")

        labels_by_block: Dict[int, List[str]] = {}
        labels_by_block[0] = list(labels)  # snapshot after first block

        logger.info("")
        logger.info("INITIAL SEED LABELS AFTER BLOCK 0 (%d labels):", len(labels))
        for i, label in enumerate(labels, 1):
            logger.info("  %d. %s", i, label)

        # Step 2: For each subsequent block, validate/expand labels using 20% sample.
        logger.info("")
        logger.info("STEP 2: Dynamic label validation and expansion for subsequent blocks")
        logger.info("-" * 80)
        logger.info("Total blocks (size=%d raw turns): %d", block_size, num_blocks)

        if num_blocks == 1:
            logger.info("Only one block of turns; no additional label validation needed.")
        else:
            for block_idx in range(1, num_blocks):
                start_idx = block_idx * block_size
                end_idx = min((block_idx + 1) * block_size, num_turns)
                block_length = end_idx - start_idx

                if block_length <= 0:
                    continue

                logger.info("")
                logger.info("Processing block %d (turn indices %d-%d, length=%d)", 
                            block_idx, start_idx + 1, end_idx, block_length)

                # Streaming-style assumption: we update labels when a block is available.
                # If the last block is partial (< block_size turns), we still validate/update
                # using the available turns so the tail isn't ignored.
                if block_length < block_size:
                    logger.info(
                        "Block %d is partial (%d turns). Validating labels with available turns.",
                        block_idx,
                        block_length,
                    )

                # Randomly sample 20% of the raw turns in this block for validation.
                indices = list(range(start_idx, end_idx))
                sample_size = max(1, int(0.2 * len(indices)))
                random.seed(42 + block_idx)  # deterministic but block-dependent
                sampled_indices = random.sample(indices, sample_size)

                logger.info(
                    "Block %d: sampling %d/%d turns (20%%) for label validation.",
                    block_idx,
                    sample_size,
                    len(indices),
                )

                batch_turns_summaries: List[str] = []
                for idx in sampled_indices:
                    turn = raw_turns[idx]
                    first_and_last = extract_first_and_last_sentence(str(turn.content))
                    batch_turns_summaries.append(first_and_last)

                batch_summary = "\n".join(
                    f"{i+1}. {sent}" for i, sent in enumerate(batch_turns_summaries)
                )

                logger.info("  Current labels before validation for block %d: %s", block_idx, labels)
                logger.info("  Calling LLM to validate/expand labels for block %d...", block_idx)

                with dspy.context(lm=self.lm):
                    validation_prediction = self._event_label_validator(
                        batch_turns_summary=batch_summary,
                        existing_event_labels=labels,
                        dataset_type=self._dataset_type,
                    )

                new_labels = (
                    validation_prediction.new_event_labels
                    if validation_prediction.new_event_labels
                    else []
                )
                logger.info(
                    "  LLM response for block %d new labels: %s",
                    block_idx,
                    new_labels if new_labels else "(no new labels needed)",
                )

                if new_labels:
                    existing_set = set(labels)
                    added = 0
                    for new_label in new_labels:
                        if new_label and new_label not in existing_set:
                            labels.append(new_label)
                            existing_set.add(new_label)
                            added += 1
                    logger.info(
                        "  ✓ Added %d new labels in block %d. Total labels now: %d",
                        added,
                        block_idx,
                        len(labels),
                    )
                else:
                    logger.info("  ○ No new labels needed for block %d.", block_idx)

                # Snapshot labels after processing this block
                labels_by_block[block_idx] = list(labels)

        # Final summary
        logger.info("=" * 80)
        logger.info("FINAL EVENT TYPE LABELS (%d labels) AFTER ALL BLOCKS:", len(labels))
        for i, label in enumerate(labels, 1):
            logger.info("  %d. %s", i, label)
        logger.info("=" * 80)
        logger.info("")

        return labels, labels_by_block

    def _assign_event_types_to_turns(
        self,
        conversation_id: str,
        turns: Sequence[Turn],
    ) -> Dict[int, List[str]]:
        """Assign event types to all turns in a conversation using async parallel processing.
        
        Uses block-level label snapshots:
        - For turns whose leading turn index falls into block b (0-based),
          we use label set L_b discovered after processing that block.
        
        Args:
            conversation_id: The conversation ID
            turns: Sequence of turns in the conversation
            
        Returns:
            Dictionary mapping turn index to list of event types
        """
        # Run the async version in an event loop
        return asyncio.run(self._assign_event_types_to_turns_async(conversation_id, turns))

    async def _assign_event_types_to_turns_async(
        self,
        conversation_id: str,
        turns: Sequence[Turn],
    ) -> Dict[int, List[str]]:
        """Async implementation of event type assignment with parallel processing.
        
        Args:
            conversation_id: The conversation ID
            turns: Sequence of turns in the conversation
            
        Returns:
            Dictionary mapping turn index to list of event types
        """
        if conversation_id not in self._conversation_event_labels:
            raise ValueError(f"Event labels not generated for conversation {conversation_id}")
        if conversation_id not in self._conversation_event_label_snapshots:
            raise ValueError(f"Block-level label snapshots not available for conversation {conversation_id}")
        if conversation_id not in self._conversation_event_embeddings_per_block:
            raise ValueError(f"Block-level embeddings not available for conversation {conversation_id}")

        labels_by_block = self._conversation_event_label_snapshots[conversation_id]
        embeddings_by_block = self._conversation_event_embeddings_per_block[conversation_id]
        final_labels = self._conversation_event_labels[conversation_id]
        final_embeddings = self._conversation_event_embeddings[conversation_id]

        logger.info("")
        logger.info("=" * 80)
        turn_groups = self._build_turn_groups(turns)
        logger.info(
            "Turn grouping by role is %s",
            "enabled" if self._group_consecutive_turns else "disabled",
        )

        logger.info(
            "ASSIGNING EVENT TYPES TO %d TURN GROUPS (via cosine similarity + async parallel)",
            len(turn_groups),
        )
        logger.info("  Max concurrent workers: %d", self._max_workers)
        logger.info("=" * 80)
        logger.info("")

        # Create semaphore to control concurrency
        semaphore = asyncio.Semaphore(self._max_workers)
        
        # Shared dictionary to store completed assignments (thread-safe with asyncio)
        # Keyed by group_idx (0-based index into turn_groups)
        completed_assignments: Dict[int, List[str]] = {}
        assignment_lock = asyncio.Lock()
        
        # Create tasks for all turn groups (wrap in asyncio.create_task for proper task tracking)
        tasks = [
            asyncio.create_task(
                self._process_turn_group_async(
                    group=group,
                    block_size=self._block_size,
                    labels_by_block=labels_by_block,
                    embeddings_by_block=embeddings_by_block,
                    final_labels=final_labels,
                    final_embeddings=final_embeddings,
                    semaphore=semaphore,
                    all_turn_groups=turn_groups,
                    current_group_idx=idx,
                    completed_assignments=completed_assignments,
                    assignment_lock=assignment_lock,
                )
            )
            for idx, group in enumerate(turn_groups)
        ]
        
        # Execute all tasks in parallel with progress tracking
        results = []
        with tqdm(total=len(tasks), desc="Assigning event types") as pbar:
            # Process tasks as they complete
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(
                    pending, 
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    try:
                        results.append(task.result())
                    except Exception as e:
                        logger.error("Task failed with error: %s", e, exc_info=True)
                        raise
                    pbar.update(1)
        
        # Aggregate results
        assignments: Dict[str, List[str]] = {}
        event_type_counts: Dict[str, int] = {}
        
        for group, selected_labels in results:
            leading_turn_index = group[0][0]
            
            # Store assignment for all turns in the group using raw turn_ids
            for _, turn in group:
                assignments[str(turn.id)] = selected_labels
            
            logger.info(
                "Turn group starting at %d | Final event types: %s",
                leading_turn_index,
                selected_labels,
            )
            
            # Track primary event type for statistics
            if selected_labels:
                primary_type = selected_labels[0]
                event_type_counts[primary_type] = event_type_counts.get(primary_type, 0) + 1

        # Log assignment summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("EVENT TYPE ASSIGNMENT SUMMARY:")
        logger.info("  Total turn groups: %d", len(turn_groups))
        logger.info("  Event type distribution (primary):")
        for event_type, count in sorted(event_type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(turn_groups) * 100) if len(turn_groups) > 0 else 0
            logger.info("    - %s: %d groups (%.1f%%)", event_type, count, percentage)
        logger.info("=" * 80)
        logger.info("")

        return assignments

    @staticmethod
    def _build_recent_event_context(
        all_turn_groups: List[List[Tuple[int, Turn]]],
        current_group_idx: int,
        completed_assignments: Dict[int, List[str]],
        num_previous_turns: int = 5,
    ) -> str:
        """Build recent event context from previous N turn groups with their event types.
        
        Args:
            all_turn_groups: List of all turn groups in the conversation
            current_group_idx: Index of the current group
            completed_assignments: Dictionary of group_idx -> event_type_labels for completed turns
            num_previous_turns: Number of previous turn groups to include
            
        Returns:
            Formatted string with recent turn summaries and event types
        """
        if current_group_idx == 0:
            return "No previous context (this is the first turn)."
        
        # Get previous N turn groups (or all available if less than N)
        start_idx = max(0, current_group_idx - num_previous_turns)
        previous_groups = all_turn_groups[start_idx:current_group_idx]
        
        if not previous_groups:
            return "No previous context (this is the first turn)."
        
        # Build context string with event types if available
        context_parts = []
        for prev_group_idx, prev_group in enumerate(previous_groups, start=start_idx):
            leading_turn_index, leading_turn = prev_group[0]
            turn_summary = extract_first_and_last_sentence(str(leading_turn.content))
            
            # Include event type if this turn has been processed
            if prev_group_idx in completed_assignments:
                event_types = completed_assignments[prev_group_idx]
                event_type_str = ", ".join(event_types)
                context_parts.append(f"Turn {leading_turn_index} [{event_type_str}]: {turn_summary}")
            else:
                # Event type not yet assigned (still processing in parallel)
                context_parts.append(f"Turn {leading_turn_index} [pending]: {turn_summary}")
        
        return "\n".join(context_parts)

    async def _process_turn_group_async(
        self,
        group: List[Tuple[int, Turn]],
        block_size: int,
        labels_by_block: Dict[int, List[str]],
        embeddings_by_block: Dict[int, np.ndarray],
        final_labels: List[str],
        final_embeddings: np.ndarray,
        semaphore: asyncio.Semaphore,
        all_turn_groups: List[List[Tuple[int, Turn]]],
        current_group_idx: int,
        completed_assignments: Dict[int, List[str]],
        assignment_lock: asyncio.Lock,
    ) -> Tuple[List[Tuple[int, Turn]], List[str]]:
        """Process a single turn group asynchronously with semaphore control.
        
        Args:
            group: List of (turn_index, Turn) tuples representing consecutive turns
            block_size: Number of raw turns per block
            labels_by_block: Map from block index to label set after that block
            embeddings_by_block: Map from block index to label embeddings
            final_labels: Final label set after all blocks (fallback)
            final_embeddings: Embeddings for final label set (fallback)
            semaphore: Semaphore to control concurrency
            all_turn_groups: List of all turn groups in the conversation
            current_group_idx: Index of the current group in all_turn_groups
            completed_assignments: Shared dictionary of completed event type assignments
            assignment_lock: Lock for safely updating completed_assignments
            
        Returns:
            Tuple of (group, selected_labels)
        """
        leading_turn_index, leading_turn = group[0]

        # Determine which block this group's leading turn belongs to (0-based)
        block_idx = (leading_turn_index - 1) // block_size
        labels = labels_by_block.get(block_idx, final_labels)
        label_embeddings = embeddings_by_block.get(block_idx, final_embeddings)

        if not labels or label_embeddings is None or len(labels) != label_embeddings.shape[0]:
            # Fallback safety: use final labels/embeddings
            logger.warning(
                "Block %d labels/embeddings missing or size mismatch; falling back to final label set.",
                block_idx,
            )
            labels = final_labels
            label_embeddings = final_embeddings

        # Extract first and last sentence from leading turn
        first_and_last = extract_first_and_last_sentence(str(leading_turn.content))
        
        logger.debug(
            "Turn %d | Assigning event types (block %d) | First+Last: %s",
            leading_turn_index,
            block_idx,
            first_and_last[:80] + "..." if len(first_and_last) > 80 else first_and_last,
        )
        
        # Build recent event context from previous 5 turn groups (with their event types if available)
        async with assignment_lock:
            recent_event_context = self._build_recent_event_context(
                all_turn_groups, current_group_idx, completed_assignments, num_previous_turns=5
            )
        
        # Get embedding for turn's first+last sentence (run in thread pool to avoid blocking)
        loop = asyncio.get_event_loop()
        turn_embedding_list = await loop.run_in_executor(
            None, self._encoder.encode, [first_and_last]
        )
        turn_embedding = np.array(turn_embedding_list[0])

        # Compute cosine similarities
        # label_embeddings: shape (num_labels, dim)
        numerator = np.dot(label_embeddings, turn_embedding)
        denom = (
            np.linalg.norm(label_embeddings, axis=1) * np.linalg.norm(turn_embedding)
        )
        # Avoid division by zero
        denom = np.where(denom == 0, 1e-8, denom)
        similarities = numerator / denom

        # Sort indices by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        top_candidate_indices = sorted_indices[: min(5, len(sorted_indices))]
        candidate_labels = [labels[idx] for idx in top_candidate_indices]
        candidate_prompt = "\n".join(
            f"{rank}. {labels[idx]} (similarity={similarities[idx]:.4f})"
            for rank, idx in enumerate(top_candidate_indices, start=1)
        )

        # Default to the highest similarity label in case LLM selection fails
        selected_labels: List[str] = [candidate_labels[0]] if candidate_labels else []

        if candidate_labels:
            # Call LLM selector with semaphore control (only control the expensive LLM call)
            async with semaphore:
                selection_prediction = await loop.run_in_executor(
                    None,
                    lambda: self._event_label_selector(
                        turn_summary=first_and_last,
                        candidate_event_labels=candidate_prompt,
                        dataset_type=self._dataset_type,
                        recent_event_context=recent_event_context,
                    ),
                )

            raw_selected = str(selection_prediction.selected_event_labels).strip()
            if raw_selected:
                # selection_prediction.selected_event_labels is already a list[str] in principle,
                # but we keep the robust parsing in case the LM outputs something odd.
                if isinstance(selection_prediction.selected_event_labels, list):
                    chosen_raw = selection_prediction.selected_event_labels
                else:
                    chosen_raw = [raw_selected]

                chosen: List[str] = []
                for item in chosen_raw:
                    if isinstance(item, str):
                        parts = [p.strip() for p in item.split(",") if p.strip()]
                        for p in parts:
                            chosen.append(p)

                # Deduplicate while preserving order and filter to candidate_labels
                deduped = list(dict.fromkeys(chosen))
                filtered = [label for label in deduped if label in candidate_labels]

                if filtered:
                    selected_labels = filtered
                elif candidate_labels:
                    selected_labels = candidate_labels[:1]

            logger.info(
                "Turn %d (block %d) | LLM-selected event types from candidates %s -> %s",
                leading_turn_index,
                block_idx,
                candidate_labels,
                selected_labels,
            )
        else:
            logger.warning("Turn %d | No candidate event labels available", leading_turn_index)

        # Store the assignment in shared dictionary for use by subsequent turns
        async with assignment_lock:
            completed_assignments[current_group_idx] = selected_labels

        return group, selected_labels

    @staticmethod
    def save_event_type_assignments(
        assignments: Dict[str, Dict[str, List[str]]],
        output_path: str | Path,
    ) -> None:
        """Save event type assignments to JSON file, ordered by conversation ID and turn index."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)


        # Keep nested structure for other datasets
        # Sort assignments: conversations by ID, turns by index
        sorted_assignments = {
            conv_id: {
                str(turn_idx): labels
                for turn_idx, labels in sorted(turn_assignments.items(), key=lambda item: item[0])
            }
            for conv_id, turn_assignments in sorted(assignments.items())
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(sorted_assignments, f, ensure_ascii=False, indent=2)

        logger.info("Saved event type assignments to %s", output_path)

    @staticmethod
    def load_event_type_assignments(
        input_path: str | Path,
    ) -> Dict[str, Dict[str, List[str]]]:
        """Load event type assignments from JSON file."""

        path = Path(input_path)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle nested format: {conversation_id: {turn_id: [labels]}}
        assignments: Dict[str, Dict[str, List[str]]] = {}
        for conv_id, turn_assignments in data.items():
            normalised: Dict[str, List[str]] = {}
            for turn_key, labels in turn_assignments.items():
                turn_id = str(turn_key).strip()
                if not turn_id:
                    continue
                normalised[turn_id] = labels
            assignments[conv_id] = normalised

        logger.info("Loaded event type assignments from %s", input_path)
        return assignments

    def _group_turns_by_conversation(
        self,
        turns: Sequence[Turn],
    ) -> Iterator[Tuple[str, List[Turn]]]:
        """Group turns by conversation ID."""

        conversations: Dict[str, List[Turn]] = {}
        for turn in turns:
            turn_id = str(turn.id)
            conversation_id = self._conversation_from_turn_id(turn_id)
            conversations.setdefault(conversation_id, []).append(turn)

        for conversation_id, dialogue_turns in conversations.items():
            yield conversation_id, dialogue_turns

    def _conversation_from_turn_id(self, turn_id: str) -> str:
        return common_conversation_from_turn_id(turn_id)

    def _build_turn_groups(
        self,
        turns: Sequence[Turn],
    ) -> List[List[Tuple[int, Turn]]]:
        """Return either grouped or per-turn units based on configuration."""

        if not self._group_consecutive_turns:
            return [[(idx, turn)] for idx, turn in enumerate(turns, start=1)]
        return self._group_consecutive_turns_by_role(turns)

    @staticmethod
    def _group_consecutive_turns_by_role(
        turns: Sequence[Turn],
    ) -> List[List[Tuple[int, Turn]]]:
        """Group consecutive turns that share the same role."""

        if not turns:
            return []

        groups: List[List[Tuple[int, Turn]]] = []
        current_group: List[Tuple[int, Turn]] = []
        current_role: Optional[str] = None

        for turn_index, turn in enumerate(turns, start=1):
            role = str(turn.role)

            if current_role is None or role != current_role:
                if current_group:
                    groups.append(current_group)
                current_group = [(turn_index, turn)]
                current_role = role
            else:
                current_group.append((turn_index, turn))

        if current_group:
            groups.append(current_group)

        return groups


def _load_config(
    config_path: str | Path,
    output_override: Optional[str],
    max_workers_override: Optional[int],
) -> Tuple[str, str, LanguageModelProviderConfig, EmbeddingModelProviderConfig, Path, int]:
    """Load and validate the event type labeler configuration."""
    config_path = Path(config_path)
    config_data = load_config(config_path)

    raw_cfg = config_data.get("event_type_labeler_config")
    if not isinstance(raw_cfg, dict):
        raise ValueError("event_type_labeler_config section is required in the config")

    cfg = ParseDict(raw_cfg, EventTypeLabelerConfig())

    if not cfg.turns_jsonl_path:
        raise ValueError("turns_jsonl_path must be provided in event_type_labeler_config")
    turns_path_str = cfg.turns_jsonl_path

    dataset_name = cfg.dataset_name

    lm_config = cfg.language_model_provider_config
    if lm_config.provider == LanguageModelProvider.LANGUAGE_MODEL_PROVIDER_UNSPECIFIED:
        raise ValueError("language_model_provider_config must specify a provider")

    if not cfg.HasField("encoder_config"):
        raise ValueError("encoder_config must be provided in event_type_labeler_config")
    encoder_config = cfg.encoder_config
    if encoder_config.provider == EmbeddingModelProvider.EMBEDDING_MODEL_PROVIDER_UNSPECIFIED:
        raise ValueError("encoder_config must specify an embedding provider")

    if output_override is not None:
        output_path = Path(output_override)
    elif cfg.event_type_assignments_output_path:
        output_path = Path(cfg.event_type_assignments_output_path)
    else:
        output_path = config_path.with_name(f"{config_path.stem}_event_type_assignments.json")

    # Get max_workers from override, config, or default
    if max_workers_override is not None:
        max_workers = max_workers_override
    elif cfg.max_workers > 0:
        max_workers = cfg.max_workers
    else:
        max_workers = 5  # Default value

    return dataset_name, turns_path_str, lm_config, encoder_config, output_path, max_workers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover and assign event type labels to conversation turns.",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to event type labeler JSON config file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output JSON path; overrides event_type_assignments_output_path in the config.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of concurrent workers for parallel processing; overrides config value.",
    )
    parser.add_argument(
        "--group-consecutive-turns",
        dest="group_consecutive_turns",
        action="store_true",
        help="Group consecutive turns by role before processing (default).",
    )
    parser.add_argument(
        "--no-group-consecutive-turns",
        dest="group_consecutive_turns",
        action="store_false",
        help="Disable role-based grouping and process every turn individually.",
    )
    parser.set_defaults(group_consecutive_turns=True)
    parser.add_argument(
        "--max-conversations",
        type=int,
        help="Limit processing to the first N conversations (useful for large datasets).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )

    args = parser.parse_args()

    dataset_name, turns_path, lm_config, encoder_config, output_path, max_workers = _load_config(
        args.config,
        args.output,
        args.max_workers,
    )

    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_path} already exists. Re-run with --overwrite to replace it."
            )
        else:
            # Delete existing file if overwrite is set
            output_path.unlink()
            logger.info("Deleted existing output file: %s", output_path)

    turns = load_turns(dataset_name, turns_path)
    logger.info(
        "Loaded %d turns for dataset '%s' from %s",
        len(turns),
        dataset_name or "<unspecified>",
        turns_path,
    )

    labeler = EventTypeLabeler(
        lm_config=lm_config,
        encoder_config=encoder_config,
        dataset_type=dataset_name,
        max_workers=max_workers,
        group_consecutive_turns=args.group_consecutive_turns,
    )

    logger.info("Starting event type discovery and assignment...")
    assignments = labeler.discover_and_assign_event_types(
        turns=turns,
        output_path=output_path,
        max_conversations=args.max_conversations,
    )

    total_conversations = len(assignments)
    total_turns = sum(len(turn_assignments) for turn_assignments in assignments.values())

    print("\n" + "=" * 70)
    print("EVENT TYPE LABELING SUMMARY:")
    print(f"  Total conversations: {total_conversations}")
    print(f"  Total turns labeled: {total_turns}")
    print(f"  Max concurrent workers: {max_workers}")
    print(f"  Output saved to: {output_path}")
    print("=" * 70 + "\n")

    total_cost = get_lm_cost(labeler.lm)
    logger.info("Event type labeling LM cost: $%.4f", total_cost)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    main()
