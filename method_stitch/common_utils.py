"""Common utility functions for hierarchical context reduction methods."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def _extract_turn_index_from_id(turn_id: str) -> Optional[int]:
    """Extract numeric turn index from a turn_id string.

    Supports legacy ids such as ``trip-xxx-turn-123`` as well as newer formats
    used in locomo datasets (e.g., ``conv-1-D4:3``) by falling back to the last
    numeric token present in the identifier.
    """
    if not turn_id:
        return None
    
    # Try to match pattern like "trip-xxx-turn-123"
    if "-turn-" in turn_id:
        parts = turn_id.split("-turn-")
        if len(parts) == 2:
            try:
                return int(parts[1])
            except (ValueError, TypeError):
                return None
    
    # Fall back to the last numeric token in the id (handles "conv-1-D4:3")
    match = re.search(r"(\d+)(?!.*\d)", turn_id)
    if match:
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None

    return None


def _conversation_from_turn_id(turn_id: str) -> str:
    """Extract a conversation identifier from a turn id.

    Supports legacy ids that contain "-turn-" as well as newer datasets
    (e.g., locomo10) whose ids follow formats such as:
        - "conv-1-D4:3" (turns)
        - "conv-1-question-1" (questions)

    Args:
        turn_id: The raw turn identifier string.

    Returns:
        The derived conversation identifier.
    """
    if not turn_id:
        return turn_id

    if "-turn-" in turn_id:
        return turn_id.split("-turn-", 1)[0]

    if turn_id.startswith("conv-"):
        match = re.match(r"^(conv-\d+)", turn_id)
        if match:
            return match.group(1)
        if "-question-" in turn_id:
            return turn_id.split("-question-", 1)[0]
        if "-D" in turn_id:
            return turn_id.split("-D", 1)[0]
        if "-process_" in turn_id:
            return turn_id.split("-process_", 1)[0]
        if ":" in turn_id:
            return turn_id.split(":", 1)[0]

    return turn_id


def load_structured_turn_notes(notes_jsonl_path: str) -> Dict[Union[str, int], Dict[str, Any]]:
    """Load structured turn notes and create a lookup by turn_id.
    
    Args:
        notes_jsonl_path: Path to structured turn notes JSONL file
        
    Returns:
        Dictionary mapping turn_id (str) to note record (dict). For backward
        compatibility, also includes a numeric alias key when a stable numeric
        index can be extracted and no collision is detected.
    """
    notes_lookup: Dict[Union[str, int], Dict[str, Any]] = {}
    
    notes_path = Path(notes_jsonl_path)
    if not notes_path.exists():
        logger.warning("Structured notes file not found: %s", notes_jsonl_path)
        return notes_lookup
    
    with notes_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                turn_id = str(record.get("turn_id") or "").strip()
                if not turn_id:
                    continue

                # Store by string turn_id to preserve dataset-specific ids
                notes_lookup[turn_id] = record

                # Best-effort numeric alias (only if not already present)
                turn_index = _extract_turn_index_from_id(turn_id)
                if turn_index is not None and turn_index not in notes_lookup:
                    notes_lookup[turn_index] = record
            except json.JSONDecodeError:
                continue
    
    logger.info("Loaded %d structured turn notes from %s", len(notes_lookup), notes_jsonl_path)
    return notes_lookup


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while handling float numbers correctly.
    
    This helper function carefully handles punctuation marks and float numbers 
    (e.g., 4.4, 35.43) to avoid splitting sentences incorrectly.
    
    Args:
        text: The input text to split
        
    Returns:
        List of sentences
    """
    if not text or not text.strip():
        return []
    
    text = text.strip()
    
    # Split into sentences while being careful about float numbers
    # Pattern explanation:
    # - Look for sentence-ending punctuation: . ! ?
    # - Followed by whitespace
    # - NOT preceded by a digit (to avoid splitting on floats like 4.4)
    # Use negative lookbehind (?<!\d) to check that the period is not after a digit
    # Use positive lookahead (?=\s) to ensure there's whitespace after the punctuation
    sentence_pattern = r'(?<![0-9])([.!?])(?=\s+|$)'
    
    # Find all sentence boundaries
    matches = list(re.finditer(sentence_pattern, text))
    
    if not matches:
        # No sentence boundaries found, return the whole text as one sentence
        return [text]
    
    sentences: List[str] = []
    start = 0
    
    for match in matches:
        end = match.end()
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
    
    # Add any remaining text as the last sentence
    if start < len(text):
        remaining = text[start:].strip()
        if remaining:
            sentences.append(remaining)
    
    return sentences if sentences else [text]


def extract_first_sentence(text: str) -> str:
    """Extract the first sentence from text.
    
    This function carefully handles punctuation marks and float numbers (e.g., 4.4, 35.43)
    to avoid splitting sentences incorrectly.
    
    Args:
        text: The input text
        
    Returns:
        First sentence from the text
    """
    if not text or not text.strip():
        return ""
    
    sentences = _split_into_sentences(text)
    return sentences[0] if sentences else text


def extract_abbreviated_utterance(text: str) -> str:
    """Extract first 2 sentences and last 1 sentence from text.
    
    This function carefully handles punctuation marks and float numbers (e.g., 4.4, 35.43)
    to avoid splitting sentences incorrectly.
    
    Args:
        text: The input text to abbreviate
        
    Returns:
        Abbreviated text containing first 2 + last 1 sentences
    """
    if not text or not text.strip():
        return ""
    
    sentences = _split_into_sentences(text)
    
    if not sentences:
        return text
    
    # Extract first 2 and last 1 sentences
    if len(sentences) <= 3:
        # If 3 or fewer sentences, return all
        return " ".join(sentences)
    else:
        # Get first 2 sentences
        first_two = sentences[:2]
        # Get last 1 sentence
        last_one = [sentences[-1]]
        # Combine with ellipsis to indicate omission
        return " ".join(first_two) + " ... " + " ".join(last_one)
