"""
CS336 Assignment 5 - Metric Parsing Functions.

Implements:
- parse_mmlu_response: Extract A/B/C/D from model output
- parse_gsm8k_response: Extract last number from model output
"""

import re
from typing import Any


def parse_mmlu_response(mmlu_example: dict[str, Any], model_output: str) -> str | None:
    """Parse MMLU model output into predicted answer letter (A/B/C/D)."""
    # Try to find pattern like "The correct answer is X" or just a standalone letter
    # Strategy 1: Look for "correct answer is X"
    match = re.search(r"(?:correct answer is|answer is)\s*([A-D])", model_output, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Strategy 2: Look for standalone letter at the beginning or after common prefixes
    match = re.search(r"^([A-D])[\.\)\s,:]", model_output.strip())
    if match:
        return match.group(1).upper()

    # Strategy 3: Look for any single letter A-D mentioned
    matches = re.findall(r"\b([A-D])\b", model_output)
    if len(matches) == 1:
        return matches[0].upper()

    return None


def parse_gsm8k_response(model_output: str) -> str | None:
    """Parse GSM8K model output by taking the last number in the output."""
    # Find all numbers (integers and decimals, possibly negative)
    # Match patterns like: 72, 3.14, -5, 1,000, etc.
    numbers = re.findall(r"-?[\d,]+\.?\d*", model_output)
    if not numbers:
        return None

    # Take the last number and clean it (remove commas)
    last_number = numbers[-1].replace(",", "")
    return last_number
