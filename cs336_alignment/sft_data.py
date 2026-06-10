"""
CS336 Assignment 5 - SFT Data Loading.

Implements:
- PackedSFTDataset: PyTorch Dataset for packed instruction tuning
- iterate_batches: Batch iterator for the dataset
"""

import gzip
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


# Alpaca SFT template
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n{response}"
)


class PackedSFTDataset(Dataset):
    """Packed instruction-tuning dataset.
    
    Concatenates all tokenized documents (separated by EOS) and splits
    into non-overlapping chunks of seq_length.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | Path,
        seq_length: int,
        shuffle: bool = False,
    ):
        self.seq_length = seq_length
        dataset_path = Path(dataset_path)

        # Load data
        examples = []
        opener = gzip.open if str(dataset_path).endswith(".gz") else open
        with opener(dataset_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

        if shuffle:
            random.shuffle(examples)

        # Format and tokenize all documents, concatenate with EOS delimiter
        # Each document gets BOS prepended (add_special_tokens=True) and EOS appended
        eos_id = tokenizer.eos_token_id
        all_token_ids = []
        for ex in examples:
            text = ALPACA_TEMPLATE.format(
                instruction=ex["prompt"],
                response=ex["response"],
            )
            token_ids = tokenizer.encode(text, add_special_tokens=True)
            all_token_ids.extend(token_ids)
            all_token_ids.append(eos_id)

        # Take consecutive non-overlapping chunks of size seq_length for input_ids
        # labels are shifted by 1, so we need seq_length + 1 tokens total per chunk
        # But the spec says: "take consecutive, non-overlapping chunks of size m"
        # and input_ids/labels are both (seq_length,) — input = chunk[:-1], label = chunk[1:]
        # Wait - re-reading: chunks of size m where each batch input is [0,1,2,3] for seq_length=4
        # So chunks are of size seq_length, and labels are offset by 1 from the STREAM
        # i.e., input = all_tokens[i*m : (i+1)*m], label = all_tokens[i*m+1 : (i+1)*m+1]
        # This means we need (n_chunks * seq_length + 1) tokens total
        n_chunks = (len(all_token_ids) - 1) // seq_length
        self.chunks = []
        for i in range(n_chunks):
            start = i * seq_length
            input_chunk = all_token_ids[start:start + seq_length]
            label_chunk = all_token_ids[start + 1:start + seq_length + 1]
            self.chunks.append((input_chunk, label_chunk))

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        input_chunk, label_chunk = self.chunks[idx]
        input_ids = torch.tensor(input_chunk, dtype=torch.long)
        labels = torch.tensor(label_chunk, dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    """Return a DataLoader that iterates over the dataset in batches."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )
