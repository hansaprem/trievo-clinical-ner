"""Sliding Window Tokenization and Chunking for Long Texts.

Enables processing texts of arbitrary length exceeding PubMedBERT's
512-subword position embedding limit using overlapping sliding token windows.
Character offsets are strictly preserved against the original input document.
"""

from dataclasses import dataclass
from typing import List, Tuple
import torch
from transformers import AutoTokenizer


@dataclass
class TokenWindowChunk:
    """Represents a single tokenized window chunk with offset mappings."""
    chunk_index: int
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    offset_mapping: List[Tuple[int, int]]


class SlidingWindowTokenizer:
    """Splits input text into overlapping token windows with exact character offset preservation."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_window_size: int = 384,
        stride: int = 64,
    ):
        assert max_window_size <= 512, "Window size must not exceed model max position (512)"
        assert stride < max_window_size, "Stride must be strictly less than window size"
        self.tokenizer = tokenizer
        self.max_window_size = max_window_size
        self.stride = stride

    def tokenize_text(self, text: str) -> List[TokenWindowChunk]:
        """
        Tokenizes text into one or more overlapping window chunks.

        Args:
            text: Raw input string.

        Returns:
            List of TokenWindowChunk objects.
        """
        if not text or not text.strip():
            return []

        # Do not use return_tensors='pt' here because the last overflowing chunk
        # may have fewer tokens than max_window_size, which causes PyTorch tensor stacking to fail.
        encoded = self.tokenizer(
            text,
            max_length=self.max_window_size,
            stride=self.stride,
            truncation=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )

        num_chunks = len(encoded["input_ids"])
        chunks = []

        for i in range(num_chunks):
            raw_offsets = encoded["offset_mapping"][i]
            offset_list = [(int(o[0]), int(o[1])) for o in raw_offsets]

            input_ids_tensor = torch.tensor(encoded["input_ids"][i], dtype=torch.long).unsqueeze(0)
            attn_mask_tensor = torch.tensor(encoded["attention_mask"][i], dtype=torch.long).unsqueeze(0)

            chunks.append(
                TokenWindowChunk(
                    chunk_index=i,
                    input_ids=input_ids_tensor,
                    attention_mask=attn_mask_tensor,
                    offset_mapping=offset_list,
                )
            )

        return chunks
