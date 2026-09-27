"""Model Inference Adapter for Clinical NER.

Wraps a single LoadedModel to execute inference over windowed chunks,
extract BIO entities, and map them to exact character offsets with authentic confidence.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import torch

from src.inference.model_loader import LoadedModel
from src.inference.sliding_window import SlidingWindowTokenizer, TokenWindowChunk


@dataclass
class RawEntitySpan:
    """Represents an extracted entity span before cross-model merging."""
    text: str
    raw_label: str
    start: int
    end: int
    source_model: str
    confidence: float
    token_count: int


class ModelInferenceAdapter:
    """Executes token classification inference for a single model across token windows."""

    def __init__(
        self,
        loaded_model: LoadedModel,
        window_size: int = 384,
        stride: int = 64,
    ):
        self.loaded_model = loaded_model
        self.model_name = loaded_model.name
        self.model = loaded_model.model
        self.tokenizer = loaded_model.tokenizer
        self.id2label = loaded_model.id2label
        self.device = loaded_model.device
        self.window_tokenizer = SlidingWindowTokenizer(
            tokenizer=self.tokenizer,
            max_window_size=window_size,
            stride=stride,
        )

    def extract_entities(self, text: str) -> List[RawEntitySpan]:
        """
        Runs model inference on text and returns extracted raw entity spans.

        Args:
            text: Full raw input text.

        Returns:
            List of RawEntitySpan objects with exact character offsets.
        """
        if not text or not text.strip():
            return []

        chunks = self.window_tokenizer.tokenize_text(text)
        if not chunks:
            return []

        chunk_entities: List[RawEntitySpan] = []

        for chunk in chunks:
            entities_in_chunk = self._infer_chunk(chunk, text)
            chunk_entities.extend(entities_in_chunk)

        # Deduplicate identical or boundary-overlapping entities across chunks of the same model
        deduped = self._deduplicate_chunk_entities(chunk_entities, text)
        return deduped

    def _infer_chunk(self, chunk: TokenWindowChunk, full_text: str) -> List[RawEntitySpan]:
        """Runs inference on a single token window and extracts entity spans."""
        input_ids = chunk.input_ids.to(self.device)
        attention_mask = chunk.attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]  # shape: (seq_len, num_labels)
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            predicted_ids = np.argmax(probabilities, axis=-1)

        extracted: List[RawEntitySpan] = []
        current_entity: Optional[Dict] = None

        for idx, (p_id, (start_char, end_char)) in enumerate(zip(predicted_ids, chunk.offset_mapping)):
            # Skip special tokens ([CLS], [SEP], [PAD])
            if start_char == end_char:
                continue

            tag = self.id2label.get(int(p_id), "O")
            token_prob = float(probabilities[idx, p_id])

            if tag != "O":
                ent_type = tag[2:] if ("-" in tag) else tag
                is_b = tag.startswith("B-")
                is_i = tag.startswith("I-")

                # Check if this token continues the current entity:
                is_subword_continuation = (
                    current_entity is not None
                    and current_entity["label"] == ent_type
                    and start_char == current_entity["end"]
                )
                is_inside_tag = (
                    current_entity is not None
                    and current_entity["label"] == ent_type
                    and is_i
                )

                if is_subword_continuation or is_inside_tag:
                    current_entity["end"] = int(end_char)
                    current_entity["token_probs"].append(token_prob)
                else:
                    if current_entity:
                        span = self._finalize_span(current_entity, full_text)
                        if span:
                            extracted.append(span)

                    current_entity = {
                        "label": ent_type,
                        "start": int(start_char),
                        "end": int(end_char),
                        "token_probs": [token_prob],
                    }
            else:
                # Outside 'O'
                if current_entity:
                    span = self._finalize_span(current_entity, full_text)
                    if span:
                        extracted.append(span)
                    current_entity = None

        if current_entity:
            span = self._finalize_span(current_entity, full_text)
            if span:
                extracted.append(span)

        return extracted

    def _finalize_span(self, ent: Dict, full_text: str) -> Optional[RawEntitySpan]:
        """Validates offsets, calculates confidence, and creates RawEntitySpan."""
        start = ent["start"]
        end = ent["end"]

        # Safeguard offset bounds
        if start < 0 or end > len(full_text) or start >= end:
            return None

        span_text = full_text[start:end]
        avg_confidence = float(np.mean(ent["token_probs"])) if ent["token_probs"] else 0.0

        return RawEntitySpan(
            text=span_text,
            raw_label=ent["label"],
            start=start,
            end=end,
            source_model=self.model_name,
            confidence=round(avg_confidence, 4),
            token_count=len(ent["token_probs"]),
        )

    def _deduplicate_chunk_entities(
        self,
        entities: List[RawEntitySpan],
        full_text: str,
    ) -> List[RawEntitySpan]:
        """
        Deduplicates entity spans found across overlapping window chunks for this model.
        Prefers the longer span if one chunk captured a boundary fragment, or the higher
        confidence span if spans are identical.
        """
        if len(entities) <= 1:
            return entities

        # Sort primarily by start offset ascending, then length descending
        entities.sort(key=lambda e: (e.start, -(e.end - e.start)))

        merged: List[RawEntitySpan] = []
        for ent in entities:
            if not merged:
                merged.append(ent)
                continue

            prev = merged[-1]

            # 1. Exact identical span and label
            if prev.start == ent.start and prev.end == ent.end and prev.raw_label == ent.raw_label:
                # Keep the one with higher confidence
                if ent.confidence > prev.confidence:
                    merged[-1] = ent
                continue

            # 2. Same label, overlapping spans across chunk seam
            if prev.raw_label == ent.raw_label and max(prev.start, ent.start) < min(prev.end, ent.end):
                # If one is a complete sub-span of the other, keep the longer span
                if prev.start <= ent.start and prev.end >= ent.end:
                    continue  # prev already contains ent
                elif ent.start <= prev.start and ent.end >= prev.end:
                    merged[-1] = ent  # ent contains prev
                    continue
                else:
                    # Partial overlapping seam: merge span boundaries
                    new_start = min(prev.start, ent.start)
                    new_end = max(prev.end, ent.end)
                    new_conf = round(float(np.mean([prev.confidence, ent.confidence])), 4)
                    merged[-1] = RawEntitySpan(
                        text=full_text[new_start:new_end],
                        raw_label=prev.raw_label,
                        start=new_start,
                        end=new_end,
                        source_model=self.model_name,
                        confidence=new_conf,
                        token_count=prev.token_count + ent.token_count,
                    )
                    continue

            merged.append(ent)

        return merged
