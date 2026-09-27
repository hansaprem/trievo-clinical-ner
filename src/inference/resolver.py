"""Conflict Resolution and Entity Merging Module.

Implements deterministic policies to resolve duplicates, overlapping spans,
and multi-model entity predictions without relying on uncalibrated cross-model
confidence comparisons.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from src.inference.adapter import RawEntitySpan
from src.inference.normalizer import normalize_label


@dataclass
class UnifiedEntity:
    """Unified entity representation conforming to benchmark requirements."""
    text: str
    label: str
    start: int
    end: int
    source_model: str
    confidence: float
    source_models: List[str] = field(default_factory=list)
    model_confidences: Dict[str, float] = field(default_factory=dict)
    sub_spans: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Converts to dictionary strictly adhering to the user schema."""
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "source_model": self.source_model,
            "confidence": self.confidence,
        }

    def to_extended_dict(self) -> Dict:
        """Extended dictionary including full provenance details."""
        d = self.to_dict()
        d["source_models"] = self.source_models
        d["model_confidences"] = self.model_confidences
        if self.sub_spans:
            d["sub_spans"] = self.sub_spans
        return d


def spans_overlap(s1: Tuple[int, int], s2: Tuple[int, int]) -> bool:
    """Checks if character spans [s1[0], s1[1]) and [s2[0], s2[1]) overlap."""
    return max(s1[0], s2[0]) < min(s1[1], s2[1])


def is_nested(inner: Tuple[int, int], outer: Tuple[int, int]) -> bool:
    """Checks if inner span is strictly or loosely contained within outer span."""
    return outer[0] <= inner[0] and inner[1] <= outer[1] and (inner != outer)


class EntityResolver:
    """
    Deterministic conflict resolution engine for multi-model Clinical NER.

    Policies:
      1. Identical Spans, Same Label:
         Merged into a single entity with joint source_models provenance and average confidence.
      2. Overlapping Spans, Same Label:
         Deterministically selects the maximal (longest) span to preserve critical clinical
         modifiers (e.g. 'acute', 'chronic'), recording shorter spans in provenance.
      3. Nested Spans, Different Labels:
         Both entities are preserved (e.g. 'prostate' [ANATOMY] within 'prostate cancer' [DISEASE],
         or 'estrogen' [CHEMICAL] within 'estrogen receptor' [PROTEIN]).
      4. Identical Spans, Different Labels:
         Both entities are preserved with their respective source models and labels.
      5. Partial Overlapping Spans, Different Labels:
         Both entities are preserved with an overlap notation, avoiding uncalibrated score comparisons.
    """

    def resolve(
        self,
        raw_spans: List[RawEntitySpan],
        full_text: str,
    ) -> List[UnifiedEntity]:
        """
        Resolves conflicts and merges raw entity spans from multiple models.

        Args:
            raw_spans: Collection of RawEntitySpan objects from all executed models.
            full_text: Original raw input text.

        Returns:
            List of UnifiedEntity objects sorted by start character offset.
        """
        if not raw_spans:
            return []

        # Step 1: Normalize labels and wrap into initial UnifiedEntity candidates
        candidates: List[UnifiedEntity] = []
        for r in raw_spans:
            norm_lbl = normalize_label(r.raw_label)
            # Ensure text matches slice
            actual_text = full_text[r.start : r.end]
            candidates.append(
                UnifiedEntity(
                    text=actual_text,
                    label=norm_lbl,
                    start=r.start,
                    end=r.end,
                    source_model=r.source_model,
                    confidence=r.confidence,
                    source_models=[r.source_model],
                    model_confidences={r.source_model: r.confidence},
                )
            )

        # Step 2: Deduplicate identical spans with identical normalized labels
        candidates = self._deduplicate_identical_spans(candidates)

        # Step 3: Resolve same-label overlaps (prefer maximal clinical span)
        candidates = self._resolve_same_label_overlaps(candidates, full_text)

        # Step 4: Sort entities by start offset ascending, then end offset descending
        candidates.sort(key=lambda e: (e.start, -e.end, e.label))

        return candidates

    def _deduplicate_identical_spans(
        self, entities: List[UnifiedEntity]
    ) -> List[UnifiedEntity]:
        """Deduplicates exact identical character spans with the same label."""
        grouped: Dict[Tuple[int, int, str], List[UnifiedEntity]] = {}
        for ent in entities:
            key = (ent.start, ent.end, ent.label)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(ent)

        deduped: List[UnifiedEntity] = []
        for (start, end, label), group in grouped.items():
            if len(group) == 1:
                deduped.append(group[0])
            else:
                # Merge provenance
                all_models = sorted(list({m for g in group for m in g.source_models}))
                model_confs: Dict[str, float] = {}
                for g in group:
                    model_confs.update(g.model_confidences)
                avg_conf = round(float(np.mean(list(model_confs.values()))), 4)

                merged_source_str = ", ".join(all_models)
                deduped.append(
                    UnifiedEntity(
                        text=group[0].text,
                        label=label,
                        start=start,
                        end=end,
                        source_model=merged_source_str,
                        confidence=avg_conf,
                        source_models=all_models,
                        model_confidences=model_confs,
                    )
                )

        return deduped

    def _resolve_same_label_overlaps(
        self, entities: List[UnifiedEntity], full_text: str
    ) -> List[UnifiedEntity]:
        """
        Resolves overlapping spans that share the same normalized entity label.
        Prefers the maximal/longest span to capture full clinical condition/medication names.
        """
        # Group entities by normalized label
        by_label: Dict[str, List[UnifiedEntity]] = {}
        for ent in entities:
            by_label.setdefault(ent.label, []).append(ent)

        resolved_all: List[UnifiedEntity] = []

        for label, group in by_label.items():
            if len(group) <= 1:
                resolved_all.extend(group)
                continue

            # Sort by start ascending, then length descending
            group.sort(key=lambda e: (e.start, -(e.end - e.start)))

            merged_group: List[UnifiedEntity] = []
            for current in group:
                if not merged_group:
                    merged_group.append(current)
                    continue

                prev = merged_group[-1]

                # Check if overlapping
                if spans_overlap((prev.start, prev.end), (current.start, current.end)):
                    # Case A: current is completely inside prev (prev is longer)
                    if prev.start <= current.start and prev.end >= current.end:
                        # Record current as a sub-span of prev
                        prev.sub_spans.append({
                            "text": current.text,
                            "start": current.start,
                            "end": current.end,
                            "source_models": current.source_models,
                            "confidence": current.confidence,
                        })
                        # Add any models from current not in prev
                        for m in current.source_models:
                            if m not in prev.source_models:
                                prev.source_models.append(m)
                                prev.source_model = ", ".join(sorted(prev.source_models))
                        prev.model_confidences.update(current.model_confidences)
                        continue

                    # Case B: prev is completely inside current (current is longer)
                    elif current.start <= prev.start and current.end >= prev.end:
                        current.sub_spans.append({
                            "text": prev.text,
                            "start": prev.start,
                            "end": prev.end,
                            "source_models": prev.source_models,
                            "confidence": prev.confidence,
                        })
                        for m in prev.source_models:
                            if m not in current.source_models:
                                current.source_models.append(m)
                                current.source_model = ", ".join(sorted(current.source_models))
                        current.model_confidences.update(prev.model_confidences)
                        merged_group[-1] = current
                        continue

                    # Case C: Partial overlap across boundaries (e.g. [0, 10] and [5, 15])
                    else:
                        # Expand to maximal envelope
                        max_start = min(prev.start, current.start)
                        max_end = max(prev.end, current.end)
                        combined_models = sorted(list(set(prev.source_models + current.source_models)))
                        combined_confs = {**prev.model_confidences, **current.model_confidences}
                        avg_c = round(float(np.mean(list(combined_confs.values()))), 4)

                        merged_group[-1] = UnifiedEntity(
                            text=full_text[max_start:max_end],
                            label=label,
                            start=max_start,
                            end=max_end,
                            source_model=", ".join(combined_models),
                            confidence=avg_c,
                            source_models=combined_models,
                            model_confidences=combined_confs,
                            sub_spans=[
                                {
                                    "text": prev.text,
                                    "start": prev.start,
                                    "end": prev.end,
                                    "source_models": prev.source_models,
                                },
                                {
                                    "text": current.text,
                                    "start": current.start,
                                    "end": current.end,
                                    "source_models": current.source_models,
                                },
                            ],
                        )
                        continue

                merged_group.append(current)

            resolved_all.extend(merged_group)

        return resolved_all
