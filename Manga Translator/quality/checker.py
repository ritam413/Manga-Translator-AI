"""
quality/checker.py
------------------
QualityChecker — the orchestrating engine for Stage 4.

It applies a configurable, ordered pipeline of heuristic filters to every
TextRegion in a PageManifest and produces a CheckReport summarising the
outcome.  The QualityChecker is *pure data*: it never writes to disk and
never loads a model.  The calling stage controller is responsible for
persisting manifest changes.

Usage:
    from quality.checker import QualityChecker, CheckReport
    from quality.heuristics import (
        BoundingBoxSizeFilter, AspectRatioFilter, OcrConfidenceFilter,
        TextDensityFilter, DuplicateRegionFilter, GarbageTextFilter,
    )
    from core.models import PageManifest

    checker = QualityChecker()          # uses all default filters
    report  = checker.check_page(page)  # returns CheckReport, mutates page in-place
    print(report.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from core.models import PageManifest, TextRegion
from quality.heuristics import (
    BaseHeuristic,
    HeuristicResult,
    AspectRatioFilter,
    BoundingBoxSizeFilter,
    DuplicateRegionFilter,
    GarbageTextFilter,
    OcrConfidenceFilter,
    TextDensityFilter,
)

logger = logging.getLogger("quality.checker")


# ──────────────────────────────────────────────────────────────────────────────
# Default filter pipeline factory
# ──────────────────────────────────────────────────────────────────────────────

def _default_filters() -> List[BaseHeuristic]:
    """Return a fresh default heuristic pipeline in evaluation order."""
    return [
        BoundingBoxSizeFilter(),       # 1. reject obviously bad boxes first
        AspectRatioFilter(),           # 2. aspect-ratio sanity
        OcrConfidenceFilter(),         # 3. OCR confidence gate
        TextDensityFilter(),           # 4. text-density plausibility
        GarbageTextFilter(),           # 5. garbage-character check
        DuplicateRegionFilter(),       # 6. duplicate/overlap detection (stateful)
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Per-region result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RegionCheckResult:
    """Aggregated quality-check outcome for a single TextRegion."""
    region_id: str
    accepted: bool                             # False → region was rejected
    needs_review: bool                         # True → accepted but flagged
    filter_results: List[HeuristicResult] = field(default_factory=list)

    @property
    def rejection_reason(self) -> str:
        """Human-readable reason for rejection (empty string if accepted)."""
        for fr in self.filter_results:
            if not fr.passed:
                return f"[{fr.filter_name}] {fr.reason}"
        return ""

    @property
    def review_reasons(self) -> List[str]:
        """List of all reasons why the region was flagged for review."""
        return [
            f"[{fr.filter_name}] {fr.reason}"
            for fr in self.filter_results
            if fr.flag_review and fr.reason
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Page-level report
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CheckReport:
    """
    Full quality-check report for one PageManifest run.

    Attributes:
        image_filename: The filename of the page that was checked.
        region_results: Per-region outcomes keyed by region ID.
        rejected_ids:   IDs of regions that failed at least one filter.
        flagged_ids:    IDs of regions that passed but need human review.
    """
    image_filename: str
    region_results: Dict[str, RegionCheckResult] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.region_results)

    @property
    def rejected_ids(self) -> List[str]:
        return [rid for rid, r in self.region_results.items() if not r.accepted]

    @property
    def flagged_ids(self) -> List[str]:
        return [rid for rid, r in self.region_results.items() if r.accepted and r.needs_review]

    @property
    def accepted_ids(self) -> List[str]:
        return [rid for rid, r in self.region_results.items() if r.accepted]

    def summary(self) -> str:
        """Return a compact one-line summary of the report."""
        return (
            f"[{self.image_filename}] "
            f"total={self.total} "
            f"accepted={len(self.accepted_ids)} "
            f"flagged={len(self.flagged_ids)} "
            f"rejected={len(self.rejected_ids)}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# QualityChecker
# ──────────────────────────────────────────────────────────────────────────────

class QualityChecker:
    """
    Orchestrates the quality-control heuristic pipeline for the Manga Translator.

    The checker runs a configurable list of :class:`BaseHeuristic` filters
    sequentially against each TextRegion in a PageManifest.  Evaluation short-
    circuits on the first *failing* (``passed=False``) result, but all
    *review-flagging* results are accumulated.

    Rejected regions are *removed* from the page's text_regions list.
    Regions that pass but trigger review flags have their ``needs_review``
    attribute set to True.

    Args:
        filters: Ordered list of heuristic filters.  Defaults to the standard
                 pipeline when None (recommended for production use).
        remove_rejected: If True (default), remove rejected regions from the
                         PageManifest in-place.  Set to False for dry-run audits.
    """

    def __init__(
        self,
        filters: Optional[Sequence[BaseHeuristic]] = None,
        remove_rejected: bool = True,
    ) -> None:
        self.filters: List[BaseHeuristic] = (
            list(filters) if filters is not None else _default_filters()
        )
        self.remove_rejected = remove_rejected

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_page(self, page: PageManifest) -> CheckReport:
        """
        Run quality control on all TextRegions in a PageManifest.

        The manifest is modified in-place (rejected regions removed, review
        flags set) unless ``remove_rejected=False``.

        Args:
            page: The PageManifest whose text_regions will be evaluated.

        Returns:
            A :class:`CheckReport` summarising the outcome for every region.
        """
        report = CheckReport(image_filename=page.image_filename)

        # Reset any stateful filters (e.g. DuplicateRegionFilter)
        self._reset_stateful_filters()

        regions_to_keep: List[TextRegion] = []

        for region in page.text_regions:
            result = self._check_region(region, page)
            report.region_results[region.id] = result

            if result.accepted:
                # Apply needs_review flag if any filter requested it
                if result.needs_review:
                    region.needs_review = True
                regions_to_keep.append(region)
                logger.debug(
                    "ACCEPTED region '%s' on '%s'%s",
                    region.id,
                    page.image_filename,
                    " [needs_review]" if result.needs_review else "",
                )
            else:
                logger.info(
                    "REJECTED region '%s' on '%s': %s",
                    region.id,
                    page.image_filename,
                    result.rejection_reason,
                )

        if self.remove_rejected:
            page.text_regions = regions_to_keep

        logger.info("Quality check complete — %s", report.summary())
        return report

    def check_region(
        self,
        region: TextRegion,
        page: Optional[PageManifest] = None,
    ) -> RegionCheckResult:
        """
        Run quality control on a single TextRegion (stateless, no mutation).

        Useful for pre-validation before inserting a new region into the manifest.
        Note: DuplicateRegionFilter state is NOT updated by this method.

        Args:
            region: The region to evaluate.
            page:   Optional page for context-aware filters.

        Returns:
            RegionCheckResult describing the outcome.
        """
        return self._check_region(region, page)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_region(
        self,
        region: TextRegion,
        page: Optional[PageManifest],
    ) -> RegionCheckResult:
        """Apply all filters to one region; short-circuit on first hard fail."""
        filter_results: List[HeuristicResult] = []
        needs_review = region.needs_review  # preserve existing flag
        accepted = True

        for f in self.filters:
            try:
                result = f.check(region, page)
            except Exception as exc:          # never crash the pipeline
                logger.error(
                    "Filter '%s' raised an exception on region '%s': %s",
                    f.name,
                    region.id,
                    exc,
                    exc_info=True,
                )
                # Treat unexpected exceptions as a soft flag, not hard rejection
                result = HeuristicResult(
                    filter_name=f.name,
                    passed=True,
                    reason=f"Filter exception (check logs): {exc}",
                    flag_review=True,
                )

            filter_results.append(result)

            if result.flag_review:
                needs_review = True

            if not result.passed:
                accepted = False
                break  # short-circuit; no point running later filters

        return RegionCheckResult(
            region_id=region.id,
            accepted=accepted,
            needs_review=needs_review,
            filter_results=filter_results,
        )

    def _reset_stateful_filters(self) -> None:
        """Call reset() on any filter that exposes it (e.g. DuplicateRegionFilter)."""
        for f in self.filters:
            if hasattr(f, "reset") and callable(f.reset):
                f.reset()
