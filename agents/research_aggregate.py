from __future__ import annotations

import logging
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from harness.base import AgentHarness, ContractViolationError
from schemas.common import ResearchFinding
from schemas.planning import ResearchExecutionPlan
from schemas.research import ConflictRecord, DimensionResearchResult, MultiDimensionResearchResult
from schemas.research_partials import ResearchPartialBatch
from schemas.state import GraphState


class ResearchAggregateInput(BaseModel):
    plan: ResearchExecutionPlan
    partials: list[ResearchPartialBatch] = Field(min_length=1, max_length=10)


class ResearchAggregationHarness(AgentHarness[ResearchAggregateInput, MultiDimensionResearchResult]):
    def __init__(
        self,
        *,
        logger: logging.Logger,
        max_retries: int,
        timeout_s: float,
    ) -> None:
        super().__init__(
            logger=logger,
            role="research_aggregate_agent",
            node="research_aggregate",
            max_retries=max_retries,
            timeout_s=timeout_s,
        )

    def pre_validate(self, *, input: ResearchAggregateInput, state: GraphState) -> None:
        if not input.partials:
            raise ContractViolationError("no_partials")
        for p in input.partials:
            if p.plan_id != input.plan.plan_id:
                raise ContractViolationError("plan_id_mismatch_in_partials")

    def post_validate(
        self, *, output: MultiDimensionResearchResult, input: ResearchAggregateInput, state: GraphState
    ) -> None:
        if output.plan_id != input.plan.plan_id:
            raise ContractViolationError("plan_id_mismatch")
        dim_ids = {d.dimension_id for d in input.plan.dimensions}
        out_dim_ids = {r.dimension_id for r in output.dimension_results}
        if dim_ids != out_dim_ids:
            raise ContractViolationError("dimension_set_mismatch")
        for dr in output.dimension_results:
            if not dr.findings:
                raise ContractViolationError("dimension_has_no_findings")
            if any(len(f.citations) < 1 for f in dr.findings):
                raise ContractViolationError("finding_missing_citation")

    def _invoke(self, *, input: ResearchAggregateInput, state: GraphState) -> MultiDimensionResearchResult:
        merged: dict[tuple[str, str], DimensionResearchResult] = {}
        for p in input.partials:
            for r in p.results:
                merged[(r.dimension_id, r.agent_type)] = r

        per_dimension: dict[str, list[DimensionResearchResult]] = {}
        for (dim_id, _), r in merged.items():
            per_dimension.setdefault(dim_id, []).append(r)

        dimension_results: list[DimensionResearchResult] = []
        for dim in input.plan.dimensions:
            dim_id = dim.dimension_id
            bundles = per_dimension.get(dim_id, [])
            if not bundles:
                continue
            combined = _combine_dimension(bundles=bundles)
            dimension_results.extend(combined)

        conflicts = _detect_conflicts(dimension_results=dimension_results)
        deduped = True

        return MultiDimensionResearchResult(
            plan_id=input.plan.plan_id,
            thesis=input.plan.thesis,
            dimension_results=dimension_results,
            conflicts=conflicts,
            deduped=deduped,
            generated_at=datetime.now(tz=timezone.utc),
        )


def _combine_dimension(*, bundles: list[DimensionResearchResult]) -> list[DimensionResearchResult]:
    output: list[DimensionResearchResult] = []
    for b in bundles:
        seen_claims: set[str] = set()
        dedup_findings: list[ResearchFinding] = []
        for f in b.findings:
            key = f.claim.strip().lower()
            if key in seen_claims:
                continue
            seen_claims.add(key)
            dedup_findings.append(f)
        output.append(
            DimensionResearchResult(
                dimension_id=b.dimension_id,
                agent_type=b.agent_type,
                findings=dedup_findings,
                sources=b.sources,
                notes=b.notes,
                validation=b.validation,
            )
        )
    return output


def _detect_conflicts(*, dimension_results: list[DimensionResearchResult]) -> list[ConflictRecord]:
    conflicts: list[ConflictRecord] = []
    claims: dict[str, list[tuple[str, str]]] = {}
    for dr in dimension_results:
        for f in dr.findings:
            key = f.claim.strip().lower()
            claims.setdefault(key, []).append((f.finding_id, dr.dimension_id))

    idx = 0
    for claim, refs in claims.items():
        if len(refs) <= 1:
            continue
        idx += 1
        conflicts.append(
            ConflictRecord(
                conflict_id=f"conflict_{idx}",
                claim_a=claim,
                claim_b=claim,
                related_finding_ids=[r[0] for r in refs][:10],
                resolution_status="unresolved",
                followup_suggestion="多个来源重复结论：需要进一步核对其适用范围与前提条件。",
            )
        )
    return conflicts[:50]

