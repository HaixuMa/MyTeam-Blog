from __future__ import annotations

from schemas.state import GraphState


def system_context(*, state: GraphState, node: str) -> str:
    ctx = state.get("project_context") or {}
    trace_id = state.get("trace_id", "")
    stage = state.get("stage", "")
    model = str(ctx.get("model_name") or "")
    provider = str(ctx.get("model_provider") or "")
    data_dir = str(ctx.get("data_dir") or "")
    outputs_root = str(ctx.get("outputs_root") or "")

    flow = (
        "planning -> research_academic -> research_tech -> research_industry -> "
        "research_competitor -> research_aggregate -> analysis -> writing -> imaging -> auditing -> publish"
    )

    return (
        "你在 MyTeam-Blog 的 Harness Engineering 流水线中工作。\n"
        f"trace_id={trace_id}\n"
        f"stage={stage}\n"
        f"node={node}\n"
        f"model_provider={provider}\n"
        f"model_name={model}\n"
        f"data_dir={data_dir}\n"
        f"outputs_root={outputs_root}\n"
        f"pipeline={flow}\n"
        "通用约束：必须严格输出结构化契约；不得编造来源；引用/URL 必须来自输入提供的 sources/citations；"
        "遇到信息不足应降低 confidence 或明确 open questions。"
    )
