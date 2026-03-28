from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from config import load_config
from llm_factory import ConfigError, create_chat_model
from logging_utils import configure_logging
from orchestrator import HarnessOrchestrator, OrchestratorResult, new_initial_state
from schemas.planning import UserResearchGoal


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MyTeam-Blog (LangChain + LangGraph + Harness Engineering)")
    p.add_argument(
        "--mode",
        choices=["full", "step"],
        default="full",
        help="full: 一次跑完整流程；step: 每次仅执行一个节点，便于调试与中断恢复。",
    )
    p.add_argument(
        "--goal",
        type=str,
        default="",
        help="研究目标（示例：'调研 RAG 系统在企业知识库中的落地架构与评测体系'）",
    )
    p.add_argument(
        "--goal-json",
        type=str,
        default="",
        help="UserResearchGoal 的 JSON 文件路径（优先级高于 --goal）。",
    )
    p.add_argument(
        "--resume-trace-id",
        type=str,
        default="",
        help="从 checkpoint 恢复执行的 trace_id。",
    )
    p.add_argument(
        "--clarifications-json",
        type=str,
        default="",
        help="澄清问题答案 JSON（dict[str,str]），用于继续执行被规划 Agent 暂停的流程。",
    )
    return p.parse_args()


def _load_goal(args: argparse.Namespace) -> UserResearchGoal:
    if args.goal_json:
        raw = Path(args.goal_json).read_text(encoding="utf-8")
        data = json.loads(raw)
        return UserResearchGoal.model_validate(data)

    if not args.goal.strip():
        return UserResearchGoal(
            research_goal="调研 RAG（Retrieval-Augmented Generation）在企业知识库落地中的典型架构、评测体系与工程实践，并输出可复用的技术白皮书。",
            user_requirements=[
                "重点覆盖近 3 年关键论文/开源框架演进",
                "包含可落地的系统架构图与流程图（需要配图）",
                "给出评测指标与 A/B 验证方法",
            ],
            deadline=None,
            output_language="zh",
        )

    return UserResearchGoal(research_goal=args.goal)


def main() -> int:
    args = _parse_args()
    project_root = Path(__file__).resolve().parent

    cfg = load_config(project_root)
    configure_logging(cfg.log_level)
    logger = logging.getLogger("myteam_blog")

    try:
        llm, model_info = create_chat_model(cfg)
        logger.info(f"model_ready: provider={model_info.provider} model={model_info.model_name}")
    except ConfigError as e:
        logger.error(str(e))
        logger.error("请先复制 .env.example 为 .env 并配置必要的 API Key。")
        return 2

    orchestrator = HarnessOrchestrator(cfg=cfg, llm=llm, logger=logger, project_root=project_root)

    if args.resume_trace_id:
        state = orchestrator.load_state(trace_id=args.resume_trace_id)
        if args.clarifications_json:
            clar = json.loads(args.clarifications_json)
            user_goal = UserResearchGoal.model_validate(state["user_goal"])
            user_goal.clarifications.update({str(k): str(v) for k, v in dict(clar).items()})
            state["user_goal"] = user_goal.model_dump(mode="json")
    else:
        goal = _load_goal(args)
        state = new_initial_state(user_goal=goal)

    if not isinstance(state.get("project_context"), dict):
        state["project_context"] = {}
    state["project_context"].update(
        {
            "project_root": str(project_root),
            "data_dir": str(cfg.data_dir),
            "outputs_root": str((cfg.data_dir / "outputs").resolve()),
            "checkpoint_sqlite_path": str(cfg.checkpoint_sqlite_path),
            "model_provider": cfg.model_provider,
            "model_name": cfg.model_name,
        }
    )
    if not isinstance(state.get("prompt_history"), list):
        state["prompt_history"] = []
    if not isinstance(state.get("memory"), dict):
        state["memory"] = {}
    if not isinstance(state.get("stage_history"), list):
        state["stage_history"] = []

    result: OrchestratorResult
    if args.mode == "full":
        result = orchestrator.run_full(state=state)
    else:
        result = orchestrator.run_step(state=state)

    _print_result(result, cfg.data_dir)
    return 0 if result.status == "completed" else 1


def _print_result(result: OrchestratorResult, data_dir: Path) -> None:
    state = result.state
    trace_id = state["trace_id"]
    print(f"status={result.status} trace_id={trace_id} stage={state.get('stage')}")
    if result.status == "halted":
        reason = state.get("halted_reason", "unknown")
        print(f"halted_reason={reason}")
        plan = state.get("plan")
        if plan and plan.get("clarification_needed"):
            print("clarification_questions:")
            for q in plan.get("clarification_questions", []):
                print(f"- {q.get('question_id')}: {q.get('question')}")
            print("继续执行方式：使用 --resume-trace-id 并通过 --clarifications-json 提供答案。")
        return

    out_dir = (data_dir / "outputs" / trace_id).resolve()
    if out_dir.exists():
        print(f"outputs_dir={out_dir}")
        print(f"- final_article.md")
        print(f"- audit_report.json")
        print(f"- execution_events.json")


if __name__ == "__main__":
    raise SystemExit(main())
