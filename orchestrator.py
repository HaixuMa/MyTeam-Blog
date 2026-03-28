from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph
from datetime import datetime, timezone

from agents.analysis import AnalysisInput, DeepResearchAgentHarness
from agents.auditing import AuditAgentHarness, AuditInput, AuditOutput
from agents.imaging import ImagingAgentHarness
from agents.planning import PlanningAgentHarness
from agents.research_aggregate import ResearchAggregateInput, ResearchAggregationHarness
from agents.research_base import ResearchSubAgentHarness
from agents.writing import TechnicalWritingAgentHarness, WritingInput
from config import AppConfig
from schemas.auditing import AuditReport, FinalPublishedArticle
from schemas.analysis import DeepResearchAnalysisReport
from schemas.imaging import ArticleWithImages
from schemas.planning import ResearchExecutionPlan, UserResearchGoal
from schemas.research import MultiDimensionResearchResult
from schemas.research_partials import ResearchPartialBatch
from schemas.state import GraphState, now_iso
from schemas.writing import TechnicalArticleDraft
from tools.image_tools import ImageToolbox
from tools.research_tools import ResearchToolbox
from memory_store import update_quality_memory


@dataclass(frozen=True)
class OrchestratorResult:
    state: GraphState
    status: Literal["completed", "halted", "failed"]


def _route_next_or_end(state: GraphState) -> str:
    if state.get("fatal_error") or state.get("stage") == "halted":
        return "end"
    return "next"


def _detect_repeating_suffix(*, seq: list[str], max_period: int = 4, min_repeats: int = 2) -> dict | None:
    if min_repeats < 2:
        min_repeats = 2
    n = len(seq)
    for period in range(1, max_period + 1):
        window = period * min_repeats
        if n < window:
            continue
        tail = seq[-window:]
        pat = tail[:period]
        if len(set(pat)) == 1:
            continue
        ok = True
        for k in range(1, min_repeats):
            if tail[k * period : (k + 1) * period] != pat:
                ok = False
                break
        if ok:
            return {"period": period, "repeats": min_repeats, "pattern": pat, "window": tail}
    return None


def new_initial_state(*, user_goal: UserResearchGoal, trace_id: str | None = None) -> GraphState:
    tid = trace_id or f"trace_{uuid.uuid4().hex[:12]}"
    return GraphState(
        trace_id=tid,
        created_at=now_iso(),
        stage="start",
        user_goal=user_goal.model_dump(mode="json"),
        execution_events=[],
        retries={},
        audit_rounds_used=0,
    )


class HarnessOrchestrator:
    def __init__(
        self,
        *,
        cfg: AppConfig,
        llm: BaseChatModel,
        logger: logging.Logger,
        project_root: Path,
    ) -> None:
        self._cfg = cfg
        self._llm = llm
        self._logger = logger
        self._project_root = project_root
        self._app = self._build_graph()

    def load_state(self, *, trace_id: str) -> GraphState:
        snapshot = self._app.get_state(config=_thread_config(trace_id))
        values = snapshot.values
        if not isinstance(values, dict):
            raise RuntimeError("checkpoint_state_invalid")
        return values  # type: ignore[return-value]

    def run_full(self, *, state: GraphState) -> OrchestratorResult:
        result_state = self._app.invoke(state, config=_thread_config(state["trace_id"]))
        return _result_from_state(result_state)

    def run_step(self, *, state: GraphState) -> OrchestratorResult:
        last_state: GraphState = state
        for updated in self._app.stream(state, config=_thread_config(state["trace_id"]), stream_mode="values"):
            last_state = updated
            break
        return _result_from_state(last_state)

    def _build_graph(self):
        graph: StateGraph[GraphState] = StateGraph(GraphState)

        graph.add_node("planning", self._planning_node)
        graph.add_node("research_academic", self._research_academic_node)
        graph.add_node("research_tech", self._research_tech_node)
        graph.add_node("research_industry", self._research_industry_node)
        graph.add_node("research_competitor", self._research_competitor_node)
        graph.add_node("research_aggregate", self._research_aggregate_node)
        graph.add_node("analysis", self._analysis_node)
        graph.add_node("writing", self._writing_node)
        graph.add_node("imaging", self._imaging_node)
        graph.add_node("auditing", self._auditing_node)
        graph.add_node("publish", self._publish_node)

        graph.set_entry_point("planning")

        graph.add_conditional_edges(
            "planning",
            self._route_after_planning,
            {
                "halted": END,
                "research_academic": "research_academic",
            },
        )

        graph.add_conditional_edges(
            "research_academic",
            _route_next_or_end,
            {"next": "research_tech", "end": END},
        )
        graph.add_conditional_edges(
            "research_tech",
            _route_next_or_end,
            {"next": "research_industry", "end": END},
        )
        graph.add_conditional_edges(
            "research_industry",
            _route_next_or_end,
            {"next": "research_competitor", "end": END},
        )
        graph.add_conditional_edges(
            "research_competitor",
            _route_next_or_end,
            {"next": "research_aggregate", "end": END},
        )
        graph.add_conditional_edges(
            "research_aggregate",
            _route_next_or_end,
            {"next": "analysis", "end": END},
        )
        graph.add_conditional_edges("analysis", _route_next_or_end, {"next": "writing", "end": END})
        graph.add_conditional_edges("writing", _route_next_or_end, {"next": "imaging", "end": END})

        graph.add_conditional_edges(
            "auditing",
            self._route_after_audit,
            {
                "publish": "publish",
                "back_to_research": "research_academic",
                "back_to_analysis": "analysis",
                "back_to_writing": "writing",
                "halted": END,
            },
        )
        graph.add_conditional_edges("imaging", _route_next_or_end, {"next": "auditing", "end": END})
        graph.add_edge("publish", END)

        sqlite_url = f"sqlite:///{self._cfg.checkpoint_sqlite_path.as_posix()}"
        from langgraph.checkpoint.sqlite import SqliteSaver

        try:
            checkpointer = SqliteSaver.from_conn_string(sqlite_url)
        except AttributeError:
            import sqlite3

            conn = sqlite3.connect(self._cfg.checkpoint_sqlite_path)
            checkpointer = SqliteSaver(conn)
        return graph.compile(checkpointer=checkpointer)

    def _bump_loop_guard(self, *, state: GraphState, node: str) -> bool:
        memory = state.get("memory")
        if not isinstance(memory, dict):
            memory = {}
            state["memory"] = memory

        guard = memory.get("loop_guard")
        if not isinstance(guard, dict):
            guard = {}
            memory["loop_guard"] = guard

        seq = guard.get("seq")
        if not isinstance(seq, list):
            seq = []
            guard["seq"] = seq
        seq.append(node)
        if len(seq) > 80:
            del seq[: len(seq) - 80]

        history = state.get("stage_history")
        if not isinstance(history, list):
            history = []
            state["stage_history"] = history
        history.append(
            {
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "from_stage": state.get("stage", ""),
                "to_node": node,
            }
        )
        if len(history) > 400:
            del history[: len(history) - 400]

        total_steps = int(guard.get("total_steps", 0)) + 1
        guard["total_steps"] = total_steps

        node_visits = guard.get("node_visits")
        if not isinstance(node_visits, dict):
            node_visits = {}
            guard["node_visits"] = node_visits
        node_visits[node] = int(node_visits.get(node, 0)) + 1

        repeat = _detect_repeating_suffix(seq=seq, max_period=4, min_repeats=2)
        if repeat is not None:
            state["stage"] = "halted"
            state["halted_reason"] = "loop_guard_repeating_pattern"
            memory["loop_guard_exceeded"] = {
                "node": node,
                "pattern": repeat.get("pattern"),
                "period": repeat.get("period"),
                "repeats": repeat.get("repeats"),
                "window": repeat.get("window"),
                "total_steps": total_steps,
            }
            state["execution_events"].append(
                {
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    "trace_id": state["trace_id"],
                    "node": node,
                    "role": "orchestrator",
                    "status": "halted",
                    "duration_ms": None,
                    "attempt": 1,
                    "message": "loop_guard_repeating_pattern",
                    "input_summary": None,
                    "output_summary": None,
                    "error_type": None,
                    "error_message": None,
                }
            )
            return False

        if total_steps > self._cfg.max_total_node_steps:
            state["stage"] = "halted"
            state["halted_reason"] = "loop_guard_total_steps_exceeded"
            memory["loop_guard_exceeded"] = {
                "node": node,
                "total_steps": total_steps,
                "max_total_node_steps": self._cfg.max_total_node_steps,
            }
            return False

        if int(node_visits[node]) > self._cfg.max_node_visits_per_node:
            state["stage"] = "halted"
            state["halted_reason"] = "loop_guard_node_visits_exceeded"
            memory["loop_guard_exceeded"] = {
                "node": node,
                "node_visits": int(node_visits[node]),
                "max_node_visits_per_node": self._cfg.max_node_visits_per_node,
                "total_steps": total_steps,
            }
            return False

        return True

    def _planning_node(self, state: GraphState) -> GraphState:
        try:
            if not self._bump_loop_guard(state=state, node="planning"):
                return state
            state["stage"] = "planning"
            user_goal = UserResearchGoal.model_validate(state["user_goal"])
            harness = PlanningAgentHarness(
                logger=self._logger,
                llm=self._llm,
                max_retries=self._cfg.max_agent_retries,
                timeout_s=self._cfg.request_timeout_s,
            )
            plan = harness.run(input=user_goal, state=state)
            state["plan"] = plan.model_dump(mode="json")
            if plan.clarification_needed:
                state["stage"] = "halted"
                state["halted_reason"] = "clarification_needed"
            return state
        except Exception as e:
            return _fatal(state, e)

    def _route_after_planning(self, state: GraphState) -> str:
        if state.get("fatal_error") or state.get("stage") == "halted":
            return "halted"
        plan = ResearchExecutionPlan.model_validate(state.get("plan") or {})
        if plan.clarification_needed:
            return "halted"
        return "research_academic"

    def _research_academic_node(self, state: GraphState) -> GraphState:
        try:
            if not self._bump_loop_guard(state=state, node="research_academic"):
                return state
            return self._run_research_subagent(state=state, agent_type="academic")
        except Exception as e:
            return _fatal(state, e)

    def _research_tech_node(self, state: GraphState) -> GraphState:
        try:
            if not self._bump_loop_guard(state=state, node="research_tech"):
                return state
            return self._run_research_subagent(state=state, agent_type="tech")
        except Exception as e:
            return _fatal(state, e)

    def _research_industry_node(self, state: GraphState) -> GraphState:
        try:
            if not self._bump_loop_guard(state=state, node="research_industry"):
                return state
            return self._run_research_subagent(state=state, agent_type="industry")
        except Exception as e:
            return _fatal(state, e)

    def _research_competitor_node(self, state: GraphState) -> GraphState:
        try:
            if not self._bump_loop_guard(state=state, node="research_competitor"):
                return state
            return self._run_research_subagent(state=state, agent_type="competitor")
        except Exception as e:
            return _fatal(state, e)

    def _run_research_subagent(
        self,
        *,
        state: GraphState,
        agent_type: Literal["academic", "tech", "industry", "competitor"],
    ) -> GraphState:
        if state.get("fatal_error") or state.get("stage") == "halted":
            return state
        state["stage"] = f"research_{agent_type}"  # type: ignore[assignment]
        plan = ResearchExecutionPlan.model_validate(state["plan"])
        user_goal = UserResearchGoal.model_validate(state["user_goal"])

        toolbox = ResearchToolbox(
            logger=self._logger,
            allowed_tools=user_goal.allowed_tools,
            rate_limit_per_minute=self._cfg.tool_rate_limit_per_minute,
            tavily_api_key_present=bool(self._cfg.tavily_api_key),
        )
        harness = ResearchSubAgentHarness(
            logger=self._logger,
            llm=self._llm,
            toolbox=toolbox,
            agent_type=agent_type,
            max_retries=self._cfg.max_agent_retries,
            timeout_s=self._cfg.request_timeout_s,
        )
        batch = harness.run(input=plan, state=state)
        partials = state.get("research_partials") or []
        partials.append(batch.model_dump(mode="json"))
        state["research_partials"] = partials
        return state

    def _research_aggregate_node(self, state: GraphState) -> GraphState:
        try:
            if not self._bump_loop_guard(state=state, node="research_aggregate"):
                return state
            state["stage"] = "research_aggregate"
            plan = ResearchExecutionPlan.model_validate(state["plan"])
            partial_dicts = state.get("research_partials") or []
            partials = [ResearchPartialBatch.model_validate(p) for p in partial_dicts]

            harness = ResearchAggregationHarness(
                logger=self._logger,
                max_retries=self._cfg.max_agent_retries,
                timeout_s=self._cfg.request_timeout_s,
            )
            research = harness.run(
                input=ResearchAggregateInput(plan=plan, partials=partials), state=state
            )
            state["research_result"] = research.model_dump(mode="json")
            return state
        except Exception as e:
            return _fatal(state, e)

    def _analysis_node(self, state: GraphState) -> GraphState:
        try:
            if not self._bump_loop_guard(state=state, node="analysis"):
                return state
            state["stage"] = "analysis"
            plan = ResearchExecutionPlan.model_validate(state["plan"])
            research = MultiDimensionResearchResult.model_validate(state["research_result"])
            harness = DeepResearchAgentHarness(
                logger=self._logger,
                llm=self._llm,
                max_retries=self._cfg.max_agent_retries,
                timeout_s=self._cfg.request_timeout_s,
            )
            report = harness.run(input=AnalysisInput(plan=plan, research=research), state=state)
            state["analysis_report"] = report.model_dump(mode="json")
            return state
        except Exception as e:
            return _fatal(state, e)

    def _writing_node(self, state: GraphState) -> GraphState:
        try:
            if not self._bump_loop_guard(state=state, node="writing"):
                return state
            state["stage"] = "writing"
            plan = ResearchExecutionPlan.model_validate(state["plan"])
            analysis = DeepResearchAnalysisReport.model_validate(state["analysis_report"])
            harness = TechnicalWritingAgentHarness(
                logger=self._logger,
                llm=self._llm,
                max_retries=self._cfg.max_agent_retries,
                timeout_s=self._cfg.request_timeout_s,
            )
            draft = harness.run(input=WritingInput(plan=plan, analysis=analysis), state=state)
            state["article_draft"] = draft.model_dump(mode="json")
            return state
        except Exception as e:
            return _fatal(state, e)

    def _imaging_node(self, state: GraphState) -> GraphState:
        try:
            if not self._bump_loop_guard(state=state, node="imaging"):
                return state
            state["stage"] = "imaging"
            user_goal = UserResearchGoal.model_validate(state["user_goal"])
            draft = state.get("article_draft") or {}
            article = TechnicalArticleDraft.model_validate(draft)

            toolbox = ImageToolbox(
                logger=self._logger,
                allowed_tools=user_goal.allowed_tools,
                rate_limit_per_minute=self._cfg.tool_rate_limit_per_minute,
                openai_api_key_present=bool(self._cfg.openai_api_key)
                and self._cfg.model_provider == "openai",
                data_dir=self._cfg.data_dir,
                request_timeout_s=self._cfg.request_timeout_s,
            )
            harness = ImagingAgentHarness(
                logger=self._logger,
                llm=self._llm,
                toolbox=toolbox,
                max_retries=self._cfg.max_agent_retries,
                timeout_s=self._cfg.request_timeout_s,
                max_image_retries=self._cfg.max_image_retries,
            )
            with_images = harness.run(input=article, state=state)
            state["article_with_images"] = with_images.model_dump(mode="json")
            return state
        except Exception as e:
            return _fatal(state, e)

    def _auditing_node(self, state: GraphState) -> GraphState:
        try:
            if not self._bump_loop_guard(state=state, node="auditing"):
                return state
            state["stage"] = "auditing"
            research = MultiDimensionResearchResult.model_validate(state["research_result"])
            article = ArticleWithImages.model_validate(state["article_with_images"])

            rounds_used = int(state.get("audit_rounds_used", 0))
            harness = AuditAgentHarness(
                logger=self._logger,
                llm=self._llm,
                max_retries=self._cfg.max_agent_retries,
                timeout_s=self._cfg.request_timeout_s,
            )
            out = harness.run(
                input=AuditInput(article=article, research=research, rounds_used=rounds_used),
                state=state,
            )
            state["final_article"] = out.final_article.model_dump(mode="json")
            state["audit_report"] = out.audit_report.model_dump(mode="json")
            memory = state.get("memory")
            if not isinstance(memory, dict):
                memory = {}
                state["memory"] = memory
            memory["last_audit_summary"] = out.audit_report.summary[:2000]
            memory["last_audit_issue_counts"] = {
                "blocker_or_high": sum(1 for i in out.audit_report.issues if i.severity in {"blocker", "high"}),
                "total": len(out.audit_report.issues),
            }
            update_quality_memory(
                data_dir=self._cfg.data_dir,
                issues=[i.model_dump(mode="json") for i in out.audit_report.issues],
            )
            return state
        except Exception as e:
            return _fatal(state, e)

    def _route_after_audit(self, state: GraphState) -> str:
        if state.get("fatal_error") or state.get("stage") == "halted":
            return "halted"
        rounds = int(state.get("audit_rounds_used", 0))
        report = AuditReport.model_validate(state.get("audit_report") or {})

        if report.passed:
            return "publish"

        rounds += 1
        state["audit_rounds_used"] = rounds
        if rounds > self._cfg.max_audit_rounds:
            state["stage"] = "halted"
            state["halted_reason"] = "audit_rounds_exceeded"
            return "halted"

        target = _pick_back_target(report)
        if target == "research":
            return "back_to_research"
        if target == "analysis":
            return "back_to_analysis"
        return "back_to_writing"

    def _publish_node(self, state: GraphState) -> GraphState:
        try:
            if not self._bump_loop_guard(state=state, node="publish"):
                return state
            state["stage"] = "publish"
            final = FinalPublishedArticle.model_validate(state["final_article"])
            audit = AuditReport.model_validate(state["audit_report"])

            out_dir = (self._cfg.data_dir / "outputs" / state["trace_id"]).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

            (out_dir / "final_article.md").write_text(final.markdown, encoding="utf-8")
            (out_dir / "audit_report.json").write_text(
                audit.model_dump_json(indent=2), encoding="utf-8"
            )
            (out_dir / "execution_events.json").write_text(
                json.dumps(state["execution_events"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return state
        except Exception as e:
            return _fatal(state, e)


def _fatal(state: GraphState, err: Exception) -> GraphState:
    state["stage"] = "failed"
    state["fatal_error"] = f"{type(err).__name__}: {err}"
    return state


def _pick_back_target(report: AuditReport) -> Literal["research", "analysis", "writing"]:
    severities = {"blocker": 4, "high": 3, "medium": 2, "low": 1}
    if not report.issues:
        return "writing"
    issue = max(report.issues, key=lambda i: severities.get(i.severity, 0))
    if issue.target_stage in {"research", "analysis", "writing"}:
        return issue.target_stage
    return "writing"


def _thread_config(trace_id: str) -> dict:
    return {"configurable": {"thread_id": trace_id}}


def _result_from_state(state: GraphState) -> OrchestratorResult:
    if state.get("fatal_error"):
        return OrchestratorResult(state=state, status="failed")
    if state.get("stage") == "halted":
        return OrchestratorResult(state=state, status="halted")
    if state.get("final_article") and state.get("audit_report"):
        report = AuditReport.model_validate(state["audit_report"])
        if report.passed:
            return OrchestratorResult(state=state, status="completed")
    return OrchestratorResult(state=state, status="completed")
