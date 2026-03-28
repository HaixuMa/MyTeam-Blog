from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph

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


@dataclass(frozen=True)
class OrchestratorResult:
    state: GraphState
    status: Literal["completed", "halted", "failed"]


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

        graph.add_edge("research_academic", "research_tech")
        graph.add_edge("research_tech", "research_industry")
        graph.add_edge("research_industry", "research_competitor")
        graph.add_edge("research_competitor", "research_aggregate")
        graph.add_edge("research_aggregate", "analysis")
        graph.add_edge("analysis", "writing")
        graph.add_edge("writing", "imaging")

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
        graph.add_edge("imaging", "auditing")
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

    def _planning_node(self, state: GraphState) -> GraphState:
        try:
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
        plan = ResearchExecutionPlan.model_validate(state.get("plan") or {})
        if plan.clarification_needed:
            return "halted"
        return "research_academic"

    def _research_academic_node(self, state: GraphState) -> GraphState:
        try:
            return self._run_research_subagent(state=state, agent_type="academic")
        except Exception as e:
            return _fatal(state, e)

    def _research_tech_node(self, state: GraphState) -> GraphState:
        try:
            return self._run_research_subagent(state=state, agent_type="tech")
        except Exception as e:
            return _fatal(state, e)

    def _research_industry_node(self, state: GraphState) -> GraphState:
        try:
            return self._run_research_subagent(state=state, agent_type="industry")
        except Exception as e:
            return _fatal(state, e)

    def _research_competitor_node(self, state: GraphState) -> GraphState:
        try:
            return self._run_research_subagent(state=state, agent_type="competitor")
        except Exception as e:
            return _fatal(state, e)

    def _run_research_subagent(
        self,
        *,
        state: GraphState,
        agent_type: Literal["academic", "tech", "industry", "competitor"],
    ) -> GraphState:
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
            state["stage"] = "imaging"
            user_goal = UserResearchGoal.model_validate(state["user_goal"])
            draft = state.get("article_draft") or {}
            article = TechnicalArticleDraft.model_validate(draft)

            toolbox = ImageToolbox(
                logger=self._logger,
                allowed_tools=user_goal.allowed_tools,
                rate_limit_per_minute=self._cfg.tool_rate_limit_per_minute,
                openai_api_key_present=bool(self._cfg.openai_api_key),
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
            return state
        except Exception as e:
            return _fatal(state, e)

    def _route_after_audit(self, state: GraphState) -> str:
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
