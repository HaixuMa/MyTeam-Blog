from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import os
from typing import Any
from urllib.parse import urlparse

_DEPS_DIR = (Path(__file__).resolve().parent / ".pydeps").resolve()
if _DEPS_DIR.exists():
    sys.path.insert(0, str(_DEPS_DIR))

@dataclass(frozen=True)
class FallbackConfig:
    log_level: str
    data_dir: Path
    checkpoint_sqlite_path: Path
    model_provider: str
    model_name: str


@dataclass
class RunRecord:
    run_id: str
    trace_id: str
    created_at: float
    updated_at: float
    status: str
    state: dict[str, Any]
    error: str | None


class RunRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, RunRecord] = {}
        self._run_lock = threading.Lock()

    def create(self, *, trace_id: str, state: dict[str, Any]) -> RunRecord:
        now = time.time()
        rec = RunRecord(
            run_id=f"run_{uuid.uuid4().hex[:12]}",
            trace_id=trace_id,
            created_at=now,
            updated_at=now,
            status="queued",
            state=state,
            error=None,
        )
        with self._lock:
            self._runs[rec.run_id] = rec
        return rec

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._runs.get(run_id)

    def update(
        self,
        *,
        run_id: str,
        status: str | None = None,
        state: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            rec = self._runs.get(run_id)
            if not rec:
                return
            rec.updated_at = time.time()
            if status is not None:
                rec.status = status
            if state is not None:
                rec.state = state
            if error is not None:
                rec.error = error

    def run_lock(self) -> threading.Lock:
        return self._run_lock


class App:
    def __init__(self, *, project_root: Path) -> None:
        self.project_root = project_root
        self._init_error: str | None = None
        self._llm_error: str | None = None
        self._llm_attempted: bool = False
        self.cfg = self._load_cfg(project_root)
        self._configure_logging(self.cfg.log_level)
        import logging

        self.logger = logging.getLogger("myteam_blog_web")
        self.registry = RunRegistry()
        self.llm = None
        self.model_info = None

    def _load_cfg(self, project_root: Path):
        try:
            from config import load_config

            return load_config(project_root)
        except ModuleNotFoundError as e:
            if not self._init_error:
                self._init_error = f"missing_dependency: {e.name}"
        except Exception as e:
            if not self._init_error:
                self._init_error = f"{type(e).__name__}: {e}"

        data_dir = (project_root / "data").resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        return FallbackConfig(
            log_level="INFO",
            data_dir=data_dir,
            checkpoint_sqlite_path=(data_dir / "checkpoints.sqlite3").resolve(),
            model_provider="unknown",
            model_name="unknown",
        )

    def _configure_logging(self, log_level: str) -> None:
        try:
            from logging_utils import configure_logging

            configure_logging(log_level)
        except Exception:
            return

    def init_error(self) -> str | None:
        return self._init_error

    def llm_error(self) -> str | None:
        return self._llm_error

    def ensure_llm(self) -> None:
        if self._llm_attempted:
            return
        self._llm_attempted = True
        try:
            from llm_factory import ConfigError, create_chat_model

            self.llm, self.model_info = create_chat_model(self.cfg)
            self._llm_error = None
        except ModuleNotFoundError as e:
            self.llm = None
            self.model_info = None
            self._llm_error = f"missing_dependency: {e.name}"
        except ConfigError as e:
            self.llm = None
            self.model_info = None
            self._llm_error = str(e)
        except Exception as e:
            self.llm = None
            self.model_info = None
            self._llm_error = f"{type(e).__name__}: {e}"

    def start_run(self, *, goal: str) -> RunRecord:
        if self.init_error():
            raise RuntimeError(self.init_error())
        self.ensure_llm()
        if self.llm is None:
            raise RuntimeError(self.llm_error() or "llm_not_ready")
        user_goal = {
            "research_goal": goal,
            "user_requirements": [],
            "deadline": None,
            "output_language": "zh",
            "max_sources_per_dimension": 8,
            "allowed_tools": ["tavily", "arxiv", "wikipedia", "web_loader", "image_generation"],
            "clarifications": {},
        }
        state = _new_initial_state(user_goal=user_goal)
        state.setdefault("project_context", {})
        state["project_context"].update(
            {
                "project_root": str(self.project_root),
                "data_dir": str(self.cfg.data_dir),
                "outputs_root": str((self.cfg.data_dir / "outputs").resolve()),
                "checkpoint_sqlite_path": str(self.cfg.checkpoint_sqlite_path),
                "model_provider": getattr(self.cfg, "model_provider", "unknown"),
                "model_name": getattr(self.cfg, "model_name", "unknown"),
            }
        )
        state.setdefault("prompt_history", [])
        state.setdefault("memory", {})
        state.setdefault("stage_history", [])

        rec = self.registry.create(trace_id=state["trace_id"], state=state)
        t = threading.Thread(target=self._run_worker, args=(rec.run_id,), daemon=True)
        t.start()
        return rec

    def _run_worker(self, run_id: str) -> None:
        init_err = self.init_error()
        if init_err:
            self.registry.update(run_id=run_id, status="failed", error=init_err)
            return
        self.ensure_llm()
        if self.llm is None:
            self.registry.update(run_id=run_id, status="failed", error=self.llm_error() or "llm_not_ready")
            return

        rec = self.registry.get(run_id)
        if not rec:
            return

        self.registry.update(run_id=run_id, status="running")
        with self.registry.run_lock():
            try:
                from orchestrator import HarnessOrchestrator

                orchestrator = HarnessOrchestrator(
                    cfg=self.cfg,
                    llm=self.llm,  # type: ignore[arg-type]
                    logger=self.logger,
                    project_root=self.project_root,
                )
                state = rec.state or {}
                scheduler_events: list[dict[str, Any]] = []

                def _push_sched_event(evt: dict[str, Any]) -> None:
                    scheduler_events.append(evt)
                    if len(scheduler_events) > 200:
                        del scheduler_events[: len(scheduler_events) - 200]
                    state["scheduler_events"] = scheduler_events

                def _set_scheduler(*, current_task: str | None, status: str, step: int | None, ts: str | None) -> None:
                    sched = state.get("scheduler")
                    if not isinstance(sched, dict):
                        sched = {}
                        state["scheduler"] = sched
                    if current_task is not None:
                        sched["current_task"] = current_task
                    if step is not None:
                        sched["step"] = step
                    if ts is not None:
                        sched["timestamp"] = ts
                    sched["status"] = status

                cfg = {"configurable": {"thread_id": str(state.get("trace_id") or rec.trace_id)}}
                stream = orchestrator._app.stream(state, config=cfg, stream_mode="debug")  # type: ignore[attr-defined]
                for ev in stream:
                    if not isinstance(ev, dict):
                        continue
                    etype = str(ev.get("type") or "")
                    step = ev.get("step")
                    ts = ev.get("timestamp")
                    payload = ev.get("payload") or {}
                    if not isinstance(payload, dict):
                        payload = {}

                    if etype == "task":
                        name = str(payload.get("name") or "")
                        if name:
                            if state.get("stage") == "start":
                                state["stage"] = name
                            _set_scheduler(current_task=name, status="running", step=int(step) if isinstance(step, int) else None, ts=str(ts) if ts else None)
                            _push_sched_event({"type": "task", "name": name, "step": step, "timestamp": ts})
                            self.registry.update(run_id=run_id, state=state)
                        continue

                    if etype == "task_result":
                        name = str(payload.get("name") or "")
                        err = payload.get("error")
                        result = payload.get("result")
                        next_state = None
                        if isinstance(result, list):
                            try:
                                next_state = dict(result)
                            except Exception:
                                next_state = None
                        elif isinstance(result, dict):
                            next_state = result
                        if isinstance(next_state, dict):
                            for k in ("scheduler", "scheduler_events"):
                                if k in state and k not in next_state:
                                    next_state[k] = state[k]
                            state = next_state
                        if name:
                            _set_scheduler(current_task=name, status="done" if not err else "failed", step=int(step) if isinstance(step, int) else None, ts=str(ts) if ts else None)
                            _push_sched_event({"type": "task_result", "name": name, "step": step, "timestamp": ts, "error": err})
                        self.registry.update(run_id=run_id, state=state)
                        continue

                    if etype == "checkpoint":
                        self.registry.update(run_id=run_id, state=state)
                        continue

                if state.get("fatal_error"):
                    self.registry.update(run_id=run_id, status="failed", error=str(state.get("fatal_error") or "failed"))
                    return
                if state.get("stage") == "halted":
                    self.registry.update(run_id=run_id, status="halted", error=str(state.get("halted_reason") or "halted"))
                    return
                if state.get("final_article") and state.get("audit_report"):
                    report = state.get("audit_report") or {}
                    if isinstance(report, dict) and report.get("passed") is True:
                        self.registry.update(run_id=run_id, status="completed")
                        return
                self.registry.update(run_id=run_id, status="completed")
                return
            except Exception as e:
                self.registry.update(run_id=run_id, status="failed", error=f"{type(e).__name__}: {e}")


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _new_initial_state(*, user_goal: dict[str, Any], trace_id: str | None = None) -> dict[str, Any]:
    tid = trace_id or f"trace_{uuid.uuid4().hex[:12]}"
    return {
        "trace_id": tid,
        "created_at": _now_iso(),
        "stage": "start",
        "user_goal": user_goal,
        "execution_events": [],
        "retries": {},
        "audit_rounds_used": 0,
    }


def _read_text_safe(path: Path) -> str | None:
    try:
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8")
        return None
    except Exception:
        return None


class Handler(BaseHTTPRequestHandler):
    server_version = "MyTeamBlogWeb/1.0"

    def _app(self) -> App:
        return self.server.app  # type: ignore[attr-defined]

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, *, path: Path, content_type: str) -> None:
        try:
            data = path.read_bytes()
        except Exception:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/version":
            self._send_json(
                200,
                {
                    "server": "myteam-blog-web",
                    "health_schema": "v2",
                    "web_server_file": str(Path(__file__).resolve()),
                    "python": sys.executable,
                    "pid": os.getpid(),
                },
            )
            return

        if path == "/api/debug":
            app = self._app()
            app.ensure_llm()
            deps = {
                "deps_dir": str(_DEPS_DIR),
                "deps_dir_exists": _DEPS_DIR.exists(),
                "sys_path_head": sys.path[:5],
            }

            def _try_import(name: str) -> dict[str, object]:
                try:
                    mod = __import__(name)
                    return {"ok": True, "file": getattr(mod, "__file__", None), "version": getattr(mod, "__version__", None)}
                except Exception as e:
                    return {"ok": False, "error": f"{type(e).__name__}: {e}"}

            def _try_import_deep(name: str) -> dict[str, object]:
                try:
                    mod = __import__(name, fromlist=["__name__"])
                    return {"ok": True, "file": getattr(mod, "__file__", None), "version": getattr(mod, "__version__", None)}
                except Exception as e:
                    return {"ok": False, "error": f"{type(e).__name__}: {e}"}

            cfg = app.cfg
            env_path = (app.project_root / ".env").resolve()
            cfg_view = {
                "project_root": str(app.project_root),
                "env_path": str(env_path),
                "env_exists": env_path.exists(),
                "model_provider": getattr(cfg, "model_provider", None),
                "model_name": getattr(cfg, "model_name", None),
                "model_base_url": getattr(cfg, "model_base_url", None),
            }

            self._send_json(
                200,
                {
                    "server_error": app.init_error(),
                    "llm_ok": (app.llm is not None and app.llm_error() is None),
                    "llm_error": app.llm_error(),
                    "cfg": cfg_view,
                    "deps": deps,
                    "modules": {
                        "orchestrator": _try_import("orchestrator"),
                    },
                    "imports": {
                        "pydantic": _try_import("pydantic"),
                        "langchain_core": _try_import("langchain_core"),
                        "langchain_openai": _try_import("langchain_openai"),
                        "openai": _try_import("openai"),
                        "tiktoken": _try_import("tiktoken"),
                        "jiter": _try_import("jiter"),
                        "langgraph": _try_import("langgraph"),
                        "langgraph.checkpoint.sqlite": _try_import_deep("langgraph.checkpoint.sqlite"),
                        "langgraph.checkpoint.memory": _try_import_deep("langgraph.checkpoint.memory"),
                    },
                },
            )
            return

        if path == "/api/health":
            app = self._app()
            app.ensure_llm()
            err = app.init_error()
            llm_err = app.llm_error()
            self._send_json(
                200,
                {
                    "ok": True,
                    "server_error": err,
                    "llm_ok": (llm_err is None and app.llm is not None),
                    "llm_error": llm_err,
                    "model_provider": getattr(app.model_info, "provider", getattr(app.cfg, "model_provider", None)),
                    "model_name": getattr(app.model_info, "model_name", getattr(app.cfg, "model_name", None)),
                },
            )
            return

        if path.startswith("/api/runs/"):
            parts = [p for p in path.split("/") if p]
            if len(parts) >= 3:
                run_id = parts[2]
                rec = self._app().registry.get(run_id)
                if not rec:
                    self._send_json(404, {"error": "run_not_found"})
                    return

                if len(parts) == 3:
                    state = rec.state or {}
                    evts = state.get("execution_events") or []
                    tail = evts[-50:] if isinstance(evts, list) else []
                    sched_tail = (state.get("scheduler_events") or [])
                    sched_tail = sched_tail[-50:] if isinstance(sched_tail, list) else []
                    self._send_json(
                        200,
                        {
                            "run_id": rec.run_id,
                            "trace_id": rec.trace_id,
                            "status": rec.status,
                            "stage": state.get("stage"),
                            "halted_reason": state.get("halted_reason"),
                            "fatal_error": state.get("fatal_error") or rec.error,
                            "updated_at": rec.updated_at,
                            "execution_events_tail": tail,
                            "stage_history_tail": (state.get("stage_history") or [])[-50:],
                            "scheduler": state.get("scheduler"),
                            "scheduler_events_tail": sched_tail,
                        },
                    )
                    return

                if len(parts) == 4 and parts[3] == "final":
                    state = rec.state or {}
                    trace_id = str(state.get("trace_id") or rec.trace_id)
                    out_dir = (self._app().cfg.data_dir / "outputs" / trace_id).resolve()
                    md_path = (out_dir / "final_article.md").resolve()
                    md = _read_text_safe(md_path)
                    if md is None:
                        final_obj = state.get("final_article") or {}
                        if isinstance(final_obj, dict):
                            md = str(final_obj.get("markdown") or "")
                    self._send_json(
                        200,
                        {
                            "run_id": rec.run_id,
                            "trace_id": rec.trace_id,
                            "status": rec.status,
                            "outputs_dir": str(out_dir),
                            "markdown": md or "",
                        },
                    )
                    return

        if path == "/" or path == "/index.html":
            web_root = (self._app().project_root / "webui").resolve()
            self._send_file(path=(web_root / "index.html"), content_type="text/html; charset=utf-8")
            return

        if path.startswith("/static/"):
            web_root = (self._app().project_root / "webui").resolve()
            rel = path[len("/static/") :]
            rel_path = Path(rel)
            if ".." in rel_path.parts:
                self.send_error(400)
                return
            file_path = (web_root / rel_path).resolve()
            if not str(file_path).startswith(str(web_root)):
                self.send_error(400)
                return
            if file_path.suffix == ".js":
                self._send_file(path=file_path, content_type="text/javascript; charset=utf-8")
                return
            if file_path.suffix == ".css":
                self._send_file(path=file_path, content_type="text/css; charset=utf-8")
                return
            if file_path.suffix in {".png", ".jpg", ".jpeg", ".svg"}:
                self._send_file(path=file_path, content_type="application/octet-stream")
                return
            self.send_error(404)
            return

        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/runs":
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                data = json.loads(raw.decode("utf-8"))
            except Exception:
                self._send_json(400, {"error": "invalid_json"})
                return

            goal = str((data or {}).get("goal") or "").strip()
            if not goal:
                self._send_json(400, {"error": "goal_required"})
                return

            try:
                rec = self._app().start_run(goal=goal)
            except Exception as e:
                self._send_json(400, {"error": f"{type(e).__name__}: {e}"})
                return

            self._send_json(200, {"run_id": rec.run_id, "trace_id": rec.trace_id})
            return

        self.send_error(404)

    def log_message(self, format: str, *args: Any) -> None:
        return


class Server(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_class, app: App):
        super().__init__(server_address, handler_class)
        self.app = app


def main() -> int:
    p = argparse.ArgumentParser(description="MyTeam-Blog Web UI")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    project_root = Path(__file__).resolve().parent
    app = App(project_root=project_root)
    srv = Server((args.host, args.port), Handler, app=app)
    print(f"web_ui=http://{args.host}:{args.port}/")
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
