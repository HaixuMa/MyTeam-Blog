const $ = (id) => document.getElementById(id);

const state = {
  runs: [],
  activeRunId: null,
  polling: false,
  pollTimer: null,
  runSnapshots: {},
};

function nowLabel(ts) {
  try {
    return new Date(ts * 1000).toLocaleString();
  } catch {
    return "";
  }
}

function loadRuns() {
  try {
    const raw = localStorage.getItem("myteam_runs");
    const data = raw ? JSON.parse(raw) : [];
    if (Array.isArray(data)) state.runs = data;
  } catch {}
}

function saveRuns() {
  try {
    localStorage.setItem("myteam_runs", JSON.stringify(state.runs));
  } catch {}
}

function setHealth(ok, text, meta) {
  const dot = $("healthDot");
  dot.classList.remove("ok", "bad");
  dot.classList.add(ok ? "ok" : "bad");
  $("healthText").textContent = text;
  $("modelMeta").textContent = meta || "";
}

async function checkHealth() {
  try {
    const resp = await fetch("/api/health");
    const data = await resp.json();
    const modelLine = `模型：${data.model_provider || "-"} / ${data.model_name || "-"}`;
    if (data.llm_ok) {
      setHealth(true, "服务正常", modelLine);
      return;
    }
    const reason = data.llm_error || "unknown";
    setHealth(true, "服务正常（模型未就绪）", `${modelLine} · ${reason}`);
  } catch (e) {
    setHealth(false, "无法连接服务", String(e));
  }
}

async function fetchDebug() {
  try {
    const resp = await fetch("/api/debug");
    const data = await resp.json();
    if (!resp.ok) return null;
    return data;
  } catch {
    return null;
  }
}

function renderRuns() {
  const el = $("runs");
  el.innerHTML = "";
  state.runs.forEach((r) => {
    const item = document.createElement("div");
    item.className = "run-item" + (r.run_id === state.activeRunId ? " active" : "");
    item.onclick = () => {
      setActiveRun(r.run_id);
    };
    const title = document.createElement("div");
    title.className = "run-title";
    title.textContent = r.title || r.goal || r.run_id;
    const meta = document.createElement("div");
    meta.className = "run-meta";
    const stage = r.stage || "-";
    const task = r.current_task ? ` · ${r.current_task}` : "";
    meta.textContent = `trace_id=${r.trace_id || "-"} · ${r.status || "unknown"} · stage=${stage}${task} · ${nowLabel(
      r.created_at || Date.now() / 1000
    )}`;
    item.appendChild(title);
    item.appendChild(meta);
    el.appendChild(item);
  });
}

function appendMsg(text, cls) {
  const msg = document.createElement("div");
  msg.className = `msg ${cls}`;
  msg.textContent = text;
  $("messages").appendChild(msg);
  $("messages").scrollTop = $("messages").scrollHeight;
}

function resetConversation() {
  $("messages").innerHTML = "";
  $("finalMarkdown").textContent = "";
  $("copyBtn").disabled = true;
}

function setActiveRun(runId) {
  state.activeRunId = runId;
  renderRuns();
  resetConversation();
  const run = state.runs.find((r) => r.run_id === runId);
  if (run) {
    appendMsg(run.goal || "", "user");
    appendMsg("任务已加载，开始拉取运行状态…", "system");
    startPolling(runId);
  }
}

function upsertRun(rec) {
  const idx = state.runs.findIndex((r) => r.run_id === rec.run_id);
  if (idx >= 0) state.runs[idx] = { ...state.runs[idx], ...rec };
  else state.runs.unshift(rec);
  saveRuns();
  renderRuns();
}

function safeText(v) {
  if (v === null || v === undefined) return "-";
  if (typeof v === "string") return v;
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}

function escapeHTML(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function stageIndex(stage) {
  const order = [
    "planning",
    "research_academic",
    "research_tech",
    "research_industry",
    "research_competitor",
    "research_aggregate",
    "analysis",
    "writing",
    "imaging",
    "auditing",
    "publish",
    "completed",
  ];
  const idx = order.indexOf(stage || "");
  return idx >= 0 ? idx : 0;
}

function renderTaskPanel(data) {
  const el = $("taskPanelBody");
  if (!data) {
    el.textContent = "未选择任务";
    return;
  }

  const sched = data.scheduler || {};
  const currentTask = sched.current_task || "-";
  const currentStatus = sched.status || "-";
  const stage = data.stage || "-";
  const status = data.status || "-";
  const halted = data.halted_reason ? `（${data.halted_reason}）` : "";
  const fatal = data.fatal_error ? `错误：${safeText(data.fatal_error)}` : "";

  const idx = stageIndex(currentTask || stage);
  const pct = Math.max(0, Math.min(100, Math.round((idx / 11) * 100)));

  const evts = Array.isArray(data.scheduler_events_tail) ? data.scheduler_events_tail.slice(-8) : [];
  const listHtml = evts
    .map((e) => {
      const type = escapeHTML(e.type || "-");
      const name = escapeHTML(e.name || "-");
      const err = e.error ? " · error" : "";
      return `<div class="list-item">${type} · ${name}${err}</div>`;
    })
    .join("");

  const exec = Array.isArray(data.execution_events_tail) ? data.execution_events_tail : [];
  const lastByRole = new Map();
  exec.forEach((e) => {
    if (e && e.role) lastByRole.set(String(e.role), e);
  });
  const agentHtml = Array.from(lastByRole.entries())
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([role, e]) => {
      const node = escapeHTML(e.node || "-");
      const st = escapeHTML(e.status || "-");
      return `<div class="list-item">agent · ${escapeHTML(role)} · ${node} · ${st}</div>`;
    })
    .join("");

  const execDetail = exec
    .slice(-10)
    .map((e) => {
      const ts = escapeHTML((e.timestamp || "").slice(11, 19) || "-");
      const role = escapeHTML(e.role || "-");
      const node = escapeHTML(e.node || "-");
      const st = escapeHTML(e.status || "-");
      const msg = escapeHTML(e.message || "-");
      const err = e.error_message ? ` · ${escapeHTML(String(e.error_message).slice(0, 160))}` : "";
      const out = e.output_summary ? `\n${escapeHTML(String(e.output_summary).slice(0, 420))}` : "";
      return `<div class="list-item">${ts} · ${role} · ${node} · ${st} · ${msg}${err}${out ? `<pre style="margin:8px 0 0 0; white-space:pre-wrap">${out}</pre>` : ""}</div>`;
    })
    .join("");

  const prompts = Array.isArray(data.prompt_history_tail) ? data.prompt_history_tail : [];
  const promptDetail = prompts
    .slice(-6)
    .map((p) => {
      const ts = escapeHTML((p.timestamp || "").slice(11, 19) || "-");
      const node = escapeHTML(p.node || "-");
      const role = escapeHTML(p.role || "-");
      const sys = escapeHTML(String(p.system_prompt || "").slice(0, 1600));
      const usr = escapeHTML(String(p.user_prompt || "").slice(0, 1600));
      return `<div class="list-item">${ts} · ${role} · ${node}<pre style="margin:8px 0 0 0; white-space:pre-wrap">SYSTEM:\n${sys}\n\nUSER:\n${usr}</pre></div>`;
    })
    .join("");

  el.innerHTML = `
    <div class="kv">
      <div class="k">status</div><div class="v">${escapeHTML(safeText(status))} ${escapeHTML(halted)}</div>
      <div class="k">stage</div><div class="v">${escapeHTML(safeText(stage))}</div>
      <div class="k">current_task</div><div class="v">${escapeHTML(safeText(currentTask))} · ${escapeHTML(
    safeText(currentStatus)
  )}</div>
      <div class="k">trace_id</div><div class="v">${escapeHTML(safeText(data.trace_id))}</div>
    </div>
    <div class="progress"><div style="width:${pct}%"></div></div>
    <div class="list">${fatal ? `<div class="list-item">${escapeHTML(fatal)}</div>` : ""}${listHtml}${agentHtml}</div>
    ${
      execDetail
        ? `<details style="margin-top:12px"><summary>阶段运行信息</summary><div class="list" style="margin-top:8px">${execDetail}</div></details>`
        : ""
    }
    ${
      promptDetail
        ? `<details style="margin-top:12px"><summary>提示快照（system/user）</summary><div class="list" style="margin-top:8px">${promptDetail}</div></details>`
        : ""
    }
  `;
}

function renderDepsPanel(debug, health) {
  const el = $("depsPanelBody");
  const modelProvider = (health && health.model_provider) || "-";
  const modelName = (health && health.model_name) || "-";
  const llmOk = health && health.llm_ok ? "ok" : "not_ready";
  const llmErr = (health && health.llm_error) || "";

  const imports = (debug && debug.imports) || {};
  const show = ["langgraph.checkpoint.sqlite", "langgraph.checkpoint.memory", "langchain_core", "openai"];
  const lines = show
    .map((k) => {
      const info = imports[k] || {};
      return `${k}: ${info.ok ? "ok" : "missing"}${info.ok && info.version ? `@${info.version}` : ""}${
        !info.ok && info.error ? ` · ${info.error}` : ""
      }`;
    })
    .join("\n");

  el.textContent = `LLM: ${llmOk} · ${modelProvider} / ${modelName}${llmErr ? ` · ${llmErr}` : ""}\n${lines}`;
}

async function createRun(goal) {
  const resp = await fetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ goal }),
  });
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.error || "create_run_failed");
  }
  return data;
}

function stopPolling() {
  state.polling = false;
  $("stopBtn").disabled = true;
  if (state.pollTimer) {
    clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }
}

async function fetchRun(runId) {
  const resp = await fetch(`/api/runs/${runId}`);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || "fetch_run_failed");
  return data;
}

async function fetchFinal(runId) {
  const resp = await fetch(`/api/runs/${runId}/final`);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || "fetch_final_failed");
  return data;
}

function formatEvent(evt) {
  const node = evt.node || "-";
  const status = evt.status || "-";
  const msg = evt.message || "";
  return `${node} · ${status} · ${msg}`;
}

async function pollOnce(runId) {
  const data = await fetchRun(runId);
  const currentTask = (data.scheduler && data.scheduler.current_task) || null;
  upsertRun({ run_id: runId, trace_id: data.trace_id, status: data.status, stage: data.stage, current_task: currentTask });
  renderTaskPanel(data);

  const prev = state.runSnapshots[runId] || {};
  const key = `${data.status}|${data.stage}|${data.halted_reason || ""}|${data.fatal_error || ""}`;
  if (prev.key !== key) {
    const stage = data.stage || "-";
    const haltedReason = data.halted_reason ? `（${data.halted_reason}）` : "";
    appendMsg(`stage=${stage} status=${data.status}${haltedReason}`, "system");
  }
  state.runSnapshots[runId] = { key };

  if (data.status === "completed") {
    const fin = await fetchFinal(runId);
    $("finalMarkdown").textContent = fin.markdown || "";
    $("copyBtn").disabled = !(fin.markdown || "").trim();
    appendMsg("生成完成。最终文档已加载。", "system");
    stopPolling();
    return;
  }

  if (data.status === "halted") {
    appendMsg(`流程暂停：${data.halted_reason || "unknown"}`, "system");
    stopPolling();
    return;
  }

  if (data.status === "failed") {
    appendMsg(`流程失败：${data.fatal_error || "unknown"}`, "system");
    stopPolling();
    return;
  }
}

function startPolling(runId) {
  stopPolling();
  state.polling = true;
  $("stopBtn").disabled = false;
  const loop = async () => {
    if (!state.polling) return;
    try {
      await pollOnce(runId);
    } catch (e) {
      appendMsg(`轮询失败：${String(e)}`, "system");
      stopPolling();
      return;
    }
    if (!state.polling) return;
    state.pollTimer = setTimeout(loop, 1200);
  };
  loop();
}

function bind() {
  $("newRunBtn").onclick = () => {
    state.activeRunId = null;
    renderRuns();
    resetConversation();
  };

  $("startBtn").onclick = async () => {
    const goal = $("goalInput").value.trim();
    if (!goal) return;
    resetConversation();
    appendMsg(goal, "user");
    appendMsg("已提交任务，等待执行…", "system");
    try {
      const { run_id, trace_id } = await createRun(goal);
      const rec = {
        run_id,
        trace_id,
        goal,
        title: goal.slice(0, 22),
        created_at: Date.now() / 1000,
        status: "queued",
      };
      upsertRun(rec);
      setActiveRun(run_id);
    } catch (e) {
      appendMsg(`创建任务失败：${String(e)}`, "system");
    }
  };

  $("stopBtn").onclick = () => {
    stopPolling();
    appendMsg("已停止轮询。", "system");
  };

  $("copyBtn").onclick = async () => {
    const text = $("finalMarkdown").textContent || "";
    if (!text.trim()) return;
    try {
      await navigator.clipboard.writeText(text);
      appendMsg("已复制到剪贴板。", "system");
    } catch {
      appendMsg("复制失败（浏览器权限限制）。", "system");
    }
  };
}

async function init() {
  loadRuns();
  renderRuns();
  bind();
  await checkHealth();
  const dbg = await fetchDebug();
  const health = await (await fetch("/api/health")).json();
  renderDepsPanel(dbg, health);
  setInterval(async () => {
    await checkHealth();
    const dbg2 = await fetchDebug();
    const health2 = await (await fetch("/api/health")).json();
    renderDepsPanel(dbg2, health2);
  }, 8000);
}

init();
