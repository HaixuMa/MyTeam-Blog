# MyTeam-Blog（LangChain + LangGraph + Harness Engineering）

MyTeam-Blog 是一套可运行的多 Agent 深度研究与技术写作流水线：研究目标 → 规划 → 多维调研 → 深度分析 → 技术文章撰写 → 配图生成与嵌入 → 审核排版 → 终稿输出。

## 目录结构

- [main.py](main.py)：程序入口（全量/分步执行、恢复）
- [orchestrator.py](orchestrator.py)：LangGraph 全局编排器 + CheckpointSaver
- [harness/base.py](harness/base.py)：单 Agent Harness（契约/校验/重试/超时/可观测事件）
- [schemas/](schemas/)：全量 Pydantic v2 契约模型
- [agents/](agents/)：7 类角色（规划、4 子调研、聚合、分析、写作、配图、审核）
- [tools/](tools/)：调研与配图工具封装（权限 + 限流 + 超时/降级）
- [config.py](config.py)：配置与环境变量加载

## Harness Engineering 标准（本项目落地约定）

本项目把“多 Agent”当作一组可编排的工程组件，而不是一段不可控的 prompt 脚本。核心目标是：强类型契约、可验证、可恢复、可观测、可停止。

### 1）强类型输入/输出契约（Contracts）

- 所有 Agent 的输入/输出都必须是 Pydantic v2 模型（`schemas/`），并通过 `with_structured_output` 强制结构化输出。
- 约束来源：
  - 契约模型：`schemas/*.py`
  - 运行期强制：`agents/*.py`（各 Agent `_invoke` 内调用 `with_structured_output`）

### 2）前置/后置校验（Pre/Post Validate）

- `pre_validate`：输入合规、状态/权限检查、必要字段存在性检查
- `post_validate`：输出质量检查（必要字段、长度、注入/安全规则），失败则触发重试或降级
- 落地位置：`harness/base.py` 的 `AgentHarness.run` 调用链 + 各 Agent Harness 自己实现

### 3）可恢复重试 + 超时（Retry/Timeout）

- 可恢复错误（ValidationError / ContractViolationError / RecoverableHarnessError / TimeoutError）按 `MAX_AGENT_RETRIES` 重试
- 单次调用统一超时：`REQUEST_TIMEOUT_S`
- 落地位置：`harness/base.py`（重试循环 + `_invoke_with_timeout`），`config.py`（统一配置）

### 4）可观测性（Observability）

- 每次 Agent 调用都会写入 `state["execution_events"]`（start/success/retry/degraded/failed）
- 事件结构：`schemas/common.py::ExecutionEvent`
- 落地位置：`harness/base.py`（事件写入 + 日志埋点）

### 5）全局编排：状态机 + Checkpoint（Orchestration）

- 全流程使用 LangGraph 的 StateGraph 编排，节点间以 `GraphState` 传递中间产物
- 支持 SQLite checkpoint：可中断、可恢复
- 落地位置：`orchestrator.py`（节点与边、路由、checkpointer），`main.py`（resume）

### 6）系统上下文注入（Project Context）

- 为了让每个 Agent “知道自己在什么流水线里”，入口会把项目环境与模型信息写入 `state["project_context"]`
- 每个 Agent 的 `SystemMessage` 会在自身 system prompt 前拼接通用上下文（trace_id/stage/node/pipeline/data_dir 等）
- 落地位置：
  - 注入：`main.py`
  - 生成：`agents/context.py`
  - 使用：各 Agent 的 `SystemMessage` 构造

### 7）提示词可追溯（Prompt Snapshots）

- 每次 Agent 调用前，会把 system/user prompt 快照写入 `state["prompt_history"]`（用于定位质量问题与迭代 prompt）
- 落地位置：`agents/prompting.py::record_prompt_snapshot` + 各 Agent `_invoke` 中调用

### 8）死循环护栏（Loop Guard）

目标：防止审稿回跳或异常路由导致隐蔽循环，保证流程“可停止”。

- 硬上限：
  - 总节点步数：`MAX_TOTAL_NODE_STEPS`
  - 单节点访问次数：`MAX_NODE_VISITS_PER_NODE`
- 模式检测：检测尾部重复序列（例如 `A->B->A->B`），命中直接 halt
- 落地位置：`orchestrator.py`（在每个节点执行前统一计数与检测，超过阈值写入 `halted_reason` 并 END）

### 9）轻量质量记忆（Quality Memory）

- 审稿输出会把 issue 分类统计累积到 `data/memory/quality_feedback.json`，用于后续“质量趋势/常见问题”分析
- 落地位置：`memory_store.py` + `orchestrator.py`（auditing 节点）

## 快速启动

### 1）安装依赖（Python 3.11+）

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2）配置环境变量

复制环境变量模板并填入 Key：

```bash
copy .env.example .env
```

必须项取决于你的模型：

- `MODEL_PROVIDER=openai`：需要 `OPENAI_API_KEY`
- `MODEL_PROVIDER=anthropic`：需要 `ANTHROPIC_API_KEY`
- `TAVILY_API_KEY` 可选（缺失时调研会降级为“无 Tavily 结果”）

### 3）运行（全量执行）

```bash
python main.py --mode full
```

你也可以传入自定义研究目标：

```bash
python main.py --mode full --goal "调研 Agentic RAG 在企业知识库中的工程落地与评测体系"
```

### 4）输出位置

运行完成后会写入：

`data/outputs/<trace_id>/`

- `final_article.md`
- `audit_report.json`
- `execution_events.json`

## Harness 相关环境变量

除模型与工具相关配置外，本项目还提供以下 Harness 控制项（见 `.env.example`）：

- `MAX_AGENT_RETRIES`：单节点 Agent 重试次数
- `MAX_IMAGE_RETRIES`：单张图片生成重试次数
- `MAX_AUDIT_ROUNDS`：审稿失败回跳次数上限
- `MAX_TOTAL_NODE_STEPS`：全流程总节点执行步数上限（死循环硬护栏）
- `MAX_NODE_VISITS_PER_NODE`：单个节点允许被执行的最大次数（死循环硬护栏）

## 分步执行与中断恢复

### 分步执行

```bash
python main.py --mode step
```

每次只推进一个 LangGraph 节点，适合调试与观察中间产物。

### 恢复执行（Checkpoint）

当规划阶段触发“需要澄清”时，流程会进入 `halted`，终端会打印 `trace_id` 与问题列表。

提供澄清答案并恢复：

```bash
python main.py --mode full --resume-trace-id trace_xxx --clarifications-json "{\"scope\":\"企业知识库（百万级文档）\",\"constraint\":\"必须支持私有化\"}"
```

## 可扩展方式

### 新增 Agent

1. 在 [schemas/](schemas/) 中新增输入/输出契约模型
2. 在 [agents/](agents/) 中实现 `AgentHarness[In, Out]`：
   - `pre_validate`：输入合规与权限/流程校验
   - `post_validate`：输出质量校验，不合格触发重试/打回
3. 在 [orchestrator.py](orchestrator.py) 中增加节点与边，纳入全局状态机

### 替换模型

修改 `.env`：

- `MODEL_PROVIDER` / `MODEL_NAME`
- 若为 OpenAI 兼容网关：设置 `MODEL_PROVIDER=openai_compatible` 与 `MODEL_BASE_URL`

### 调整 Harness 规则

统一在：

- [config.py](config.py) 中配置重试/超时/限流
- 各 Harness 的 `pre_validate` / `post_validate` 内增改规则
