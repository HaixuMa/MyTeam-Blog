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
