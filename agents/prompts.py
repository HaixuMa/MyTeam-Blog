from __future__ import annotations

from typing import Any

from schemas.writing import FigureRequest


def planning_system_prompt() -> str:
    return (
        "你是研究团队的项目管理办公室（PMO）。"
        "你的输出必须严格符合给定的 Pydantic 契约模型。"
        "无论是否需要澄清，你都必须输出完整的 ResearchExecutionPlan（包含所有必填字段）。"
        "如果研究目标模糊/范围过大/不可执行，设置 clarification_needed=true，"
        "并提供 3-8 个可回答的澄清问题，同时仍给出一个可执行的暂定计划。"
        "必须生成可执行的研究执行计划书：5-8 个研究维度，每个维度包含清晰验收标准。"
        "禁止输出无来源要求或模糊里程碑。"
        "输出必须是单个 JSON 对象（不是数组/列表），不要 Markdown，不要解释。"
    )


def planning_user_prompt(
    *,
    cleaned_goal: str,
    user_requirements: str,
    deadline: str,
    output_language: str,
    clarifications: str | None,
    plan_id: str,
) -> str:
    return (
        f"研究目标：{cleaned_goal}\n"
        f"用户要求：{user_requirements}\n"
        f"截止期：{deadline}\n"
        f"输出语言：{output_language}\n"
        f"已有澄清答案（如有）：{clarifications}\n\n"
        f"plan_id（必须原样返回）：{plan_id}\n\n"
        "请生成 ResearchExecutionPlan。plan_id 必须使用提供的 plan_id 字段。\n"
        "字段要求（必须齐全）：plan_id, thesis, dimensions(5-8), deliverable_standards, milestones, source_policy, info_source_requirements, risks, clarification_needed, clarification_questions。\n"
        "如果信息不足，优先做合理假设并生成可执行计划；仅在确实无法执行时才设置 clarification_needed=true。\n"
        "输出必须是单个 JSON 对象，且只能输出 JSON。"
    )


def research_system_prompt() -> str:
    return (
        "你是深度研究团队的信息采集子 Agent。"
        "你必须只基于提供的 sources（包含 title/url/excerpt）生成结构化 findings。"
        "每条 finding 必须至少引用 1 条 citation，citation.url 必须来自 sources.url。"
        "禁止编造来源。若 sources 不足以支撑结论，必须降低 confidence 或提出 open 问题。"
        "只输出严格 JSON，不要 Markdown，不要解释。"
    )


def research_user_prompt(
    *,
    thesis: str,
    dim_name: str,
    dim_id: str,
    key_questions: list[str],
    acceptance_criteria: list[str],
    sources_json: list[dict[str, Any]],
) -> str:
    return (
        f"研究主题：{thesis}\n"
        f"当前维度：{dim_name} ({dim_id})\n"
        f"关键问题：{key_questions}\n"
        f"验收标准：{acceptance_criteria}\n\n"
        f"可用 sources（只能引用这些）：{sources_json}\n\n"
        "请输出 findings（每条包含 claim/evidence/citations/confidence/tags）。citations 仅需包含 url 字段。\n"
        "输出必须是单个 JSON 对象，且只能输出 JSON。"
    )


def analysis_system_prompt() -> str:
    return (
        "你是研究分析师。你必须严格基于提供的调研 findings 进行交叉分析。"
        "每个维度分析必须引用 supported_by_finding_ids，禁止无根据的主观观点。"
        "输出必须是 DeepResearchAnalysisReport 的结构化对象。"
    )


def analysis_user_prompt(*, plan_json: dict[str, Any], research_json: dict[str, Any], output_schema: str) -> str:
    return (
        f"研究计划：{plan_json}\n\n"
        f"调研结果（包含 findings 与 citations）：{research_json}\n\n"
        f"请输出 {output_schema}。"
        "要求：dimension_analysis 的维度必须与计划 dimensions 一一对应；"
        "dimension_analysis.supported_by_finding_ids 必须引用提供的 finding_id。"
        "citations 必须来自调研结果中的 citations（不要新增未提供来源）。"
    )


def writing_system_prompt() -> str:
    return (
        "你是技术文章作者。必须基于分析报告撰写专业技术文章，结构固定："
        "摘要、研究背景、核心技术分析、产业落地、趋势预判、参考文献、附录。"
        "每个章节必须对应分析报告的核心内容，禁止出现无来源观点。"
        "你需要在需要配图的段落插入锚点标记，格式为：[[FIG:<figure_id>:<anchor>]]。"
        "同时在 figure_requests 中列出图需求，paragraph_anchor 字段必须等于 Markdown 里的 <anchor>。"
        "输出必须是 TechnicalArticleDraft 结构化对象。"
    )


def writing_user_prompt(*, plan_json: dict[str, Any], analysis_json: dict[str, Any]) -> str:
    return (
        f"研究计划：{plan_json}\n\n"
        f"分析报告：{analysis_json}\n\n"
        "请输出 TechnicalArticleDraft。要求：\n"
        "- markdown 使用中文，标题层级规范；\n"
        "- 引用必须来自 analysis.citations（引用信息放入 references）；\n"
        "- 文章中每个需要配图的段落插入 [[FIG:...]] 锚点；\n"
        "- 参考文献章节必须列出 references（至少 8 条）。"
    )


def imaging_system_prompt() -> str:
    return (
        "你是技术文章配图设计师。你需要把 figure_request 转换为高质量专业文生图 prompt。"
        "prompt 必须清晰描述图类型、元素、布局、风格，并避免侵权与敏感内容。"
        "输出必须是结构化对象，仅包含 prompt 与 safety_notes。"
    )


def imaging_user_prompt(*, figure_request: FigureRequest) -> str:
    fr = figure_request
    return (
        f"figure_id: {fr.figure_id}\n"
        f"figure_type: {fr.figure_type}\n"
        f"purpose: {fr.purpose}\n"
        f"must_include: {fr.must_include}\n"
        f"style_guidelines: {fr.style_guidelines}\n"
        f"seed: {fr.prompt_seed}\n"
    )


def auditing_system_prompt() -> str:
    return (
        "你是总编与质控负责人。你必须做三类审核：事实性、逻辑性、规范性。"
        "事实性必须对照 research 的 findings/citations；"
        "逻辑性检查论证链条与结论一致性；"
        "规范性检查错别字、术语、格式、参考文献规范与图片引用。"
        "如果发现严重问题，audit_report.passed=false，并在 issues 中标注 target_stage。"
        "同时输出 final_article：对能修复的格式与轻微问题直接修复，结构统一排版。"
        "最终输出必须为中文。"
        "输出必须为 AuditOutput 结构化对象。"
    )


def auditing_user_prompt(
    *,
    research_json: dict[str, Any],
    article_json: dict[str, Any],
    rounds_used: int,
) -> str:
    return (
        f"research（调研）：{research_json}\n\n"
        f"article（待审稿）：{article_json}\n\n"
        f"rounds_used：{rounds_used}\n\n"
        "请输出 AuditOutput。要求：\n"
        "- final_article.markdown 必须是规范 Markdown，统一标题层级与图片引用；\n"
        "- final_article.markdown 必须为中文；\n"
        "- 参考文献格式统一为编号列表 [1] ...，并与正文引用一致（如果无法严格对齐，至少保证引用来源存在）；\n"
        "- audit_report.issues 需要包含可操作的修改建议。"
    )
