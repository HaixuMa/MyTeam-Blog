from __future__ import annotations

import re


def sanitize_user_text(text: str) -> str:
    """
    基础 prompt 注入防护：
    - 剔除可疑指令片段（并不依赖黑名单完全防御，核心仍以结构化输出契约与 Harness 校验为主）
    - 限制超长输入，避免上下文挤占
    """
    cleaned = text.strip()
    cleaned = re.sub(r"(?i)\b(ignore|bypass|override)\b.{0,40}\b(system|policy|instruction)\b", "", cleaned)
    cleaned = re.sub(r"(?i)\b(disable|turn off)\b.{0,40}\b(safety|guardrails)\b", "", cleaned)
    cleaned = cleaned.replace("\u0000", "")
    return cleaned[:2000]


def enforce_markdown_no_html(md: str) -> str:
    return re.sub(r"<script[\s\S]*?</script>", "", md, flags=re.IGNORECASE)

