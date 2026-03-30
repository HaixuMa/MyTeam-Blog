from __future__ import annotations

import json
import re
import socket
import time
import urllib.request
import ssl
from datetime import datetime, timezone
from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel
import certifi

from schemas.state import GraphState

TModel = TypeVar("TModel", bound=BaseModel)


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


def record_prompt_snapshot(
    *,
    state: GraphState,
    node: str,
    role: str,
    system_prompt: str,
    user_prompt: str,
) -> None:
    history = state.get("prompt_history")
    if not isinstance(history, list):
        history = []
        state["prompt_history"] = history
    history.append(
        {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "node": node,
            "role": role,
            "system_prompt": system_prompt[:12000],
            "user_prompt": user_prompt[:12000],
        }
    )


def invoke_structured_output(
    *,
    llm: BaseChatModel,
    schema: type[TModel],
    messages: list[Any],
) -> TModel:
    raw_cfg = getattr(llm, "_raw_openai_compatible", None)
    if isinstance(raw_cfg, dict) and raw_cfg.get("base_url") and raw_cfg.get("api_key") and raw_cfg.get("model"):
        try:
            content = _raw_openai_compatible_chat(
                base_url=str(raw_cfg["base_url"]),
                api_key=str(raw_cfg["api_key"]),
                model=str(raw_cfg["model"]),
                temperature=float(raw_cfg.get("temperature", 0.2)),
                timeout_s=float(raw_cfg.get("timeout_s", 60.0)),
                messages=messages,
            )
            return _parse_content_to_schema(schema=schema, content=content)
        except Exception as e:
            try:
                msg = llm.invoke(messages)
                content = getattr(msg, "content", "")
                if not isinstance(content, str):
                    content = str(content)
                return _parse_content_to_schema(schema=schema, content=content)
            except Exception as e2:
                raise RuntimeError(f"{e2}; raw={e}") from e2

    methods: list[str | None] = ["json_mode", "function_calling", None]
    last_error: Exception | None = None
    for m in methods:
        try:
            if m is None:
                structured = llm.with_structured_output(schema)
            else:
                structured = llm.with_structured_output(schema, method=m)
            return structured.invoke(messages)
        except TypeError as e:
            last_error = e
            if "NoneType" in str(e) and "iterable" in str(e):
                continue
            raise
        except Exception as e:
            last_error = e
            continue

    try:
        msg = llm.invoke(messages)
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            content = str(content)
        return _parse_content_to_schema(schema=schema, content=content)
    except Exception as e:
        raise RuntimeError(str(last_error) if last_error else str(e))


def _raw_openai_compatible_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout_s: float,
    messages: list[Any],
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    oa_messages: list[dict[str, str]] = []
    for m in messages:
        if isinstance(m, dict) and "role" in m and "content" in m:
            oa_messages.append({"role": str(m["role"]), "content": str(m["content"])})
            continue
        msg_type = getattr(m, "type", None)
        content = getattr(m, "content", "")
        if msg_type == "system":
            oa_messages.append({"role": "system", "content": str(content)})
        elif msg_type == "human":
            oa_messages.append({"role": "user", "content": str(content)})
        elif msg_type == "ai":
            oa_messages.append({"role": "assistant", "content": str(content)})
        else:
            oa_messages.append({"role": "user", "content": str(content)})

    body = {
        "model": model,
        "messages": oa_messages,
        "temperature": temperature,
        "stream": False,
        "max_tokens": 4096,
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("Authorization", "Bearer " + api_key)

    ctx = ssl.create_default_context(cafile=certifi.where())
    resp = urllib.request.urlopen(req, timeout=timeout_s, context=ctx)
    raw = resp.read().decode("utf-8", errors="replace")
    obj = json.loads(raw) if raw.strip() else {}
    if isinstance(obj, dict) and obj.get("error"):
        err = obj.get("error")
        if isinstance(err, (dict, list)):
            raise RuntimeError(json.dumps(err, ensure_ascii=False)[:1200])
        raise RuntimeError(str(err)[:1200])
    choices = obj.get("choices") or []
    if not choices:
        raise RuntimeError(f"no_choices: {raw[:1200]}")
    msg = (choices[0] or {}).get("message") or {}
    c = msg.get("content")
    if isinstance(c, str) and c.strip():
        return c.strip()
    rc = msg.get("reasoning_content")
    if isinstance(rc, str) and rc.strip():
        return rc.strip()
    text = str(c or rc or "").strip()
    return text


def _extract_json(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
        t = t.strip()
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        return t[start : end + 1]
    start = t.find("[")
    end = t.rfind("]")
    if start >= 0 and end > start:
        return t[start : end + 1]
    return t


def _parse_content_to_schema(*, schema: type[TModel], content: str) -> TModel:
    last: Exception | None = None

    for cand in _iter_json_candidates(content):
        try:
            data = json.loads(cand)
        except Exception as e:
            last = e
            continue

        try:
            return schema.model_validate(data)
        except Exception as e:
            last = e
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, (dict, list)):
                        continue
                    try:
                        return schema.model_validate(item)
                    except Exception as e2:
                        last = e2
                        continue

            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, dict):
                        try:
                            return schema.model_validate(v)
                        except Exception as e2:
                            last = e2
                            continue
                    if isinstance(v, list):
                        for item in v:
                            if not isinstance(item, (dict, list)):
                                continue
                            try:
                                return schema.model_validate(item)
                            except Exception as e3:
                                last = e3
                                continue
            continue

    raise RuntimeError(str(last) if last else "structured_output_parse_failed")


def _iter_json_candidates(text: str) -> list[str]:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
        t = t.strip()

    candidates: list[str] = []
    candidates.append(t)

    extracted = _extract_json(t)
    if extracted != t:
        candidates.append(extracted)

    candidates.extend(_extract_braced_blocks(t, open_ch="{", close_ch="}"))
    candidates.extend(_extract_braced_blocks(t, open_ch="[", close_ch="]"))

    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        c2 = c.strip()
        if not c2 or c2 in seen:
            continue
        seen.add(c2)
        out.append(c2)
    return out


def _extract_braced_blocks(text: str, *, open_ch: str, close_ch: str) -> list[str]:
    blocks: list[str] = []
    depth = 0
    start: int | None = None
    in_str = False
    escape = False

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == open_ch:
            if depth == 0:
                start = i
            depth += 1
            continue
        if ch == close_ch and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                blocks.append(text[start : i + 1])
                start = None
    return blocks

