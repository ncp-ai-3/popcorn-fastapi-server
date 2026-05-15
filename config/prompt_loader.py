"""프롬프트 YAML 로드 및 ChatML 조립."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")
_DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _prompts_dir() -> Path:
    return Path(os.getenv("PROMPTS_DIR", str(_DEFAULT_PROMPTS_DIR)))


@lru_cache(maxsize=1)
def load_prompts() -> dict[str, Any]:
    root = _prompts_dir()
    merged: dict[str, Any] = {}
    if not root.is_dir():
        raise FileNotFoundError(f"prompts directory not found: {root}")
    for path in sorted(root.glob("**/*.yaml")):
        with path.open(encoding="utf-8") as f:
            chunk = yaml.safe_load(f) or {}
        if not isinstance(chunk, dict):
            continue
        merged.update(chunk)
    return merged


def clear_prompt_cache() -> None:
    load_prompts.cache_clear()


def chatml_tokens() -> dict[str, str]:
    cfg = load_prompts().get("chatml") or {}
    return {
        "im_start": cfg.get("im_start", "<|im_start|>"),
        "im_end": cfg.get("im_end", "<|im_end|>"),
    }


def render_template(text: str, **kwargs: Any) -> str:
    """{name} 플레이스홀더만 치환(JSON 중괄호는 그대로 둠)."""

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in kwargs:
            return str(kwargs[key])
        return match.group(0)

    return _PLACEHOLDER_RE.sub(repl, text or "")


def build_chatml(
    *,
    system: str,
    turns: list[dict[str, str]] | None = None,
    final_user: str | None = None,
    assistant_prefix: str = "",
) -> str:
    t = chatml_tokens()
    im_start, im_end = t["im_start"], t["im_end"]
    parts: list[str] = [f"{im_start}system\n{system}{im_end}\n"]
    for turn in turns or []:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        parts.append(f"{im_start}{role}\n{content}{im_end}\n")
    if final_user is not None:
        parts.append(f"{im_start}user\n{final_user}{im_end}\n")
    parts.append(f"{im_start}assistant\n{assistant_prefix}")
    return "".join(parts)
