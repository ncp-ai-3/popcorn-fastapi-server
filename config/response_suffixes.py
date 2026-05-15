"""추천 답변 말미 고정 문장 — LLM이 아닌 후처리 전용."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).with_name("response_suffixes.yaml")


def _suffixes_path() -> Path:
    return Path(os.getenv("RESPONSE_SUFFIXES_PATH", str(_DEFAULT_PATH)))


@lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    path = _suffixes_path()
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def clear_response_suffixes_cache() -> None:
    _load.cache_clear()


def missing_slot_suffix(vec: tuple[int, int, int] | None) -> str:
    if not vec or len(vec) != 3:
        return ""
    table = _load().get("missing_slot_suffix") or {}
    key = f"{int(vec[0])},{int(vec[1])},{int(vec[2])}"
    return str(table.get(key) or "").strip()


def non_action_popup_invite() -> str:
    return str(_load().get("non_action_popup_invite") or "").strip()
