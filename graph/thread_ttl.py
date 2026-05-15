"""LangGraph checkpointer thread idle TTL (in-process MemorySaver)."""

from __future__ import annotations

import asyncio
import logging
import time
from threading import Lock
from typing import Any

from config.chat_llm import chat_state_ttl_seconds

logger = logging.getLogger(__name__)

_last_access: dict[str, float] = {}
_lock = Lock()


def touch_thread(thread_id: str) -> None:
    with _lock:
        _last_access[thread_id] = time.monotonic()


def _forget_thread(thread_id: str) -> None:
    with _lock:
        _last_access.pop(thread_id, None)


def _idle_seconds(thread_id: str) -> float | None:
    with _lock:
        ts = _last_access.get(thread_id)
    if ts is None:
        return None
    return time.monotonic() - ts


async def _delete_checkpoint_thread(checkpointer: Any, thread_id: str) -> None:
    if checkpointer is None:
        return
    if hasattr(checkpointer, "adelete_thread"):
        await checkpointer.adelete_thread(thread_id)
        return
    if hasattr(checkpointer, "delete_thread"):
        await asyncio.to_thread(checkpointer.delete_thread, thread_id)
        return
    logger.warning(
        "[thread_ttl] checkpointer %s has no delete_thread; skip thread_id=%s",
        type(checkpointer).__name__,
        thread_id,
    )


async def expire_thread_if_idle(checkpointer: Any, thread_id: str) -> bool:
    """마지막 성공 /chat 이후 TTL 초과 시 체크포인트 삭제. True면 만료·삭제됨."""
    ttl = chat_state_ttl_seconds()
    if ttl <= 0:
        return False
    idle = _idle_seconds(thread_id)
    if idle is None or idle < ttl:
        return False
    await _delete_checkpoint_thread(checkpointer, thread_id)
    _forget_thread(thread_id)
    logger.info(
        "[thread_ttl] expired thread_id=%s idle_sec=%.0f ttl_sec=%.0f",
        thread_id,
        idle,
        ttl,
    )
    return True
