import logging
import os
import time
from typing import List

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_DEFAULT_PATH = "/api/v1/embed"
_DEFAULT_TIMEOUT = 30.0


def _embed_url() -> str:
    base = os.getenv("EMBED_API_BASE_URL", "").strip().rstrip("/")
    if not base:
        raise ValueError("EMBED_API_BASE_URL is not set (cloud 임베딩 서비스 오리진)")
    path = os.getenv("EMBED_API_PATH", _DEFAULT_PATH).strip()
    if not path.startswith("/"):
        path = "/" + path
    return f"{base}{path}"


def _verify_tls() -> bool:
    return os.getenv("EMBED_API_VERIFY_TLS", "true").lower() not in ("0", "false", "no")


def _timeout_seconds() -> float:
    return float(os.getenv("EMBED_API_TIMEOUT_SECONDS", str(_DEFAULT_TIMEOUT)))


def _expected_dimension() -> int:
    return int(os.getenv("EMBEDDING_DIMENSION", "768"))


async def fetch_query_embedding(text: str) -> List[float]:
    """클라우드 임베딩 API(POST JSON {\"text\": ...}) 호출 후 벡터 반환."""
    url = _embed_url()
    timeout = httpx.Timeout(_timeout_seconds())
    payload = {"text": text or ""}
    raw = text or ""
    logger.debug(
        "[embedding:client] POST %s text_len=%d text_preview=%r",
        url,
        len(raw),
        raw[:200] + ("…" if len(raw) > 200 else ""),
    )
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout, verify=_verify_tls()) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    http_ms = (time.perf_counter() - t0) * 1000
    embedding = data.get("embedding")
    if not isinstance(embedding, list) or len(embedding) == 0:
        raise ValueError("embedding API response: missing or empty 'embedding'")

    vec = [float(x) for x in embedding]
    expected = _expected_dimension()
    if len(vec) != expected:
        raise ValueError(
            f"embedding dimension mismatch: expected {expected}, got {len(vec)}"
        )
    logger.debug(
        "[embedding:client] http_elapsed_ms=%.1f dim=%d",
        http_ms,
        len(vec),
    )
    return vec
