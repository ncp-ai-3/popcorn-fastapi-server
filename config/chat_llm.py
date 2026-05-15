"""채팅·추천(generate) LLM 설정 — 로컬 Llama / Vertex AI."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

RecommendBackend = Literal["local", "vertex"]


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_int(key: str, default: int, *, minimum: int | None = None) -> int:
    raw = _env(key)
    try:
        v = int(raw) if raw else default
    except ValueError:
        logger.warning("%s=%r invalid; using %d", key, raw, default)
        v = default
    if minimum is not None:
        v = max(minimum, v)
    return v


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    try:
        return float(raw) if raw else default
    except ValueError:
        logger.warning("%s=%r invalid; using %s", key, raw, default)
        return default


def _parse_llm_backend(raw: str, *, env_key: str) -> RecommendBackend:
    """local | vertex (gemini/google 은 vertex 로 호환)."""
    b = (raw or "local").lower()
    if b in ("vertex", "vertex_ai", "gemini", "google"):
        if b in ("gemini", "google"):
            logger.info(
                "%s=%r → vertex (API 키 대신 Vertex AI 인증 사용)",
                env_key,
                raw,
            )
        return "vertex"
    return "local"


def _parse_recommend_backend(raw: str) -> RecommendBackend:
    return _parse_llm_backend(raw, env_key="RECOMMEND_LLM_BACKEND")


def _parse_extract_backend(raw: str) -> RecommendBackend:
    return _parse_llm_backend(raw, env_key="EXTRACT_LLM_BACKEND")


@dataclass(frozen=True)
class VertexAIConfig:
    """Vertex AI Gemini — ADC 또는 GOOGLE_APPLICATION_CREDENTIALS."""

    project_id: str
    location: str
    model: str
    temperature: float
    credentials_path: str | None

    def validate(self) -> None:
        if not self.project_id:
            raise ValueError(
                "Vertex AI 사용 시 VERTEX_AI_PROJECT_ID 또는 GOOGLE_CLOUD_PROJECT 가 필요합니다."
            )
        if not self.location:
            raise ValueError("VERTEX_AI_LOCATION 이 필요합니다 (예: asia-northeast3).")


@dataclass(frozen=True)
class ChatLLMConfig:
    """추천(generate)·추출(extract) LLM 및 관련 생성 파라미터."""

    recommend_backend: RecommendBackend
    extract_backend: RecommendBackend
    generate_max_tokens: int
    extract_intent_max_tokens: int
    popup_desc_chars: int
    vertex: VertexAIConfig | None

    @property
    def uses_vertex_for_recommend(self) -> bool:
        return self.recommend_backend == "vertex"

    @property
    def uses_vertex_for_extract(self) -> bool:
        return self.extract_backend == "vertex"

    @property
    def uses_vertex(self) -> bool:
        """Vertex 설정·init 이 필요한지 (추천 또는 추출 중 하나라도 vertex)."""
        return self.uses_vertex_for_recommend or self.uses_vertex_for_extract

    @property
    def uses_local_llama(self) -> bool:
        return self.recommend_backend == "local" and self.extract_backend == "local"


def _resolve_credentials_path(raw: str | None) -> str | None:
    if not raw:
        return None
    p = os.path.expanduser(os.path.expandvars(raw.strip()))
    if not os.path.isabs(p):
        # 프로젝트 루트 = config/ 의 상위
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        p = os.path.normpath(os.path.join(root, p))
    if not os.path.isfile(p):
        raise ValueError(
            f"GOOGLE_APPLICATION_CREDENTIALS 파일이 없습니다: {p} "
            "(secrets/gcp-vertex-sa.json 을 두었는지 확인)"
        )
    return p


def _load_vertex_config() -> VertexAIConfig:
    project = (
        _env("VERTEX_AI_PROJECT_ID")
        or _env("GOOGLE_CLOUD_PROJECT")
        or _env("GCP_PROJECT")
    )
    creds = _resolve_credentials_path(_env("GOOGLE_APPLICATION_CREDENTIALS") or None)
    return VertexAIConfig(
        project_id=project,
        location=_env("VERTEX_AI_LOCATION", "asia-northeast3"),
        model=_env("VERTEX_AI_MODEL") or _env("GEMINI_MODEL", "gemini-2.0-flash"),
        temperature=_env_float("VERTEX_AI_TEMPERATURE", _env_float("GEMINI_TEMPERATURE", 0.7)),
        credentials_path=creds,
    )


@lru_cache(maxsize=1)
def get_chat_llm_config() -> ChatLLMConfig:
    recommend_backend = _parse_recommend_backend(_env("RECOMMEND_LLM_BACKEND", "local"))
    extract_backend = _parse_extract_backend(_env("EXTRACT_LLM_BACKEND", "local"))
    vertex = (
        _load_vertex_config()
        if recommend_backend == "vertex" or extract_backend == "vertex"
        else None
    )
    if vertex is not None:
        vertex.validate()
    cfg = ChatLLMConfig(
        recommend_backend=recommend_backend,
        extract_backend=extract_backend,
        generate_max_tokens=_env_int("LLM_GENERATE_MAX_TOKENS", 1024, minimum=256),
        extract_intent_max_tokens=_env_int("EXTRACT_INTENT_MAX_TOKENS", 200, minimum=64),
        popup_desc_chars=_env_int("LLM_POPUP_DESC_CHARS", 480, minimum=80),
        vertex=vertex,
    )
    v = cfg.vertex
    cred = (
        f"file={os.path.basename(v.credentials_path)}"
        if v and v.credentials_path
        else ("ADC" if v else "n/a")
    )
    vertex_line = ""
    if v:
        vertex_line = (
            f" vertex_project={v.project_id} vertex_location={v.location} "
            f"vertex_model={v.model} vertex_temp={v.temperature:.2f} auth={cred}"
        )
    logger.info(
        "[chat_llm_config] recommend_backend=%s extract_backend=%s "
        "generate_max_tokens=%d extract_intent_max_tokens=%d popup_desc_chars=%d "
        "chat_timeout_s=%.0f llm_call_timeout_s=%.0f%s",
        cfg.recommend_backend,
        cfg.extract_backend,
        cfg.generate_max_tokens,
        cfg.extract_intent_max_tokens,
        cfg.popup_desc_chars,
        chat_request_timeout_seconds(),
        llm_call_timeout_seconds(),
        vertex_line,
    )
    return cfg


def recommend_llm_backend_label() -> str:
    """로그·에러 메시지용 백엔드 문자열."""
    return get_chat_llm_config().recommend_backend


def chat_request_timeout_seconds() -> float:
    """POST /chat LangGraph 전체 상한(초). 초과 시 연결 종료."""
    return max(1.0, _env_float("CHAT_REQUEST_TIMEOUT_SECONDS", 40.0))


def llm_call_timeout_seconds() -> float:
    """단일 LLM 호출(Vertex·로컬 Llama) 상한(초)."""
    return max(1.0, _env_float("LLM_CALL_TIMEOUT_SECONDS", 40.0))
