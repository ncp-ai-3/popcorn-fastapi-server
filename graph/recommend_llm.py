"""추천(generate) 단계 전용 텍스트 완성 — 로컬 Llama 또는 Vertex AI."""

from __future__ import annotations

import concurrent.futures
import logging
import os
import time
from typing import Any, Callable, TypeVar

from config.chat_llm import (
    ChatLLMConfig,
    RecommendBackend,
    get_chat_llm_config,
    llm_call_timeout_seconds,
)

T = TypeVar("T")

_LLM_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="llm_call",
)

logger = logging.getLogger(__name__)

_vertex_initialized = False

# gemini-2.5-* 는 thinking 토큰이 max_output_tokens 안에서 소비됨.
# ChatML stop_sequences 를 Vertex에 넘기면 출력 0건·MAX_TOKENS 로 resp.text 가 실패하는 경우가 있음.
_VERTEX_THINKING_MODEL_MARKERS = ("2.5", "2-5")


class LLMCallTimeoutError(TimeoutError):
    """단일 LLM 호출이 LLM_CALL_TIMEOUT_SECONDS 를 초과."""

    def __init__(self, *, log_tag: str, timeout_seconds: float):
        self.log_tag = log_tag
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"{log_tag}: LLM 호출이 {timeout_seconds:.0f}초 내에 끝나지 않았습니다."
        )


def _run_with_timeout(
    fn: Callable[[], T],
    *,
    timeout_seconds: float,
    log_tag: str,
) -> T:
    future = _LLM_EXECUTOR.submit(fn)
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError as e:
        future.cancel()
        logger.error(
            "[%s] timeout after %.1fs (LLM_CALL_TIMEOUT_SECONDS)",
            log_tag,
            timeout_seconds,
        )
        raise LLMCallTimeoutError(
            log_tag=log_tag, timeout_seconds=timeout_seconds
        ) from e


def _vertex_max_output_tokens(requested: int, model: str) -> int:
    """thinking 모델용 출력 상한 — 요청값만으로는 내부 추론 토큰에 밀릴 수 있음."""
    cap = min(max(1, requested), 8192)
    if any(m in (model or "") for m in _VERTEX_THINKING_MODEL_MARKERS):
        return min(max(cap, cap + 1024, 2048), 8192)
    return cap


def _apply_stop_sequences(text: str, stop_sequences: list[str]) -> str:
    if not text:
        return text
    out = text
    for stop in stop_sequences:
        if not stop:
            continue
        idx = out.find(stop)
        if idx >= 0:
            out = out[:idx]
    return out.strip()


def recommend_llm_backend() -> str:
    return get_chat_llm_config().recommend_backend


def extract_llm_backend() -> str:
    return get_chat_llm_config().extract_backend


def complete_recommend_prompt(
    prompt: str,
    *,
    max_tokens: int | None = None,
    stop_sequences: list[str],
) -> str:
    """ChatML 형태 프롬프트 전체를 받아 assistant 연속 텍스트만 반환 (추천 generate)."""
    cfg = get_chat_llm_config()
    cap = max_tokens if max_tokens is not None else cfg.generate_max_tokens
    return _complete_chatml_prompt(
        prompt,
        max_tokens=cap,
        stop_sequences=stop_sequences,
        backend=cfg.recommend_backend,
        log_tag="recommend_llm",
    )


def complete_extract_prompt(
    prompt: str,
    *,
    max_tokens: int,
    stop_sequences: list[str],
) -> str:
    """의도 JSON·키워드·임베딩 문장 등 extract 단계 ChatML 완성."""
    cfg = get_chat_llm_config()
    return _complete_chatml_prompt(
        prompt,
        max_tokens=max_tokens,
        stop_sequences=stop_sequences,
        backend=cfg.extract_backend,
        log_tag="extract_llm",
    )


def _complete_chatml_prompt(
    prompt: str,
    *,
    max_tokens: int,
    stop_sequences: list[str],
    backend: RecommendBackend,
    log_tag: str,
) -> str:
    logger.info(
        "[%s] start backend=%s prompt_chars=%d max_tokens=%d",
        log_tag,
        backend,
        len(prompt),
        max_tokens,
    )
    timeout_s = llm_call_timeout_seconds()
    if backend == "vertex":
        cfg = get_chat_llm_config()
        assert cfg.vertex is not None
        return _run_with_timeout(
            lambda: _complete_vertex(
                prompt,
                cfg=cfg,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences,
                log_tag=log_tag,
            ),
            timeout_seconds=timeout_s,
            log_tag=log_tag,
        )
    return _run_with_timeout(
        lambda: _complete_local_llama(
            prompt,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            log_tag=log_tag,
        ),
        timeout_seconds=timeout_s,
        log_tag=log_tag,
    )


def _count_prompt_tokens(llm: Any, prompt: str) -> int:
    try:
        return len(llm.tokenize(prompt.encode("utf-8")))
    except Exception:
        return max(1, len(prompt) // 3)


def _fit_local_prompt_and_max_tokens(
    llm: Any,
    prompt: str,
    *,
    requested_max_tokens: int,
    log_tag: str,
) -> tuple[str, int]:
    """n_ctx 초과 방지: 프롬프트 토큰 수에 맞춰 잘라내고 max_tokens 캡."""
    from graph.model import llm_n_ctx

    n_ctx = llm_n_ctx()
    margin = 48
    min_output = 128
    prompt_tokens = _count_prompt_tokens(llm, prompt)
    budget = n_ctx - margin

    if prompt_tokens + min_output > budget:
        keep_tokens = max(256, budget - min_output)
        try:
            tokens = llm.tokenize(prompt.encode("utf-8"))[:keep_tokens]
            prompt = llm.detokenize(tokens)
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8", errors="ignore")
            prompt_tokens = len(tokens)
            logger.warning(
                "[%s:local] prompt truncated to %d tokens (n_ctx=%d)",
                log_tag,
                prompt_tokens,
                n_ctx,
            )
        except Exception as e:
            logger.warning("[%s:local] prompt truncate failed: %s", log_tag, e)
            prompt = prompt[: keep_tokens * 3]

    max_out = min(requested_max_tokens, max(min_output, budget - prompt_tokens))
    if max_out < requested_max_tokens:
        logger.warning(
            "[%s:local] max_tokens capped %d -> %d (n_ctx=%d prompt_tokens≈%d)",
            log_tag,
            requested_max_tokens,
            max_out,
            n_ctx,
            prompt_tokens,
        )
    return prompt, max_out


def _complete_local_llama(
    prompt: str,
    *,
    max_tokens: int,
    stop_sequences: list[str],
    log_tag: str = "recommend_llm",
) -> str:
    from graph.model import get_llm

    llm = get_llm()
    prompt, max_tokens = _fit_local_prompt_and_max_tokens(
        llm, prompt, requested_max_tokens=max_tokens, log_tag=log_tag
    )
    if log_tag == "recommend_llm":
        # Vertex(full) = "\n[팝업 목록]:"  /  Local(lean) = "[팝업]\n"
        full_idx = prompt.rfind("\n[팝업 목록]:")
        lean_idx = prompt.rfind("[팝업]\n")
        if full_idx >= 0:
            popup_idx, popup_kind = full_idx, "full"
        elif lean_idx >= 0:
            popup_idx, popup_kind = lean_idx, "lean"
        else:
            popup_idx, popup_kind = -1, "none"
        logger.info(
            "[%s:local] prompt_after_fit chars=%d has_popup_block=%s popup_block_kind=%s popup_block_offset=%s",
            log_tag,
            len(prompt),
            popup_idx >= 0,
            popup_kind,
            popup_idx if popup_idx >= 0 else None,
        )
        if popup_idx >= 0:
            tail = prompt[popup_idx:].replace("\n", " ")
            if len(tail) > 400:
                tail = tail[:399] + "…"
            logger.debug("[%s:local] prompt_popup_tail=%r", log_tag, tail)
    t0 = time.perf_counter()
    out: dict[str, Any] = llm(
        prompt,
        max_tokens=max_tokens,
        stop=stop_sequences,
        echo=False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    choices = out.get("choices") or []
    if not choices:
        logger.error(
            "[%s:local] empty choices keys=%s elapsed_ms=%.1f",
            log_tag,
            list(out.keys()) if isinstance(out, dict) else type(out),
            elapsed_ms,
        )
        return ""
    text = (choices[0].get("text") or "").strip()
    logger.info(
        "[%s:local] done chars_out=%d elapsed_ms=%.1f",
        log_tag,
        len(text),
        elapsed_ms,
    )
    return text


def _ensure_vertex_init() -> None:
    """vertexai.init 1회 — 서비스 계정 JSON 또는 ADC."""
    global _vertex_initialized
    if _vertex_initialized:
        return
    import vertexai

    cfg = get_chat_llm_config()
    assert cfg.vertex is not None
    v = cfg.vertex

    if v.credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = v.credentials_path
        logger.info(
            "[recommend_llm:vertex] credentials_file=%s project=%s location=%s",
            os.path.basename(v.credentials_path),
            v.project_id,
            v.location,
        )
    else:
        logger.info(
            "[recommend_llm:vertex] ADC(기본 자격증명) project=%s location=%s",
            v.project_id,
            v.location,
        )

    vertexai.init(project=v.project_id, location=v.location)
    _vertex_initialized = True


def _complete_vertex(
    prompt: str,
    *,
    cfg: ChatLLMConfig,
    max_tokens: int,
    stop_sequences: list[str],
    log_tag: str = "recommend_llm",
) -> str:
    try:
        from vertexai.generative_models import GenerationConfig, GenerativeModel
    except ImportError as e:
        raise RuntimeError(
            "RECOMMEND_LLM_BACKEND=vertex 인데 google-cloud-aiplatform 이 없습니다. "
            "`pip install google-cloud-aiplatform` 후 다시 시도하세요."
        ) from e

    assert cfg.vertex is not None
    v = cfg.vertex
    _ensure_vertex_init()

    stops = [s for s in stop_sequences if s and str(s).strip()][:5]
    max_output_tokens = _vertex_max_output_tokens(max_tokens, v.model)
    # Vertex/Gemini: stop_sequences 는 API에 넘기지 않고 응답 후처리 (2.5 thinking + stop 충돌 방지)
    gen_cfg = GenerationConfig(
        max_output_tokens=max_output_tokens,
        temperature=v.temperature,
    )

    logger.info(
        "[%s:vertex] calling Gemini model=%s project=%s location=%s "
        "prompt_chars=%d max_output_tokens=%d (requested=%d) temperature=%.2f stop_postprocess=%s",
        log_tag,
        v.model,
        v.project_id,
        v.location,
        len(prompt),
        max_output_tokens,
        max_tokens,
        v.temperature,
        stops or None,
    )

    model = GenerativeModel(v.model)
    t0 = time.perf_counter()
    resp = model.generate_content(prompt, generation_config=gen_cfg)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    text = _apply_stop_sequences(_vertex_response_text(resp), stops)
    preview = (text[:80] + "…") if len(text) > 80 else text
    logger.info(
        "[%s:vertex] done model=%s chars_out=%d elapsed_ms=%.1f answer_preview=%r",
        log_tag,
        v.model,
        len(text),
        elapsed_ms,
        preview.replace("\n", " "),
    )
    return text


def _vertex_response_text(resp: Any) -> str:
    try:
        return (resp.text or "").strip()
    except ValueError as e:
        logger.warning(
            "[vertex] resp.text failed: %s | %s",
            e,
            _vertex_response_debug(resp),
        )
        parts_text = _vertex_collect_parts_text(resp)
        if parts_text:
            logger.info(
                "[vertex] recovered text from candidate parts chars=%d",
                len(parts_text),
            )
            return parts_text
        return ""
    except AttributeError:
        return ""


def _vertex_response_debug(resp: Any) -> str:
    bits: list[str] = []
    um = getattr(resp, "usage_metadata", None)
    if um is not None:
        thoughts = getattr(um, "thoughts_token_count", None)
        total = getattr(um, "total_token_count", None)
        if thoughts is not None:
            bits.append(f"thoughts_token_count={thoughts}")
        if total is not None:
            bits.append(f"total_token_count={total}")
    cands = getattr(resp, "candidates", None) or []
    bits.append(f"candidates={len(cands)}")
    for i, c in enumerate(cands[:2]):
        fr = getattr(c, "finish_reason", None)
        bits.append(f"cand[{i}].finish_reason={fr}")
        content = getattr(c, "content", None)
        n_parts = len(getattr(content, "parts", None) or []) if content else 0
        bits.append(f"cand[{i}].parts={n_parts}")
    return " ".join(bits)


def _vertex_collect_parts_text(resp: Any) -> str:
    chunks: list[str] = []
    for c in getattr(resp, "candidates", None) or []:
        content = getattr(c, "content", None)
        if not content:
            continue
        for p in getattr(content, "parts", None) or []:
            t = getattr(p, "text", None)
            if t:
                chunks.append(str(t))
    return "".join(chunks).strip()
