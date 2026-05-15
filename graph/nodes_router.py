"""라우터·임베딩 준비."""

import asyncio
import json
import logging
import time
from typing import Any

from graph.extract import (
    augment_action_extracted,
    coerce_intent_for_popup_queries,
    compose_embedding_text_from_search_conditions,
    extract_intent_and_conditions,
    normalize_extracted_category,
    strip_ungrounded_category,
)
from graph.routing import (
    condition_vector_from_extracted,
    merge_search_conditions_carry_forward,
    resolve_route,
    retrieval_mode_for_route,
)
from graph.state import AgentState

logger = logging.getLogger(__name__)

# re-export for tests / dynamic imports
__all__ = ["router_node", "prepare_embedding"]


def router_node(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()
    query = state.get("user_query") or ""
    logger.debug("[Node:enter] router user_query_preview=%r", query[:200])

    extracted_fresh = extract_intent_and_conditions(query)
    extracted_fresh = coerce_intent_for_popup_queries(query, extracted_fresh)
    extracted_fresh = augment_action_extracted(query, extracted_fresh)
    extracted_fresh = normalize_extracted_category(extracted_fresh)
    extracted_fresh = strip_ungrounded_category(query, extracted_fresh)
    prev_sc = state.get("search_conditions")
    prev_dict = prev_sc if isinstance(prev_sc, dict) else None
    extracted = merge_search_conditions_carry_forward(prev_dict, extracted_fresh)
    vec_fresh = condition_vector_from_extracted(extracted_fresh)
    route, condition_vector = resolve_route(extracted)
    mode = retrieval_mode_for_route(route)

    logger.info(
        "[router] condition_vector=%s carry_fresh_vec=%s route=%s search_conditions=%s",
        condition_vector,
        vec_fresh,
        route,
        json.dumps(extracted, ensure_ascii=False),
    )
    logger.debug(
        "[Router] route=%s condition_vector=%s raw_intent=%r elapsed_ms=%.1f",
        route,
        condition_vector,
        extracted.get("intent"),
        (time.perf_counter() - t0) * 1000,
    )
    return {
        "intent": str(extracted.get("intent") or ""),
        "route": route,
        "search_conditions": extracted,
        "condition_vector": condition_vector,
        "retrieval_mode": mode,
    }


async def prepare_embedding(state: AgentState) -> dict[str, Any]:
    """action 경로: 클라이언트 벡터가 없으면 키워드 추출 + 임베딩 API."""
    if state.get("skip_embedding") and state.get("query_vector"):
        logger.info(
            "[prepare_embedding] 클라이언트 query_vector 사용 (임베딩 API 호출 없음)"
        )
        logger.debug("[prepare_embedding] skip (client query_vector)")
        return {"embed_text_used": "<client_query_vector>"}
    from embedding_client import fetch_query_embedding

    uq = state.get("user_query") or ""
    sc = state.get("search_conditions")
    sc_dict = sc if isinstance(sc, dict) else None
    search_keywords = await asyncio.to_thread(
        compose_embedding_text_from_search_conditions, uq, sc_dict
    )
    raw = search_keywords or ""
    logger.info(
        "[prepare_embedding] 임베딩_API_text=%r len=%d",
        raw[:500] + ("…" if len(raw) > 500 else ""),
        len(raw),
    )
    logger.debug(
        "[prepare_embedding] embed_text=%r (search_conditions 기반 재구성)",
        search_keywords,
    )
    query_vector = await fetch_query_embedding(search_keywords)
    return {"query_vector": query_vector, "embed_text_used": search_keywords}
