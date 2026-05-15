"""RAG 검색·진행 중 팝업 최근 3건."""

import logging
import time
from typing import Any

from graph.db import get_db_connection, rag_cosine_distance_max
from graph.routing import ROUTE_ACTION
from graph.state import AgentState

logger = logging.getLogger(__name__)


def _popup_preview_50(name: str, desc: str) -> str:
    raw = f"{name or ''}: {desc or ''}".replace("\n", " ").strip()
    if len(raw) <= 50:
        return raw
    return raw[:50] + "…"


def _row_to_popup_dict(
    row: tuple,
    *,
    cosine_distance: float | None,
    is_below_threshold: bool | None,
) -> dict[str, Any]:
    return {
        "id": row[0],
        "name": row[1],
        "desc": row[2],
        "lat": row[3],
        "lon": row[4],
        "cosine_distance": cosine_distance,
        "is_below_threshold": is_below_threshold,
    }


def _fetch_recent_rows(cursor, exclude_ids: set[int], limit: int) -> list[tuple]:
    q = """
        SELECT p.id, p.title, p.description, p.latitude, p.longitude
        FROM popup p
        WHERE p.start_date <= CURRENT_DATE AND p.end_date >= CURRENT_DATE
    """
    params: list[Any] = []
    if exclude_ids:
        q += " AND p.id NOT IN (" + ",".join(str(int(i)) for i in exclude_ids) + ")"
    q += " ORDER BY p.start_date DESC NULLS LAST LIMIT %s"
    params.append(limit)
    cursor.execute(q, params)
    return cursor.fetchall()


def retrieve_recent_three(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()
    route = state.get("route", ROUTE_ACTION)
    if route != ROUTE_ACTION:
        logger.debug(
            "[Node:enter] retrieve_recent_three skip DB route=%s (non-action: no popup list)",
            route,
        )
        return {
            "retrieved_popups": [],
            "matched_popup_ids": [],
            "retrieval_mode": "non_action_no_db",
        }
    logger.debug("[Node:enter] retrieve_recent_three")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        rows = _fetch_recent_rows(cursor, set(), 3)
        db_result = [
            _row_to_popup_dict(r, cosine_distance=None, is_below_threshold=False)
            for r in rows
        ]
        cursor.close()
        conn.close()
        if len(db_result) < 3:
            logger.warning(
                "retrieve_recent_three: got %d rows (target 3)",
                len(db_result),
            )
        matched_ids = [p["id"] for p in db_result]
        logger.debug(
            "[Node:exit] retrieve_recent_three elapsed_ms=%.1f ids=%s",
            (time.perf_counter() - t0) * 1000,
            matched_ids,
        )
        return {
            "retrieved_popups": db_result,
            "matched_popup_ids": matched_ids,
            "retrieval_mode": "recent_three",
        }
    except Exception as e:
        logger.exception("retrieve_recent_three 실패: %s", e)
        return {
            "retrieved_popups": [],
            "matched_popup_ids": [],
            "retrieval_mode": "recent_three",
        }


def retrieve_popups(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()
    query_vector = state.get("query_vector")
    route = state.get("route", ROUTE_ACTION)
    logger.debug(
        "[Node:enter] retrieve_popups query_vector_len=%s route=%s",
        len(query_vector) if query_vector else 0,
        route,
    )
    try:
        if not query_vector:
            logger.warning("질문 벡터가 비어있습니다. 최근 3건 폴백 시도.")
            return _fallback_recent_from_state(state, t0)

        threshold = rag_cosine_distance_max()
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
            WITH ranked AS (
                SELECT 
                    p.id, p.title, p.description, p.latitude, p.longitude,
                    (pe.embedding <=> %s::vector) AS cosine_distance
                FROM popup_embedding pe
                JOIN popup p ON pe.popup_id = p.id
                WHERE p.start_date <= CURRENT_DATE AND p.end_date >= CURRENT_DATE
            ),
            ordered AS (
                SELECT
                    id, title, description, latitude, longitude, cosine_distance,
                    COUNT(*) FILTER (WHERE cosine_distance < %s) OVER () AS below_threshold_total,
                    (cosine_distance < %s) AS is_below_threshold,
                    ROW_NUMBER() OVER (
                        ORDER BY (cosine_distance < %s) DESC, cosine_distance ASC
                    ) AS rn
                FROM ranked
            )
            SELECT id, title, description, latitude, longitude, cosine_distance, is_below_threshold, below_threshold_total
            FROM ordered
            WHERE rn <= 3
            ORDER BY rn;
        """

        cursor.execute(query, (query_vector, threshold, threshold, threshold))
        rows = cursor.fetchall()

        below_threshold_total = int(rows[0][7]) if rows else 0

        db_result: list[dict[str, Any]] = []
        for row in rows:
            dist = float(row[5])
            below = row[6]
            is_below = bool(below) if below is not None else False
            db_result.append(
                _row_to_popup_dict(
                    row[:5],
                    cosine_distance=dist,
                    is_below_threshold=is_below,
                )
            )

        if route == ROUTE_ACTION and len(db_result) < 3:
            have = {p["id"] for p in db_result}
            need = 3 - len(db_result)
            extra_rows = _fetch_recent_rows(cursor, have, need)
            for row in extra_rows:
                db_result.append(
                    _row_to_popup_dict(
                        row,
                        cosine_distance=None,
                        is_below_threshold=False,
                    )
                )

        cursor.close()
        conn.close()

        ids_str = ", ".join(str(p["id"]) for p in db_result) if db_result else "(없음)"
        matched_ids = [p["id"] for p in db_result]
        similar_n = sum(1 for p in db_result if p.get("is_below_threshold"))
        logger.debug(
            "DB RAG+pad: threshold=%.4f 미만 전체(기간 내) %d건, 반환 %d건(유사 %d / 일반·보충 %d), matched_popup_ids=[%s]",
            threshold,
            below_threshold_total,
            len(db_result),
            similar_n,
            len(db_result) - similar_n,
            ids_str,
        )
        for rank, p in enumerate(db_result, start=1):
            title = p.get("name") or ""
            desc = p.get("desc") or ""
            d = p.get("cosine_distance")
            tier = "유사" if p.get("is_below_threshold") else "일반"
            logger.debug(
                "DB RAG: rank=%d id=%s tier=%s cosine_distance=%s preview_50=%s",
                rank,
                p["id"],
                tier,
                d if d is not None else "n/a",
                _popup_preview_50(title, desc),
            )
        logger.debug(
            "[Node:exit] retrieve_popups elapsed_ms=%.1f",
            (time.perf_counter() - t0) * 1000,
        )
        return {
            "retrieved_popups": db_result,
            "matched_popup_ids": matched_ids,
            "retrieval_mode": "rag",
        }

    except Exception as e:
        logger.exception("DB 조인 검색 실패: %s", e)
        return _fallback_recent_from_state(state, t0)


def _fallback_recent_from_state(state: AgentState, t0: float) -> dict[str, Any]:
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        rows = _fetch_recent_rows(cursor, set(), 3)
        db_result = [
            _row_to_popup_dict(r, cosine_distance=None, is_below_threshold=False)
            for r in rows
        ]
        cursor.close()
        conn.close()
        logger.debug(
            "[Node:exit] retrieve_popups fallback recent_three elapsed_ms=%.1f",
            (time.perf_counter() - t0) * 1000,
        )
        return {
            "retrieved_popups": db_result,
            "matched_popup_ids": [p["id"] for p in db_result],
            "retrieval_mode": "recent_three_fallback",
        }
    except Exception:
        logger.exception("폴백 최근 3건 실패")
        return {"retrieved_popups": [], "matched_popup_ids": [], "retrieval_mode": "error"}
