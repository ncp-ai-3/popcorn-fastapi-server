"""RAG 검색·진행 중 팝업 최근 3건."""

import logging
import time
from typing import Any

from graph.db import get_db_connection, rag_cosine_distance_max
from graph.popup_date_filter import popup_date_where_sql, visit_window_from_search_conditions
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


def _category_exists_sql(
    search_conditions: dict[str, Any] | None,
    *,
    table_alias: str = "p",
) -> tuple[str, list[Any]]:
    """search_conditions['category'] 가 있으면 popup 행에 대한 EXISTS 절·파라미터 반환.

    DB 매핑: popup_category.cateroty_id → category.id, popup_category.popup_id → popup.id
    (cateroty_id 는 실제 DB 컬럼명 — Flyway 마이그레이션 오타 그대로 유지)
    """
    if not search_conditions:
        return "", []
    cat = search_conditions.get("category")
    if cat is None or not str(cat).strip():
        return "", []
    cat_s = str(cat).strip()
    clause = (
        f" AND EXISTS ("
        f"SELECT 1 FROM popup_category pc "
        f"JOIN category c ON c.id = pc.cateroty_id "
        f"WHERE pc.popup_id = {table_alias}.id AND c.name = %s"
        f")"
    )
    return clause, [cat_s]


def _fetch_recent_rows(
    cursor,
    exclude_ids: set[int],
    limit: int,
    search_conditions: dict[str, Any] | None = None,
) -> list[tuple]:
    date_clause, date_params, _ = popup_date_where_sql(search_conditions)
    cat_clause, cat_params = _category_exists_sql(search_conditions)
    q = f"""
        SELECT p.id, p.title, p.description, p.latitude, p.longitude
        FROM popup p
        WHERE {date_clause}{cat_clause}
    """
    params: list[Any] = list(date_params) + list(cat_params)
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
        rows = _fetch_recent_rows(cursor, set(), 3, state.get("search_conditions"))
        db_result = [
            _row_to_popup_dict(r, cosine_distance=None, is_below_threshold=False)
            for r in rows
        ]
        cursor.close()
        conn.close()
        matched_ids = [p["id"] for p in db_result]
        ids_str = ", ".join(str(i) for i in matched_ids) if matched_ids else "(없음)"
        logger.info(
            "[retrieve_recent_three] returned=%d (target 3) matched_popup_ids=[%s]",
            len(db_result),
            ids_str,
        )
        if len(db_result) < 3:
            logger.warning(
                "retrieve_recent_three: got %d rows (target 3)",
                len(db_result),
            )
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
    qv = query_vector
    qv_len = len(qv) if isinstance(qv, (list, tuple)) else 0
    logger.debug(
        "[Node:enter] retrieve_popups query_vector_len=%s route=%s",
        qv_len,
        route,
    )
    try:
        if not query_vector:
            logger.warning("질문 벡터가 비어있습니다. 최근 3건 폴백 시도.")
            return _fallback_recent_from_state(state, t0)

        threshold = rag_cosine_distance_max()
        sc = state.get("search_conditions")
        sc_dict = sc if isinstance(sc, dict) else None
        date_clause, date_params, date_mode = popup_date_where_sql(sc_dict)
        cat_clause, cat_params = _category_exists_sql(sc_dict)
        ws, we, _ = visit_window_from_search_conditions(sc_dict)
        if date_mode == "extracted_range":
            logger.info(
                "[retrieve_popups] date_prefilter=extracted_range popup overlap "
                "user_start=%s user_end=%s (before vector rank)",
                ws,
                we,
            )
        else:
            logger.info(
                "[retrieve_popups] date_prefilter=current_open (no extracted dates)"
            )
        sc_cat_log = (
            str(sc_dict.get("category")).strip()
            if isinstance(sc_dict, dict) and sc_dict.get("category")
            else None
        )
        if cat_clause:
            logger.info(
                "[retrieve_popups] category_prefilter=on category=%r",
                sc_cat_log,
            )
        else:
            logger.info("[retrieve_popups] category_prefilter=off")

        conn = get_db_connection()
        cursor = conn.cursor()

        query = f"""
            WITH ranked AS (
                SELECT 
                    p.id, p.title, p.description, p.latitude, p.longitude,
                    (pe.embedding <=> %s::vector) AS cosine_distance
                FROM popup_embedding pe
                JOIN popup p ON pe.popup_id = p.id
                WHERE {date_clause}{cat_clause}
            ),
            ordered AS (
                SELECT
                    id, title, description, latitude, longitude, cosine_distance,
                    COUNT(*) FILTER (WHERE cosine_distance < %s) OVER () AS below_threshold_total,
                    COUNT(*) OVER () AS total_candidates,
                    (cosine_distance < %s) AS is_below_threshold,
                    ROW_NUMBER() OVER (
                        ORDER BY (cosine_distance < %s) DESC, cosine_distance ASC
                    ) AS rn
                FROM ranked
            )
            SELECT id, title, description, latitude, longitude, cosine_distance,
                   is_below_threshold, below_threshold_total, total_candidates
            FROM ordered
            WHERE rn <= 3
            ORDER BY rn;
        """

        cursor.execute(
            query,
            (query_vector, *date_params, *cat_params, threshold, threshold, threshold),
        )
        rows = cursor.fetchall()

        below_threshold_total = int(rows[0][7]) if rows else 0
        total_candidates = int(rows[0][8]) if rows else 0

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

        rag_returned = len(db_result)
        similar_n = sum(1 for p in db_result if p.get("is_below_threshold"))
        logger.info(
            "[retrieve_popups] db_search candidates=%d below_threshold=%d "
            "returned=%d (유사 %d / 일반 %d) threshold=%.4f",
            total_candidates,
            below_threshold_total,
            rag_returned,
            similar_n,
            rag_returned - similar_n,
            threshold,
        )

        pad_added = 0
        if route == ROUTE_ACTION and len(db_result) < 3:
            have = {p["id"] for p in db_result}
            need = 3 - len(db_result)
            extra_rows = _fetch_recent_rows(cursor, have, need, sc_dict)
            for row in extra_rows:
                db_result.append(
                    _row_to_popup_dict(
                        row,
                        cosine_distance=None,
                        is_below_threshold=False,
                    )
                )
            pad_added = len(extra_rows)
            logger.info(
                "[retrieve_popups] pad_with_recent need=%d added=%d total_after=%d",
                need,
                pad_added,
                len(db_result),
            )

        cursor.close()
        conn.close()

        ids_str = ", ".join(str(p["id"]) for p in db_result) if db_result else "(없음)"
        matched_ids = [p["id"] for p in db_result]
        logger.info(
            "[retrieve_popups] final returned=%d (rag=%d + pad=%d) matched_popup_ids=[%s]",
            len(db_result),
            rag_returned,
            pad_added,
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
        rows = _fetch_recent_rows(
            cursor, set(), 3, state.get("search_conditions")
        )
        db_result = [
            _row_to_popup_dict(r, cosine_distance=None, is_below_threshold=False)
            for r in rows
        ]
        cursor.close()
        conn.close()
        matched_ids = [p["id"] for p in db_result]
        ids_str = ", ".join(str(i) for i in matched_ids) if matched_ids else "(없음)"
        logger.info(
            "[retrieve_popups] fallback_recent returned=%d matched_popup_ids=[%s]",
            len(db_result),
            ids_str,
        )
        logger.debug(
            "[Node:exit] retrieve_popups fallback recent_three elapsed_ms=%.1f",
            (time.perf_counter() - t0) * 1000,
        )
        return {
            "retrieved_popups": db_result,
            "matched_popup_ids": matched_ids,
            "retrieval_mode": "recent_three_fallback",
        }
    except Exception:
        logger.exception("폴백 최근 3건 실패")
        return {"retrieved_popups": [], "matched_popup_ids": [], "retrieval_mode": "error"}
