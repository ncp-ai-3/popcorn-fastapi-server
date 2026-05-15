"""조건 벡터·라우트 결정·조건부 엣지."""

from typing import Any, Mapping

from graph.state import (
    ROUTE_ACTION,
    ROUTE_CHITCHAT,
    ROUTE_CLARIFICATION_ALL,
    ROUTE_OUT_OF_DOMAIN,
)


def _nonempty_str(v: Any) -> bool:
    return v is not None and str(v).strip() != ""


def _has_time_window(d: Mapping[str, Any]) -> bool:
    return _nonempty_str(d.get("start_date")) or _nonempty_str(d.get("end_date"))


def merge_search_conditions_carry_forward(
    prev: dict[str, Any] | None,
    new: dict[str, Any],
) -> dict[str, Any]:
    """이번 턴 LLM 추출(new)이 비운 슬롯은 이전 턴(prev) 값으로 채움.

    intent 등 new의 나머지 키는 그대로 두고, category·날짜·location만 누적 보강.
    예: 이전 (1,0,0) + 이번 (0,0,1) 추출 → 병합 후 (1,0,1).
    """
    out = dict(new)
    if not prev:
        return out
    if _nonempty_str(prev.get("category")) and not _nonempty_str(out.get("category")):
        out["category"] = prev.get("category")
    if _nonempty_str(prev.get("location")) and not _nonempty_str(out.get("location")):
        out["location"] = prev.get("location")
    if _has_time_window(prev) and not _has_time_window(out):
        out["start_date"] = prev.get("start_date")
        out["end_date"] = prev.get("end_date")
    return out


def condition_vector_from_extracted(extracted: dict[str, Any]) -> tuple[int, int, int]:
    """(category, time, location) 이진 벡터."""
    cat = extracted.get("category")
    has_cat = 1 if (cat is not None and str(cat).strip()) else 0
    sd = extracted.get("start_date")
    ed = extracted.get("end_date")
    has_time = 1 if (sd or ed) else 0
    loc = extracted.get("location")
    has_loc = 1 if (loc is not None and str(loc).strip()) else 0
    return (has_cat, has_time, has_loc)


def resolve_route(extracted: dict[str, Any]) -> tuple[str, tuple[int, int, int]]:
    """LLM JSON → route 문자열 + condition_vector."""
    raw_intent = (extracted.get("intent") or "chitchat").strip().lower()
    vec = condition_vector_from_extracted(extracted)

    if raw_intent == "chitchat":
        return ROUTE_CHITCHAT, vec
    if raw_intent == "out_of_domain":
        return ROUTE_OUT_OF_DOMAIN, vec
    if raw_intent == "action":
        match vec:
            case (0, 0, 0):
                return ROUTE_CLARIFICATION_ALL, vec
            case (1, 0, 0):
                # 종류만 있어도 임베딩 RAG로 검색 (지역 미정이면 답변에서만 가볍게 물어볼 수 있음)
                return ROUTE_ACTION, vec
            case _:
                return ROUTE_ACTION, vec
    return ROUTE_CHITCHAT, vec


def route_after_router(state: dict) -> str:
    """조건부 엣지: 다음 노드 이름."""
    r = state.get("route", ROUTE_ACTION)
    if r == ROUTE_ACTION:
        return "prepare_embedding"
    return "retrieve_recent_three"


def retrieval_mode_for_route(route: str) -> str:
    if route == ROUTE_ACTION:
        return "rag"
    return "recent_three"
