"""search_conditions 날짜 → popup 테이블 기간 겹침 SQL."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)


def reference_now_for_extract() -> datetime:
    """의도 추출 LLM에 넘길 기준 시각(서버 로컬)."""
    return datetime.now()


def format_reference_context(now: datetime | None = None) -> str:
    """추출 프롬프트용 기준 시각·요일 문자열."""
    now = now or reference_now_for_extract()
    weekdays = ("월", "화", "수", "목", "금", "토", "일")
    wd = weekdays[now.weekday()]
    return (
        f"기준 시각: {now.strftime('%Y-%m-%d %H:%M')} ({wd}요일, 서버 로컬)\n"
        f"기준 날짜(YYYY-MM-DD): {now.strftime('%Y-%m-%d')}"
    )


def parse_iso_date(value: Any) -> date | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw.lower() == "null":
        return None
    try:
        return datetime.strptime(raw[:10], "%Y-%m-%d").date()
    except ValueError:
        logger.warning("popup_date_filter: invalid date %r", value)
        return None


def visit_window_from_search_conditions(
    sc: dict[str, Any] | None,
    *,
    today: date | None = None,
) -> tuple[date | None, date | None, str]:
    """사용자 방문 희망 구간 (시작, 끝). 둘 다 None이면 DB는 '오늘 진행 중' 모드."""
    today = today or date.today()
    if not sc:
        return None, None, "current_open"

    window_start = parse_iso_date(sc.get("start_date"))
    window_end = parse_iso_date(sc.get("end_date"))

    if window_start is None and window_end is None:
        return None, None, "current_open"

    if window_start is None:
        window_start = today
    if window_end is None:
        window_end = window_start
    if window_start > window_end:
        window_start, window_end = window_end, window_start

    return window_start, window_end, "extracted_range"


def popup_date_where_sql(
    sc: dict[str, Any] | None,
    *,
    table_alias: str = "p",
) -> tuple[str, list[Any], str]:
    """popup 행 기간 조건 SQL 조각과 바인드 파라미터.

    겹침: popup.start_date <= user_end AND popup.end_date >= user_start
    """
    window_start, window_end, mode = visit_window_from_search_conditions(sc)
    alias = table_alias
    if mode == "extracted_range" and window_start and window_end:
        clause = f"{alias}.start_date <= %s AND {alias}.end_date >= %s"
        return clause, [window_end, window_start], mode
    clause = (
        f"{alias}.start_date <= CURRENT_DATE AND {alias}.end_date >= CURRENT_DATE"
    )
    return clause, [], "current_open"
