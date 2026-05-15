"""LangGraph 상태·상수."""

from typing import Any, List, NotRequired, Tuple, TypedDict

# 추천 LLM: 프롬프트에 넣을 과거 대화 — 직전 1턴(User·AI 한 쌍)만 = 2줄
RECOMMEND_HISTORY_TURNS = 1

# 라우트 문자열
ROUTE_ACTION = "action"
ROUTE_CLARIFICATION_ALL = "clarification_all"
ROUTE_CLARIFICATION_LOC = "clarification_loc"
ROUTE_CHITCHAT = "chitchat"
ROUTE_OUT_OF_DOMAIN = "out_of_domain"


class AgentState(TypedDict):
    user_query: str
    query_vector: NotRequired[List[float]]
    skip_embedding: NotRequired[bool]
    intent: NotRequired[str]
    route: NotRequired[str]
    search_conditions: NotRequired[dict[str, Any]]
    condition_vector: NotRequired[Tuple[int, int, int]]
    retrieval_mode: NotRequired[str]
    retrieved_popups: NotRequired[List[dict]]
    history: NotRequired[List[str]]
    final_answer: NotRequired[str]
    matched_popup_ids: NotRequired[List[int]]
    # 요약 로그용 (prepare_embedding에서 설정)
    embed_text_used: NotRequired[str]
