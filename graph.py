import logging
import os
import time
import psycopg2
from dotenv import load_dotenv
from datetime import datetime
from llama_cpp import Llama
from typing import TypedDict, List
from typing_extensions import NotRequired
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import numpy as np

# 환경 변수 로드
load_dotenv()

logger = logging.getLogger(__name__)


def _preview_for_log(text: str, max_chars: int = 4000) -> str:
    """로그용: 긴 프롬프트는 앞부분만 남기고 길이 표시."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [{len(text) - max_chars} chars truncated]"


def _db_password() -> str:
    raw = os.getenv("DB_PASS") or os.getenv("DB_PASSWORD") or ""
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        return raw[1:-1]
    return raw


def _db_connect_kwargs():
    host = os.getenv("DB_HOST")
    database = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = _db_password()
    port = os.getenv("DB_PORT")
    missing = [k for k, v in [
        ("DB_HOST", host), ("DB_NAME", database), ("DB_USER", user),
        ("DB_PASS or DB_PASSWORD", password or None), ("DB_PORT", port),
    ] if not v]
    if missing:
        logger.error("DB 설정 누락: %s", ", ".join(missing))
    return host, database, user, password, port


def _llm_model_path() -> str:
    return os.getenv(
        "LLM_MODEL_PATH",
        "./qwen2-1_5b-instruct-q4_k_m.gguf",
    ).strip()


def _llm_n_gpu_layers() -> int:
    raw = os.getenv("LLM_N_GPU_LAYERS", "-1").strip()
    try:
        return int(raw)
    except ValueError:
        logger.warning("LLM_N_GPU_LAYERS=%r invalid; using 0", raw)
        return 0


def _llm_n_ctx() -> int:
    raw = os.getenv("LLM_N_CTX", "2048").strip()
    try:
        return max(256, int(raw))
    except ValueError:
        return 2048


def _rag_cosine_distance_max() -> float:
    """pgvector 코사인 거리(<=>) 상한. 이보다 작은(더 유사한) 행만 RAG 후보."""
    raw = os.getenv("RAG_COSINE_DISTANCE_MAX", "0.4").strip()
    try:
        return float(raw)
    except ValueError:
        logger.warning("RAG_COSINE_DISTANCE_MAX=%r invalid; using 0.4", raw)
        return 0.4


_model_path = _llm_model_path()
_n_gpu = _llm_n_gpu_layers()
logger.info(
    "Loading LLM model_path=%s n_gpu_layers=%d n_ctx=%d",
    _model_path,
    _n_gpu,
    _llm_n_ctx(),
)
llm = Llama(
    model_path=_model_path,
    n_gpu_layers=_n_gpu,
    n_ctx=_llm_n_ctx(),
    verbose=False,
)

# DB 연결 함수
def get_db_connection():
    host, database, user, password, port = _db_connect_kwargs()
    return psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password,
        port=port,
    )

# 랭그래프 가방(상태) 정의
class AgentState(TypedDict):
    user_query: str
    query_vector: List[float]
    intent: str
    retrieved_popups: List[dict]
    history: List[str]
    final_answer: str
    matched_popup_ids: NotRequired[List[int]]

# [Node 1] 의도 파악
def analyze_intent(state: AgentState):
    t0 = time.perf_counter()
    uq = state.get("user_query") or ""
    logger.info("[Node:enter] analyze_intent user_query_preview=%r", uq[:200])
    out = {"intent": "popup_search"}
    logger.info(
        "[Node:exit] analyze_intent elapsed_ms=%.1f",
        (time.perf_counter() - t0) * 1000,
    )
    return out

# [추가] 사용자의 원본 질문에서 RAG 검색용 핵심 키워드만 추출하는 필터 함수
def extract_search_keywords(query: str) -> str:
    prompt = f"""<|im_start|>system
당신은 검색 키워드 추출기입니다. 
사용자의 질문에서 불필요한 서술어, 조사, 인사말은 모두 제거하고 띄어쓰기로만 구분된 핵심 명사(지역, 장소, 날짜, 이벤트 종류 등)만 출력하세요.
<|im_end|>
<|im_start|>user
이번 주말에 부산에서 열리는 뷰티 팝업스토어 알려줘<|im_end|>
<|im_start|>assistant
이번 주말 부산 뷰티 팝업스토어<|im_end|>
<|im_start|>user
오늘 성수동에서 하는 캐릭터 팝업스토어 추천해줄래?<|im_end|>
<|im_start|>assistant
오늘 성수동 캐릭터 팝업스토어<|im_end|>
<|im_start|>user
내일 여의도 더현대에서 하는 음식 팝업 있어?<|im_end|>
<|im_start|>assistant
내일 여의도 더현대 음식 팝업<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
    logger.info(
        "[LLM:keyword_extract] to_llm: original_query=%r prompt_len=%d prompt=\n%s",
        query,
        len(prompt),
        _preview_for_log(prompt, max_chars=6000),
    )
    # 추출은 짧게 끝내므로 max_tokens를 작게 설정
    t_llm = time.perf_counter()
    output = llm(prompt, max_tokens=64, stop=["<|im_end|>"], echo=False)
    llm_ms = (time.perf_counter() - t_llm) * 1000
    extracted = output["choices"][0]["text"].strip()
    logger.info("[LLM:keyword_extract] from_llm: extracted_for_embedding=%r", extracted)
    logger.info(
        "[keyword_extract:done] llm_elapsed_ms=%.1f original_len=%d extracted_len=%d "
        "original=%r -> extracted=%r",
        llm_ms,
        len(query),
        len(extracted),
        query,
        extracted,
    )
    return extracted

def _popup_preview_50(name: str, desc: str) -> str:
    """로그용: 제목·설명 앞 50자(개행은 공백으로)."""
    raw = f"{name or ''}: {desc or ''}".replace("\n", " ").strip()
    if len(raw) <= 50:
        return raw
    return raw[:50] + "…"


# [Node 2] 시현님 DB 구조에 맞춘 RAG 검색
def retrieve_popups(state: AgentState):
    t0 = time.perf_counter()
    query_vector = state.get("query_vector")
    logger.info(
        "[Node:enter] retrieve_popups query_vector_len=%s",
        len(query_vector) if query_vector else 0,
    )
    try:
        logger.info("[Node: RAG] 유사도 검색 시작 (벡터 길이=%s)", len(query_vector) if query_vector else 0)
    
        if not query_vector:
            logger.warning("질문 벡터가 비어있습니다.")
            return {"retrieved_popups": [], "matched_popup_ids": []}

        threshold = _rag_cosine_distance_max()
        conn = get_db_connection()
        cursor = conn.cursor()

        # popup_embedding(pe)과 popup(p) 조인.
        # pe.embedding <=> 쿼리벡터 : pgvector 코사인 거리(작을수록 유사).
        # threshold 미만을 거리순 우선, 부족하면 threshold 이상 중 거리순으로 채워 최대 3건.
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

        logger.info(
            "[Node: RAG] threshold=%.4f 기준 유사 우선 후 거리순 최대 3건",
            threshold,
        )
        cursor.execute(query, (query_vector, threshold, threshold, threshold))
        rows = cursor.fetchall()

        below_threshold_total = int(rows[0][7]) if rows else 0

        db_result = []
        for row in rows:
            dist = float(row[5])
            below = row[6]
            is_below = bool(below) if below is not None else False
            db_result.append({
                "id": row[0],
                "name": row[1],
                "desc": row[2],
                "lat": row[3],
                "lon": row[4],
                "cosine_distance": dist,
                "is_below_threshold": is_below,
            })
            
        cursor.close()
        conn.close()
        ids_str = ", ".join(str(p["id"]) for p in db_result) if db_result else "(없음)"
        matched_ids = [p["id"] for p in db_result]
        similar_n = sum(1 for p in db_result if p.get("is_below_threshold"))
        logger.info(
            "DB RAG: threshold=%.4f 미만 전체(기간 내) %d건, 반환 %d건(유사 %d / 일반 %d), matched_popup_ids=[%s]",
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
            logger.info(
                "DB RAG: rank=%d id=%s tier=%s cosine_distance=%.6f preview_50=%s",
                rank,
                p["id"],
                tier,
                d if d is not None else float("nan"),
                _popup_preview_50(title, desc),
            )
        return {"retrieved_popups": db_result, "matched_popup_ids": matched_ids}

    except Exception as e:
        logger.exception("DB 조인 검색 실패: %s", e)
        return {"retrieved_popups": [], "matched_popup_ids": []}
    finally:
        logger.info(
            "[Node:exit] retrieve_popups elapsed_ms=%.1f",
            (time.perf_counter() - t0) * 1000,
        )

# [Node 3] 답변 생성
def generate_recommendation(state: AgentState):
    t0 = time.perf_counter()
    query = state["user_query"]
    popups = state.get("retrieved_popups", [])
    history = state.get("history") or []
    logger.info(
        "[Node:enter] generate popup_count=%d history_turns=%d user_query_preview=%r",
        len(popups),
        len(history),
        (query or "")[:200],
    )

    # 팝업 목록: 메타 문구(=== 등) 없이 데이터만. 순서는 DB 반환(유사 우선) 유지.
    context_lines: list[str] = []
    for i, p in enumerate(popups):
        if p.get("is_below_threshold"):
            context_lines.append(f"[추천 {i + 1}] {p['name']} - {p['desc']}")
        else:
            context_lines.append(f"[참고 {i + 1}] {p['name']} - {p['desc']}")
    if not context_lines:
        context_lines.append("(해당하는 팝업스토어 없음)")
    context = "\n".join(context_lines)
    history_context = "\n".join(history[-5:])
    
    # 현재 시간을 가져와서 프롬프트에 주입
    current_time = datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")

    prompt = f"""<|im_start|>system
당신은 팝업스토어를 소개하는 친절하고 자연스러운 대화형 AI 가이드입니다.
현재 시간은 {current_time}입니다. 일정은 이 시간을 기준으로 판단하세요.

[답변 작성 규칙]
1. [팝업 목록]에 있는 정보만 사용하여 답변하세요. 목록이 비었거나 안내 문구만 있으면, 그 사실을 짧게 말하고 지어내지 마세요.
2. 옆 사람에게 말하듯 문장으로만 이어 쓰세요. "1. 2. 3." 식 번호 매기기나 마크다운 표는 쓰지 마세요. 목록 앞의 [추천 N], [참고 N]은 구분용이니 **답변에는 그대로 가져오지 말고** 팝업 이름으로만 부르세요.
3. [추천]이 붙은 줄을 먼저 자세히 소개하고, [참고]가 붙은 줄은 뒤에 짧게 덧붙이거나 생략해도 됩니다.
4. 설명이 길면 한두 문장으로 핵심만 요약해 자연스럽게 말해 주세요. 전시·매장·행사 등 설명에 맞는 말로 쓰면 됩니다.

[대화 예시]
User: 성수동 근처에 구경할 만한 곳 있어?
Assistant: 성수 쪽이시면 '무드 라이트 팝업'을 가장 추천해 드려요. 예쁜 조명과 소품을 직접 체험해 볼 수 있거든요. 시간이 조금 남으신다면 근처에서 한정 굿즈를 파는 '스프링 굿즈 마켓'도 가볍게 둘러보시기 좋아요.

[팝업 목록]:
{context}

[이전 대화]:
{history_context}
<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
    logger.info(
        "[LLM:recommend] to_llm: injected_current_time=%r user_query=%r "
        "popup_count=%d history_lines=%d context_chars=%d history_chars=%d prompt_len=%d",
        current_time,
        query,
        len(popups),
        len(history),
        len(context),
        len(history_context),
        len(prompt),
    )
    logger.info("[LLM:recommend] to_llm: context_block=\n%s", _preview_for_log(context, 2500))
    logger.info(
        "[LLM:recommend] to_llm: history_block=\n%s",
        _preview_for_log(history_context or "(비어 있음)", 1500),
    )
    logger.info("[LLM:recommend] to_llm: full_prompt=\n%s", _preview_for_log(prompt, 8000))
    output = llm(prompt, max_tokens=512, stop=["<|im_end|>"], echo=False)
    answer = output["choices"][0]["text"].strip()
    if answer.startswith("AI:"):
        answer = answer[3:].lstrip()

    new_history = history + [f"User: {query}", f"AI: {answer}"]

    logger.info(
        "[Node:exit] generate elapsed_ms=%.1f answer_chars=%d",
        (time.perf_counter() - t0) * 1000,
        len(answer),
    )
    return {
        "final_answer": answer,
        "history": new_history
    }

# 워크플로우 조립
workflow = StateGraph(AgentState)
workflow.add_node("analyze_intent", analyze_intent)
workflow.add_node("retrieve_popups", retrieve_popups)
workflow.add_node("generate", generate_recommendation)

workflow.set_entry_point("analyze_intent")
workflow.add_edge("analyze_intent", "retrieve_popups")
workflow.add_edge("retrieve_popups", "generate")
workflow.add_edge("generate", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)