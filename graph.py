import logging
import os
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
    return {"intent": "popup_search"}

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
    # 추출은 짧게 끝내므로 max_tokens를 작게 설정
    output = llm(prompt, max_tokens=64, stop=["<|im_end|>"], echo=False)
    return output["choices"][0]["text"].strip()

def _popup_preview_50(name: str, desc: str) -> str:
    """로그용: 제목·설명 앞 50자(개행은 공백으로)."""
    raw = f"{name or ''}: {desc or ''}".replace("\n", " ").strip()
    if len(raw) <= 50:
        return raw
    return raw[:50] + "…"


# [Node 2] 시현님 DB 구조에 맞춘 RAG 검색
def retrieve_popups(state: AgentState):
    query_vector = state.get("query_vector")
    logger.info("[Node: RAG] 유사도 검색 시작 (벡터 길이=%s)", len(query_vector) if query_vector else 0)
    
    if not query_vector:
        logger.warning("질문 벡터가 비어있습니다.")
        return {"retrieved_popups": [], "matched_popup_ids": []}

    threshold = _rag_cosine_distance_max()
    try:
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

# [Node 3] 답변 생성
def generate_recommendation(state: AgentState):
    query = state["user_query"]
    popups = state.get("retrieved_popups", [])
    history = state.get("history") or []
    
    # 팝업 정보: 유사(threshold 미만) / 일반(거리순 보충) 태그로 구분
    context_lines = []
    for p in popups:
        if p.get("is_below_threshold"):
            context_lines.append(f"- [유사 추천] {p['name']}: {p['desc']}")
        else:
            context_lines.append(f"- [일반 추천] {p['name']}: {p['desc']}")
    context = "\n".join(context_lines)
    history_context = "\n".join(history[-5:])
    
    # 현재 시간을 가져와서 프롬프트에 주입
    current_time = datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")

    prompt = f"""<|im_start|>system
당신은 팝업스토어 전문 가이드입니다. 
    현재 시간은 {current_time}입니다. 사용자가 '오늘', '내일', '이번 주' 등의 일정을 물어보면 이 시간을 기준으로 답변하세요.
제공된 [팝업 목록]의 정보를 바탕으로 사용자의 질문에 친절하게 답하세요. 
반드시 목록에 있는 장소들을 중심으로 설명해야 합니다.

출력 형식 규칙:
- [유사 추천] 항목은 질문과 관련이 높다고 보고, 본문에서 먼저·자세히 추천하세요.
- [일반 추천] 항목이 있으면, 해당 장소를 소개할 때 문장을 "원하시는 정보는 아니지만, 추천할 만한 이벤트가 있어요."로 시작한 뒤 이어서 간단히 설명하세요. (일반 추천이 여러 개면 각각에 동일한 도입을 쓰지 말고 자연스럽게 묶거나 번갈아 표현해도 됩니다.)

[팝업 목록]:
{context}

[이전 대화]:
{history_context}
<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
    output = llm(prompt, max_tokens=512, stop=["<|im_end|>"], echo=False)
    answer = output["choices"][0]["text"].strip()
    
    new_history = history + [f"User: {query}", f"AI: {answer}"]
    
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