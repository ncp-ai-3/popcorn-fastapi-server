import logging
import os
import psycopg2
from dotenv import load_dotenv
from llama_cpp import Llama
from typing import TypedDict, List, NotRequired
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
    
    prompt = f"""<|im_start|>system
당신은 팝업스토어 전문 가이드입니다. 
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