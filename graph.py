import logging
import os
import psycopg2
from dotenv import load_dotenv
from llama_cpp import Llama
from typing import TypedDict, List
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

# AI 모델 로드 (M4 GPU 활용)
print("⏳ M4 GPU를 사용하여 LLM 로드 중...")
llm = Llama(
    model_path="./qwen2-1_5b-instruct-q4_k_m.gguf", 
    n_gpu_layers=-1, 
    n_ctx=2048,
    verbose=False
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

# [Node 1] 의도 파악
def analyze_intent(state: AgentState):
    return {"intent": "popup_search"}

# [Node 2] 시현님 DB 구조에 맞춘 RAG 검색
def retrieve_popups(state: AgentState):
    query_vector = state.get("query_vector")
    logger.info("[Node: RAG] 유사도 검색 시작 (벡터 길이=%s)", len(query_vector) if query_vector else 0)
    
    if not query_vector:
        logger.warning("질문 벡터가 비어있습니다.")
        return {"retrieved_popups": []}

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 시현님이 주신 popup_embedding(pe)과 실제 정보가 있는 popup(p) 조인 쿼리
        query = """
            SELECT 
                p.id, p.title, p.description, p.latitude, p.longitude
            FROM popup_embedding pe
            JOIN popup p ON pe.popup_id = p.id
            WHERE p.start_date <= CURRENT_DATE AND p.end_date >= CURRENT_DATE
            ORDER BY pe.embedding <=> %s::vector
            LIMIT 3;
        """
        
        cursor.execute(query, (query_vector,))
        rows = cursor.fetchall()
        
        db_result = []
        for row in rows:
            db_result.append({
                "id": row[0],
                "name": row[1],
                "desc": row[2],
                "lat": row[3],
                "lon": row[4]
            })
            
        cursor.close()
        conn.close()
        logger.info("DB RAG: 유사도 Top %d 팝업 매칭 완료.", len(db_result))
        return {"retrieved_popups": db_result}

    except Exception as e:
        logger.exception("DB 조인 검색 실패: %s", e)
        return {"retrieved_popups": []}

# [Node 3] 답변 생성
def generate_recommendation(state: AgentState):
    query = state["user_query"]
    popups = state.get("retrieved_popups", [])
    history = state.get("history") or []
    
    # 팝업 정보를 텍스트로 변환
    context = "\n".join([f"- {p['name']}: {p['desc']}" for p in popups])
    history_context = "\n".join(history[-5:])
    
    prompt = f"""<|im_start|>system
당신은 성수동 팝업스토어 전문 가이드입니다. 
제공된 [팝업 목록]의 정보를 바탕으로 사용자의 질문에 친절하게 답하세요. 
반드시 목록에 있는 장소들을 중심으로 설명해야 합니다.

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