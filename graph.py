import os
import psycopg2
from dotenv import load_dotenv
from llama_cpp import Llama
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 1. 환경 변수 및 모델 로드
load_dotenv()

print("⏳ M4 GPU를 사용하여 모델을 로드하는 중...")
llm = Llama(
    model_path="./qwen2-1_5b-instruct-q4_k_m.gguf", 
    n_gpu_layers=-1, 
    verbose=False
)

# 2. 상태 정의 (history는 없을 수 있으므로 초기값 처리가 중요합니다)
class AgentState(TypedDict):
    user_query: str
    intent: str
    retrieved_popups: List[dict]
    history: List[str]  # 대화 기록 저장
    final_answer: str

# 3. 노드 함수들
def analyze_intent(state: AgentState):
    # 나중에 의도 분류 로직을 추가할 수 있습니다.
    return {"intent": "popup_search"}

def retrieve_popups(state: AgentState):
    print("[Node: RAG 검색] DB에서 데이터를 가져오는 중...")
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            dbname=os.getenv("DB_NAME")
        )
        cursor = conn.cursor()
        
        # Spring application.yml에 적힌 DB 정보를 바탕으로 쿼리 실행
        cursor.execute("SELECT id, title, description, latitude, longitude FROM popup LIMIT 3;")
        rows = cursor.fetchall()
        
        db_result = []
        for row in rows:
            db_result.append({
                "id": row[0],
                "name": row[1],
                "desc": row[2],
                "lat": float(row[3]) if row[3] else 0.0,
                "lng": float(row[4]) if row[4] else 0.0
            })
            
        cursor.close()
        conn.close()
        print(f"✅ DB 검색 완료! {len(db_result)}개의 팝업을 찾았습니다.")
        return {"retrieved_popups": db_result}
        
    except Exception as e:
        print(f"🚨 DB 연결 에러: {e}")
        return {"retrieved_popups": []}

def generate_recommendation(state: AgentState):
    query = state["user_query"]
    popups = state.get("retrieved_popups", [])
    history = state.get("history") or []
    
    # 팝업 목록을 번호를 붙여서 더 명확하게 전달
    context = "\n".join([f"{i+1}. {p['name']}: {p['desc']}" for i, p in enumerate(popups)])
    history_context = "\n".join(history[-5:])
    
    prompt = f"""<|im_start|>system
당신은 성수동 팝업스토어 가이드입니다. 
당신은 반드시 아래 [팝업 목록]에 제공된 3개의 장소를 하나도 빠짐없이 모두 추천해야 합니다. 
만약 목록이 3개보다 적다면 있는 것만이라도 상세히 설명하세요.

[이전 대화]:
{history_context}

[팝업 목록]:
{context}
<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
    output = llm(prompt, max_tokens=512, stop=["<|im_end|>"], echo=False)
    answer = output["choices"][0]["text"].strip()
    
    # 새로운 대화를 history에 누적
    new_history = history + [f"User: {query}", f"AI: {answer}"]
    
    return {
        "final_answer": answer,
        "history": new_history
    }

# 4. 워크플로우 구성
workflow = StateGraph(AgentState)

workflow.add_node("analyze_intent", analyze_intent)
workflow.add_node("retrieve_popups", retrieve_popups)
workflow.add_node("generate", generate_recommendation)

workflow.set_entry_point("analyze_intent")
workflow.add_edge("analyze_intent", "retrieve_popups")
workflow.add_edge("retrieve_popups", "generate")
workflow.add_edge("generate", END)

# 5. 체크포인터(단기기억 저장소) 연결 및 컴파일
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)