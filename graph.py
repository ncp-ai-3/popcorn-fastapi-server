import os
import psycopg2
from dotenv import load_dotenv
from llama_cpp import Llama
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# .env 파일에서 DB 비밀번호 등 환경변수 불러오기
load_dotenv()

# 모델 로드 (GPU 가속)
print("⏳ M4 GPU를 사용하여 모델을 로드하는 중...")
llm = Llama(model_path="./qwen2-1_5b-instruct-q4_k_m.gguf", n_gpu_layers=-1, verbose=False)

class AgentState(TypedDict):
    user_query: str
    intent: str
    retrieved_popups: List[dict]
    final_answer: str

# 1. 의도 분석 노드
def analyze_intent(state: AgentState):
    return {"intent": "popup_search"}

# 2. RAG 검색 노드 (진짜 DB 연결 🚀)
def retrieve_popups(state: AgentState):
    print("[Node: RAG 검색] NCP DB에서 데이터를 가져오는 중...")
    
    try:
        # 1) DB 연결 (비밀번호는 .env에서 쏙 가져옵니다)
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            dbname=os.getenv("DB_NAME")
        )
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, title, description, latitude, longitude FROM popup LIMIT 3;") 
        
        rows = cursor.fetchall()
        # 3) DB에서 가져온 데이터를 우리 바구니 모양에 맞게 포장
        # (참고) 데이터를 바구니에 담는 부분
        db_result = []
        for row in rows:
            db_result.append({
                "id": row[0],
                "name": row[1],         # DB의 title을 name으로
                "desc": row[2],         # DB의 description을 desc로
                "lat": float(row[3]) if row[3] else 0.0,
                "lng": float(row[4]) if row[4] else 0.0
            })
            
        cursor.close()
        conn.close()
        
        print(f"✅ DB 검색 완료! {len(db_result)}개의 팝업을 찾았습니다.")
        return {"retrieved_popups": db_result}
        
    except Exception as e:
        print(f"🚨 DB 연결 또는 조회 에러 발생: {e}")
        # 에러가 나도 서버가 죽지 않도록 빈 리스트를 반환합니다.
        return {"retrieved_popups": []}

# 3. 답변 생성 노드
def generate_recommendation(state: AgentState):
    query = state["user_query"]
    popups = state.get("retrieved_popups", [])
    
    context = "\n".join([f"- {p['name']}: {p['desc']}" for p in popups])
    
    prompt = f"""<|im_start|>system
당신은 성수동 팝업스토어 가이드입니다. 제공된 [팝업 목록]만 사용해서 추천하세요.
[팝업 목록]:
{context}
<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
    output = llm(prompt, max_tokens=512, stop=["<|im_end|>"], echo=False)
    return {"final_answer": output["choices"][0]["text"].strip()}

# 워크플로우 조립
workflow = StateGraph(AgentState)
workflow.add_node("analyze_intent", analyze_intent)
workflow.add_node("retrieve_popups", retrieve_popups)
workflow.add_node("generate", generate_recommendation)

workflow.set_entry_point("analyze_intent")
workflow.add_edge("analyze_intent", "retrieve_popups")
workflow.add_edge("retrieve_popups", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()