from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from graph import app as app_graph # 랭그래프 가져오기

app = FastAPI()

# 1. Spring에서 오는 요청 데이터 형식 (숫자 1이 와도 문자열 "1"로 자동 변환됨)
class SpringRequest(BaseModel):
    userId: str 
    question: str

@app.post("/recommend")
async def recommend(data: SpringRequest):
    print(f"🚀 [Spring -> FastAPI] userId: {data.userId}, question: {data.question}")
    
    config = {"configurable": {"thread_id": data.userId}}
    inputs = {"user_query": data.question} 
    
    # 2. 랭그래프 실행
    result = app_graph.invoke(inputs, config=config)
    
    # 3. 💡 핵심! 랭그래프의 가방(retrieved_popups)에서 'id' 숫자만 쏙쏙 뽑아내기
    popups = result.get("retrieved_popups", [])
    popup_ids = [popup["id"] for popup in popups] 
    
    return {
        "answer": result["final_answer"],
        "popupIds": popup_ids
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)