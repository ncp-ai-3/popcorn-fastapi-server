import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from graph import app as langgraph_app

app = FastAPI()

# Spring에서 보내주는 데이터를 담는 그릇
class SpringRequest(BaseModel):
    userId: str
    queryVector: List[float]  # 시현님이 주시는 768차원 숫자 리스트
    question: str             # 유저가 입력한 실제 텍스트 질문

@app.post("/recommend")
async def recommend(request: SpringRequest):
    # 랭그래프 실행을 위한 입력값 세팅
    # 체크포인트를 위한 thread_id는 유저별로 고유하게 관리합니다.
    config = {"configurable": {"thread_id": request.userId}}
    
    inputs = {
        "user_query": request.question,
        "query_vector": request.queryVector
    }
    
    # 랭그래프 가동!
    result = langgraph_app.invoke(inputs, config=config)
    
    # 최종 결과 반환
    return {
        "answer": result.get("final_answer"),
        "retrieved_popups": result.get("retrieved_popups")
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)