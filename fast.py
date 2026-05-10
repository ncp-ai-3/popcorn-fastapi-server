from fastapi import FastAPI
from pydantic import BaseModel # 추가됨!
import uvicorn
from graph import app as app_graph

app = FastAPI()

# 1. 받을 데이터의 '규칙(모양)'을 명시합니다.
class SpringRequest(BaseModel):
    query: str
    user_id: str = "default_user"  # 값을 안 보내면 'default_user'로 처리

# 2. Request 대신 우리가 만든 규칙(SpringRequest)을 사용합니다.
@app.post("/recommend")
async def recommend(data: SpringRequest): # 여기가 핵심!
    
    # 이제 data.query 처럼 아주 편하게 꺼내 쓸 수 있습니다.
    print("🚀 [스프링에서 넘어온 데이터 확인]:", data)
    
    user_query = data.query
    user_id = data.user_id

    # 랭그래프 실행 시 config에 thread_id 전달
    config = {"configurable": {"thread_id": user_id}}
    
    inputs = {"user_query": user_query}
    result = app_graph.invoke(inputs, config=config) 
    
    return {"answer": result["final_answer"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)