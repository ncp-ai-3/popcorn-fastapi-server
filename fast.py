from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from graph import app as app_graph # 랭그래프 컴파일된 앱 가져오기

app = FastAPI()

# 1. 스프링이 보내는 이름(userId, question)과 똑같이 맞춥니다.
class SpringRequest(BaseModel):
    userId: str
    question: str

@app.post("/recommend") # 스프링 application.yml에 등록할 주소
async def recommend(data: SpringRequest):
    # 터미널에서 데이터 잘 오는지 확인
    print(f"🚀 [Spring -> FastAPI] userId: {data.userId}, question: {data.question}")
    
    # 2. 랭그래프에 전달할 설정 (thread_id로 userId 사용)
    config = {"configurable": {"thread_id": data.userId}}
    
    # 3. 랭그래프 입력값 (graph.py의 AgentState 필드명과 맞춰야 함)
    inputs = {"user_query": data.question} 
    
    # 4. 랭그래프 실행
    result = app_graph.invoke(inputs, config=config)
    
    # 5. 스프링이 받기 편하게 JSON으로 반환
    return {
        "answer": result["final_answer"],
        "status": "success"
    }

if __name__ == "__main__":
    # NCP 서버에서 돌릴 때를 대비해 host를 0.0.0.0으로 설정
    uvicorn.run(app, host="0.0.0.0", port=8000)