from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from graph import app as app_graph

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/recommend")
async def recommend_popups(request: QueryRequest):
    # AI 에이전트 실행
    initial_state = {"user_query": request.query}
    result = app_graph.invoke(initial_state) 

    popups_data = result.get("retrieved_popups", [])
    extracted_ids = [popup["id"] for popup in popups_data]
    
    return {
        "answer": result["final_answer"],
        "popupIds": extracted_ids 
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)