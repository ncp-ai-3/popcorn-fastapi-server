import logging
import uvicorn
import httpx
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import List, Optional

from embedding_client import fetch_query_embedding
from graph import app as langgraph_app, extract_search_keywords

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()


class SpringRequest(BaseModel):
    # Spring에서 userId가 숫자(Long)로 오면 JSON에 숫자가 됨 → str로 통일
    userId: str
    question: str
    # 제공되면 임베딩 API를 호출하지 않고 해당 벡터 사용 (기존·로컬 테스트용)
    queryVector: Optional[List[float]] = None

    @field_validator("userId", mode="before")
    @classmethod
    def user_id_as_str(cls, v):
        return str(v) if v is not None else v


def _embedding_preview_50(vec: List[float]) -> str:
    head = vec[:12]
    s = "[" + ", ".join(f"{x:.4f}" for x in head)
    if len(vec) > len(head):
        s += ", ..."
    s += "]"
    return s[:50]


@app.post("/chat")
async def chat(request: SpringRequest):
    config = {"configurable": {"thread_id": request.userId}}

    logger.info(
        "[chat] userId=%s user_message=%s",
        request.userId,
        request.question,
    )

    if request.queryVector is not None:
        query_vector = request.queryVector
        logger.info("[embedding] source=client_body dim=%d preview_50=%s", len(query_vector), _embedding_preview_50(query_vector))
    else:
        try:
            # 1. RAG 검색 품질을 높이기 위해 LLM으로 핵심 키워드 먼저 추출
            search_keywords = await asyncio.to_thread(extract_search_keywords, request.question)
            logger.info("[embedding] extracted keywords: '%s' (original: '%s')", search_keywords, request.question)
            
            # 2. 추출된 정제 키워드로 벡터 임베딩 생성
            query_vector = await fetch_query_embedding(search_keywords)
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=502,
                detail=f"embedding API HTTP {e.response.status_code}",
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503,
                detail=f"embedding API unreachable: {e.__class__.__name__}",
            )
        logger.info(
            "[embedding] source=embed_api dim=%d preview_50=%s",
            len(query_vector),
            _embedding_preview_50(query_vector),
        )

    inputs = {
        "user_query": request.question,
        "query_vector": query_vector,
    }

    result = langgraph_app.invoke(inputs, config=config)

    return {
        "answer": result.get("final_answer"),
        "retrieved_popups": result.get("retrieved_popups"),
        "matched_popup_ids": result.get("matched_popup_ids", []),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
