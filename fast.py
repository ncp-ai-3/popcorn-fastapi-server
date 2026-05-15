import logging
from typing import Any, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from graph import app as langgraph_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()


class SpringRequest(BaseModel):
    userId: str
    question: str
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


def _popups_db_summary(popups: list[dict[str, Any]]) -> str:
    """로그 한 줄용: id, title, cosine_distance, desc 앞 50자."""
    parts: list[str] = []
    for p in popups or []:
        cd = p.get("cosine_distance")
        cos = "n/a" if cd is None else f"{float(cd):.4f}"
        title = (p.get("name") or "")[:80]
        desc = (p.get("desc") or "").replace("\n", " ")[:50]
        parts.append(f"id={p['id']} title={title!r} cos={cos} desc50={desc!r}")
    return " | ".join(parts) if parts else "(없음)"


def _log_chat_summary(*, user_id: str, question: str, result: dict[str, Any]) -> None:
    route = result.get("route")
    vec = result.get("condition_vector")
    mode = result.get("retrieval_mode")
    embed_text = result.get("embed_text_used")
    if not embed_text:
        embed_text = "(임베딩 단계 없음·최근목록 경로 등)"
    qv = result.get("query_vector")
    if isinstance(qv, list) and qv:
        emb = f"dim={len(qv)} preview={_embedding_preview_50(qv)}"
    else:
        emb = "벡터없음"
    pops = result.get("retrieved_popups") or []
    ans = (result.get("final_answer") or "")[:50].replace("\n", " ")
    logger.info(
        "[chat] userId=%s 입력=%r 라우트=%s condition_vector=%s retrieval_mode=%s "
        "임베딩문장=%r 임베딩=%s DB=[%s] 답변50자=%r",
        user_id,
        question,
        route,
        vec,
        mode,
        embed_text,
        emb,
        _popups_db_summary(pops),
        ans + ("…" if len(result.get("final_answer") or "") > 50 else ""),
    )


@app.post("/chat")
async def chat(request: SpringRequest):
    config = {"configurable": {"thread_id": request.userId}}

    inputs: dict = {
        "user_query": request.question,
        "skip_embedding": False,
    }
    if request.queryVector is not None:
        inputs["query_vector"] = request.queryVector
        inputs["skip_embedding"] = True
        inputs["embed_text_used"] = "<client_query_vector>"

    try:
        result = await langgraph_app.ainvoke(inputs, config=config)
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

    _log_chat_summary(user_id=request.userId, question=request.question, result=result)

    return {
        "answer": result.get("final_answer"),
        "retrieved_popups": result.get("retrieved_popups"),
        "matched_popup_ids": result.get("matched_popup_ids", []),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
