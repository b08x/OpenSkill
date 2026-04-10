from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 3
    mode: str = "auto"  # "injection" or "verbalization"


@router.post("/")
async def retrieve_guidance(req: RetrieveRequest, request: Request):
    client = request.app.state.client

    try:
        # 1. S-Path-RAG + TurboQuant Retrieval
        guidance = await client.retriever.retrieve(req.query, top_k=req.top_k)

        response_data = {
            "query": req.query,
            "confidence": guidance.confidence,
            "trace": guidance.reasoning_trace,
            "skills": guidance.skills_meta,
            "best_path": guidance.best_path_ids
        }

        # 2. Generation (If local model is loaded, performs Vector Injection)
        if hasattr(client.llm, 'generate_with_guidance'):
            gen_resp = await client.llm.generate_with_guidance(
                query=req.query,
                guidance=guidance,
                mode=req.mode
            )
            response_data["answer"] = gen_resp.content
            response_data["generation_mode"] = gen_resp.raw.get("mode")
        else:
            response_data["answer"] = None
            response_data["message"] = "Local LLM not loaded for generation."

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))