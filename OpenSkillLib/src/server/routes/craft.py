from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

router = APIRouter()


class CraftRequest(BaseModel):
    task: str
    weak_model: Optional[str] = "openai/gpt-4o-mini"
    strong_model: Optional[str] = "anthropic/claude-3-5-sonnet"
    context_docs: Optional[list[str]] = None  # Additional documents to use as context


@router.post("/")
async def craft_skill(req: CraftRequest, request: Request):
    client = request.app.state.client
    try:
        meta = await client.craft(
            task=req.task,
            weak_model=req.weak_model,
            strong_model=req.strong_model,
            context_docs=req.context_docs,
        )
        return {"status": "success", "skill": meta.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))