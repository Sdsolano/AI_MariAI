"""
Experimental HYBRID Responses API teacher endpoint.

POST /api/v1/chat/teacher/responses
  1) deterministic SQL shortcuts
  2) Responses API agentic loop (reasoning.effort + previous_response_id + tools)

Does NOT replace Mao (/chat/teacher/) or Chat-Completions hybrid (/multiagent).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

from app.services import teacher_context_service as teacher_ctx
from app.services.teacher_responses_service import (
    MAX_ROUNDS,
    MODEL,
    REASONING_EFFORT,
    run_responses_teacher_qa,
)

logger = logging.getLogger(__name__)
router = APIRouter()

client = OpenAI(api_key=os.environ["API_KEY"])


class ResponsesRequest(BaseModel):
    user_id: str
    message: str
    db_url: str
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    conversation_history: Optional[List[Dict[str, str]]] = None
    max_rounds: Optional[int] = None
    model: Optional[str] = None
    reasoning_effort: Optional[str] = None


class ResponsesResponse(BaseModel):
    response: str
    user_id: str
    conversation_id: Optional[str] = None
    satisfactory: bool
    score: float
    rounds_used: int
    max_rounds: int
    elapsed_ms: int
    mode: str = "responses_api"
    model: Optional[str] = None
    reasoning_effort: Optional[str] = None
    plan: Optional[Dict[str, Any]] = None
    critic: Optional[Dict[str, Any]] = None
    trace: Optional[List[Dict[str, Any]]] = None
    timestamp: str
    confidence: Optional[float] = None


@router.post("/responses", response_model=ResponsesResponse)
async def teacher_responses(request: ResponsesRequest):
    try:
        iddocente = int(request.user_id)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail="user_id debe ser iddocente") from e

    if not (request.message or "").strip():
        raise HTTPException(status_code=400, detail="message es requerido")
    if not request.db_url:
        raise HTTPException(status_code=400, detail="db_url es requerido")

    teacher_info = await teacher_ctx.get_teacher_info(iddocente, request.db_url)
    if not teacher_info:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontró docente activo con id={iddocente}",
        )

    max_rounds = max(1, min(12, int(request.max_rounds or MAX_ROUNDS)))
    model = (request.model or MODEL).strip()
    effort = (request.reasoning_effort or REASONING_EFFORT).strip()

    logger.info(
        "ResponsesAPI start iddocente=%s model=%s effort=%s q=%s",
        iddocente,
        model,
        effort,
        (request.message or "")[:120],
    )

    result = await run_responses_teacher_qa(
        client=client,
        db_url=request.db_url,
        iddocente=iddocente,
        question=request.message.strip(),
        teacher_name=teacher_info.get("nombre_completo") or "",
        max_rounds=max_rounds,
        model=model,
        reasoning_effort=effort,
    )

    return ResponsesResponse(
        response=result["response"],
        user_id=str(iddocente),
        conversation_id=request.conversation_id,
        satisfactory=bool(result.get("satisfactory")),
        score=float(result.get("score") or 0),
        rounds_used=int(result.get("rounds_used") or 0),
        max_rounds=int(result.get("max_rounds") or max_rounds),
        elapsed_ms=int(result.get("elapsed_ms") or 0),
        mode=result.get("mode") or "responses_api",
        model=result.get("model") or model,
        reasoning_effort=result.get("reasoning_effort"),
        plan=result.get("plan"),
        critic=result.get("critic"),
        trace=result.get("trace"),
        timestamp=result.get("timestamp") or "",
        confidence=float(result.get("score") or 0),
    )


@router.get("/responses/health")
async def responses_health():
    return {
        "status": "operational",
        "service": "Mao hybrid Responses reasoner (deterministic + Responses API)",
        "model_default": MODEL,
        "reasoning_effort_default": REASONING_EFFORT,
        "paths": ["hybrid_responses_deterministic", "hybrid_responses_agentic"],
        "features": [
            "deterministic_sql",
            "responses.create",
            "reasoning.effort",
            "previous_response_id",
            "function_calling",
            "agentic_tool_loop",
        ],
        "note": "Experimental — does not replace /chat/teacher/ or /multiagent",
    }
