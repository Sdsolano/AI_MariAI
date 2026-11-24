"""
Experimental multi-agent teacher endpoint (query loop until satisfactory).

POST /api/v1/chat/teacher/multiagent
Same payload shape as Mao teacher chat (JSON). For comparison only.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

from app.services import teacher_context_service as teacher_ctx
from app.services.teacher_multiagent_service import (
    MAX_ROUNDS,
    run_multiagent_teacher_qa,
)

logger = logging.getLogger(__name__)
router = APIRouter()

client = OpenAI(api_key=os.environ["API_KEY"])


class MultiAgentRequest(BaseModel):
    user_id: str  # iddocente
    message: str
    db_url: str
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    conversation_history: Optional[List[Dict[str, str]]] = None
    max_rounds: Optional[int] = None


class MultiAgentResponse(BaseModel):
    response: str
    user_id: str
    conversation_id: Optional[str] = None
    satisfactory: bool
    score: float
    rounds_used: int
    max_rounds: int
    elapsed_ms: int
    mode: str = "multiagent_query_loop"
    plan: Optional[Dict[str, Any]] = None
    critic: Optional[Dict[str, Any]] = None
    trace: Optional[List[Dict[str, Any]]] = None
    timestamp: str
    confidence: Optional[float] = None


@router.post("/multiagent", response_model=MultiAgentResponse)
async def teacher_multiagent(request: MultiAgentRequest):
    """
    Multi-agente: planifica → consulta SQL → critica → reintenta
    hasta respuesta satisfactoria o max_rounds.
    """
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

    max_rounds = int(request.max_rounds or MAX_ROUNDS)
    max_rounds = max(1, min(12, max_rounds))

    logger.info(
        "Multiagent start iddocente=%s max_rounds=%s q=%s",
        iddocente,
        max_rounds,
        (request.message or "")[:120],
    )

    result = await run_multiagent_teacher_qa(
        client=client,
        db_url=request.db_url,
        iddocente=iddocente,
        question=request.message.strip(),
        teacher_name=teacher_info.get("nombre_completo") or "",
        max_rounds=max_rounds,
    )

    return MultiAgentResponse(
        response=result["response"],
        user_id=str(iddocente),
        conversation_id=request.conversation_id,
        satisfactory=bool(result.get("satisfactory")),
        score=float(result.get("score") or 0),
        rounds_used=int(result.get("rounds_used") or 0),
        max_rounds=int(result.get("max_rounds") or max_rounds),
        elapsed_ms=int(result.get("elapsed_ms") or 0),
        mode=result.get("mode") or "multiagent_query_loop",
        plan=result.get("plan"),
        critic=result.get("critic"),
        trace=result.get("trace"),
        timestamp=result.get("timestamp") or "",
        confidence=float(result.get("score") or 0),
    )


@router.get("/multiagent/health")
async def multiagent_health():
    return {
        "status": "operational",
        "service": "Mao hybrid reasoner (deterministic SQL + agentic loop)",
        "model": "gpt-4o-mini",
        "max_rounds_default": MAX_ROUNDS,
        "paths": ["deterministic", "hybrid_agentic"],
        "agents": ["deterministic", "query", "critic", "writer"],
        "note": "Experimental — does not replace /chat/teacher/",
    }
