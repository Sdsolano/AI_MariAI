"""
Experimental HYBRID teacher reasoner using OpenAI Responses API.

Pipeline:
  1) Deterministic SQL shortcuts (same as multiagent hybrid)
  2) If miss → Responses API agentic loop:
       - client.responses.create
       - reasoning.effort
       - previous_response_id
       - function calling

Does NOT replace Mao (/chat/teacher/) nor the Chat-Completions hybrid (/multiagent).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import OpenAI

from app.services import teacher_sql_service as teacher_sql
from app.services.teacher_multiagent_service import try_deterministic

logger = logging.getLogger(__name__)

# Reasoning-capable models available on the project key (probed 2026-11-23)
MODEL = "gpt-5-mini"
REASONING_EFFORT = "medium"
MAX_ROUNDS = 8

INSTRUCTIONS = """Eres Mao Responses, asistente de un docente colombiano.
Responde SOLO con datos de las herramientas (notas reales). No inventes.
Alcance: solo cursos del docente autenticado (iddocente ya filtrado en tools).

Estrategia:
1) list_periods si necesitas id de periodo (Periodo 1/2/3 ≠ id numérico).
2) Preguntas de actividad → list_activities o activity_extreme_grade (mode max|min|avg).
3) Ranking mejor/peor estudiante → rank_students (order DESC|ASC, asignatura opcional).
4) Mejor/peor nota de un alumno → student_best_grade.
5) Promedio global → global_average.
6) SQL libre (run_sql) solo si las tools anteriores no bastan; debe ser SELECT con :iddocente.

Responde en español, breve, con nombres y cifras exactas.
Si varios alumnos empatan en la nota extrema, menciónalos.
No pidas aclaraciones si puedes responder con tools: si falta periodo,
usa todos o lista_periods y elige el más reciente / el que encaje por nombre.
"""


TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "name": "list_periods",
        "description": "Lista periodos académicos ligados a las actividades del docente.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "list_activities",
        "description": "Lista/filtra actividades del docente (por asignatura, periodo, nombre).",
        "parameters": {
            "type": "object",
            "properties": {
                "asignatura": {"type": "string"},
                "idperiodo": {"type": "integer"},
                "actividad": {"type": "string", "description": "Fragmento del nombre de la actividad"},
                "limit": {"type": "integer"},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "activity_extreme_grade",
        "description": "Mejor/peor nota (o promedio) de una actividad del docente.",
        "parameters": {
            "type": "object",
            "properties": {
                "actividad": {"type": "string"},
                "asignatura": {"type": "string"},
                "idperiodo": {"type": "integer"},
                "mode": {"type": "string", "enum": ["max", "min", "avg"]},
            },
            "required": ["actividad", "mode"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "rank_students",
        "description": "Ranking de estudiantes por promedio en un periodo (opcionalmente por asignatura).",
        "parameters": {
            "type": "object",
            "properties": {
                "idperiodo": {"type": "integer"},
                "asignatura": {"type": "string"},
                "order": {"type": "string", "enum": ["ASC", "DESC"]},
                "limit": {"type": "integer"},
            },
            "required": ["idperiodo"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "student_best_grade",
        "description": "Mejor o peor nota individual de un estudiante (opcionalmente en una asignatura).",
        "parameters": {
            "type": "object",
            "properties": {
                "nombre": {"type": "string"},
                "asignatura": {"type": "string"},
                "worst": {"type": "boolean"},
            },
            "required": ["nombre"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "global_average",
        "description": "Promedio global de todas las notas del docente.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "best_per_subject",
        "description": "Mejor (o peor) estudiante por cada asignatura/grupo del docente.",
        "parameters": {
            "type": "object",
            "properties": {
                "worst": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_sql",
        "description": "Ejecuta un SELECT de solo lectura con :iddocente obligatorio.",
        "parameters": {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
            "additionalProperties": False,
        },
    },
]


def _trim_payload(payload: Dict[str, Any], max_rows: int = 12) -> Dict[str, Any]:
    out = dict(payload or {})
    rows = out.get("result")
    if isinstance(rows, list) and len(rows) > max_rows:
        out["result"] = rows[:max_rows]
        out["truncated"] = True
        out["total_before_trim"] = len(rows)
    # drop huge sql_executed blobs
    if "sql_executed" in out and isinstance(out["sql_executed"], str) and len(out["sql_executed"]) > 400:
        out["sql_executed"] = out["sql_executed"][:400] + "…"
    return out


async def _run_tool(
    *,
    db_url: str,
    iddocente: int,
    name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    args = arguments or {}
    try:
        if name == "list_periods":
            return await teacher_sql.list_periods(db_url, iddocente)
        if name == "list_activities":
            return await teacher_sql.list_activities(
                db_url,
                iddocente,
                asignatura=args.get("asignatura"),
                idperiodo=args.get("idperiodo"),
                actividad=args.get("actividad"),
                limit=int(args.get("limit") or 30),
            )
        if name == "activity_extreme_grade":
            return await teacher_sql.activity_extreme_grade(
                db_url,
                iddocente,
                actividad=str(args.get("actividad") or ""),
                asignatura=args.get("asignatura"),
                idperiodo=args.get("idperiodo"),
                mode=str(args.get("mode") or "max"),
            )
        if name == "rank_students":
            return await teacher_sql.rank_students_by_period(
                db_url,
                iddocente,
                int(args["idperiodo"]),
                order=str(args.get("order") or "DESC"),
                limit=int(args.get("limit") or 5),
                asignatura=args.get("asignatura"),
            )
        if name == "student_best_grade":
            return await teacher_sql.student_best_grade(
                db_url,
                iddocente,
                nombre=str(args.get("nombre") or ""),
                asignatura=args.get("asignatura"),
                worst=bool(args.get("worst")),
            )
        if name == "global_average":
            return await teacher_sql.global_average(db_url, iddocente)
        if name == "best_per_subject":
            return await teacher_sql.best_student_per_subject_group(
                db_url,
                iddocente,
                order="ASC" if args.get("worst") else "DESC",
            )
        if name == "run_sql":
            return await teacher_sql.run_teacher_sql(
                db_url, iddocente, str(args.get("sql") or "")
            )
        return {"error": f"herramienta desconocida: {name}"}
    except Exception as e:
        logger.exception("tool %s failed", name)
        return {"error": str(e), "result": [], "count": 0}


def _function_calls(response: Any) -> List[Any]:
    out = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "function_call":
            out.append(item)
    return out


async def _run_responses_agentic(
    *,
    client: OpenAI,
    db_url: str,
    iddocente: int,
    question: str,
    teacher_name: str,
    max_rounds: int,
    model: str,
    reasoning_effort: str,
    prior_trace: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Responses API tool loop (fallback after deterministic miss)."""
    trace: List[Dict[str, Any]] = list(prior_trace or [])
    effort = reasoning_effort

    user_input = (
        f"Docente: {teacher_name or iddocente}\n"
        f"Pregunta: {question.strip()}"
    )

    create_kwargs: Dict[str, Any] = {
        "model": model,
        "instructions": INSTRUCTIONS,
        "tools": TOOLS,
        "input": user_input,
        "store": True,
    }
    if effort:
        create_kwargs["reasoning"] = {"effort": effort}

    try:
        response = client.responses.create(**create_kwargs)
    except Exception as e:
        if "reasoning" in str(e).lower():
            create_kwargs.pop("reasoning", None)
            response = client.responses.create(**create_kwargs)
            effort = ""
        else:
            raise

    trace.append(
        {
            "agent": "responses",
            "round": 0,
            "response_id": getattr(response, "id", None),
            "tool_calls": len(_function_calls(response)),
        }
    )

    rounds_used = 0
    for round_idx in range(1, max_rounds + 1):
        calls = _function_calls(response)
        if not calls:
            break
        rounds_used = round_idx
        tool_outputs: List[Dict[str, Any]] = []
        for call in calls:
            raw_args = getattr(call, "arguments", None) or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                args = {}
            payload = await _run_tool(
                db_url=db_url,
                iddocente=iddocente,
                name=str(getattr(call, "name", "")),
                arguments=args if isinstance(args, dict) else {},
            )
            trimmed = _trim_payload(payload)
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": getattr(call, "call_id", None),
                    "output": json.dumps(trimmed, ensure_ascii=False, default=str),
                }
            )
            trace.append(
                {
                    "agent": "tool",
                    "round": round_idx,
                    "name": getattr(call, "name", None),
                    "args": args,
                    "ok": not bool(trimmed.get("error") or trimmed.get("rejected")),
                    "count": trimmed.get("count"),
                }
            )

        follow_kwargs: Dict[str, Any] = {
            "model": model,
            "tools": TOOLS,
            "previous_response_id": response.id,
            "input": tool_outputs,
            "store": True,
        }
        if effort:
            follow_kwargs["reasoning"] = {"effort": effort}
        try:
            response = client.responses.create(**follow_kwargs)
        except Exception as e:
            if "reasoning" in str(e).lower():
                follow_kwargs.pop("reasoning", None)
                response = client.responses.create(**follow_kwargs)
            else:
                raise
        trace.append(
            {
                "agent": "responses",
                "round": round_idx,
                "response_id": getattr(response, "id", None),
                "tool_calls": len(_function_calls(response)),
            }
        )

    answer = (getattr(response, "output_text", None) or "").strip()
    if not answer:
        answer = "No alcancé una respuesta con las herramientas disponibles."
    pending = bool(_function_calls(response))
    return {
        "response": answer,
        "satisfactory": bool(answer) and not pending,
        "score": 1.0 if (answer and not pending) else 0.4,
        "rounds_used": rounds_used,
        "trace": trace,
        "critic": {
            "satisfactory": not pending,
            "score": 1.0 if not pending else 0.4,
            "reason": "responses_api_loop",
        },
        "previous_response_id": getattr(response, "id", None),
        "reasoning_effort": effort or None,
    }


async def run_responses_teacher_qa(
    *,
    client: OpenAI,
    db_url: str,
    iddocente: int,
    question: str,
    teacher_name: str = "",
    max_rounds: int = MAX_ROUNDS,
    model: str = MODEL,
    reasoning_effort: str = REASONING_EFFORT,
) -> Dict[str, Any]:
    started = datetime.now()
    max_rounds = max(1, min(12, int(max_rounds)))

    # 1) Deterministic hybrid shortcuts (same as /multiagent)
    det = await try_deterministic(db_url, iddocente, question)
    if det and det.get("satisfactory") and det.get("response"):
        elapsed_ms = int((datetime.now() - started).total_seconds() * 1000)
        return {
            "response": det["response"],
            "satisfactory": True,
            "score": float(det.get("score") or 1.0),
            "rounds_used": 0,
            "max_rounds": max_rounds,
            "trace": det.get("trace") or [],
            "critic": {"satisfactory": True, "score": 1.0, "reason": "deterministic"},
            "plan": {
                "path": "hybrid_deterministic",
                "model": model,
                "reasoning_effort": None,
                "teacher": teacher_name,
            },
            "elapsed_ms": elapsed_ms,
            "mode": "hybrid_responses_deterministic",
            "model": model,
            "reasoning_effort": None,
            "previous_response_id": None,
            "timestamp": datetime.now().isoformat(),
        }

    # 2) Responses API agentic fallback
    prior = (det or {}).get("trace") or []
    agentic = await _run_responses_agentic(
        client=client,
        db_url=db_url,
        iddocente=iddocente,
        question=question,
        teacher_name=teacher_name,
        max_rounds=max_rounds,
        model=model,
        reasoning_effort=reasoning_effort,
        prior_trace=prior,
    )
    elapsed_ms = int((datetime.now() - started).total_seconds() * 1000)
    return {
        "response": agentic["response"],
        "satisfactory": bool(agentic.get("satisfactory")),
        "score": float(agentic.get("score") or 0),
        "rounds_used": int(agentic.get("rounds_used") or 0),
        "max_rounds": max_rounds,
        "trace": agentic.get("trace") or [],
        "critic": agentic.get("critic"),
        "plan": {
            "path": "hybrid_responses_agentic",
            "model": model,
            "reasoning_effort": agentic.get("reasoning_effort"),
            "teacher": teacher_name,
        },
        "elapsed_ms": elapsed_ms,
        "mode": "hybrid_responses_agentic",
        "model": model,
        "reasoning_effort": agentic.get("reasoning_effort"),
        "previous_response_id": agentic.get("previous_response_id"),
        "timestamp": datetime.now().isoformat(),
    }
