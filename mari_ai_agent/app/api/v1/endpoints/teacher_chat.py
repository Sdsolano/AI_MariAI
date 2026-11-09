"""
Mao — teacher chat endpoint (isolated from student /chat/).
POST /api/v1/chat/teacher/

Hybrid:
- Server always loads a compact academic snapshot (asignaturas/notas/asistencia)
  and injects it into the prompt for the LLM to resolve.
- Detail tools only for students-at-risk, PDM and RAG.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from openai import OpenAI
from pydantic import BaseModel

from app.services import teacher_context_service as teacher_ctx
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter()

client = OpenAI(api_key=os.environ["API_KEY"])
rag_service = RAGService()

DATABASE_FILE = "teacher_chat_history.db"

TEACHER_ASSISTANT_PROMPT = """
Eres Mao, el asistente de IA de MariAI para docentes.

PERSONALIDAD: profesional, claro, cercano, conciso. Español colombiano.

DATOS:
En cada turno recibes un bloque JSON cerrado DATOS_ACADEMICOS_DEL_DOCENTE con:
- asignaturas
- calificaciones (promedio, total_notas, estudiantes_con_nota)
- asistencia (presentes, ausentes, tarde, excusas)
- origenes

REGLAS DE CIFRAS (OBLIGATORIAS):
1. Para notas, asistencia o listado de grupos/asignaturas: responde SOLO con cifras que aparezcan en ese JSON. Prohibido inventar, redondear de más o estimar.
2. Si el usuario pide algo que no está en el JSON (ej. un grupo que no aparece), dilo: "en tus datos no aparece X".
3. Ignora mayúsculas/tildes al buscar (Matematicas = Matemáticas, 5A = 5°A).
4. NUNCA modifiques notas ni digas que ya guardaste algo. Borradores = sugerencia con aprobación humana.
5. No clasifiques estudiantes con etiquetas definitivas.
6. No muestres iddocente / idmatricula salvo que el usuario lo pida.
7. Cita el origen del JSON (ej. acad_actividades_notas).
8. No agregues grado/grupo que el usuario no pidió.
9. Si una herramienta de detalle devuelve filas (count>0), lista esas filas; prohibido decir que no hay datos.

HERRAMIENTAS (solo detalle fino):
- get_students_needing_attention: lista estudiantes con promedio bajo (no viene en el snapshot).
- list_pdm: planes de mejoramiento.
- search_academic_resources: material pedagógico / RAG por grado.
No uses tools para promedios ni asistencia agregada: eso ya está en el JSON.
"""

TEACHER_FUNCTION_DEFINITIONS = [
    {
        "name": "get_students_needing_attention",
        "description": (
            "Detalle: estudiantes del docente con promedio bajo el umbral. "
            "Usar solo si preguntan por alumnos en riesgo / que necesitan atención."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "consulta": {
                    "type": "string",
                    "description": "Filtro opcional, ej. '5°A Matematicas'",
                },
                "threshold": {"type": "number"},
                "limit": {"type": "integer"},
            },
            "required": [],
        },
    },
    {
        "name": "list_pdm",
        "description": "Detalle: planes de mejoramiento (PDM) del docente",
        "parameters": {
            "type": "object",
            "properties": {
                "consulta": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": [],
        },
    },
    {
        "name": "search_academic_resources",
        "description": "Busca recursos académicos por grado (RAG) para planeación",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "grade": {"type": "string"},
                "context_type": {
                    "type": "string",
                    "enum": ["academic", "administrative", "policies"],
                },
            },
            "required": ["query"],
        },
    },
]


class TeacherChatRequest(BaseModel):
    user_id: str  # iddocente
    message: str
    db_url: str
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = {}


class TeacherChatResponse(BaseModel):
    response: str
    user_id: str
    conversation_id: str
    function_calls_made: Optional[List[str]] = []
    snapshot_counts: Optional[Dict[str, int]] = None
    timestamp: str
    confidence: Optional[float] = None


def init_teacher_db():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS historial_chat_teacher (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            iddocente INTEGER NOT NULL,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_historial_teacher_conv
        ON historial_chat_teacher (iddocente, conversation_id, timestamp)
        """
    )
    conn.commit()
    conn.close()


def save_message(iddocente: int, conversation_id: str, role: str, content: str):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO historial_chat_teacher (iddocente, conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
        (iddocente, conversation_id, role, content, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def load_history(iddocente: int, conversation_id: str, limit: int = 5) -> List[Dict[str, str]]:
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT role, content
        FROM historial_chat_teacher
        WHERE iddocente = ? AND conversation_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (iddocente, conversation_id, limit),
    )
    history = [dict(row) for row in reversed(cursor.fetchall())]
    conn.close()
    return history


def trim_history(iddocente: int, conversation_id: str, max_size: int = 5):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        DELETE FROM historial_chat_teacher
        WHERE id IN (
            SELECT id FROM historial_chat_teacher
            WHERE iddocente = ? AND conversation_id = ?
            ORDER BY timestamp ASC
            LIMIT MAX(0, (SELECT COUNT(*) FROM historial_chat_teacher WHERE iddocente = ? AND conversation_id = ?) - ?)
        )
        """,
        (iddocente, conversation_id, iddocente, conversation_id, max_size),
    )
    conn.commit()
    conn.close()


def format_subjects_answer(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No tienes asignaturas asignadas en el sistema. (origen: acad_gradosasignaturas)"
    by_group: Dict[str, List[str]] = {}
    for r in rows:
        key = f"{r.get('grado', '')} / {r.get('grupo', '')}".strip(" /")
        by_group.setdefault(key, [])
        asig = r.get("asignatura")
        if asig and asig not in by_group[key]:
            by_group[key].append(asig)
    lines = [
        f"Tus asignaturas son estas ({len(rows)} asignaciones):",
        "",
    ]
    for grupo, asignaturas in by_group.items():
        lines.append(f"**{grupo}**")
        for a in asignaturas:
            lines.append(f"- {a}")
        lines.append("")
    lines.append("(origen: acad_gradosasignaturas)")
    return "\n".join(lines)


def format_attention_answer(rows: List[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None) -> str:
    if not rows:
        return (
            "No encontré estudiantes por debajo del umbral en tus asignaturas "
            "con los filtros indicados. (origen: acad_actividades_notas)"
        )
    lines = [
        f"Estudiantes que necesitan atención (promedio bajo umbral "
        f"{rows[0].get('umbral', 3.5)}), ordenados del más bajo al más alto:",
        "",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"{i}. {r.get('estudiante')} — {r.get('asignatura')} "
            f"{r.get('grupo')} — promedio {r.get('promedio')} "
            f"({r.get('total_notas')} notas)"
        )
    lines.append("")
    lines.append("(origen: acad_actividades_notas)")
    if meta and meta.get("total_filtrado"):
        lines.append(
            f"Mostrando {len(rows)} de {meta.get('total_filtrado')} coincidencias."
        )
    return "\n".join(lines)


def format_pdm_answer(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No tienes planes de mejoramiento (PDM) registrados. (origen: acad_pdm_estudiantes)"
    lines = ["Planes de mejoramiento (PDM):", ""]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"{i}. {r.get('estudiante')} — {r.get('asignatura')} {r.get('grupo')} "
            f"— periodo {r.get('idperiodo')} — {r.get('archivo') or 'sin archivo'}"
        )
    lines.append("")
    lines.append("(origen: acad_pdm_estudiantes)")
    return "\n".join(lines)


async def execute_teacher_tool(
    function_name: str,
    arguments: str,
    db_url: str,
    iddocente: int,
    user_message: str = "",
) -> Dict[str, Any]:
    try:
        args = json.loads(arguments) if arguments else {}
        # Nunca confiar en el filtro inventado por el LLM (ej. mete "5°A" solo).
        # El servidor parsea el mensaje literal del usuario.
        consulta = user_message or args.get("consulta") or None
        logger.info(
            "Mao detail-tool %s args=%s consulta_user=%s iddocente=%s",
            function_name,
            args,
            consulta,
            iddocente,
        )

        if function_name == "get_students_needing_attention":
            payload = await teacher_ctx.get_students_needing_attention(
                iddocente,
                db_url,
                threshold=float(args.get("threshold", 3.5)),
                limit=int(args.get("limit", 15)),
                consulta=consulta,
            )
            rows = payload.get("result") or []
            return {
                "function": function_name,
                "count": len(rows),
                "meta": payload.get("meta"),
                "result": rows,
            }

        if function_name == "list_pdm":
            payload = await teacher_ctx.list_pdm_for_teacher(
                iddocente,
                db_url,
                limit=int(args.get("limit", 30)),
                consulta=consulta,
            )
            rows = payload.get("result") or []
            return {
                "function": function_name,
                "count": len(rows),
                "meta": payload.get("meta"),
                "result": rows,
            }

        if function_name == "search_academic_resources":
            query = args.get("query")
            if not query:
                return {"error": "Missing query parameter"}
            grade = args.get("grade")
            rag_result = await rag_service.query_by_grade(
                query=query,
                db_url=db_url,
                grade=grade,
                context_type=args.get("context_type", "academic"),
            )
            return {
                "function": function_name,
                "query": query,
                "grade": grade,
                "result": rag_result,
            }

        return {"error": f"Unknown function: {function_name}"}
    except Exception as e:
        logger.error("Teacher tool error %s: %s", function_name, e)
        return {"error": f"Function execution failed: {str(e)}"}


init_teacher_db()


@router.post("/", response_model=TeacherChatResponse)
async def chat_with_mao(request: TeacherChatRequest):
    """Chat de Mao para docentes. user_id = iddocente. Hybrid snapshot + detail tools."""
    try:
        iddocente = int(request.user_id)
        MAX_MESSAGES = 5
        conversation_id = request.conversation_id or f"teacher:{request.db_url}"

        logger.info("Mao chat (hybrid) iddocente=%s", iddocente)

        teacher_info = await teacher_ctx.get_teacher_info(iddocente, request.db_url)
        if not teacher_info:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontró docente activo con id={iddocente}",
            )

        # --- Hybrid core: always load compact snapshot server-side ---
        snapshot = await teacher_ctx.get_academic_snapshot(iddocente, request.db_url)
        snapshot_json = json.dumps(snapshot, ensure_ascii=False, default=str)
        logger.info(
            "Mao snapshot iddocente=%s counts=%s",
            iddocente,
            snapshot.get("conteos"),
        )

        # Deterministic match so the LLM cannot "miss" rows that exist
        lookup = teacher_ctx.lookup_snapshot(snapshot, request.message)
        lookup_json = (
            json.dumps(lookup, ensure_ascii=False, default=str) if lookup else None
        )
        if lookup:
            logger.info("Mao lookup tipo=%s filas=%s", lookup.get("tipo"), len(lookup.get("filas") or []))

        history = load_history(iddocente, conversation_id, limit=MAX_MESSAGES)
        teacher_name = teacher_info["nombre_completo"]

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": TEACHER_ASSISTANT_PROMPT},
            {
                "role": "system",
                "content": (
                    f"Docente autenticado: {teacher_name}.\n"
                    f"DATOS_ACADEMICOS_DEL_DOCENTE (JSON cerrado, única fuente de cifras agregadas):\n"
                    f"{snapshot_json}"
                ),
            },
        ]
        # Asignaturas: respuesta determinística (lista explícita de nombres)
        if lookup and lookup.get("tipo") == "asignaturas":
            final_content = format_subjects_answer(lookup.get("filas") or [])
            save_message(iddocente, conversation_id, "user", request.message)
            save_message(iddocente, conversation_id, "assistant", final_content)
            trim_history(iddocente, conversation_id, max_size=MAX_MESSAGES)
            return TeacherChatResponse(
                response=final_content,
                user_id=request.user_id,
                conversation_id=conversation_id,
                function_calls_made=[],
                snapshot_counts=snapshot.get("conteos"),
                timestamp=datetime.now().isoformat(),
                confidence=0.99,
            )

        if lookup_json:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "COINCIDENCIA_SERVIDOR (obligatoria para esta pregunta). "
                        "Si tipo=calificaciones|asistencia, responde con esas filas "
                        "(promedio/conteos exactos). Si tipo=sin_coincidencia, di que no aparece.\n"
                        f"{lookup_json}"
                    ),
                }
            )
        messages.extend(history)
        messages.append(
            {"role": "user", "content": request.message}
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=[
                {"type": "function", "function": func}
                for func in TEACHER_FUNCTION_DEFINITIONS
            ],
            tool_choice="auto",
            temperature=0.2,
            max_tokens=900,
        )

        assistant_message = response.choices[0].message
        function_calls_made: List[str] = []
        final_content = ""

        # Detail tools: server formats structured lists (LLM must not rewrite cifras).
        if assistant_message.tool_calls:
            tool_call = assistant_message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = tool_call.function.arguments
            function_result = await execute_teacher_tool(
                function_name,
                function_args,
                request.db_url,
                iddocente,
                user_message=request.message,
            )
            function_calls_made.append(function_name)
            rows = function_result.get("result") or []
            meta = function_result.get("meta")

            if function_name == "get_students_needing_attention":
                final_content = format_attention_answer(rows, meta)
            elif function_name == "list_pdm":
                final_content = format_pdm_answer(rows)
            else:
                # RAG / other: let the model phrase from tool JSON
                messages.append(assistant_message)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(
                            function_result, ensure_ascii=False, default=str
                        ),
                    }
                )
                follow = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.2,
                    max_tokens=1200,
                )
                final_content = follow.choices[0].message.content or ""
        else:
            final_content = assistant_message.content or ""

        save_message(iddocente, conversation_id, "user", request.message)
        save_message(iddocente, conversation_id, "assistant", final_content)
        trim_history(iddocente, conversation_id, max_size=MAX_MESSAGES)

        return TeacherChatResponse(
            response=final_content,
            user_id=request.user_id,
            conversation_id=conversation_id,
            function_calls_made=function_calls_made,
            snapshot_counts=snapshot.get("conteos"),
            timestamp=datetime.now().isoformat(),
            confidence=0.95,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Mao chat error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error al procesar la solicitud: {str(e)}"
        )


@router.get("/health")
async def teacher_chat_health():
    return {
        "status": "operational",
        "service": "Mao Teacher Chat (hybrid snapshot)",
        "model": "gpt-4o-mini",
        "mode": "snapshot_injected + detail_tools",
        "detail_functions": [f["name"] for f in TEACHER_FUNCTION_DEFINITIONS],
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/snapshot/{iddocente}")
async def teacher_snapshot_debug(iddocente: int, db_url: str):
    """Debug: ver el JSON que recibe el LLM (solo local/dev)."""
    info = await teacher_ctx.get_teacher_info(iddocente, db_url)
    if not info:
        raise HTTPException(status_code=404, detail="Docente no encontrado")
    snap = await teacher_ctx.get_academic_snapshot(iddocente, db_url)
    return {"teacher": info, "snapshot": snap}
