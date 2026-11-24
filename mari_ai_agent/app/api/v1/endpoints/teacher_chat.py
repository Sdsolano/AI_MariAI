"""
Mao — teacher chat endpoint (isolated from student /chat/).
POST /api/v1/chat/teacher/

Hybrid:
- Server always loads a compact academic snapshot (asignaturas/notas/asistencia)
  and injects it into the prompt for the LLM to resolve.
- Detail tools with multi-round agent loop (retry / chain).
- Optional document attachment (MATERIAL_ORIGINAL) for support materials.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request
from openai import OpenAI
from pydantic import BaseModel

from app.services import teacher_context_service as teacher_ctx
from app.services import teacher_sql_service as teacher_sql
from app.services.document_extract_service import extract_text_from_upload
from app.services.rag_service import RAGService
from app.services.support_material_pdf_service import (
    build_support_material_attachment,
)

logger = logging.getLogger(__name__)
router = APIRouter()

client = OpenAI(api_key=os.environ["API_KEY"])
rag_service = RAGService()

DATABASE_FILE = "teacher_chat_history.db"
MAX_TOOL_ROUNDS = 5
STRUCTURED_TOOLS = {
    "get_students_needing_attention",
    "get_student_detail",
    "list_pdm",
}

TEACHER_ASSISTANT_PROMPT = """
Eres Mao, el asistente de IA de MariAI para docentes.

PERSONALIDAD: profesional, claro, cercano, conciso. Español colombiano.

DATOS:
En cada turno recibes un bloque JSON cerrado DATOS_ACADEMICOS_DEL_DOCENTE con:
- asignaturas
- calificaciones (promedio, total_notas, estudiantes_con_nota)
- asistencia (presentes, ausentes, tarde, excusas)

REGLAS DE CIFRAS (OBLIGATORIAS):
1. Para notas, asistencia o listado de grupos/asignaturas: responde SOLO con cifras que aparezcan en ese JSON. Prohibido inventar, redondear de más o estimar.
2. Si el usuario pide algo que no está en el JSON (ej. un grupo que no aparece), dilo: "en tus datos no aparece X".
3. Ignora mayúsculas/tildes al buscar (Matematicas = Matemáticas, 5A = 5°A).
4. NUNCA modifiques notas ni digas que ya guardaste algo. Borradores = sugerencia con aprobación humana.
5. No clasifiques estudiantes con etiquetas definitivas.
6. No muestres iddocente / idmatricula salvo que el usuario lo pida.
7. No menciones tablas, orígenes técnicos ni nombres de columnas (nada como acad_*, student_attendance, origen: …). Habla en lenguaje natural.
8. No agregues grado/grupo que el usuario no pidió.
9. Si una herramienta de detalle devuelve filas (count>0), lista esas filas; prohibido decir que no hay datos.

RAZONAMIENTO CON HERRAMIENTAS (OBLIGATORIO):
- Puedes llamar varias herramientas y reintentar en el mismo turno.
- Si una tool falla, viene vacía (count=0) o el nombre no coincide: NO te rindas. Reintenta con otro argumento (nombre parcial, sin tildes, otra consulta) u otra tool.
- Encadena tools cuando haga falta (ej. list_periods → run_teacher_sql → interpretar).
- No inventes datos si las tools no los dieron. Cifras SOLO del snapshot, tools fijas o resultset SQL.

HERRAMIENTAS:
Fijas (rápidas):
- get_students_needing_attention: alumnos con promedio bajo.
- get_student_detail: UN estudiante por nombre.
- list_pdm: planes de mejoramiento.
- search_academic_resources: material pedagógico / RAG por grado.
Exploratorias (SQL de solo lectura, acotado a TUS cursos):
- describe_teacher_schema: esquema/joins permitidos (úsalo si no sabes cómo armar el SQL).
- list_periods: periodos (id/nombre) para no adivinar el "tercer periodo".
- run_teacher_sql: ejecuta SELECT con placeholder obligatorio :iddocente y paso por acad_gradosasignaturas.
SQL EXPLORATORIO (OBLIGATORIO en estos casos — no digas "no aparece" sin consultar):
- mejor / peor estudiante, top N, ranking
- mejor/peor estudiante de cada asignatura (o por asignatura)
- cómo le fue a un estudiante en un periodo (ej. primer periodo)
- filtros por periodo (Periodo 3, tercer periodo, etc.)
- promedio global de todas tus notas
- cualquier ranking que no esté en el snapshot
Flujo: list_periods (si hay periodo) → run_teacher_sql con :iddocente → responde SOLO con las filas.
Prohibido responder "no aparece el periodo" sin haber llamado list_periods/run_teacher_sql.
No uses get_students_needing_attention para "top" o "mejores" (eso es lo contrario).
Si piden umbral (ej. bajo 3.0), pasa threshold=3.0 a get_students_needing_attention.
No uses tools para promedios ni asistencia agregada por grupo: eso ya está en el JSON.
Si preguntan por un alumno concreto y no usas get_student_detail ni SQL, NO digas que "no aparece" solo porque no está en el JSON agregado.
"notas de Educación Artística" es una ASIGNATURA (usa el JSON), no un estudiante.

MATERIAL DE APOYO (refuerzo / taller / recursos de una actividad):
1. Sin MATERIAL_ORIGINAL_ADJUNTADO: SOLO pide que adjunte el documento original (PDF/DOCX/PPTX/guía).
   Prohibido inventar talleres, ejercicios, claves o lecturas genéricas.
   Prohibido rellenar con search_academic_resources si no hay documento adjunto.
2. Con MATERIAL_ORIGINAL_ADJUNTADO: adapta el apoyo a ESE texto (puedes complementar con RAG).
3. Con material original, entrega algo accionable en Markdown claro:
   - Taller corto alineado al documento
   - 2–3 ejercicios RESUELTOS paso a paso
   - 3–5 ejercicios PARA EL ESTUDIANTE (clave al final para el docente)
   - Lectura breve del concepto
4. Indica uso en clase (≈15–20 min) y como tarea.
5. El sistema convertirá tu respuesta a PDF descargable: usa títulos ## / ### y listas.
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
        "name": "get_student_detail",
        "description": (
            "Detalle académico de UN estudiante del docente por nombre "
            "(promedio por asignatura). Usar si preguntan cómo le va / notas de alguien."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "nombre": {
                    "type": "string",
                    "description": "Nombre completo o parcial del estudiante",
                },
            },
            "required": ["nombre"],
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
    {
        "name": "describe_teacher_schema",
        "description": (
            "Devuelve el esquema/joins permitidos para consultas SQL del docente. "
            "Usar antes de run_teacher_sql si no conoces las tablas."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_periods",
        "description": (
            "Lista periodos académicos (id, nombre) relevantes para el docente. "
            "Usar cuando pregunten por un periodo (ej. tercer periodo) antes del SQL."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_teacher_sql",
        "description": (
            "Ejecuta SQL de SOLO LECTURA (SELECT/WITH) acotado al docente. "
            "Obligatorio: placeholder :iddocente y JOIN/FROM acad_gradosasignaturas. "
            "Para consultas ad-hoc: rankings, mejor estudiante, filtros por periodo, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": (
                        "Consulta SELECT/WITH. Debe incluir :iddocente "
                        "(ej. WHERE ga.iddocente = :iddocente). Opcional :idperiodo."
                    ),
                },
                "idperiodo": {
                    "type": "integer",
                    "description": "Bind opcional si el SQL usa :idperiodo",
                },
            },
            "required": ["sql"],
        },
    },
]


class TeacherChatRequest(BaseModel):
    user_id: str  # iddocente
    message: str
    db_url: str
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = {}
    conversation_history: Optional[List[Dict[str, str]]] = None


class TeacherChatResponse(BaseModel):
    response: str
    user_id: str
    conversation_id: str
    function_calls_made: Optional[List[str]] = []
    snapshot_counts: Optional[Dict[str, int]] = None
    charts: Optional[List[Dict[str, Any]]] = None
    attachment: Optional[Dict[str, Any]] = None
    timestamp: str
    confidence: Optional[float] = None


def _label_asig(row: Dict[str, Any]) -> str:
    asig = (row.get("asignatura") or "").strip()
    grupo = (row.get("grupo") or "").strip()
    grado = (row.get("grado") or "").strip()
    tail = " ".join(p for p in (grado, grupo) if p).strip()
    if asig and tail:
        return f"{asig} ({tail})"
    return asig or tail or "—"


def charts_from_grades(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Barras de promedio por asignatura/grupo."""
    usable = [
        r
        for r in rows
        if r.get("promedio") is not None and _label_asig(r)
    ]
    if len(usable) < 1:
        return []
    usable = usable[:12]
    return [
        {
            "type": "bar",
            "title": "Promedio por asignatura",
            "categories": [_label_asig(r) for r in usable],
            "series": [
                {
                    "name": "Promedio",
                    "data": [float(r["promedio"]) for r in usable],
                }
            ],
            "yMax": 5,
            "suffix": "",
            "decimals": 1,
        }
    ]


def charts_from_attendance(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Barras agrupadas: presentes vs ausentes (y % asistencia)."""
    usable = [r for r in rows if _label_asig(r)]
    if len(usable) < 1:
        return []
    usable = usable[:10]
    cats = [_label_asig(r) for r in usable]
    presentes = [int(r.get("presentes") or 0) for r in usable]
    ausentes = [int(r.get("ausentes") or 0) for r in usable]
    pct = []
    for r in usable:
        p = int(r.get("presentes") or 0)
        a = int(r.get("ausentes") or 0)
        t = int(r.get("tarde") or 0)
        e = int(r.get("excusas") or 0)
        total = p + a + t + e
        pct.append(round(100.0 * p / total, 1) if total else 0.0)
    charts: List[Dict[str, Any]] = [
        {
            "type": "bar-grouped",
            "title": "Asistencia: presentes vs ausentes",
            "categories": cats,
            "series": [
                {"name": "Presentes", "data": presentes},
                {"name": "Ausentes", "data": ausentes},
            ],
            "suffix": "",
            "decimals": 0,
        }
    ]
    if any(pct):
        charts.append(
            {
                "type": "bar",
                "title": "% de asistencia (presentes)",
                "categories": cats,
                "series": [{"name": "% asistencia", "data": pct}],
                "yMax": 100,
                "suffix": "%",
                "decimals": 0,
            }
        )
    return charts


def charts_from_attention(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Barras horizontales: estudiantes con promedio bajo."""
    usable = [
        r
        for r in rows
        if r.get("promedio") is not None and (r.get("estudiante") or "").strip()
    ]
    if len(usable) < 1:
        return []
    usable = usable[:12]
    return [
        {
            "type": "bar-horizontal",
            "title": "Estudiantes que necesitan atención",
            "categories": [
                f"{(r.get('estudiante') or '').strip()}"
                f"{' · ' + (r.get('asignatura') or '') if r.get('asignatura') else ''}"
                for r in usable
            ],
            "series": [
                {
                    "name": "Promedio",
                    "data": [float(r["promedio"]) for r in usable],
                }
            ],
            "yMax": 5,
            "suffix": "",
            "decimals": 1,
        }
    ]


def charts_from_lookup(lookup: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not lookup:
        return []
    rows = lookup.get("filas") or []
    tipo = lookup.get("tipo")
    if tipo == "calificaciones":
        return charts_from_grades(rows)
    if tipo == "asistencia":
        return charts_from_attendance(rows)
    return []


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
        return "No tienes asignaturas asignadas en este momento."
    by_group: Dict[str, List[str]] = {}
    for r in rows:
        key = f"{r.get('grado', '')} / {r.get('grupo', '')}".strip(" /")
        by_group.setdefault(key, [])
        asig = r.get("asignatura")
        if asig and asig not in by_group[key]:
            by_group[key].append(asig)
    lines = [
        f"Dictas **{len(rows)}** asignaturas/grupos. Detalle:",
        "",
    ]
    for grupo, asignaturas in by_group.items():
        lines.append(f"**{grupo}**")
        for a in asignaturas:
            lines.append(f"- {a}")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_grades_answer(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "En tus datos no aparecen calificaciones agregadas."
    lines = ["Avance de tus grupos (promedio por asignatura):", ""]
    for r in rows:
        lines.append(
            f"- {_label_asig(r)}: promedio **{r.get('promedio')}** "
            f"({r.get('total_notas')} notas, {r.get('estudiantes_con_nota')} estudiantes)"
        )
    return "\n".join(lines)


def format_rank_answer(
    rows: List[Dict[str, Any]],
    *,
    title: str,
) -> str:
    if not rows:
        return f"No encontré resultados para: {title}."
    lines = [title, ""]
    for i, r in enumerate(rows, 1):
        extra = ""
        if r.get("asignatura"):
            extra = f" — {r.get('asignatura')} {r.get('grado') or ''}{r.get('grupo') or ''}"
        notes = f" ({r.get('total_notas')} notas)" if r.get("total_notas") is not None else ""
        lines.append(
            f"{i}. {r.get('estudiante')}{extra} — promedio **{r.get('promedio')}**{notes}"
        )
    return "\n".join(lines)


def format_best_per_subject_answer(
    rows: List[Dict[str, Any]],
    *,
    worst: bool = False,
) -> str:
    if not rows:
        return "No encontré estudiantes con notas para armar ese listado."
    kind = "Peor" if worst else "Mejor"
    lines = [
        f"{kind} estudiante de cada asignatura/grupo (según promedio):",
        "",
    ]
    for r in rows:
        lines.append(
            f"- **{r.get('asignatura')}** ({r.get('grado')} {r.get('grupo')}): "
            f"{r.get('estudiante')} — promedio **{r.get('promedio')}** "
            f"({r.get('total_notas')} notas)"
        )
    return "\n".join(lines)


def _parse_top_n(message: str, default: int = 5) -> int:
    m = re.search(r"\btop\s*(\d+)\b", message or "", re.I)
    if m:
        return max(1, min(20, int(m.group(1))))
    m = re.search(r"\b(\d+)\s+estudiantes?\b", message or "", re.I)
    if m:
        return max(1, min(20, int(m.group(1))))
    return default


def _guess_subject_from_message(message: str) -> Optional[str]:
    subjects = [
        "Educación Física",
        "Educación Artística",
        "Religión",
        "Ética",
        "Matemáticas",
        "Geometría",
        "Español",
        "Inglés",
        "Ciencias",
        "Sociales",
        "Física",
        "Química",
        "Biología",
    ]
    fmsg = teacher_ctx.fold(message or "")
    # Tolera typos tipo "mattematicas" (letras repetidas)
    fmsg_soft = re.sub(r"(.)\1+", r"\1", fmsg)
    # 1) match completo / casi completo primero
    for s in subjects:
        fs = teacher_ctx.fold(s)
        fs_soft = re.sub(r"(.)\1+", r"\1", fs)
        if fs in fmsg or fs_soft in fmsg_soft:
            return s
    # 2) tokens distintivos (evita que "educación" confunda Física vs Artística)
    token_map = [
        ("fisica", "Educación Física"),
        ("edfisica", "Educación Física"),
        ("artistica", "Educación Artística"),
        ("matematic", "Matemáticas"),
        ("geometr", "Geometría"),
        ("religion", "Religión"),
        ("etica", "Ética"),
        ("ingles", "Inglés"),
        ("espanol", "Español"),
        ("castellano", "Español"),
    ]
    for tok, label in token_map:
        if tok in fmsg_soft:
            return label
    return None


def _guess_grupo_grado(message: str) -> Tuple[Optional[str], Optional[str]]:
    f = teacher_ctx.fold(message or "")
    m = re.search(r"\b(\d{1,2})\s*[°ºo]?\s*([a-d])\b", f, re.I)
    if not m:
        return None, None
    # Patrones flexibles para ILIKE (6°A / 6A / 6 A)
    return m.group(1), f"{m.group(1)}%{m.group(2)}"


def format_attention_answer(rows: List[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None) -> str:
    if not rows:
        return (
            "No encontré estudiantes por debajo del umbral en tus asignaturas "
            "con los filtros indicados."
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
    if meta and meta.get("total_filtrado"):
        lines.append("")
        lines.append(
            f"Mostrando {len(rows)} de {meta.get('total_filtrado')} coincidencias."
        )
    return "\n".join(lines)


def format_pdm_answer(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No tienes planes de mejoramiento (PDM) registrados."
    lines = ["Planes de mejoramiento (PDM):", ""]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"{i}. {r.get('estudiante')} — {r.get('asignatura')} {r.get('grupo')} "
            f"— periodo {r.get('idperiodo')} — {r.get('archivo') or 'sin archivo'}"
        )
    return "\n".join(lines)


def format_student_detail_answer(
    rows: List[Dict[str, Any]],
    meta: Optional[Dict[str, Any]] = None,
    query_name: Optional[str] = None,
    user_message: Optional[str] = None,
    activity_notes: Optional[List[Dict[str, Any]]] = None,
) -> str:
    name = (meta or {}).get("estudiante") or query_name or "el estudiante"
    filtro_asig = None
    if rows and user_message:
        rows, filtro_asig = teacher_ctx.filter_rows_by_asignatura_in_message(
            rows, user_message
        )

    if not rows:
        q = query_name or name
        if filtro_asig:
            return (
                f"No encontré notas de **{q}** en **{filtro_asig}** "
                "entre tus asignaturas registradas."
            )
        return (
            f"No encontré a **{q}** en tus asignaturas con notas registradas. "
            "Prueba con nombre y apellido (ej. Salomé Sofía Daza)."
        )

    # Varias personas distintas con el mismo nombre corto
    by_id: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        by_id.setdefault(r.get("idmatricula") or r.get("estudiante"), []).append(r)
    if len(by_id) > 1:
        lines = [
            f"Encontré **{len(by_id)}** estudiantes que coinciden con "
            f"“{query_name or name}”. ¿A cuál te refieres?",
            "",
        ]
        for i, (_id, group) in enumerate(by_id.items(), 1):
            full = group[0].get("estudiante")
            asigs = ", ".join(
                f"{g.get('asignatura')} ({g.get('promedio')})" for g in group[:4]
            )
            lines.append(f"{i}. **{full}** — {asigs}")
        lines.append("")
        lines.append("Puedes preguntar con el nombre completo.")
        return "\n".join(lines)

    if filtro_asig:
        lines = [
            f"Así va **{name}** en **{filtro_asig}**:",
            "",
        ]
    else:
        lines = [
            f"Así va **{name}** en tus asignaturas:",
            "",
        ]
    for r in rows:
        grupo = f"{r.get('grado') or ''}{r.get('grupo') or ''}".strip()
        lines.append(
            f"- {r.get('asignatura')} {grupo} — promedio **{r.get('promedio')}** "
            f"({r.get('total_notas')} notas)"
        )

    # Detalle de notas por actividad (cuando filtró por asignatura o pidió "notas")
    notes = activity_notes or []
    if notes:
        lines.append("")
        lines.append("Notas por actividad:")
        for n in notes[:40]:
            nota = n.get("nota")
            nota_txt = f"**{nota:.2f}**" if isinstance(nota, (int, float)) else "—"
            act = n.get("actividad") or "Actividad"
            if filtro_asig:
                lines.append(f"- {act}: {nota_txt}")
            else:
                lines.append(
                    f"- {n.get('asignatura')}: {act} — {nota_txt}"
                )
        if len(notes) > 40:
            lines.append(f"_…y {len(notes) - 40} notas más._")

    return "\n".join(lines)


def charts_from_student_detail(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    usable = [
        r for r in rows if r.get("promedio") is not None and r.get("asignatura")
    ]
    if not usable:
        return []
    return [
        {
            "type": "bar",
            "title": f"Promedios de {(usable[0].get('estudiante') or 'estudiante')}",
            "categories": [
                f"{r.get('asignatura')} {r.get('grupo') or ''}".strip()
                for r in usable[:12]
            ],
            "series": [
                {
                    "name": "Promedio",
                    "data": [float(r["promedio"]) for r in usable[:12]],
                }
            ],
            "yMax": 5,
            "suffix": "",
            "decimals": 1,
        }
    ]


def charts_from_activity_notes(
    notes: List[Dict[str, Any]],
    *,
    estudiante: Optional[str] = None,
    asignatura: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Barras con cada nota de actividad (no solo el promedio)."""
    usable = [
        n for n in notes if isinstance(n.get("nota"), (int, float))
    ]
    if not usable:
        return []

    # Contar nombres repetidos para etiquetar (1), (2)…
    name_totals: Dict[str, int] = {}
    for n in usable:
        key = (n.get("actividad") or "Actividad").strip() or "Actividad"
        name_totals[key] = name_totals.get(key, 0) + 1

    name_seen: Dict[str, int] = {}
    categories: List[str] = []
    data: List[float] = []
    for n in usable[:20]:
        raw = (n.get("actividad") or "Actividad").strip() or "Actividad"
        name_seen[raw] = name_seen.get(raw, 0) + 1
        short = raw if len(raw) <= 26 else raw[:23] + "…"
        if name_totals.get(raw, 0) > 1:
            label = f"{short} ({name_seen[raw]})"
        else:
            label = short
        categories.append(label)
        data.append(round(float(n["nota"]), 2))

    who = estudiante or (usable[0].get("estudiante") if usable else "estudiante")
    title = f"Notas de {who}"
    if asignatura:
        title = f"Notas · {asignatura}"

    return [
        {
            "type": "bar",
            "title": title,
            "categories": categories,
            "series": [{"name": "Nota", "data": data}],
            "yMax": 5,
            "suffix": "",
            "decimals": 1,
        }
    ]


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
            threshold = args.get("threshold")
            if threshold is None:
                threshold = teacher_ctx.extract_attention_threshold(
                    user_message or "", default=3.5
                )
            else:
                try:
                    threshold = float(threshold)
                except (TypeError, ValueError):
                    threshold = teacher_ctx.extract_attention_threshold(
                        user_message or "", default=3.5
                    )
            payload = await teacher_ctx.get_students_needing_attention(
                iddocente,
                db_url,
                threshold=float(threshold),
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

        if function_name == "get_student_detail":
            nombre = args.get("nombre") or consulta or ""
            payload = await teacher_ctx.get_student_detail_for_teacher(
                iddocente, db_url, str(nombre)
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

        if function_name == "describe_teacher_schema":
            return teacher_sql.describe_teacher_schema()

        if function_name == "list_periods":
            return await teacher_sql.list_periods(db_url, iddocente)

        if function_name == "run_teacher_sql":
            sql = args.get("sql") or ""
            extra_binds: Dict[str, Any] = {}
            if args.get("idperiodo") is not None:
                try:
                    extra_binds["idperiodo"] = int(args["idperiodo"])
                except (TypeError, ValueError):
                    return {
                        "function": function_name,
                        "error": "idperiodo debe ser un entero",
                        "result": [],
                        "count": 0,
                    }
            # Extra binds from args except control keys
            for k, v in args.items():
                if k in ("sql", "idperiodo", "consulta", "nombre", "query"):
                    continue
                if isinstance(v, (str, int, float, bool)) or v is None:
                    extra_binds[k] = v
            return await teacher_sql.run_teacher_sql(
                db_url, iddocente, str(sql), extra_binds=extra_binds or None
            )

        return {"error": f"Unknown function: {function_name}"}
    except Exception as e:
        logger.error("Teacher tool error %s: %s", function_name, e)
        return {"error": f"Function execution failed: {str(e)}"}


def _assistant_message_to_dict(assistant_message: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "role": "assistant",
        "content": assistant_message.content or None,
    }
    if assistant_message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in assistant_message.tool_calls
        ]
    return payload


def _tool_row_count(function_result: Dict[str, Any]) -> int:
    if isinstance(function_result.get("count"), int):
        return function_result["count"]
    rows = function_result.get("result")
    if isinstance(rows, list):
        return len(rows)
    return 0


async def _format_structured_tool_answer(
    function_name: str,
    function_result: Dict[str, Any],
    *,
    iddocente: int,
    db_url: str,
    user_message: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    rows = function_result.get("result") or []
    if not isinstance(rows, list):
        rows = []
    meta = function_result.get("meta")
    charts: List[Dict[str, Any]] = []

    if function_name == "get_students_needing_attention":
        return format_attention_answer(rows, meta), charts_from_attention(rows)

    if function_name == "list_pdm":
        return format_pdm_answer(rows), []

    if function_name == "get_student_detail":
        filtered_rows, filtro_asig = teacher_ctx.filter_rows_by_asignatura_in_message(
            rows, user_message
        )
        qname = (
            (meta or {}).get("query_name")
            or (meta or {}).get("estudiante")
            or ""
        )
        activity_notes: List[Dict[str, Any]] = []
        if qname and (
            filtro_asig
            or re.search(
                r"\b(notas?|calificaciones?|detalle)\b",
                user_message or "",
                re.IGNORECASE,
            )
        ):
            activity_notes = await teacher_ctx.get_student_activity_notes_for_teacher(
                iddocente,
                db_url,
                qname,
                asignatura=filtro_asig,
            )
        final_content = format_student_detail_answer(
            rows,
            meta=meta,
            query_name=qname or None,
            user_message=user_message,
            activity_notes=activity_notes,
        )
        if activity_notes:
            charts = charts_from_activity_notes(
                activity_notes,
                estudiante=qname or None,
                asignatura=filtro_asig,
            )
        else:
            charts = charts_from_student_detail(filtered_rows)
        return final_content, charts

    return "", []


def _parse_history_json(raw: Any) -> Optional[List[Dict[str, str]]]:
    if raw is None or raw == "":
        return None
    if isinstance(raw, list):
        return raw  # type: ignore[return-value]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else None
        except json.JSONDecodeError:
            return None
    return None


async def _parse_teacher_request(
    http_request: Request,
) -> Tuple[TeacherChatRequest, Optional[str], Optional[str]]:
    """Parse JSON or multipart teacher chat. Returns (request, doc_text, doc_name)."""
    content_type = (http_request.headers.get("content-type") or "").lower()

    if "multipart/form-data" in content_type:
        form = await http_request.form()
        user_id = str(form.get("user_id") or "").strip()
        message = str(form.get("message") or "").strip()
        db_url = str(form.get("db_url") or "").strip()
        conversation_id = form.get("conversation_id")
        context_raw = form.get("context")
        history_raw = form.get("conversation_history")

        context: Dict[str, Any] = {}
        if isinstance(context_raw, str) and context_raw.strip():
            try:
                parsed_ctx = json.loads(context_raw)
                if isinstance(parsed_ctx, dict):
                    context = parsed_ctx
            except json.JSONDecodeError:
                context = {}

        document_text: Optional[str] = None
        document_name: Optional[str] = None
        upload = form.get("file")
        if upload is not None and hasattr(upload, "read"):
            file_bytes = await upload.read()
            filename = getattr(upload, "filename", None) or "documento"
            try:
                document_text, document_name = await extract_text_from_upload(
                    file_bytes, filename
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            except Exception as e:
                logger.error("Mao document extract error: %s", e, exc_info=True)
                raise HTTPException(
                    status_code=400,
                    detail="No pude leer el documento adjunto. Prueba PDF, DOCX, TXT o PPTX.",
                ) from e

        if not user_id or not message or not db_url:
            raise HTTPException(
                status_code=400,
                detail="user_id, message y db_url son requeridos",
            )

        req = TeacherChatRequest(
            user_id=user_id,
            message=message,
            db_url=db_url,
            conversation_id=str(conversation_id) if conversation_id else None,
            context=context,
            conversation_history=_parse_history_json(history_raw),
        )
        return req, document_text, document_name

    try:
        body = await http_request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="JSON inválido") from e
    try:
        req = TeacherChatRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Payload inválido: {e}") from e
    return req, None, None


init_teacher_db()


@router.post("/", response_model=TeacherChatResponse)
async def chat_with_mao(http_request: Request):
    """Chat de Mao para docentes. JSON o multipart con archivo opcional."""
    try:
        request, document_text, document_name = await _parse_teacher_request(
            http_request
        )
        iddocente = int(request.user_id)
        MAX_MESSAGES = 5
        conversation_id = request.conversation_id or f"teacher:{request.db_url}"

        logger.info(
            "Mao chat (hybrid+loop) iddocente=%s has_doc=%s",
            iddocente,
            bool(document_text),
        )

        teacher_info = await teacher_ctx.get_teacher_info(iddocente, request.db_url)
        if not teacher_info:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontró docente activo con id={iddocente}",
            )

        snapshot = await teacher_ctx.get_academic_snapshot(iddocente, request.db_url)
        snapshot_json = json.dumps(snapshot, ensure_ascii=False, default=str)
        logger.info(
            "Mao snapshot iddocente=%s counts=%s",
            iddocente,
            snapshot.get("conteos"),
        )

        lookup = teacher_ctx.lookup_snapshot(snapshot, request.message)
        lookup_json = (
            json.dumps(lookup, ensure_ascii=False, default=str) if lookup else None
        )
        if lookup:
            logger.info(
                "Mao lookup tipo=%s filas=%s",
                lookup.get("tipo"),
                len(lookup.get("filas") or []),
            )

        history = load_history(iddocente, conversation_id, limit=MAX_MESSAGES)
        if not history and request.conversation_history:
            cleaned: List[Dict[str, str]] = []
            for m in request.conversation_history[-MAX_MESSAGES:]:
                role = (m or {}).get("role")
                content = (m or {}).get("content")
                if role in ("user", "assistant") and content:
                    cleaned.append({"role": role, "content": str(content)})
            if (
                cleaned
                and cleaned[-1]["role"] == "user"
                and cleaned[-1]["content"].strip() == request.message.strip()
            ):
                cleaned = cleaned[:-1]
            history = cleaned[-MAX_MESSAGES:]
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

        # Con documento adjunto: priorizar material original (saltar shortcuts de BD)
        skip_db_shortcuts = bool(document_text)

        # Sin adjunto: no inventar talleres — pedir el documento original
        if (
            not document_text
            and teacher_ctx.is_material_support_request(request.message)
        ):
            ask = teacher_ctx.ASK_ORIGINAL_MATERIAL_MSG
            save_message(iddocente, conversation_id, "user", request.message)
            save_message(iddocente, conversation_id, "assistant", ask)
            trim_history(iddocente, conversation_id, max_size=MAX_MESSAGES)
            return TeacherChatResponse(
                response=ask,
                user_id=request.user_id,
                conversation_id=conversation_id,
                function_calls_made=[],
                snapshot_counts=snapshot.get("conteos"),
                charts=None,
                timestamp=datetime.now().isoformat(),
                confidence=0.99,
            )

        if not skip_db_shortcuts and lookup and lookup.get("tipo") == "asignaturas":
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
                charts=[],
                timestamp=datetime.now().isoformat(),
                confidence=0.99,
            )

        # Avance/resumen de grupos: responder con promedios del snapshot (no solo lista)
        if (
            not skip_db_shortcuts
            and lookup
            and lookup.get("tipo") == "calificaciones"
            and teacher_ctx.is_progress_summary_query(request.message)
            and not teacher_ctx._parse_consulta(request.message).get("asignatura")
        ):
            grades_rows = lookup.get("filas") or []
            final_content = format_grades_answer(grades_rows)
            charts = charts_from_grades(grades_rows)
            save_message(iddocente, conversation_id, "user", request.message)
            save_message(iddocente, conversation_id, "assistant", final_content)
            trim_history(iddocente, conversation_id, max_size=MAX_MESSAGES)
            return TeacherChatResponse(
                response=final_content,
                user_id=request.user_id,
                conversation_id=conversation_id,
                function_calls_made=[],
                snapshot_counts=snapshot.get("conteos"),
                charts=charts or None,
                timestamp=datetime.now().isoformat(),
                confidence=0.98,
            )

        # Consultas ad-hoc: ranking / periodo / top / promedio global (SQL scoped)
        if not skip_db_shortcuts and teacher_ctx.needs_exploratory_sql(request.message):
            msg = request.message or ""
            # Colapsar typos "segundoo"/"estudiantte" para matching estable
            msg_l = re.sub(r"(.)\1+", r"\1", teacher_ctx.fold(msg))
            exploratory_content: Optional[str] = None
            exploratory_tools: List[str] = []
            exploratory_charts: List[Dict[str, Any]] = []

            if re.search(r"promedio\s+global|global\s+de\s+todas", msg_l):
                g = await teacher_sql.global_average(request.db_url, iddocente)
                exploratory_tools.append("run_teacher_sql")
                rows = g.get("result") or []
                if rows:
                    r0 = rows[0]
                    exploratory_content = (
                        f"Tu promedio global es **{r0.get('promedio_global')}** "
                        f"({r0.get('total_notas')} notas, "
                        f"{r0.get('estudiantes_con_nota')} estudiantes con nota)."
                    )
                else:
                    exploratory_content = (
                        g.get("error")
                        or "No pude calcular el promedio global con tus datos."
                    )

            elif teacher_ctx.is_best_per_subject_query(msg):
                worst = bool(re.search(r"\bpeor|peores\b", msg_l))
                ranked = await teacher_sql.best_student_per_subject_group(
                    request.db_url,
                    iddocente,
                    order="ASC" if worst else "DESC",
                )
                exploratory_tools.append("run_teacher_sql")
                if ranked.get("error"):
                    exploratory_content = (
                        f"No pude calcular el listado: {ranked.get('error')}"
                    )
                else:
                    exploratory_content = format_best_per_subject_answer(
                        ranked.get("result") or [],
                        worst=worst,
                    )

            elif re.search(r"\btop\b|mejores\s+estudiantes", msg_l) and not re.search(
                r"periodo|per[ií]odo", msg_l
            ):
                asig = _guess_subject_from_message(msg)
                grado, grupo = _guess_grupo_grado(msg)
                if asig:
                    top = await teacher_sql.top_students_by_subject_group(
                        request.db_url,
                        iddocente,
                        asignatura=asig,
                        grado=grado,
                        grupo=grupo,
                        limit=_parse_top_n(msg, 5),
                    )
                    exploratory_tools.append("run_teacher_sql")
                    label = f"Top estudiantes — {asig}"
                    if grado and grupo:
                        letter = str(grupo).split("%")[-1]
                        label += f" {grado}°{letter.upper()}"
                    elif grado:
                        label += f" {grado}°"
                    exploratory_content = format_rank_answer(
                        top.get("result") or [], title=label
                    )

            elif re.search(
                r"mejor|peor|mejores|peores|ranking", msg_l
            ) and re.search(r"periodo|per[ií]odo|global", msg_l):
                periods_payload = await teacher_sql.list_periods(
                    request.db_url, iddocente
                )
                exploratory_tools.append("list_periods")
                periods = periods_payload.get("result") or []
                pid = teacher_sql.resolve_period_id(periods, msg)
                if pid is None and periods:
                    # default: last period if they said periodo but unresolved
                    pid = teacher_sql.resolve_period_id(periods, "periodo 3") or int(
                        periods[-1].get("idperiodo")
                    )
                if pid is not None:
                    order = (
                        "ASC"
                        if re.search(r"peor|peores", msg_l)
                        else "DESC"
                    )
                    asig = _guess_subject_from_message(msg)
                    singular = bool(
                        re.search(
                            r"\b(el|la)\s+(peor|mejor)\s+estudiant",
                            msg_l,
                        )
                    ) and not re.search(r"\b(peores|mejores)\b", msg_l)
                    lim = 1 if singular else _parse_top_n(msg, 5)
                    ranked = await teacher_sql.rank_students_by_period(
                        request.db_url,
                        iddocente,
                        pid,
                        order=order,
                        limit=lim,
                        asignatura=asig,
                    )
                    exploratory_tools.append("run_teacher_sql")
                    pname = next(
                        (
                            p.get("nombre")
                            for p in periods
                            if int(p.get("idperiodo") or 0) == int(pid)
                        ),
                        f"periodo id={pid}",
                    )
                    kind = "Peor" if order == "ASC" else "Mejor"
                    kind_list = "Peores" if order == "ASC" else "Mejores"
                    scope = f"en **{asig}**" if asig else "en el global de tus cursos"
                    rows = ranked.get("result") or []
                    if singular:
                        top = rows[0] if rows else None
                        if top:
                            exploratory_content = (
                                f"El {kind.lower()} estudiante {scope} en "
                                f"**{pname}** fue **{top.get('estudiante')}**, "
                                f"con promedio **{top.get('promedio')}** "
                                f"({top.get('total_notas')} notas)."
                            )
                        else:
                            exploratory_content = (
                                f"No encontré notas {scope} en {pname}."
                            )
                    else:
                        title = f"{kind_list} estudiantes — {pname}"
                        if asig:
                            title += f" · {asig}"
                        else:
                            title += " (global de tus cursos)"
                        exploratory_content = format_rank_answer(
                            rows, title=title
                        )
                else:
                    exploratory_content = (
                        "No pude identificar el periodo en tus actividades. "
                        "Periodos disponibles: "
                        + ", ".join(
                            str(p.get("nombre")) for p in periods[:10]
                        )
                    )

            if exploratory_content:
                save_message(iddocente, conversation_id, "user", request.message)
                save_message(
                    iddocente, conversation_id, "assistant", exploratory_content
                )
                trim_history(iddocente, conversation_id, max_size=MAX_MESSAGES)
                return TeacherChatResponse(
                    response=exploratory_content,
                    user_id=request.user_id,
                    conversation_id=conversation_id,
                    function_calls_made=exploratory_tools,
                    snapshot_counts=snapshot.get("conteos"),
                    charts=exploratory_charts or None,
                    timestamp=datetime.now().isoformat(),
                    confidence=0.97,
                )

        student_query = None
        if not skip_db_shortcuts:
            student_query = teacher_ctx.extract_student_name(request.message)
            if not student_query and teacher_ctx.looks_like_student_followup(
                request.message
            ):
                student_query = teacher_ctx.resolve_student_name_from_history(history)

        if student_query:
            # Mejor/peor nota individual (opcionalmente en una materia)
            if teacher_ctx.is_best_grade_query(request.message):
                worst = bool(
                    re.search(
                        r"\b(peor\s+nota|nota\s+m[aá]s\s+baja|m[ií]nima\s+nota|nota\s+m[ií]nima)\b",
                        request.message,
                        re.I,
                    )
                )
                asig = _guess_subject_from_message(request.message)
                best = await teacher_sql.student_best_grade(
                    request.db_url,
                    iddocente,
                    nombre=student_query,
                    asignatura=asig,
                    worst=worst,
                )
                brows = best.get("result") or []
                if brows:
                    top = brows[0]
                    who = top.get("estudiante") or student_query
                    kind = "Peor" if worst else "Mejor"
                    where = (
                        f" en **{top.get('asignatura')}**"
                        if top.get("asignatura")
                        else ""
                    )
                    periodo = (
                        f" ({top.get('periodo')})" if top.get("periodo") else ""
                    )
                    final_content = (
                        f"{kind} nota de **{who}**{where}: "
                        f"**{top.get('nota')}** en «{top.get('actividad')}»"
                        f"{periodo}."
                    )
                else:
                    mat = f" en {asig}" if asig else ""
                    final_content = (
                        f"No encontré notas de **{student_query}**{mat} "
                        f"en tus asignaturas."
                    )
                save_message(iddocente, conversation_id, "user", request.message)
                save_message(
                    iddocente, conversation_id, "assistant", final_content
                )
                trim_history(iddocente, conversation_id, max_size=MAX_MESSAGES)
                return TeacherChatResponse(
                    response=final_content,
                    user_id=request.user_id,
                    conversation_id=conversation_id,
                    function_calls_made=["run_teacher_sql"],
                    snapshot_counts=snapshot.get("conteos"),
                    charts=None,
                    timestamp=datetime.now().isoformat(),
                    confidence=0.97 if brows else 0.9,
                )

            # Estudiante + periodo → promedios solo de ese periodo
            if teacher_ctx.message_mentions_period(request.message):
                periods_payload = await teacher_sql.list_periods(
                    request.db_url, iddocente
                )
                periods = periods_payload.get("result") or []
                pid = teacher_sql.resolve_period_id(periods, request.message)
                if pid is not None:
                    by_period = await teacher_sql.student_grades_by_period(
                        request.db_url,
                        iddocente,
                        nombre=student_query,
                        idperiodo=int(pid),
                    )
                    prows = by_period.get("result") or []
                    pname = next(
                        (
                            p.get("nombre")
                            for p in periods
                            if int(p.get("idperiodo") or 0) == int(pid)
                        ),
                        f"periodo {pid}",
                    )
                    if prows:
                        who = prows[0].get("estudiante") or student_query
                        lines = [
                            f"Así le fue a **{who}** en **{pname}**:",
                            "",
                        ]
                        for r in prows:
                            lines.append(
                                f"- {r.get('asignatura')} "
                                f"{r.get('grado') or ''}{r.get('grupo') or ''} "
                                f"— promedio **{r.get('promedio')}** "
                                f"({r.get('total_notas')} notas)"
                            )
                        final_content = "\n".join(lines)
                    else:
                        final_content = (
                            f"No encontré notas de **{student_query}** en {pname} "
                            f"dentro de tus asignaturas."
                        )
                    charts = charts_from_student_detail(
                        [
                            {
                                "asignatura": r.get("asignatura"),
                                "grado": r.get("grado"),
                                "grupo": r.get("grupo"),
                                "promedio": r.get("promedio"),
                            }
                            for r in prows
                        ]
                    )
                    save_message(iddocente, conversation_id, "user", request.message)
                    save_message(
                        iddocente, conversation_id, "assistant", final_content
                    )
                    trim_history(iddocente, conversation_id, max_size=MAX_MESSAGES)
                    return TeacherChatResponse(
                        response=final_content,
                        user_id=request.user_id,
                        conversation_id=conversation_id,
                        function_calls_made=["list_periods", "run_teacher_sql"],
                        snapshot_counts=snapshot.get("conteos"),
                        charts=charts or None,
                        timestamp=datetime.now().isoformat(),
                        confidence=0.97 if prows else 0.9,
                    )

            detail = await teacher_ctx.get_student_detail_for_teacher(
                iddocente, request.db_url, student_query
            )
            rows = detail.get("result") or []
            meta = detail.get("meta") or {}
            filtered_rows, filtro_asig = teacher_ctx.filter_rows_by_asignatura_in_message(
                rows, request.message
            )
            wants_notes = bool(
                filtro_asig
                or re.search(
                    r"\b(notas?|calificaciones?|detalle)\b",
                    request.message or "",
                    re.IGNORECASE,
                )
            )
            activity_notes: List[Dict[str, Any]] = []
            if wants_notes and rows:
                activity_notes = await teacher_ctx.get_student_activity_notes_for_teacher(
                    iddocente,
                    request.db_url,
                    student_query,
                    asignatura=filtro_asig,
                )
            final_content = format_student_detail_answer(
                rows,
                meta=meta,
                query_name=student_query,
                user_message=request.message,
                activity_notes=activity_notes,
            )
            if activity_notes:
                charts = charts_from_activity_notes(
                    activity_notes,
                    estudiante=(meta or {}).get("estudiante") or student_query,
                    asignatura=filtro_asig,
                )
            else:
                charts = charts_from_student_detail(filtered_rows)
            save_message(iddocente, conversation_id, "user", request.message)
            save_message(iddocente, conversation_id, "assistant", final_content)
            trim_history(iddocente, conversation_id, max_size=MAX_MESSAGES)
            return TeacherChatResponse(
                response=final_content,
                user_id=request.user_id,
                conversation_id=conversation_id,
                function_calls_made=["get_student_detail"],
                snapshot_counts=snapshot.get("conteos"),
                charts=charts or None,
                timestamp=datetime.now().isoformat(),
                confidence=0.98 if rows else 0.9,
            )

        if document_text:
            doc_label = document_name or "documento"
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "MATERIAL_ORIGINAL_ADJUNTADO (fuente prioritaria para material de apoyo).\n"
                        f"Archivo: {doc_label}\n"
                        "Usa este texto para adaptar talleres/ejercicios. "
                        "No digas que falta el material original.\n"
                        f"---\n{document_text}\n---"
                    ),
                }
            )

        if lookup_json and not skip_db_shortcuts:
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
        messages.append({"role": "user", "content": request.message})

        openai_tools = [
            {"type": "function", "function": func}
            for func in TEACHER_FUNCTION_DEFINITIONS
        ]
        function_calls_made: List[str] = []
        final_content = ""
        charts: List[Dict[str, Any]] = []
        last_structured: Optional[Tuple[str, Dict[str, Any]]] = None
        force_sql = teacher_ctx.needs_exploratory_sql(request.message)

        if force_sql:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "CONSULTA_SQL_OBLIGATORIA: usa list_periods y/o run_teacher_sql "
                        "antes de responder. Prohibido decir que no hay datos sin ejecutar tools."
                    ),
                }
            )

        material_with_doc = bool(document_text) and teacher_ctx.is_material_support_request(
            request.message
        )
        # Con material original, permitir respuesta más larga (irá a PDF)
        token_budget = 1800 if material_with_doc else (1200 if force_sql else 900)

        for round_idx in range(MAX_TOOL_ROUNDS):
            choice: Any = "auto"
            if force_sql and round_idx == 0 and not function_calls_made:
                choice = "required"
            # Con documento de apoyo no hace falta tools de BD
            tools_arg = None if material_with_doc else openai_tools
            tool_choice_arg = None if material_with_doc else choice
            create_kwargs: Dict[str, Any] = {
                "model": "gpt-4o-mini",
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": token_budget if round_idx == 0 else max(token_budget, 1200),
            }
            if tools_arg is not None:
                create_kwargs["tools"] = tools_arg
                create_kwargs["tool_choice"] = tool_choice_arg
            response = client.chat.completions.create(**create_kwargs)
            assistant_message = response.choices[0].message

            if not assistant_message.tool_calls:
                final_content = assistant_message.content or ""
                break

            messages.append(_assistant_message_to_dict(assistant_message))

            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = tool_call.function.arguments or "{}"
                function_result = await execute_teacher_tool(
                    function_name,
                    function_args,
                    request.db_url,
                    iddocente,
                    user_message=request.message,
                )
                function_calls_made.append(function_name)
                logger.info(
                    "Mao tool round=%s name=%s count=%s",
                    round_idx + 1,
                    function_name,
                    _tool_row_count(function_result),
                )

                if (
                    function_name in STRUCTURED_TOOLS
                    and _tool_row_count(function_result) > 0
                    and "error" not in function_result
                ):
                    last_structured = (function_name, function_result)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(
                            function_result, ensure_ascii=False, default=str
                        ),
                    }
                )
        else:
            # Agotó rondas aún pidiendo tools: una pasada final sin tools
            follow = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2,
                max_tokens=1200,
            )
            final_content = follow.choices[0].message.content or ""

        # Cifras: solo formatear si la última tool del turno fue estructurada con filas
        if (
            last_structured
            and function_calls_made
            and function_calls_made[-1] == last_structured[0]
        ):
            structured_name, structured_result = last_structured
            formatted, structured_charts = await _format_structured_tool_answer(
                structured_name,
                structured_result,
                iddocente=iddocente,
                db_url=request.db_url,
                user_message=request.message,
            )
            if formatted:
                final_content = formatted
                charts = structured_charts

        if not final_content:
            final_content = (
                "No pude completar la respuesta. ¿Puedes reformular la pregunta?"
            )

        if not charts and not skip_db_shortcuts:
            charts = charts_from_lookup(lookup)
            if not charts:
                msg_l = (request.message or "").lower()
                if any(
                    k in msg_l
                    for k in (
                        "resumen",
                        "avance",
                        "cómo van",
                        "como van",
                        "mis notas",
                        "mis grupos",
                    )
                ):
                    grades = snapshot.get("calificaciones") or []
                    charts = charts_from_grades(grades)
                elif any(k in msg_l for k in ("asistencia", "faltas", "ausenc")):
                    att = snapshot.get("asistencia") or []
                    charts = charts_from_attendance(att)

        attachment: Optional[Dict[str, Any]] = None
        if material_with_doc and final_content:
            src = document_name or "documento"
            attachment = build_support_material_attachment(
                final_content,
                title="Material de apoyo",
                source_name=src,
            )
            if attachment:
                final_content = (
                    f"{final_content.rstrip()}\n\n"
                    f"📄 Te dejé el material en PDF: **{attachment['filename']}** "
                    f"(botón de descarga abajo)."
                )

        save_message(iddocente, conversation_id, "user", request.message)
        save_message(iddocente, conversation_id, "assistant", final_content)
        trim_history(iddocente, conversation_id, max_size=MAX_MESSAGES)

        return TeacherChatResponse(
            response=final_content,
            user_id=request.user_id,
            conversation_id=conversation_id,
            function_calls_made=function_calls_made,
            snapshot_counts=snapshot.get("conteos"),
            charts=charts or None,
            attachment=attachment,
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
        "mode": "snapshot_injected + multi_round_tools + optional_document",
        "max_tool_rounds": MAX_TOOL_ROUNDS,
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
