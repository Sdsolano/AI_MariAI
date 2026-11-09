"""
Teacher context helpers for Mao (docente agent).

Design: SQL fetches the teacher's full dataset (scoped by iddocente only).
Filters are resolved in Python (accents, 5°A as grupo, fuzzy asignatura)
so the LLM cannot invent broken ILIKE params that return false empties.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from app.db.connection import db_manager

logger = logging.getLogger(__name__)


def fold(s: Optional[str]) -> str:
    """Lowercase + strip accents for comparisons."""
    if not s:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _parse_consulta(
    consulta: Optional[str] = None,
    grado: Optional[str] = None,
    grupo: Optional[str] = None,
    asignatura: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Resolve natural-language / LLM filter args into grado, grupo, asignatura.
    Examples:
      "Matematicas en 5°A" -> asignatura=matematicas, grupo=5a, grado=5
      grado="5°A" (misplaced by LLM) -> grupo=5a
    """
    bits = [consulta, grado, grupo, asignatura]
    raw = " ".join(b for b in bits if b).strip()
    f = fold(raw)

    out_grado = fold(grado) if grado else None
    out_grupo = fold(grupo) if grupo else None
    out_asig = fold(asignatura) if asignatura else None

    # Grupo patterns: 5°A, 5A, 5ºA, 5 a
    grupo_match = re.search(
        r"\b(\d{1,2})\s*[°ºo]?\s*[-]?\s*([a-d])\b", f, re.IGNORECASE
    )
    if grupo_match:
        out_grupo = f"{grupo_match.group(1)}{grupo_match.group(2).lower()}"
        out_grado = grupo_match.group(1)

    # Grado alone: 5°, grado 5
    if not out_grado:
        grado_match = re.search(r"\b(?:grado\s*)?(\d{1,2})\s*[°ºo]?\b", f)
        if grado_match:
            out_grado = grado_match.group(1)

    # If LLM put "5°A" into grado field
    if out_grado and re.fullmatch(r"\d{1,2}\s*[a-d]", out_grado.replace("°", "").replace("º", "")):
        cleaned = re.sub(r"[^\da-d]", "", out_grado)
        m = re.match(r"(\d{1,2})([a-d])", cleaned)
        if m:
            out_grupo = f"{m.group(1)}{m.group(2)}"
            out_grado = m.group(1)

    # Asignatura: strip grado/grupo tokens from consulta and use remainder
    if consulta and not out_asig:
        asig_text = fold(consulta)
        asig_text = re.sub(
            r"\b(\d{1,2})\s*[°ºo]?\s*[-]?\s*[a-d]\b", " ", asig_text
        )
        asig_text = re.sub(
            r"\b(grado|grupo|en|de|del|la|el|las|los|un|una|me|mi|mis|"
            r"dame|dime|quiero|necesito|cual|cuales|como|van|voy|que|quien|quienes|"
            r"estudiantes?|alumnos?|necesitan|atencion|riesgo|bajo|bajos|"
            r"promedio|notas?|calificaciones?|asistencia|resumen|breve|pdm|planes?|"
            r"asignaturas?|materias?|dicto|dicta|son|lista|listar|mostrar)\b",
            " ",
            asig_text,
        )
        asig_text = re.sub(r"\s+", " ", asig_text).strip()
        if asig_text:
            out_asig = asig_text

    # Misplaced: grado looks like a subject name
    if out_grado and not out_grado.isdigit() and re.search(r"[a-z]{3,}", out_grado):
        if not out_asig:
            out_asig = out_grado
        out_grado = None

    return {"grado": out_grado, "grupo": out_grupo, "asignatura": out_asig}


def _row_matches(row: Dict[str, Any], filters: Dict[str, Optional[str]]) -> bool:
    g = fold(row.get("grado"))
    gg = fold(row.get("grupo"))
    a = fold(row.get("asignatura"))

    # Normalize grupo "5°A" / "5A" / "5ºA" -> digits+letter for compare
    def grupo_key(x: str) -> str:
        m = re.search(r"(\d{1,2})\s*[°ºo]?\s*([a-d])?", x)
        if not m:
            return x
        return f"{m.group(1)}{(m.group(2) or '').lower()}"

    if filters.get("grupo"):
        if grupo_key(filters["grupo"]) not in grupo_key(gg) and grupo_key(gg) not in grupo_key(
            filters["grupo"]
        ):
            # also allow substring
            if fold(filters["grupo"]) not in gg and gg not in fold(filters["grupo"]):
                return False

    if filters.get("grado"):
        fg = filters["grado"]
        # grado "5" matches "5°"
        if fg not in g and g not in fg:
            # if grupo already matched tightly, grado is redundant
            if not filters.get("grupo"):
                return False

    if filters.get("asignatura"):
        fa = filters["asignatura"]
        # token containment either way (matematicas <-> matematicas / matematic)
        if fa not in a and a not in fa:
            # try all tokens
            tokens = [t for t in fa.split() if len(t) > 2]
            if tokens and not all(t in a for t in tokens):
                return False

    return True


def _apply_filters(
    rows: List[Dict[str, Any]],
    *,
    consulta: Optional[str] = None,
    grado: Optional[str] = None,
    grupo: Optional[str] = None,
    asignatura: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    filters = _parse_consulta(consulta, grado, grupo, asignatura)
    has_filter = any(filters.values())
    if not has_filter:
        return rows, {"filtros_aplicados": {}, "total_sin_filtro": len(rows)}

    matched = [r for r in rows if _row_matches(r, filters)]
    meta = {
        "filtros_aplicados": filters,
        "total_sin_filtro": len(rows),
        "total_filtrado": len(matched),
    }
    # Never pretend empty if the teacher has data: return matches, or flag
    if not matched and rows:
        meta["aviso"] = (
            "Ninguna fila coincidió con el filtro; se anexan todas las filas del docente "
            "para que no se invente un 'sin datos'."
        )
        return rows, meta
    return matched, meta


_SUBJECT_LIST_RE = re.compile(
    r"\b(asignaturas?|materias?|que\s+dicto|mis\s+grupos|carga\s+academica)\b",
    re.IGNORECASE,
)
_ATTENDANCE_RE = re.compile(r"\b(asistencia|faltas?|ausenc)", re.IGNORECASE)
_GRADES_RE = re.compile(
    r"\b(promedio|nota|notas|calificacion)",
    re.IGNORECASE,
)


def is_subject_list_query(consulta: str) -> bool:
    f = fold(consulta)
    if not f:
        return False
    # "promedio de Matematicas" is NOT a subject-list question
    if _GRADES_RE.search(f) or _ATTENDANCE_RE.search(f):
        return False
    return bool(_SUBJECT_LIST_RE.search(consulta or ""))


def lookup_snapshot(snapshot: Dict[str, Any], consulta: str) -> Optional[Dict[str, Any]]:
    """
    Deterministic resolve against snapshot rows (same filter rules as tools).
    Returns a compact match payload the LLM must use, or None.
    """
    subjects = snapshot.get("asignaturas") or []
    grades = snapshot.get("calificaciones") or []
    attendance = snapshot.get("asistencia") or []

    # Explicit: "qué asignaturas tengo / resumen de mis asignaturas"
    if is_subject_list_query(consulta):
        return {
            "tipo": "asignaturas",
            "filtros": {},
            "filas": subjects,
            "origen": "acad_gradosasignaturas",
        }

    filters = _parse_consulta(consulta)
    if not any(filters.values()):
        return None

    g_match, g_meta = _apply_filters(grades, consulta=consulta)
    a_match, a_meta = _apply_filters(attendance, consulta=consulta)

    if _ATTENDANCE_RE.search(consulta or ""):
        if a_match and a_meta.get("total_filtrado", len(a_match)) > 0 and not a_meta.get("aviso"):
            return {
                "tipo": "asistencia",
                "filtros": filters,
                "filas": a_match,
                "origen": "student_attendance",
            }

    if g_match and g_meta.get("total_filtrado", len(g_match)) > 0 and not g_meta.get("aviso"):
        return {
            "tipo": "calificaciones",
            "filtros": filters,
            "filas": g_match,
            "origen": "acad_actividades_notas",
        }
    if a_match and a_meta.get("total_filtrado", len(a_match)) > 0 and not a_meta.get("aviso"):
        return {
            "tipo": "asistencia",
            "filtros": filters,
            "filas": a_match,
            "origen": "student_attendance",
        }

    if filters.get("grupo") or filters.get("asignatura"):
        return {
            "tipo": "sin_coincidencia",
            "filtros": filters,
            "filas": [],
            "origen": None,
        }
    return None


async def get_academic_snapshot(iddocente: int, db_url: str) -> Dict[str, Any]:
    """
    Compact academic package for the LLM: subjects + grade aggregates + attendance.
    No LLM-driven SQL filters — the model reads this JSON and answers.
    """
    subjects = await list_teacher_subjects(iddocente, db_url)
    grades_payload = await get_grades_summary(iddocente, db_url)
    attendance_payload = await get_attendance_summary(iddocente, db_url)
    grades = grades_payload.get("result") or []
    attendance = attendance_payload.get("result") or []
    return {
        "iddocente": iddocente,
        "asignaturas": [
            {
                "grado": s["grado"],
                "grupo": s["grupo"],
                "asignatura": s["asignatura"],
            }
            for s in subjects
        ],
        "calificaciones": [
            {
                "grado": g["grado"],
                "grupo": g["grupo"],
                "asignatura": g["asignatura"],
                "promedio": g["promedio"],
                "total_notas": g["total_notas"],
                "estudiantes_con_nota": g["estudiantes_con_nota"],
            }
            for g in grades
        ],
        "asistencia": [
            {
                "grado": a["grado"],
                "grupo": a["grupo"],
                "asignatura": a["asignatura"],
                "registros": a["registros"],
                "presentes": a["presentes"],
                "ausentes": a["ausentes"],
                "tarde": a["tarde"],
                "excusas": a["excusas"],
            }
            for a in attendance
        ],
        "origenes": {
            "asignaturas": "acad_gradosasignaturas",
            "calificaciones": "acad_actividades_notas",
            "asistencia": "student_attendance",
        },
        "conteos": {
            "asignaturas": len(subjects),
            "filas_calificaciones": len(grades),
            "filas_asistencia": len(attendance),
        },
    }


async def get_teacher_info(iddocente: int, db_url: str) -> Optional[Dict[str, Any]]:
    try:
        with db_manager.get_session_for_url(db_url) as session:
            row = session.execute(
                text(
                    """
                    SELECT
                        d.id AS iddocente,
                        d.primer_nombre,
                        d.segundo_nombre,
                        d.primer_apellido,
                        d.segundo_apellido,
                        d.correo,
                        d.idusuario
                    FROM doce_docentes d
                    WHERE d.id = :iddocente
                      AND COALESCE(d.estado, 'on') = 'on'
                    LIMIT 1
                    """
                ),
                {"iddocente": iddocente},
            ).fetchone()
            if not row:
                return None
            nombre = " ".join(
                p
                for p in [
                    row.primer_nombre,
                    row.segundo_nombre,
                    row.primer_apellido,
                    row.segundo_apellido,
                ]
                if p
            ).strip()
            return {
                "iddocente": row.iddocente,
                "nombre_completo": nombre,
                "correo": row.correo,
                "idusuario": row.idusuario,
            }
    except Exception as e:
        logger.error("get_teacher_info failed: %s", e)
        return None


async def list_teacher_subjects(iddocente: int, db_url: str) -> List[Dict[str, Any]]:
    try:
        with db_manager.get_session_for_url(db_url) as session:
            rows = session.execute(
                text(
                    """
                    SELECT
                        ga.id AS id_grado_asignatura,
                        ga.idasignatura,
                        a.nombre AS asignatura,
                        ga.idgrados_grupos,
                        g.nombre AS grado,
                        gg.nombre AS grupo
                    FROM acad_gradosasignaturas ga
                    JOIN acad_asignaturas a ON a.id = ga.idasignatura
                    JOIN acad_gradosgrupos gg ON gg.id = ga.idgrados_grupos
                    JOIN acad_grados g ON g.id = gg.idgrado
                    WHERE ga.iddocente = :iddocente
                      AND COALESCE(ga.estado, 'on') = 'on'
                    ORDER BY g.nombre, gg.nombre, a.nombre
                    """
                ),
                {"iddocente": iddocente},
            ).fetchall()
            return [
                {
                    "id_grado_asignatura": r.id_grado_asignatura,
                    "idasignatura": r.idasignatura,
                    "asignatura": r.asignatura,
                    "idgrados_grupos": r.idgrados_grupos,
                    "grado": r.grado,
                    "grupo": r.grupo,
                    "origen": "acad_gradosasignaturas",
                }
                for r in rows
            ]
    except Exception as e:
        logger.error("list_teacher_subjects failed: %s", e)
        return []


async def _fetch_all_grades(iddocente: int, db_url: str) -> List[Dict[str, Any]]:
    with db_manager.get_session_for_url(db_url) as session:
        rows = session.execute(
            text(
                """
                SELECT
                    g.nombre AS grado,
                    gg.nombre AS grupo,
                    a.nombre AS asignatura,
                    ROUND(AVG(an.nota)::numeric, 2) AS promedio,
                    COUNT(an.id) AS total_notas,
                    COUNT(DISTINCT an.idmatricula) AS estudiantes_con_nota
                FROM acad_gradosasignaturas ga
                JOIN acad_asignaturas a ON a.id = ga.idasignatura
                JOIN acad_gradosgrupos gg ON gg.id = ga.idgrados_grupos
                JOIN acad_grados g ON g.id = gg.idgrado
                JOIN acad_actividades act ON act.idgrados_asignatura = ga.id
                JOIN acad_actividades_notas an ON an.idactividad = act.id
                WHERE ga.iddocente = :iddocente
                  AND COALESCE(ga.estado, 'on') = 'on'
                GROUP BY g.nombre, gg.nombre, a.nombre
                ORDER BY g.nombre, gg.nombre, a.nombre
                """
            ),
            {"iddocente": iddocente},
        ).fetchall()
        return [
            {
                "grado": r.grado,
                "grupo": r.grupo,
                "asignatura": r.asignatura,
                "promedio": float(r.promedio) if r.promedio is not None else None,
                "total_notas": r.total_notas,
                "estudiantes_con_nota": r.estudiantes_con_nota,
                "origen": "acad_actividades_notas",
            }
            for r in rows
        ]


async def get_grades_summary(
    iddocente: int,
    db_url: str,
    consulta: Optional[str] = None,
    grado: Optional[str] = None,
    grupo: Optional[str] = None,
    asignatura: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Always loads ALL grade aggregates for the teacher from Postgres,
    then filters in Python. Returns {result, meta}.
    """
    try:
        all_rows = await _fetch_all_grades(iddocente, db_url)
        matched, meta = _apply_filters(
            all_rows,
            consulta=consulta,
            grado=grado,
            grupo=grupo,
            asignatura=asignatura,
        )
        return {"result": matched, "meta": meta}
    except Exception as e:
        logger.error("get_grades_summary failed: %s", e)
        return {"result": [], "meta": {"error": str(e)}}


async def _fetch_all_attendance(iddocente: int, db_url: str) -> List[Dict[str, Any]]:
    with db_manager.get_session_for_url(db_url) as session:
        rows = session.execute(
            text(
                """
                SELECT
                    g.nombre AS grado,
                    gg.nombre AS grupo,
                    a.nombre AS asignatura,
                    COUNT(*) AS registros,
                    SUM(CASE WHEN sa.attended THEN 1 ELSE 0 END) AS presentes,
                    SUM(CASE WHEN sa.not_attended THEN 1 ELSE 0 END) AS ausentes,
                    SUM(CASE WHEN sa.late THEN 1 ELSE 0 END) AS tarde,
                    SUM(CASE WHEN sa.excuse THEN 1 ELSE 0 END) AS excusas
                FROM student_attendance sa
                JOIN acad_gradosasignaturas ga
                  ON ga.idgrados_grupos = sa.course_id
                 AND ga.idasignatura = sa.subject_id
                JOIN acad_asignaturas a ON a.id = ga.idasignatura
                JOIN acad_gradosgrupos gg ON gg.id = ga.idgrados_grupos
                JOIN acad_grados g ON g.id = gg.idgrado
                WHERE ga.iddocente = :iddocente
                  AND COALESCE(ga.estado, 'on') = 'on'
                GROUP BY g.nombre, gg.nombre, a.nombre
                ORDER BY registros DESC
                """
            ),
            {"iddocente": iddocente},
        ).fetchall()
        return [
            {
                "grado": r.grado,
                "grupo": r.grupo,
                "asignatura": r.asignatura,
                "registros": r.registros,
                "presentes": int(r.presentes or 0),
                "ausentes": int(r.ausentes or 0),
                "tarde": int(r.tarde or 0),
                "excusas": int(r.excusas or 0),
                "origen": "student_attendance",
            }
            for r in rows
        ]


async def get_attendance_summary(
    iddocente: int,
    db_url: str,
    consulta: Optional[str] = None,
    grado: Optional[str] = None,
    grupo: Optional[str] = None,
    asignatura: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        all_rows = await _fetch_all_attendance(iddocente, db_url)
        matched, meta = _apply_filters(
            all_rows,
            consulta=consulta,
            grado=grado,
            grupo=grupo,
            asignatura=asignatura,
        )
        return {"result": matched, "meta": meta}
    except Exception as e:
        logger.error("get_attendance_summary failed: %s", e)
        return {"result": [], "meta": {"error": str(e)}}


async def get_students_needing_attention(
    iddocente: int,
    db_url: str,
    threshold: float = 3.5,
    limit: int = 15,
    consulta: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Low averages by idmatricula. Note: some restored DBs have notas whose
    idmatricula no longer exists in acad_estumatricula; we still return the
    academic signal without requiring the name join.
    """
    try:
        with db_manager.get_session_for_url(db_url) as session:
            # an.idmatricula → acad_matricula_academica.id → idestumatricula → nombre
            rows = session.execute(
                text(
                    """
                    SELECT
                        em.id AS idmatricula,
                        TRIM(
                            COALESCE(e.primer_nombre, '') || ' ' ||
                            COALESCE(e.segundo_nombre, '') || ' ' ||
                            COALESCE(e.primer_apellido, '') || ' ' ||
                            COALESCE(e.segundo_apellido, '')
                        ) AS estudiante,
                        a.nombre AS asignatura,
                        g.nombre AS grado,
                        gg.nombre AS grupo,
                        ROUND(AVG(an.nota)::numeric, 2) AS promedio,
                        COUNT(an.id) AS total_notas
                    FROM acad_gradosasignaturas ga
                    JOIN acad_asignaturas a ON a.id = ga.idasignatura
                    JOIN acad_gradosgrupos gg ON gg.id = ga.idgrados_grupos
                    JOIN acad_grados g ON g.id = gg.idgrado
                    JOIN acad_actividades act ON act.idgrados_asignatura = ga.id
                    JOIN acad_actividades_notas an ON an.idactividad = act.id
                    JOIN acad_matricula_academica ma ON ma.id = an.idmatricula
                    JOIN acad_estumatricula em ON em.id = ma.idestumatricula
                    JOIN estu_estudiantes e ON e.id = em.idestudiante
                    WHERE ga.iddocente = :iddocente
                      AND COALESCE(ga.estado, 'on') = 'on'
                    GROUP BY em.id, e.primer_nombre, e.segundo_nombre,
                             e.primer_apellido, e.segundo_apellido,
                             a.nombre, g.nombre, gg.nombre
                    HAVING AVG(an.nota) < :threshold
                    ORDER BY promedio ASC
                    LIMIT 200
                    """
                ),
                {"iddocente": iddocente, "threshold": threshold},
            ).fetchall()
            all_rows = [
                {
                    "idmatricula": r.idmatricula,
                    "estudiante": " ".join(r.estudiante.split()),
                    "asignatura": r.asignatura,
                    "grado": r.grado,
                    "grupo": r.grupo,
                    "promedio": float(r.promedio) if r.promedio is not None else None,
                    "total_notas": r.total_notas,
                    "umbral": threshold,
                    "origen": "acad_actividades_notas",
                }
                for r in rows
            ]
        matched, meta = _apply_filters(all_rows, consulta=consulta)
        return {"result": matched[:limit], "meta": meta}
    except Exception as e:
        logger.error("get_students_needing_attention failed: %s", e)
        return {"result": [], "meta": {"error": str(e)}}


async def list_pdm_for_teacher(
    iddocente: int,
    db_url: str,
    limit: int = 30,
    consulta: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        with db_manager.get_session_for_url(db_url) as session:
            rows = session.execute(
                text(
                    """
                    SELECT
                        pe.id AS id_pdm,
                        e.primer_nombre || ' ' || e.primer_apellido AS estudiante,
                        a.nombre AS asignatura,
                        g.nombre AS grado,
                        gg.nombre AS grupo,
                        pe.idperiodo,
                        pe.fecha_creacion,
                        pe.file_original_name
                    FROM acad_pdm_estudiantes pe
                    JOIN acad_pdm_ensenanzas_propositos ep ON ep.id = pe.idproposito
                    JOIN acad_pdm_ensenanzas en ON en.id = ep.idensenanza
                    JOIN acad_gradosasignaturas ga ON ga.id = en.idgradoasignatura
                    JOIN acad_asignaturas a ON a.id = ga.idasignatura
                    JOIN acad_gradosgrupos gg ON gg.id = COALESCE(en.idgrados_grupos, ga.idgrados_grupos)
                    JOIN acad_grados g ON g.id = gg.idgrado
                    JOIN acad_estumatricula em ON em.id = pe.idmatricula
                    JOIN estu_estudiantes e ON e.id = em.idestudiante
                    WHERE ga.iddocente = :iddocente
                    ORDER BY pe.fecha_creacion DESC NULLS LAST
                    LIMIT 200
                    """
                ),
                {"iddocente": iddocente},
            ).fetchall()
            all_rows = [
                {
                    "id_pdm": r.id_pdm,
                    "estudiante": r.estudiante,
                    "asignatura": r.asignatura,
                    "grado": r.grado,
                    "grupo": r.grupo,
                    "idperiodo": r.idperiodo,
                    "fecha_creacion": str(r.fecha_creacion) if r.fecha_creacion else None,
                    "archivo": r.file_original_name,
                    "origen": "acad_pdm_estudiantes",
                }
                for r in rows
            ]
        matched, meta = _apply_filters(all_rows, consulta=consulta)
        return {"result": matched[:limit], "meta": meta}
    except Exception as e:
        logger.error("list_pdm_for_teacher failed: %s", e)
        return {"result": [], "meta": {"error": str(e)}}
