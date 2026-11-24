"""
Teacher context helpers for Mao (docente agent).

Design: SQL fetches the teacher's full dataset (scoped by iddocente only).
Filters are resolved in Python (accents, 5°A as grupo, fuzzy asignatura)
so the LLM cannot invent broken ILIKE params that return false empties.
"""
from __future__ import annotations

import difflib
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


def _collapse_repeats(s: str) -> str:
    """mattematicas -> matematicas (corrige letras duplicadas)."""
    return re.sub(r"(.)\1+", r"\1", s or "")


def _similar_token(a: str, b: str, cutoff: float = 0.78) -> bool:
    """Match flexible entre tokens (typos / letras repetidas)."""
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    ca, cb = _collapse_repeats(a), _collapse_repeats(b)
    if ca == cb or ca in cb or cb in ca:
        return True
    if abs(len(ca) - len(cb)) > max(3, len(cb) // 2):
        return False
    return difflib.SequenceMatcher(None, ca, cb).ratio() >= cutoff


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
            r"\b(grado|grupos?|en|de|del|la|el|las|los|un|una|me|mi|mis|"
            r"dame|dime|quiero|necesito|cual|cuales|como|van|voy|que|quien|quienes|"
            r"estudiantes?|alumnos?|necesitan|atencion|riesgo|bajo|bajos|"
            r"promedio|notas?|calificaciones?|asistencia|resumen|avance|breve|pdm|planes?|"
            r"asignaturas?|materias?|dicto|dicta|son|lista|listar|mostrar|aqui|aquí|tienes)\b",
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
    r"\b(asignaturas?|materias?|que\s+dicto|carga\s+academica|"
    r"cu[aá]ntas?\s+asignaturas?|cu[aá]ntos?\s+grupos?)\b",
    re.IGNORECASE,
)
_ATTENDANCE_RE = re.compile(r"\b(asistencia|faltas?|ausenc)", re.IGNORECASE)
_GRADES_RE = re.compile(
    r"\b(promedio|nota|notas|calificacion)",
    re.IGNORECASE,
)
_PROGRESS_RE = re.compile(
    r"\b("
    r"avance"
    r"|resumen"
    r"|c[oó]mo\s+van"
    r"|como\s+van"
    r"|mis\s+grupos"
    r"|mis\s+notas"
    r")\b",
    re.IGNORECASE,
)
_SQL_EXPLORATORY_RE = re.compile(
    r"\b("
    r"mejor\s+estudiante"
    r"|peor\s+estudiante"
    r"|mejores\s+estudiantes"
    r"|peores\s+estudiantes"
    r"|top\s*\d*"
    r"|ranking"
    r"|periodo\s*\d"
    r"|per[ií]odo\s*\d"
    r"|tercer\s+periodo"
    r"|segundo\s+periodo"
    r"|primer\s+periodo"
    r"|promedio\s+global"
    r"|global\s+del"
    r"|en\s+el\s+global"
    r"|cada\s+asignatura"
    r"|por\s+asignatura"
    r"|por\s+cada\s+asignatura"
    r"|mejor\s+de\s+cada"
    r")\b",
    re.IGNORECASE,
)

_BEST_GRADE_RE = re.compile(
    r"\b("
    r"mejor\s+nota"
    r"|nota\s+m[aá]s\s+alta"
    r"|m[aá]xima\s+nota"
    r"|nota\s+m[aá]xima"
    r"|peor\s+nota"
    r"|nota\s+m[aá]s\s+baja"
    r"|m[ií]nima\s+nota"
    r"|nota\s+m[ií]nima"
    r")\b",
    re.IGNORECASE,
)


def is_best_grade_query(consulta: str) -> bool:
    """Mejor/peor nota de un estudiante (opcionalmente en una materia)."""
    return bool(consulta and _BEST_GRADE_RE.search(consulta))


def is_best_per_subject_query(consulta: str) -> bool:
    """Mejor (o peor) estudiante por cada asignatura/grupo del docente."""
    if not consulta:
        return False
    if not re.search(
        r"\b(cada\s+asignatura|por\s+asignatura|por\s+cada\s+asignatura|"
        r"mejor\s+de\s+cada|peor\s+de\s+cada|en\s+cada\s+materia)\b",
        consulta,
        re.IGNORECASE,
    ):
        return False
    return bool(
        re.search(r"\b(mejor|peor|mejores|peores|top)\b", consulta, re.IGNORECASE)
    )

# Palabras típicas de asignatura (no son nombres de persona)
_SUBJECT_TOKEN_STOP = {
    "educacion", "educación", "artistica", "artística", "religion", "religión",
    "etica", "ética", "valores", "matematicas", "matemáticas", "espanol",
    "español", "ingles", "inglés", "ciencias", "sociales", "fisica", "física",
    "quimica", "química", "biologia", "biología", "filosofia", "filosofía",
    "tecnologia", "tecnología", "informatica", "informática", "musica", "música",
    "educacion fisica", "edfisica", "lengua", "castellano", "geografia",
    "geografía", "historia", "convivencia", "recreacion", "recreación",
}


def is_subject_list_query(consulta: str) -> bool:
    f = fold(consulta)
    if not f:
        return False
    # Promedios / asistencia / avance / ranking NO son listado de asignaturas
    if _GRADES_RE.search(f) or _ATTENDANCE_RE.search(f):
        return False
    if _PROGRESS_RE.search(consulta or "") and not re.search(
        r"\b(asignaturas?|materias?|que\s+dicto)\b", consulta or "", re.I
    ):
        return False
    if _SQL_EXPLORATORY_RE.search(consulta or ""):
        return False
    return bool(_SUBJECT_LIST_RE.search(consulta or ""))


def is_progress_summary_query(consulta: str) -> bool:
    """Resumen/avance de grupos → devolver todas las calificaciones del snapshot."""
    if not consulta:
        return False
    if _ATTENDANCE_RE.search(consulta):
        return False
    if _SQL_EXPLORATORY_RE.search(consulta):
        return False
    # "cuántas asignaturas" no es avance
    if re.search(r"\bcu[aá]nt[ao]s?\s+(asignaturas?|materias?|grupos?)\b", consulta, re.I):
        return False
    if is_subject_list_query(consulta) and not _PROGRESS_RE.search(consulta):
        return False
    return bool(_PROGRESS_RE.search(consulta))


def needs_exploratory_sql(consulta: str) -> bool:
    """Ranking / periodo / top / promedio global → SQL exploratorio."""
    if not consulta:
        return False
    if _SQL_EXPLORATORY_RE.search(consulta):
        return True
    # Typos con letras repetidas: "segundoo periodo", "peor estudiantte"
    soft = _collapse_repeats(fold(consulta))
    return bool(_SQL_EXPLORATORY_RE.search(soft))


def extract_attention_threshold(consulta: str, default: float = 3.5) -> float:
    """Lee umbral de frases como 'bajo 3.0', 'menor a 3', 'umbral 2.5'."""
    if not consulta:
        return default
    m = re.search(
        r"(?:bajo|menor(?:es)?\s+a|menor(?:es)?\s+que|umbral|threshold|<"
        r"|promedio\s*(?:<|>|bajo|menor))\s*(\d+(?:[.,]\d+)?)",
        consulta,
        re.IGNORECASE,
    )
    if not m:
        m = re.search(r"\b(\d+[.,]\d+)\b", consulta)
        if m and re.search(r"\b(bajo|menor|umbral|threshold)\b", consulta, re.I):
            pass
        else:
            return default
    try:
        val = float(m.group(1).replace(",", "."))
        if 0 < val <= 5:
            return val
    except (TypeError, ValueError):
        pass
    return default


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

    # "avance / resumen de mis grupos" → todas las calificaciones agregadas
    if is_progress_summary_query(consulta) and grades:
        return {
            "tipo": "calificaciones",
            "filtros": {},
            "filas": grades,
            "origen": "acad_actividades_notas",
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


# Preguntas de listado (NO son un alumno concreto)
_ATTENTION_LIST_RE = re.compile(
    r"\b("
    r"necesitan?\s+atenci[oó]n"
    r"|en\s+riesgo"
    r"|promedio\s+bajo"
    r"|quienes?\b.*\b(necesitan|atenci[oó]n|riesgo)"
    r"|(que|qu[eé])\s+estudiantes?"
    r"|estudiantes?\s+que\s+"
    r"|alumnos?\s+que\s+"
    r")\b",
    re.IGNORECASE,
)

# Solo disparadores claros de “un alumno por nombre”
_STUDENT_ASK_RE = re.compile(
    r"(?:"
    r"c[oó]mo\s+le\s+(?:va|fue|ha\s+ido)(?:\s+a)?"
    r"|qu[eé]\s+tal\s+(?:le\s+(?:va|fue)\s+a)?"
    r"|mejor\s+nota\s+(?:de|del|de\s+la|de\s+el)"
    r"|peor\s+nota\s+(?:de|del|de\s+la|de\s+el)"
    r"|nota\s+m[aá]s\s+alta\s+(?:de|del|de\s+la|de\s+el)"
    r"|nota\s+m[aá]xima\s+(?:de|del|de\s+la|de\s+el)"
    r"|m[aá]xima\s+nota\s+(?:de|del|de\s+la|de\s+el)"
    r"|nota\s+m[aá]s\s+baja\s+(?:de|del|de\s+la|de\s+el)"
    r"|nota\s+m[ií]nima\s+(?:de|del|de\s+la|de\s+el)"
    r"|m[ií]nima\s+nota\s+(?:de|del|de\s+la|de\s+el)"
    r"|notas?\s+(?:de|del|de\s+la|de\s+el)"
    r"|promedio\s+(?:de|del|de\s+la)\s+(?!matem|espa|cienc|social|ingles|ingl[eé]s|geo)"
    r"|detalle\s+(?:de|del|de\s+la)\s+(?:el\s+|la\s+)?(?:estudiante|alumn[oa])?"
    r"|info(?:rmaci[oó]n)?\s+(?:de|del|sobre)\s+(?:el\s+|la\s+)?(?:estudiante|alumn[oa])?"
    r"|sobre\s+(?:el\s+|la\s+)?(?:estudiante|alumn[oa])"
    r"|(?:el|la)\s+(?:estudiante|alumn[oa])"
    r")\s+(.+?)$",
    re.IGNORECASE,
)

_PERIOD_MENTION_RE = re.compile(
    r"\b("
    r"periodo\s*\d+"
    r"|per[ií]odo\s*\d+"
    r"|primer\s+periodo"
    r"|primero\s+periodo"
    r"|1er\s+periodo"
    r"|segundo\s+periodo"
    r"|2do\s+periodo"
    r"|tercer\s+periodo"
    r"|3er\s+periodo"
    r"|cuarto\s+periodo"
    r"|4to\s+periodo"
    r")\b",
    re.IGNORECASE,
)


def message_mentions_period(consulta: str) -> bool:
    return bool(consulta and _PERIOD_MENTION_RE.search(consulta))

_NAME_TOPIC_STOP = {
    "como", "le", "va", "que", "tal", "notas", "nota", "promedio",
    "dame", "dime", "sobre", "estudiante", "estudiantes", "alumno", "alumna",
    "alumnos", "detalle", "informacion", "información", "de", "del", "la", "el",
    "los", "las", "en", "por", "con", "mis", "mi", "necesitan", "necesita",
    "atencion", "atención", "riesgo", "bajo", "bajos", "quienes", "quien",
    "lista", "listar", "mostrar", "cuales", "cual", "grupo", "grupos",
    "asignatura", "asignaturas", "matematicas", "matemáticas",
}


_MATERIAL_REQUEST_RE = re.compile(
    r"("
    r"\[?\s*PEDIDO_MATERIAL_APOYO\s*\]?"
    r"|\bmaterial\s+de\s+apoyo\b"
    r"|\badjunt[eé]\s+el\s+documento\b"
    r"|\bdocumento\s+original\b"
    r"|\bsearch_academic_resources\b"
    r"|\btaller\s+(breve|corto)\b"
    r"|\bejercicios?\s+resueltos?\b"
    r"|\btarea\s+de\s+refuerzo\b"
    r"|\brecomienda(?:r|me)?\s+material\b"
    r"|\brefuerzo\s+(para|de)\b"
    r"|\brecursos?\s+acad[eé]micos?\b"
    r")",
    re.IGNORECASE,
)


def is_material_support_request(consulta: str) -> bool:
    """Pedido de taller / material de apoyo / refuerzo pedagógico."""
    return bool(consulta and _MATERIAL_REQUEST_RE.search(consulta))


ASK_ORIGINAL_MATERIAL_MSG = (
    "Para armarte un material de apoyo útil y alineado a tu clase, "
    "**adjúntame el documento original** de la actividad "
    "(guía, taller, PDF, DOCX o PPTX) con el clip del chat.\n\n"
    "Con ese archivo te preparo: taller corto, ejercicios resueltos, "
    "ejercicios para el estudiante y una clave para ti.\n\n"
    "Si quieres, en el mismo mensaje dime el **grado** y cuánto tiempo tienes "
    "(≈15–20 min)."
)


def _looks_like_subject_name(candidate: str) -> bool:
    """True si el candidato parece asignatura (Educación Artística, Religión…)."""
    f = fold(candidate)
    if not f:
        return False
    if f in _SUBJECT_TOKEN_STOP:
        return True
    tokens = [fold(t) for t in re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", candidate) if len(t) > 2]
    if not tokens:
        return False
    # Cualquier token de asignatura → no es persona
    if any(t in _SUBJECT_TOKEN_STOP for t in tokens):
        return True
    # Frases multi-palabra sin apellido típico y con & / "valores"
    if "&" in candidate or " y " in f:
        if any(t in ("valores", "etica", "ética", "artistica", "artística") for t in tokens):
            return True
    return False


def _looks_like_person_name(candidate: str) -> bool:
    """Evita frases tipo 'estudiantes necesitan atención'; permite 'Salomé'."""
    # Nombres reales son cortos; instrucciones largas no son nombres
    if len(candidate) > 80 or "(" in candidate or ")" in candidate:
        return False
    if re.search(r"\d", candidate):
        return False
    if _looks_like_subject_name(candidate):
        return False
    tokens = [t for t in re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", candidate) if len(t) > 1]
    if not tokens or len(tokens) > 6:
        return False
    meaningful = [
        t
        for t in tokens
        if fold(t) not in _NAME_TOPIC_STOP and fold(t) not in _SUBJECT_TOKEN_STOP
    ]
    if not meaningful:
        return False
    # Un solo nombre (Salomé, Sofy…) si es lo bastante largo
    if len(meaningful) == 1:
        return len(meaningful[0]) >= 4
    # Si más de la mitad son palabras de tema, no es un nombre
    if len(meaningful) < len(tokens) * 0.6:
        return False
    return True


def extract_student_name(consulta: str) -> Optional[str]:
    """
    Extrae un nombre de estudiante de preguntas como
    'cómo le va a SOFY JULIETH VERGARA HERNANDEZ' o 'cómo le va a Salomé'.
    No captura listados ('qué estudiantes necesitan atención') ni pedidos de material.
    """
    if not consulta:
        return None
    raw = consulta.strip().strip("¿?").strip()
    if _ATTENTION_LIST_RE.search(raw):
        return None
    if _MATERIAL_REQUEST_RE.search(raw):
        return None
    # Ranking/top/global sin nombre de alumno → no extraer estudiante
    if needs_exploratory_sql(raw) and not _STUDENT_ASK_RE.search(raw):
        return None

    m = _STUDENT_ASK_RE.search(raw)
    if not m:
        return None
    candidate = m.group(1).strip(" .,?¿¡")
    candidate = re.sub(
        r"^(el|la|los|las|estudiante|alumn[oa]s?)\s+",
        "",
        candidate,
        flags=re.IGNORECASE,
    ).strip()
    # Solo la primera frase / hasta puntuación fuerte (evita arrastrar el resto del prompt)
    candidate = re.split(r"[.!?；;\n]|\)|\(", candidate, maxsplit=1)[0].strip()
    # Si pidieron "notas de X en matemáticas / en el primer periodo", quedarse con el nombre
    candidate = re.split(
        r"\s+(?:en|de|del|para)\s+",
        candidate,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    # Cortar si el resto arrastra "el primer periodo..."
    candidate = re.split(
        r"\s+(?:el|la|los|las)\s+(?:primer|segundo|tercer|cuarto|periodo|per[ií]odo)",
        candidate,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    if not candidate or not _looks_like_person_name(candidate):
        return None
    return re.sub(r"\s+", " ", candidate).strip()


_FOLLOWUP_STUDENT_RE = re.compile(
    r"\b("
    r"sus?\s+notas?"
    r"|solo\s+(quiero|ver|muestra|mostrar|dame|saber)?"
    r"|unica?mente|solamente"
    r"|quiero\s+saber"
    r"|en\s+[a-záéíóúüñ]{4,}"
    r"|esa\s+(materia|asignatura)"
    r"|de\s+esa\s+materia"
    r")\b",
    re.IGNORECASE,
)

_ASIG_STOP = {
    "como", "le", "va", "que", "tal", "notas", "nota", "promedio",
    "dame", "dime", "sobre", "estudiante", "estudiantes", "alumno", "alumna",
    "alumnos", "detalle", "informacion", "de", "del", "la", "el", "los", "las",
    "en", "por", "con", "mis", "mi", "su", "sus", "quiero", "saber", "ver",
    "solo", "unica", "unicamente", "solamente", "muestra", "mostrar",
    "asignatura", "asignaturas", "materia", "materias", "nada", "mas",
    "unicamnete", "unnicamente", "general", "todas", "todos", "todo",
    "resumen", "detalle", "promedios",
}


def looks_like_student_followup(consulta: str) -> bool:
    """True si el mensaje parece un seguimiento sobre un alumno ya mencionado."""
    if not consulta:
        return False
    return bool(_FOLLOWUP_STUDENT_RE.search(consulta))


def resolve_student_name_from_history(
    history: Optional[List[Dict[str, str]]],
) -> Optional[str]:
    """Busca el último estudiante mencionado en el historial reciente."""
    if not history:
        return None
    for m in reversed(history[-12:]):
        role = (m or {}).get("role")
        content = str((m or {}).get("content") or "")
        if not content:
            continue
        if role == "user":
            name = extract_student_name(content)
            if name:
                return name
        if role == "assistant":
            m_name = re.search(
                r"As[ií] va\s+\*\*(.+?)\*\*",
                content,
                flags=re.IGNORECASE,
            )
            if m_name:
                return m_name.group(1).strip()
    return None


def _message_asignatura_candidates(message: str) -> List[str]:
    """Tokens del mensaje que podrían ser una asignatura."""
    fmsg = fold(message)
    found: List[str] = []
    for m in re.finditer(
        r"\b(?:en|de|del|solo|unica?mente|solamente)\s+([a-záéíóúüñ]{4,})\b",
        fmsg,
    ):
        found.append(m.group(1))
    for tok in re.findall(r"[a-záéíóúüñ]{4,}", fmsg):
        if tok in _ASIG_STOP:
            continue
        if tok not in found:
            found.append(tok)
    return found


def filter_rows_by_asignatura_in_message(
    rows: List[Dict[str, Any]],
    message: str,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Si el mensaje nombra una asignatura que aparece en las filas, filtra.
    Tolera typos (mattematicas ≈ matematicas).
    Retorna (filas, etiqueta_filtro). Si no hay asignatura mencionada, (rows, None).
    """
    if not rows or not message:
        return rows, None

    fmsg = fold(message)
    candidates = _message_asignatura_candidates(message)
    if not candidates and not re.search(
        r"\b(unica?mente|solamente|solo|en)\b", fmsg
    ):
        return rows, None

    scored: List[Tuple[float, str, Dict[str, Any]]] = []
    for r in rows:
        asig = str(r.get("asignatura") or "").strip()
        if not asig:
            continue
        fa = fold(asig)
        best = 0.0
        if fa and fa in fmsg:
            best = 1.0
        else:
            subj_tokens = [t for t in fa.split() if len(t) >= 3]
            for st in subj_tokens or ([fa] if fa else []):
                if st in fmsg:
                    best = max(best, 0.98)
                    continue
                for cand in candidates:
                    if _similar_token(cand, st):
                        ratio = difflib.SequenceMatcher(
                            None, _collapse_repeats(cand), _collapse_repeats(st)
                        ).ratio()
                        best = max(best, max(ratio, 0.85))
        if best >= 0.78:
            scored.append((best, asig, r))

    if not scored:
        # Solo vaciar si el docente pidió explícitamente filtrar (solo/únicamente)
        # y además escribió un candidato que no encaja con ninguna materia.
        hard_filter = bool(
            re.search(r"\b(unica?mente|solamente|\bsolo\b)\b", fmsg)
        )
        if hard_filter and candidates:
            return [], candidates[0]
        return rows, None

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[0][0]
    matched_rows: List[Dict[str, Any]] = []
    labels: List[str] = []
    for score, asig, r in scored:
        if score < top - 0.05:
            continue
        matched_rows.append(r)
        if asig not in labels:
            labels.append(asig)
    return matched_rows, ", ".join(labels)


def student_name_matches(estudiante: str, query_name: str) -> bool:
    """
    Match flexible:
    - varios tokens: todos deben aparecer en el nombre
    - un solo token (>=4): coincide con cualquier parte del nombre (ej. Salomé)
    """
    eq = fold(estudiante).split()
    qq = [t for t in fold(query_name).split() if len(t) > 1]
    if not qq or not eq:
        return False
    if len(qq) == 1:
        t = qq[0]
        if len(t) < 4:
            return False
        return any(t == e or e.startswith(t) or t.startswith(e) for e in eq)
    return all(any(t == e or t in e or e.startswith(t) for e in eq) for t in qq)


async def get_student_detail_for_teacher(
    iddocente: int,
    db_url: str,
    student_name: str,
) -> Dict[str, Any]:
    """
    Promedios por asignatura del estudiante (bajo el docente),
    buscando por nombre flexible (sin tildes / orden de tokens).
    """
    try:
        with db_manager.get_session_for_url(db_url) as session:
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
                    ORDER BY a.nombre, g.nombre, gg.nombre
                    """
                ),
                {"iddocente": iddocente},
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
            }
            for r in rows
        ]
        matched = [
            r for r in all_rows if student_name_matches(r["estudiante"], student_name)
        ]
        # Nombre canónico (el más frecuente / primero)
        canonical = matched[0]["estudiante"] if matched else None
        return {
            "result": matched,
            "meta": {
                "query_name": student_name,
                "estudiante": canonical,
                "total": len(matched),
            },
        }
    except Exception as e:
        logger.error("get_student_detail_for_teacher failed: %s", e)
        return {"result": [], "meta": {"error": str(e), "query_name": student_name}}


async def get_student_activity_notes_for_teacher(
    iddocente: int,
    db_url: str,
    student_name: str,
    asignatura: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Notas individuales (actividad → nota) del estudiante bajo el docente.
    Si `asignatura` viene, filtra por esa materia (fuzzy).
    """
    try:
        with db_manager.get_session_for_url(db_url) as session:
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
                        COALESCE(NULLIF(TRIM(act.descripcion), ''), 'Actividad') AS actividad,
                        an.nota AS nota,
                        act.fecha_creacion AS fecha
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
                      AND COALESCE(act.estado, 'on') = 'on'
                      AND COALESCE(an.estado, 'on') = 'on'
                      AND an.nota IS NOT NULL
                    ORDER BY a.nombre, act.fecha_creacion ASC NULLS LAST, act.id ASC
                    """
                ),
                {"iddocente": iddocente},
            ).fetchall()

        all_rows = [
            {
                "idmatricula": r.idmatricula,
                "estudiante": " ".join(str(r.estudiante).split()),
                "asignatura": r.asignatura,
                "grado": r.grado,
                "grupo": r.grupo,
                "actividad": r.actividad,
                "nota": float(r.nota) if r.nota is not None else None,
            }
            for r in rows
        ]
        matched = [
            r for r in all_rows if student_name_matches(r["estudiante"], student_name)
        ]
        if asignatura:
            filtered, label = filter_rows_by_asignatura_in_message(
                matched, f"en {asignatura}"
            )
            if label:
                matched = filtered
        return matched
    except Exception as e:
        logger.error("get_student_activity_notes_for_teacher failed: %s", e)
        return []
