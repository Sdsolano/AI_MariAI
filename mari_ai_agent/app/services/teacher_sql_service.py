"""
Read-only SQL for Mao, scoped to the authenticated teacher (iddocente).

Guardrails:
- Single SELECT / WITH … SELECT
- Forbidden write/DDL keywords
- Table whitelist
- Mandatory :iddocente bind placeholder
- statement_timeout + row cap
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set

from sqlalchemy import text

from app.db.connection import db_manager

logger = logging.getLogger(__name__)


def _fold_plain(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(s or ""))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _asig_ilike_pattern(asignatura: str) -> str:
    """Accent-folded ILIKE pattern ('Matemáticas' ↔ 'Matematicas')."""
    return f"%{_fold_plain(asignatura)}%"


def _fold_sql(expr: str) -> str:
    """Postgres accent-fold for ILIKE against _asig_ilike_pattern binds."""
    return (
        f"translate(lower({expr}), "
        f"'áéíóúüñäëïöüàèìòùÁÉÍÓÚÜÑ', "
        f"'aeiouunaeiouaeiouaeiouun')"
    )


def _asig_match_sql(column: str = "a.nombre") -> str:
    return f"{_fold_sql(column)} ILIKE lower(:asig)"

MAX_ROWS = 100
STATEMENT_TIMEOUT_MS = 5000

ALLOWED_TABLES: Set[str] = {
    "acad_gradosasignaturas",
    "acad_asignaturas",
    "acad_grados",
    "acad_gradosgrupos",
    "acad_actividades",
    "acad_actividades_notas",
    "acad_matricula_academica",
    "acad_estumatricula",
    "acad_periodos",
    "acad_anio_lectivo",
    "acad_anios_lectivos",
    "acad_pdm",
    "acad_planesmejoramiento",
    "acad_docentes",
    "estu_estudiantes",
    "student_attendance",
}

# Also allow any acad_* table that appears in FROM/JOIN (school schemas vary)
ALLOWED_TABLE_PREFIXES = ("acad_",)

FORBIDDEN_RE = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE|"
    r"COPY|EXECUTE|CALL|SET|INTO|MERGE|REPLACE|VACUUM|ANALYZE|"
    r"COMMENT|SECURITY|OWNER|LISTEN|NOTIFY|LOAD|PREPARE|DEALLOCATE|"
    r"DO\b|PERFORM|pg_sleep|dblink|lo_import|lo_export"
    r")\b",
    re.IGNORECASE,
)

# FROM / JOIN table references (simple; enough for LLM-generated SQL)
TABLE_REF_RE = re.compile(
    r"(?i)\b(?:FROM|JOIN)\s+([a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*)?)",
)

# CTE names in WITH cte AS ( ... )
CTE_NAME_RE = re.compile(
    r"\bWITH\s+([a-zA-Z_][\w]*)\s+AS\b"
    r"|,\s*([a-zA-Z_][\w]*)\s+AS\s*\(",
    re.IGNORECASE,
)

MULTI_STATEMENT_RE = re.compile(r";\s*\S")

TEACHER_SCHEMA_CHEATSHEET = {
    "scope_rule": (
        "TODA consulta DEBE incluir el placeholder :iddocente y filtrar por "
        "acad_gradosasignaturas (ga.iddocente = :iddocente). "
        "Nunca inventes un iddocente numérico."
    ),
    "tables": {
        "acad_gradosasignaturas": (
            "Puente docente↔asignatura↔grupo. Columnas clave: id, iddocente, "
            "idasignatura, idgrados_grupos, estado."
        ),
        "acad_asignaturas": "id, nombre, abr",
        "acad_grados": "id, nombre",
        "acad_gradosgrupos": "id, nombre, idgrado",
        "acad_actividades": (
            "id, idgrados_asignatura, idperiodo, nombre/descripcion. "
            "El periodo de la nota suele venir de act.idperiodo."
        ),
        "acad_actividades_notas": (
            "id, idactividad, idmatricula (→ acad_matricula_academica.id), nota, estado"
        ),
        "acad_matricula_academica": (
            "id, idestumatricula, idgrados_asignatura, idperiodo"
        ),
        "acad_estumatricula": "id, idestudiante, idgrados_grupos",
        "estu_estudiantes": (
            "id, primer_nombre, segundo_nombre, primer_apellido, segundo_apellido"
        ),
        "acad_periodos": "id, nombre, idanio_lectivo, porcentaje, estado",
        "student_attendance": (
            "course_id (=idgrados_grupos), subject_id (=idasignatura), "
            "student_id, period_id, attended, not_attended, late, excuse"
        ),
    },
    "join_pattern_notas": (
        "acad_gradosasignaturas ga "
        "JOIN acad_asignaturas a ON a.id = ga.idasignatura "
        "JOIN acad_gradosgrupos gg ON gg.id = ga.idgrados_grupos "
        "JOIN acad_grados g ON g.id = gg.idgrado "
        "JOIN acad_actividades act ON act.idgrados_asignatura = ga.id "
        "JOIN acad_actividades_notas an ON an.idactividad = act.id "
        "JOIN acad_matricula_academica ma ON ma.id = an.idmatricula "
        "JOIN acad_estumatricula em ON em.id = ma.idestumatricula "
        "JOIN estu_estudiantes e ON e.id = em.idestudiante "
        "LEFT JOIN acad_periodos p ON p.id = act.idperiodo "
        "WHERE ga.iddocente = :iddocente AND COALESCE(ga.estado,'on')='on'"
    ),
    "example_mejor_estudiante_periodo": (
        "SELECT TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre, "
        "e.primer_apellido, e.segundo_apellido)) AS estudiante, "
        "ROUND(AVG(an.nota)::numeric, 2) AS promedio, COUNT(an.id) AS total_notas "
        "FROM acad_gradosasignaturas ga "
        "JOIN acad_actividades act ON act.idgrados_asignatura = ga.id "
        "JOIN acad_actividades_notas an ON an.idactividad = act.id "
        "JOIN acad_matricula_academica ma ON ma.id = an.idmatricula "
        "JOIN acad_estumatricula em ON em.id = ma.idestumatricula "
        "JOIN estu_estudiantes e ON e.id = em.idestudiante "
        "JOIN acad_periodos p ON p.id = act.idperiodo "
        "WHERE ga.iddocente = :iddocente AND COALESCE(ga.estado,'on')='on' "
        "AND an.nota IS NOT NULL "
        "AND (p.nombre ILIKE '%3%' OR p.nombre ILIKE '%tercer%' OR act.idperiodo = :idperiodo) "
        "GROUP BY e.id, e.primer_nombre, e.segundo_nombre, e.primer_apellido, e.segundo_apellido "
        "ORDER BY promedio DESC NULLS LAST LIMIT 10"
    ),
    "notes": [
        "Usa list_periods para resolver el id/nombre del periodo antes de filtrar.",
        "Para promedios globales del docente, no filtres por un solo grupo salvo que el usuario lo pida.",
        "Responde al docente en lenguaje natural; no menciones nombres de tablas.",
    ],
}


def describe_teacher_schema() -> Dict[str, Any]:
    return {
        "function": "describe_teacher_schema",
        "schema": TEACHER_SCHEMA_CHEATSHEET,
    }


def _strip_sql_comments(sql: str) -> str:
    # Remove /* ... */ and -- line comments
    no_block = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    no_line = re.sub(r"--.*?$", " ", no_block, flags=re.MULTILINE)
    return no_line


def _normalize_sql(sql: str) -> str:
    cleaned = _strip_sql_comments(sql or "").strip()
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()
    return cleaned


def _extract_cte_names(sql: str) -> Set[str]:
    names: Set[str] = set()
    for m in CTE_NAME_RE.finditer(sql):
        for g in m.groups():
            if g:
                names.add(g.lower())
    return names


def _extract_tables(sql: str) -> Set[str]:
    tables: Set[str] = set()
    ctes = _extract_cte_names(sql)
    for m in TABLE_REF_RE.finditer(sql):
        raw = m.group(1)
        name = raw.split(".")[-1].lower()
        if name in ctes:
            continue  # CTE alias, not a physical table
        tables.add(name)
    return tables


def _table_allowed(name: str) -> bool:
    if name in ALLOWED_TABLES:
        return True
    return any(name.startswith(p) for p in ALLOWED_TABLE_PREFIXES)


def validate_teacher_sql(sql: str) -> Dict[str, Any]:
    """
    Returns {ok: True, sql: str} or {ok: False, error: str}.
    """
    cleaned = _normalize_sql(sql)
    if not cleaned:
        return {"ok": False, "error": "SQL vacío."}

    if MULTI_STATEMENT_RE.search(cleaned):
        return {"ok": False, "error": "Solo se permite una sentencia SQL."}

    upper = cleaned.lstrip().upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return {
            "ok": False,
            "error": "Solo se permiten consultas SELECT o WITH … SELECT.",
        }

    if FORBIDDEN_RE.search(cleaned):
        return {
            "ok": False,
            "error": "SQL rechazado: contiene operaciones no permitidas (solo lectura).",
        }

    # Block SELECT INTO explicitly (INTO is in FORBIDDEN but keep clear msg)
    if re.search(r"(?i)\bINTO\b", cleaned):
        return {"ok": False, "error": "SELECT INTO no está permitido."}

    if ":iddocente" not in cleaned:
        return {
            "ok": False,
            "error": (
                "La consulta DEBE incluir el placeholder :iddocente "
                "(ej. WHERE ga.iddocente = :iddocente)."
            ),
        }

    # Discourage hardcoding a numeric teacher id instead of the bind
    if re.search(r"(?i)iddocente\s*=\s*\d+", cleaned) and ":iddocente" not in cleaned:
        return {"ok": False, "error": "Usa :iddocente, no un id numérico fijo."}

    tables = _extract_tables(cleaned)
    if not tables:
        return {
            "ok": False,
            "error": "No se detectaron tablas en FROM/JOIN. Revisa el SQL.",
        }

    bad = sorted(t for t in tables if not _table_allowed(t))
    if bad:
        return {
            "ok": False,
            "error": (
                f"Tablas no permitidas: {', '.join(bad)}. "
                f"Usa describe_teacher_schema para ver las permitidas."
            ),
        }

    if "acad_gradosasignaturas" not in tables:
        return {
            "ok": False,
            "error": (
                "La consulta debe incluir acad_gradosasignaturas para acotar "
                "al docente (ga.iddocente = :iddocente)."
            ),
        }

    # Ensure LIMIT (numeric or bind :lim / :limit)
    final_sql = cleaned
    if re.search(r"(?i)\bLIMIT\s+:\w+", final_sql):
        pass
    elif not re.search(r"(?i)\bLIMIT\s+\d+", final_sql):
        final_sql = f"{final_sql}\nLIMIT {MAX_ROWS}"
    else:
        # Cap excessive LIMIT
        def _cap_limit(m: re.Match) -> str:
            n = int(m.group(1))
            return f"LIMIT {min(n, MAX_ROWS)}"

        final_sql = re.sub(r"(?i)\bLIMIT\s+(\d+)", _cap_limit, final_sql, count=1)

    return {"ok": True, "sql": final_sql}


def _row_to_dict(row: Any) -> Dict[str, Any]:
    mapping = getattr(row, "_mapping", None)
    if mapping is not None:
        data = dict(mapping)
    else:
        data = dict(row)
    out: Dict[str, Any] = {}
    for k, v in data.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        elif isinstance(v, (bytes, bytearray)):
            out[k] = v.decode("utf-8", errors="replace")
        else:
            # Decimal / numeric → float when possible
            try:
                from decimal import Decimal

                if isinstance(v, Decimal):
                    out[k] = float(v)
                    continue
            except Exception:
                pass
            out[k] = v
    return out


def resolve_period_id(periods: List[Dict[str, Any]], consulta: str) -> Optional[int]:
    """Map 'Periodo 3' / 'tercer periodo' → idperiodo from list_periods rows."""
    if not periods or not consulta:
        return None
    # Lower + colapsar letras repetidas (segundoo → segundo)
    f = re.sub(r"(.)\1+", r"\1", (consulta or "").lower())
    # explicit number
    m = re.search(r"(?:periodo|per[ií]odo)\s*(\d+)", f, re.I)
    ordinal = {
        "primer": 1,
        "primero": 1,
        "1er": 1,
        "segundo": 2,
        "2do": 2,
        "tercer": 3,
        "tercero": 3,
        "3er": 3,
        "cuarto": 4,
        "4to": 4,
    }
    want_num: Optional[int] = int(m.group(1)) if m else None
    if want_num is None:
        for word, num in ordinal.items():
            if re.search(rf"\b{word}\b", f):
                want_num = num
                break
    if want_num is None:
        return None

    # Prefer nombre containing the number
    for p in periods:
        nombre = str(p.get("nombre") or "")
        if re.search(rf"\b{want_num}\b", nombre):
            return int(p["idperiodo"])
        if want_num == 3 and re.search(r"tercer", nombre, re.I):
            return int(p["idperiodo"])
    # Fallback: nth by id order
    ordered = sorted(periods, key=lambda x: int(x.get("idperiodo") or 0))
    if 1 <= want_num <= len(ordered):
        return int(ordered[want_num - 1]["idperiodo"])
    return None


async def rank_students_by_period(
    db_url: str,
    iddocente: int,
    idperiodo: int,
    *,
    order: str = "DESC",
    limit: int = 5,
    asignatura: Optional[str] = None,
) -> Dict[str, Any]:
    order_sql = "DESC" if str(order).upper().startswith("D") else "ASC"
    joins = ""
    clauses = [
        "ga.iddocente = :iddocente",
        "COALESCE(ga.estado, 'on') = 'on'",
        "act.idperiodo = :idperiodo",
        "an.nota IS NOT NULL",
    ]
    binds: Dict[str, Any] = {
        "idperiodo": idperiodo,
        "lim": int(limit),
    }
    select_asig = ""
    group_asig = ""
    if asignatura:
        joins = "JOIN acad_asignaturas a ON a.id = ga.idasignatura"
        clauses.append(_asig_match_sql("a.nombre"))
        binds["asig"] = _asig_ilike_pattern(asignatura)
        select_asig = "a.nombre AS asignatura,"
        group_asig = ", a.nombre"
    where = " AND ".join(clauses)
    sql = f"""
        SELECT
            TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre,
                           e.primer_apellido, e.segundo_apellido)) AS estudiante,
            {select_asig}
            ROUND(AVG(an.nota)::numeric, 2) AS promedio,
            COUNT(an.id) AS total_notas
        FROM acad_gradosasignaturas ga
        {joins}
        JOIN acad_actividades act ON act.idgrados_asignatura = ga.id
        JOIN acad_actividades_notas an ON an.idactividad = act.id
        JOIN acad_matricula_academica ma ON ma.id = an.idmatricula
        JOIN acad_estumatricula em ON em.id = ma.idestumatricula
        JOIN estu_estudiantes e ON e.id = em.idestudiante
        WHERE {where}
        GROUP BY e.id, e.primer_nombre, e.segundo_nombre,
                 e.primer_apellido, e.segundo_apellido{group_asig}
        ORDER BY promedio {order_sql} NULLS LAST
        LIMIT :lim
    """
    return await run_teacher_sql(
        db_url,
        iddocente,
        sql,
        extra_binds=binds,
    )


async def top_students_by_subject_group(
    db_url: str,
    iddocente: int,
    *,
    asignatura: str,
    grado: Optional[str] = None,
    grupo: Optional[str] = None,
    limit: int = 5,
    order: str = "DESC",
    idperiodo: Optional[int] = None,
) -> Dict[str, Any]:
    order_sql = "DESC" if str(order).upper().startswith("D") else "ASC"
    clauses = [
        "ga.iddocente = :iddocente",
        "COALESCE(ga.estado, 'on') = 'on'",
        "an.nota IS NOT NULL",
        _asig_match_sql("a.nombre"),
    ]
    binds: Dict[str, Any] = {
        "iddocente": iddocente,
        "asig": _asig_ilike_pattern(asignatura),
        "lim": int(limit),
    }
    if grado:
        clauses.append("g.nombre ILIKE :grado")
        binds["grado"] = f"%{grado}%"
    if grupo:
        clauses.append("gg.nombre ILIKE :grupo")
        binds["grupo"] = f"%{grupo}%"
    if idperiodo is not None:
        clauses.append("act.idperiodo = :idperiodo")
        binds["idperiodo"] = int(idperiodo)
    where = " AND ".join(clauses)
    sql = f"""
        SELECT
            TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre,
                           e.primer_apellido, e.segundo_apellido)) AS estudiante,
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
        WHERE {where}
        GROUP BY e.id, e.primer_nombre, e.segundo_nombre, e.primer_apellido,
                 e.segundo_apellido, a.nombre, g.nombre, gg.nombre
        ORDER BY promedio {order_sql} NULLS LAST
        LIMIT :lim
    """
    return await run_teacher_sql(db_url, iddocente, sql, extra_binds=binds)


async def global_average(db_url: str, iddocente: int) -> Dict[str, Any]:
    sql = """
        SELECT
            ROUND(AVG(an.nota)::numeric, 2) AS promedio_global,
            COUNT(an.id) AS total_notas,
            COUNT(DISTINCT an.idmatricula) AS estudiantes_con_nota
        FROM acad_gradosasignaturas ga
        JOIN acad_actividades act ON act.idgrados_asignatura = ga.id
        JOIN acad_actividades_notas an ON an.idactividad = act.id
        WHERE ga.iddocente = :iddocente
          AND COALESCE(ga.estado, 'on') = 'on'
          AND an.nota IS NOT NULL
        LIMIT 1
    """
    return await run_teacher_sql(db_url, iddocente, sql)


async def student_best_grade(
    db_url: str,
    iddocente: int,
    *,
    nombre: str,
    asignatura: Optional[str] = None,
    worst: bool = False,
) -> Dict[str, Any]:
    """Mejor (o peor) nota individual de un estudiante del docente."""
    order = "ASC" if worst else "DESC"
    clauses = [
        "ga.iddocente = :iddocente",
        "COALESCE(ga.estado, 'on') = 'on'",
        "an.nota IS NOT NULL",
        """TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre,
                           e.primer_apellido, e.segundo_apellido))
            ILIKE :nombre_pat""",
    ]
    binds: Dict[str, Any] = {
        "iddocente": iddocente,
        "nombre_pat": f"%{' '.join(str(nombre).split()).replace(' ', '%')}%",
    }
    if asignatura:
        clauses.append(_asig_match_sql("a.nombre"))
        binds["asig"] = _asig_ilike_pattern(asignatura)
    where = " AND ".join(clauses)
    sql = f"""
        SELECT
            TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre,
                           e.primer_apellido, e.segundo_apellido)) AS estudiante,
            a.nombre AS asignatura,
            g.nombre AS grado,
            gg.nombre AS grupo,
            COALESCE(NULLIF(TRIM(act.descripcion), ''), 'Actividad') AS actividad,
            p.nombre AS periodo,
            an.nota
        FROM acad_gradosasignaturas ga
        JOIN acad_asignaturas a ON a.id = ga.idasignatura
        JOIN acad_gradosgrupos gg ON gg.id = ga.idgrados_grupos
        JOIN acad_grados g ON g.id = gg.idgrado
        JOIN acad_actividades act ON act.idgrados_asignatura = ga.id
        LEFT JOIN acad_periodos p ON p.id = act.idperiodo
        JOIN acad_actividades_notas an ON an.idactividad = act.id
        JOIN acad_matricula_academica ma ON ma.id = an.idmatricula
        JOIN acad_estumatricula em ON em.id = ma.idestumatricula
        JOIN estu_estudiantes e ON e.id = em.idestudiante
        WHERE {where}
        ORDER BY an.nota {order} NULLS LAST, act.fecha_creacion DESC NULLS LAST
        LIMIT 5
    """
    return await run_teacher_sql(db_url, iddocente, sql, extra_binds=binds)


async def list_activities(
    db_url: str,
    iddocente: int,
    *,
    asignatura: Optional[str] = None,
    idperiodo: Optional[int] = None,
    actividad: Optional[str] = None,
    limit: int = 40,
) -> Dict[str, Any]:
    """Lista actividades del docente (opcionalmente filtradas)."""
    clauses = [
        "ga.iddocente = :iddocente",
        "COALESCE(ga.estado, 'on') = 'on'",
    ]
    binds: Dict[str, Any] = {"iddocente": iddocente, "lim": int(limit)}
    if asignatura:
        clauses.append(_asig_match_sql("a.nombre"))
        binds["asig"] = _asig_ilike_pattern(asignatura)
    if idperiodo is not None:
        clauses.append("act.idperiodo = :idperiodo")
        binds["idperiodo"] = int(idperiodo)
    if actividad:
        # tokens flexibles: "CALENTAMIENTO RECORRIDO" → %calentamiento%recorrido%
        toks = [
            t
            for t in re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]{3,}", actividad)
            if _fold_plain(t) not in {"del", "los", "las", "por", "para", "una", "con"}
        ]
        if toks:
            folded_desc = _fold_sql("COALESCE(act.descripcion, '')")
            clauses.append(f"{folded_desc} ILIKE :act_pat")
            binds["act_pat"] = "%" + "%".join(_fold_plain(t) for t in toks[:6]) + "%"
    where = " AND ".join(clauses)
    sql = f"""
        SELECT
            act.id AS idactividad,
            COALESCE(NULLIF(TRIM(act.descripcion), ''), 'Actividad') AS actividad,
            a.nombre AS asignatura,
            p.nombre AS periodo,
            act.idperiodo,
            COUNT(an.id) AS total_notas,
            ROUND(AVG(an.nota)::numeric, 2) AS promedio,
            ROUND(MAX(an.nota)::numeric, 2) AS max_nota,
            ROUND(MIN(an.nota)::numeric, 2) AS min_nota
        FROM acad_gradosasignaturas ga
        JOIN acad_asignaturas a ON a.id = ga.idasignatura
        JOIN acad_actividades act ON act.idgrados_asignatura = ga.id
        LEFT JOIN acad_periodos p ON p.id = act.idperiodo
        LEFT JOIN acad_actividades_notas an
          ON an.idactividad = act.id AND an.nota IS NOT NULL
        WHERE {where}
        GROUP BY act.id, act.descripcion, a.nombre, p.nombre, act.idperiodo
        ORDER BY total_notas DESC NULLS LAST, act.id DESC
        LIMIT :lim
    """
    return await run_teacher_sql(db_url, iddocente, sql, extra_binds=binds)


async def activity_extreme_grade(
    db_url: str,
    iddocente: int,
    *,
    actividad: str,
    asignatura: Optional[str] = None,
    idperiodo: Optional[int] = None,
    mode: str = "max",
) -> Dict[str, Any]:
    """
    Mejor/peor nota (estudiante) o promedio de una actividad.
    mode: max | min | avg
    """
    mode_l = (mode or "max").lower()
    acts = await list_activities(
        db_url,
        iddocente,
        asignatura=asignatura,
        idperiodo=idperiodo,
        actividad=actividad,
        limit=10,
    )
    rows = acts.get("result") or []
    # Relax filters: period → subject → name-only (activity titles are distinctive)
    if not rows and idperiodo is not None:
        acts = await list_activities(
            db_url, iddocente, asignatura=asignatura, actividad=actividad, limit=10
        )
        rows = acts.get("result") or []
    if not rows and asignatura:
        acts = await list_activities(
            db_url, iddocente, idperiodo=idperiodo, actividad=actividad, limit=10
        )
        rows = acts.get("result") or []
    if not rows:
        acts = await list_activities(
            db_url, iddocente, actividad=actividad, limit=10
        )
        rows = acts.get("result") or []
    if not rows:
        return {
            "function": "activity_extreme_grade",
            "error": "actividad_no_encontrada",
            "result": [],
            "count": 0,
            "candidates": [],
        }
    # Prefer exact title (incl. spacing quirks) when the teacher quoted it;
    # otherwise expand to token-similar peers across groups.
    def _norm_title(s: str) -> str:
        t = _fold_plain(s)
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    def _norm_tight(s: str) -> str:
        return re.sub(r"[\s\(\)\[\]\{\}\-_/]+", "", _fold_plain(s))

    target = _fold_plain(actividad)
    target_norm = _norm_title(actividad)
    target_tight = _norm_tight(actividad)
    toks = [
        t
        for t in re.findall(r"[a-z0-9]{4,}", target)
        if t not in {"para", "como", "sobre", "entre", "hacia", "desde"}
    ] or [t for t in re.findall(r"[a-z0-9]{3,}", target)]

    exact = [
        r
        for r in rows
        if _norm_title(str(r.get("actividad") or "")) == target_norm
    ]
    if not exact and target_tight:
        # same letters ignoring spaces/parens — still treat as exact family
        exact = [
            r
            for r in rows
            if _norm_tight(str(r.get("actividad") or "")) == target_tight
        ]

    def score(r: Dict[str, Any]) -> float:
        name = _fold_plain(str(r.get("actividad") or ""))
        if not toks:
            return float(r.get("total_notas") or 0)
        hit = sum(1 for t in toks if t in name)
        coverage = hit / len(toks)
        return coverage * 1000 + float(r.get("total_notas") or 0) / 100.0

    if exact:
        # Quoted/exact title: only those rows (may still be multi-group clones)
        peers = exact[:12]
    else:
        rows_sorted = sorted(rows, key=score, reverse=True)
        top_score = score(rows_sorted[0])
        peers = [
            r for r in rows_sorted if score(r) >= max(top_score * 0.7, 600.0)
        ][:12]
        if not peers:
            peers = rows_sorted[:1]
    chosen = peers[0]
    aids = [int(r["idactividad"]) for r in peers]

    if mode_l == "avg":
        # weighted average across peer activities (same title / groups)
        sql_avg = """
            SELECT
                ROUND(AVG(an.nota)::numeric, 2) AS promedio,
                COUNT(an.id) AS total_notas,
                ROUND(MAX(an.nota)::numeric, 2) AS max_nota,
                ROUND(MIN(an.nota)::numeric, 2) AS min_nota
            FROM acad_actividades_notas an
            JOIN acad_actividades act ON act.id = an.idactividad
            JOIN acad_gradosasignaturas ga ON ga.id = act.idgrados_asignatura
            WHERE ga.iddocente = :iddocente
              AND act.id = ANY(:aids)
              AND an.nota IS NOT NULL
        """
        # run_teacher_sql may not bind arrays well — expand IN list
        in_list = ", ".join(str(i) for i in aids)
        sql_avg = f"""
            SELECT
                ROUND(AVG(an.nota)::numeric, 2) AS promedio,
                COUNT(an.id) AS total_notas,
                ROUND(MAX(an.nota)::numeric, 2) AS max_nota,
                ROUND(MIN(an.nota)::numeric, 2) AS min_nota
            FROM acad_actividades_notas an
            JOIN acad_actividades act ON act.id = an.idactividad
            JOIN acad_gradosasignaturas ga ON ga.id = act.idgrados_asignatura
            WHERE ga.iddocente = :iddocente
              AND act.id IN ({in_list})
              AND an.nota IS NOT NULL
        """
        payload = await run_teacher_sql(db_url, iddocente, sql_avg)
        row0 = (payload.get("result") or [{}])[0]
        return {
            "function": "activity_extreme_grade",
            "mode": "avg",
            "matched_activity": chosen,
            "matched_ids": aids,
            "result": [
                {
                    "actividad": chosen.get("actividad"),
                    "asignatura": chosen.get("asignatura"),
                    "periodo": chosen.get("periodo"),
                    "promedio": row0.get("promedio"),
                    "total_notas": row0.get("total_notas"),
                    "max_nota": row0.get("max_nota"),
                    "min_nota": row0.get("min_nota"),
                }
            ],
            "count": 1,
        }

    order = "DESC" if mode_l == "max" else "ASC"
    in_list = ", ".join(str(i) for i in aids)
    # First get the extreme value, then all students tied at that value
    sql_ext = f"""
        SELECT an.nota AS extrema
        FROM acad_actividades act
        JOIN acad_gradosasignaturas ga ON ga.id = act.idgrados_asignatura
        JOIN acad_actividades_notas an ON an.idactividad = act.id
        WHERE ga.iddocente = :iddocente
          AND act.id IN ({in_list})
          AND an.nota IS NOT NULL
        ORDER BY an.nota {order} NULLS LAST
        LIMIT 1
    """
    ext_payload = await run_teacher_sql(db_url, iddocente, sql_ext)
    ext_rows = ext_payload.get("result") or []
    if not ext_rows:
        return {
            "function": "activity_extreme_grade",
            "mode": mode_l,
            "matched_activity": chosen,
            "matched_ids": aids,
            "result": [],
            "count": 0,
            "error": "sin_notas",
        }
    extrema = ext_rows[0]["extrema"]
    sql = f"""
        SELECT DISTINCT
            TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre,
                           e.primer_apellido, e.segundo_apellido)) AS estudiante,
            an.nota,
            COALESCE(NULLIF(TRIM(act.descripcion), ''), 'Actividad') AS actividad,
            a.nombre AS asignatura,
            p.nombre AS periodo
        FROM acad_actividades act
        JOIN acad_gradosasignaturas ga ON ga.id = act.idgrados_asignatura
        JOIN acad_asignaturas a ON a.id = ga.idasignatura
        LEFT JOIN acad_periodos p ON p.id = act.idperiodo
        JOIN acad_actividades_notas an ON an.idactividad = act.id
        JOIN acad_matricula_academica ma ON ma.id = an.idmatricula
        JOIN acad_estumatricula em ON em.id = ma.idestumatricula
        JOIN estu_estudiantes e ON e.id = em.idestudiante
        WHERE ga.iddocente = :iddocente
          AND act.id IN ({in_list})
          AND an.nota IS NOT NULL
          AND an.nota = :extrema
        ORDER BY estudiante
        LIMIT 20
    """
    payload = await run_teacher_sql(
        db_url, iddocente, sql, extra_binds={"extrema": extrema}
    )
    payload["mode"] = mode_l
    payload["matched_activity"] = chosen
    payload["matched_ids"] = aids
    payload["function"] = "activity_extreme_grade"
    payload["extrema"] = extrema
    return payload


async def student_grades_by_period(
    db_url: str,
    iddocente: int,
    *,
    nombre: str,
    idperiodo: int,
) -> Dict[str, Any]:
    """Promedio por asignatura de UN estudiante del docente en un periodo."""
    sql = """
        SELECT
            TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre,
                           e.primer_apellido, e.segundo_apellido)) AS estudiante,
            a.nombre AS asignatura,
            g.nombre AS grado,
            gg.nombre AS grupo,
            p.nombre AS periodo,
            ROUND(AVG(an.nota)::numeric, 2) AS promedio,
            COUNT(an.id) AS total_notas
        FROM acad_gradosasignaturas ga
        JOIN acad_asignaturas a ON a.id = ga.idasignatura
        JOIN acad_gradosgrupos gg ON gg.id = ga.idgrados_grupos
        JOIN acad_grados g ON g.id = gg.idgrado
        JOIN acad_actividades act ON act.idgrados_asignatura = ga.id
        JOIN acad_periodos p ON p.id = act.idperiodo
        JOIN acad_actividades_notas an ON an.idactividad = act.id
        JOIN acad_matricula_academica ma ON ma.id = an.idmatricula
        JOIN acad_estumatricula em ON em.id = ma.idestumatricula
        JOIN estu_estudiantes e ON e.id = em.idestudiante
        WHERE ga.iddocente = :iddocente
          AND COALESCE(ga.estado, 'on') = 'on'
          AND act.idperiodo = :idperiodo
          AND an.nota IS NOT NULL
          AND (
            TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre,
                           e.primer_apellido, e.segundo_apellido))
            ILIKE :nombre_pat
          )
        GROUP BY e.id, e.primer_nombre, e.segundo_nombre, e.primer_apellido,
                 e.segundo_apellido, a.nombre, g.nombre, gg.nombre, p.nombre
        ORDER BY a.nombre
        LIMIT 50
    """
    # Flexible name match: collapse spaces, wrap with %
    compact = " ".join(str(nombre).split())
    return await run_teacher_sql(
        db_url,
        iddocente,
        sql,
        extra_binds={
            "idperiodo": int(idperiodo),
            "nombre_pat": f"%{compact.replace(' ', '%')}%",
        },
    )


async def best_student_per_subject_group(
    db_url: str,
    iddocente: int,
    *,
    order: str = "DESC",
) -> Dict[str, Any]:
    """
    Best (or worst) student within each asignatura+grado+grupo of the teacher.
    Uses ROW_NUMBER window; returns one row per course section.
    """
    order_sql = "DESC" if str(order).upper().startswith("D") else "ASC"
    sql = f"""
        WITH ranked AS (
            SELECT
                a.nombre AS asignatura,
                g.nombre AS grado,
                gg.nombre AS grupo,
                TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre,
                               e.primer_apellido, e.segundo_apellido)) AS estudiante,
                ROUND(AVG(an.nota)::numeric, 2) AS promedio,
                COUNT(an.id) AS total_notas,
                ROW_NUMBER() OVER (
                    PARTITION BY a.nombre, g.nombre, gg.nombre
                    ORDER BY AVG(an.nota) {order_sql} NULLS LAST
                ) AS rn
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
              AND an.nota IS NOT NULL
            GROUP BY a.nombre, g.nombre, gg.nombre,
                     e.id, e.primer_nombre, e.segundo_nombre,
                     e.primer_apellido, e.segundo_apellido
        )
        SELECT asignatura, grado, grupo, estudiante, promedio, total_notas
        FROM ranked
        WHERE rn = 1
        ORDER BY asignatura, grado, grupo
        LIMIT 100
    """
    return await run_teacher_sql(db_url, iddocente, sql)


async def list_periods(db_url: str, iddocente: int) -> Dict[str, Any]:
    """
    Periods linked to the teacher's activities (current relevant years),
    falling back to all on periods if join yields nothing.
    """
    try:
        with db_manager.get_session_for_url(db_url) as session:
            rows = session.execute(
                text(
                    """
                    SELECT DISTINCT
                        p.id AS idperiodo,
                        p.nombre,
                        p.idanio_lectivo,
                        p.porcentaje,
                        p.estado
                    FROM acad_periodos p
                    WHERE EXISTS (
                        SELECT 1
                        FROM acad_actividades act
                        JOIN acad_gradosasignaturas ga
                          ON ga.id = act.idgrados_asignatura
                        WHERE act.idperiodo = p.id
                          AND ga.iddocente = :iddocente
                          AND COALESCE(ga.estado, 'on') = 'on'
                    )
                    ORDER BY p.idanio_lectivo DESC NULLS LAST, p.id
                    LIMIT 50
                    """
                ),
                {"iddocente": iddocente},
            ).fetchall()

            if not rows:
                rows = session.execute(
                    text(
                        """
                        SELECT
                            p.id AS idperiodo,
                            p.nombre,
                            p.idanio_lectivo,
                            p.porcentaje,
                            p.estado
                        FROM acad_periodos p
                        WHERE COALESCE(p.estado, 'on') = 'on'
                        ORDER BY p.idanio_lectivo DESC NULLS LAST, p.id
                        LIMIT 50
                        """
                    )
                ).fetchall()

            result = [_row_to_dict(r) for r in rows]
            return {
                "function": "list_periods",
                "count": len(result),
                "result": result,
                "hint": (
                    "Usa idperiodo o nombre al filtrar. "
                    "Para 'tercer periodo' busca nombre ILIKE '%3%' / '%tercer%'."
                ),
            }
    except Exception as e:
        logger.error("list_periods failed: %s", e, exc_info=True)
        return {"function": "list_periods", "error": str(e), "result": [], "count": 0}


async def run_teacher_sql(
    db_url: str,
    iddocente: int,
    sql: str,
    extra_binds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    check = validate_teacher_sql(sql)
    if not check.get("ok"):
        return {
            "function": "run_teacher_sql",
            "error": check.get("error"),
            "rejected": True,
            "result": [],
            "count": 0,
        }

    final_sql = check["sql"]
    binds: Dict[str, Any] = {"iddocente": iddocente}
    if extra_binds:
        # Never allow overriding iddocente
        for k, v in extra_binds.items():
            if k == "iddocente":
                continue
            binds[k] = v

    # Only pass binds that appear in SQL (sqlalchemy can complain otherwise
    # depending on version; safer to filter)
    used_binds = {
        k: v
        for k, v in binds.items()
        if re.search(rf":{re.escape(k)}\b", final_sql)
    }
    # Always keep iddocente
    used_binds["iddocente"] = iddocente

    try:
        with db_manager.get_session_for_url(db_url) as session:
            session.execute(
                text(f"SET LOCAL statement_timeout = {int(STATEMENT_TIMEOUT_MS)}")
            )
            result = session.execute(text(final_sql), used_binds)
            columns = list(result.keys())
            raw_rows = result.fetchmany(MAX_ROWS + 1)
            truncated = len(raw_rows) > MAX_ROWS
            raw_rows = raw_rows[:MAX_ROWS]
            rows = [_row_to_dict(r) for r in raw_rows]
            return {
                "function": "run_teacher_sql",
                "columns": columns,
                "result": rows,
                "count": len(rows),
                "truncated": truncated,
                "sql_executed": final_sql,
            }
    except Exception as e:
        logger.error("run_teacher_sql failed: %s | sql=%s", e, final_sql, exc_info=True)
        return {
            "function": "run_teacher_sql",
            "error": f"Error al ejecutar SQL: {e}",
            "sql_attempted": final_sql,
            "result": [],
            "count": 0,
        }
