#!/usr/bin/env python3
"""Compare 20 teacher questions: ground truth SQL vs Mao agent."""
from __future__ import annotations

import json
import time
import urllib.request
from typing import Any, Dict, List, Tuple

import psycopg2
import psycopg2.extras

DB = dict(host="127.0.0.1", port=5432, dbname="aca_2", user="postgres", password="123")
DB_URL = "postgresql://postgres:123@127.0.0.1:5432/aca_2"
IDDOCENTE = 10
MAO_URL = "http://127.0.0.1:8000/api/v1/chat/teacher/"


def q(sql: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    with psycopg2.connect(**DB) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params or {"iddocente": IDDOCENTE})
            return [dict(r) for r in cur.fetchall()]


def mao(message: str, conv: str) -> Dict[str, Any]:
    payload = {
        "user_id": str(IDDOCENTE),
        "message": message,
        "db_url": DB_URL,
        "conversation_id": conv,
        "context": {"role": "teacher", "iddocente": IDDOCENTE},
        "conversation_history": [],
    }
    req = urllib.request.Request(
        MAO_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        body = ""
        if hasattr(e, "read"):
            try:
                body = e.read().decode()
            except Exception:
                pass
        return {"error": str(e), "body": body, "response": None}


def fmt_rows(rows: List[Dict[str, Any]], limit: int = 8) -> str:
    if not rows:
        return "(sin filas)"
    lines = []
    for r in rows[:limit]:
        lines.append(" | ".join(f"{k}={v}" for k, v in r.items()))
    if len(rows) > limit:
        lines.append(f"... +{len(rows) - limit} más")
    return "\n".join(lines)


# Ground-truth builders
def gt_asignaturas():
    return q(
        """
        SELECT a.nombre AS asignatura, g.nombre AS grado, gg.nombre AS grupo
        FROM acad_gradosasignaturas ga
        JOIN acad_asignaturas a ON a.id=ga.idasignatura
        JOIN acad_gradosgrupos gg ON gg.id=ga.idgrados_grupos
        JOIN acad_grados g ON g.id=gg.idgrado
        WHERE ga.iddocente=%(iddocente)s AND COALESCE(ga.estado,'on')='on'
        ORDER BY 1,2,3
        """
    )


def gt_promedios(filtro_asig: str | None = None, grado: str | None = None, grupo: str | None = None):
    clauses = ["ga.iddocente=%(iddocente)s", "COALESCE(ga.estado,'on')='on'", "an.nota IS NOT NULL"]
    params: Dict[str, Any] = {"iddocente": IDDOCENTE}
    if filtro_asig:
        clauses.append("a.nombre ILIKE %(asig)s")
        params["asig"] = f"%{filtro_asig}%"
    if grado:
        clauses.append("g.nombre ILIKE %(grado)s")
        params["grado"] = f"%{grado}%"
    if grupo:
        clauses.append("gg.nombre ILIKE %(grupo)s")
        params["grupo"] = f"%{grupo}%"
    where = " AND ".join(clauses)
    return q(
        f"""
        SELECT a.nombre AS asignatura, g.nombre AS grado, gg.nombre AS grupo,
               ROUND(AVG(an.nota)::numeric,2) AS promedio,
               COUNT(an.id) AS total_notas,
               COUNT(DISTINCT an.idmatricula) AS estudiantes
        FROM acad_gradosasignaturas ga
        JOIN acad_asignaturas a ON a.id=ga.idasignatura
        JOIN acad_gradosgrupos gg ON gg.id=ga.idgrados_grupos
        JOIN acad_grados g ON g.id=gg.idgrado
        JOIN acad_actividades act ON act.idgrados_asignatura=ga.id
        JOIN acad_actividades_notas an ON an.idactividad=act.id
        WHERE {where}
        GROUP BY 1,2,3 ORDER BY 1,2,3
        """,
        params,
    )


def gt_student(nombre_ilike: str, asig: str | None = None):
    params: Dict[str, Any] = {"iddocente": IDDOCENTE, "nom": f"%{nombre_ilike}%"}
    extra = ""
    if asig:
        extra = " AND a.nombre ILIKE %(asig)s"
        params["asig"] = f"%{asig}%"
    return q(
        f"""
        SELECT TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre, e.primer_apellido, e.segundo_apellido)) AS estudiante,
               a.nombre AS asignatura, g.nombre AS grado, gg.nombre AS grupo,
               ROUND(AVG(an.nota)::numeric,2) AS promedio, COUNT(*) AS total_notas
        FROM acad_gradosasignaturas ga
        JOIN acad_asignaturas a ON a.id=ga.idasignatura
        JOIN acad_gradosgrupos gg ON gg.id=ga.idgrados_grupos
        JOIN acad_grados g ON g.id=gg.idgrado
        JOIN acad_actividades act ON act.idgrados_asignatura=ga.id
        JOIN acad_actividades_notas an ON an.idactividad=act.id
        JOIN acad_matricula_academica ma ON ma.id=an.idmatricula
        JOIN acad_estumatricula em ON em.id=ma.idestumatricula
        JOIN estu_estudiantes e ON e.id=em.idestudiante
        WHERE ga.iddocente=%(iddocente)s
          AND (e.primer_nombre || ' ' || COALESCE(e.segundo_nombre,'') || ' ' || e.primer_apellido || ' ' || COALESCE(e.segundo_apellido,''))
              ILIKE %(nom)s
          {extra}
        GROUP BY 1,2,3,4 ORDER BY 2
        """,
        params,
    )


def gt_attention(threshold: float = 3.5, limit: int = 10):
    return q(
        """
        SELECT TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre, e.primer_apellido, e.segundo_apellido)) AS estudiante,
               a.nombre AS asignatura, ROUND(AVG(an.nota)::numeric,2) AS promedio
        FROM acad_gradosasignaturas ga
        JOIN acad_asignaturas a ON a.id=ga.idasignatura
        JOIN acad_actividades act ON act.idgrados_asignatura=ga.id
        JOIN acad_actividades_notas an ON an.idactividad=act.id
        JOIN acad_matricula_academica ma ON ma.id=an.idmatricula
        JOIN acad_estumatricula em ON em.id=ma.idestumatricula
        JOIN estu_estudiantes e ON e.id=em.idestudiante
        WHERE ga.iddocente=%(iddocente)s AND COALESCE(ga.estado,'on')='on'
        GROUP BY e.id, e.primer_nombre, e.segundo_nombre, e.primer_apellido, e.segundo_apellido, a.nombre
        HAVING AVG(an.nota) < %(th)s
        ORDER BY promedio ASC
        LIMIT %(lim)s
        """,
        {"iddocente": IDDOCENTE, "th": threshold, "lim": limit},
    )


def gt_rank_periodo(periodo_id: int, order: str = "DESC", limit: int = 5):
    return q(
        f"""
        SELECT TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre, e.primer_apellido, e.segundo_apellido)) AS estudiante,
               ROUND(AVG(an.nota)::numeric,2) AS promedio, COUNT(an.id) AS total_notas
        FROM acad_gradosasignaturas ga
        JOIN acad_actividades act ON act.idgrados_asignatura=ga.id
        JOIN acad_actividades_notas an ON an.idactividad=act.id
        JOIN acad_matricula_academica ma ON ma.id=an.idmatricula
        JOIN acad_estumatricula em ON em.id=ma.idestumatricula
        JOIN estu_estudiantes e ON e.id=em.idestudiante
        WHERE ga.iddocente=%(iddocente)s AND COALESCE(ga.estado,'on')='on'
          AND act.idperiodo=%(pid)s AND an.nota IS NOT NULL
        GROUP BY e.id, e.primer_nombre, e.segundo_nombre, e.primer_apellido, e.segundo_apellido
        ORDER BY promedio {order} NULLS LAST
        LIMIT %(lim)s
        """,
        {"iddocente": IDDOCENTE, "pid": periodo_id, "lim": limit},
    )


def gt_top_grupo(asig: str, grado: str, grupo: str, limit: int = 5):
    return q(
        """
        SELECT TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre, e.primer_apellido, e.segundo_apellido)) AS estudiante,
               ROUND(AVG(an.nota)::numeric,2) AS promedio
        FROM acad_gradosasignaturas ga
        JOIN acad_asignaturas a ON a.id=ga.idasignatura
        JOIN acad_gradosgrupos gg ON gg.id=ga.idgrados_grupos
        JOIN acad_grados g ON g.id=gg.idgrado
        JOIN acad_actividades act ON act.idgrados_asignatura=ga.id
        JOIN acad_actividades_notas an ON an.idactividad=act.id
        JOIN acad_matricula_academica ma ON ma.id=an.idmatricula
        JOIN acad_estumatricula em ON em.id=ma.idestumatricula
        JOIN estu_estudiantes e ON e.id=em.idestudiante
        WHERE ga.iddocente=%(iddocente)s AND a.nombre ILIKE %(asig)s
          AND g.nombre ILIKE %(grado)s AND gg.nombre ILIKE %(grupo)s AND an.nota IS NOT NULL
        GROUP BY e.id, e.primer_nombre, e.segundo_nombre, e.primer_apellido, e.segundo_apellido
        ORDER BY promedio DESC LIMIT %(lim)s
        """,
        {
            "iddocente": IDDOCENTE,
            "asig": f"%{asig}%",
            "grado": f"%{grado}%",
            "grupo": f"%{grupo}%",
            "lim": limit,
        },
    )


def gt_global():
    return q(
        """
        SELECT ROUND(AVG(an.nota)::numeric,2) AS promedio_global, COUNT(*) AS total_notas
        FROM acad_gradosasignaturas ga
        JOIN acad_actividades act ON act.idgrados_asignatura=ga.id
        JOIN acad_actividades_notas an ON an.idactividad=act.id
        WHERE ga.iddocente=%(iddocente)s AND an.nota IS NOT NULL
        """
    )


def gt_attendance():
    return q(
        """
        SELECT a.nombre AS asignatura, g.nombre AS grado, gg.nombre AS grupo,
               SUM(CASE WHEN sa.attended THEN 1 ELSE 0 END) AS presentes,
               SUM(CASE WHEN sa.not_attended THEN 1 ELSE 0 END) AS ausentes
        FROM student_attendance sa
        JOIN acad_gradosasignaturas ga
          ON ga.idgrados_grupos=sa.course_id AND ga.idasignatura=sa.subject_id
        JOIN acad_asignaturas a ON a.id=ga.idasignatura
        JOIN acad_gradosgrupos gg ON gg.id=ga.idgrados_grupos
        JOIN acad_grados g ON g.id=gg.idgrado
        WHERE ga.iddocente=%(iddocente)s AND COALESCE(ga.estado,'on')='on'
        GROUP BY 1,2,3 ORDER BY 1,2,3
        """
    )


def gt_pdm():
    return q(
        """
        SELECT COUNT(*) AS total_pdm
        FROM acad_pdm_estudiantes pe
        JOIN acad_pdm_ensenanzas_propositos ep ON ep.id = pe.idproposito
        JOIN acad_pdm_ensenanzas en ON en.id = ep.idensenanza
        JOIN acad_gradosasignaturas ga ON ga.id = en.idgradoasignatura
        WHERE ga.iddocente=%(iddocente)s
        """
    )


def gt_periodos():
    return q(
        """
        SELECT DISTINCT p.id AS idperiodo, p.nombre
        FROM acad_periodos p
        JOIN acad_actividades act ON act.idperiodo=p.id
        JOIN acad_gradosasignaturas ga ON ga.id=act.idgrados_asignatura
        WHERE ga.iddocente=%(iddocente)s
        ORDER BY p.id
        """
    )


QUESTIONS: List[Tuple[str, Any]] = [
    ("¿Cuáles son mis asignaturas?", gt_asignaturas),
    ("Dame un resumen del avance de mis grupos", lambda: gt_promedios()),
    ("¿Cómo van las notas de Educación Artística?", lambda: gt_promedios("Artística")),
    ("¿Cuál es el promedio de Religión en 6°A?", lambda: gt_promedios("Religión", "6", "6°A")),
    ("¿Cómo va Ética & Valores en 7°A?", lambda: gt_promedios("Ética", "7", "7°A")),
    ("Muéstrame la asistencia de mis grupos", gt_attendance),
    ("¿Qué estudiantes necesitan atención?", lambda: gt_attention(3.5, 10)),
    ("¿Cómo le va a AARON ELIAS GALEANO?", lambda: gt_student("AARON%GALEANO")),
    ("¿Cómo le va a ANA SOFIA DONADO en Religión?", lambda: gt_student("ANA SOFIA%DONADO", "Religión")),
    ("Lista mis planes de mejoramiento (PDM)", gt_pdm),
    ("¿Quién es el mejor estudiante en el global del Periodo 3?", lambda: gt_rank_periodo(4, "DESC", 5)),
    ("¿Quiénes son los 5 estudiantes con peor promedio en el Periodo 3?", lambda: gt_rank_periodo(4, "ASC", 5)),
    ("Top 5 estudiantes de Educación Artística en 6°A", lambda: gt_top_grupo("Artística", "6", "6°A", 5)),
    ("¿Cuántos estudiantes con nota tengo en Religión 11°A?", lambda: gt_promedios("Religión", "11", "11°A")),
    ("¿Cuál es el promedio global de todas mis notas?", gt_global),
    ("¿Cuántas asignaturas/grupos dicto?", lambda: [{"total": len(gt_asignaturas())}]),
    ("¿Qué periodos académicos aparecen en mis actividades?", gt_periodos),
    ("Estudiantes con promedio bajo 3.0 en mis materias", lambda: gt_attention(3.0, 10)),
    ("¿Cómo le va a HALLYS ANDREA SALGADO?", lambda: gt_student("HALLYS%SALGADO")),
    ("Promedio de Educación Artística en 8°A", lambda: gt_promedios("Artística", "8", "8°A")),
]


def main():
    results = []
    print(f"Docente id={IDDOCENTE} | Mao={MAO_URL}\n")
    for i, (question, gt_fn) in enumerate(QUESTIONS, 1):
        conv = f"mao_test_20_{i}_{int(time.time())}"
        print(f"\n===== Q{i}/20 =====")
        print(f"PREGUNTA: {question}")
        try:
            truth = gt_fn()
        except Exception as e:
            truth = [{"error": str(e)}]
        print("--- YO (DB) ---")
        print(fmt_rows(truth if isinstance(truth, list) else [truth]))

        t0 = time.time()
        resp = mao(question, conv)
        elapsed = time.time() - t0
        answer = resp.get("response") if isinstance(resp, dict) else None
        tools = resp.get("function_calls_made") if isinstance(resp, dict) else None
        print(f"--- MAO ({elapsed:.1f}s) tools={tools} ---")
        if answer:
            print(answer[:1200] + ("…" if len(answer) > 1200 else ""))
        else:
            print(json.dumps(resp, ensure_ascii=False)[:800])

        results.append(
            {
                "n": i,
                "question": question,
                "ground_truth": truth,
                "mao_response": answer,
                "mao_tools": tools,
                "mao_error": resp.get("error") if isinstance(resp, dict) else None,
                "seconds": round(elapsed, 2),
            }
        )

    out = "/tmp/mao_compare_20.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nGuardado {out}")


if __name__ == "__main__":
    main()
