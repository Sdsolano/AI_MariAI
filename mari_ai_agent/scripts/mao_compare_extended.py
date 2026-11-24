#!/usr/bin/env python3
"""Extended DB vs Mao comparison test."""
from __future__ import annotations

import json
import re
import time
import urllib.request
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras

DB = dict(host="127.0.0.1", port=5432, dbname="aca_2", user="postgres", password="123")
DB_URL = "postgresql://postgres:123@127.0.0.1:5432/aca_2"
IDDOCENTE = 10
MAO_URL = "http://127.0.0.1:8000/api/v1/chat/teacher/"


def q(sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    with psycopg2.connect(**DB) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params or {"iddocente": IDDOCENTE})
            rows = []
            for r in cur.fetchall():
                d = dict(r)
                for k, v in list(d.items()):
                    if isinstance(v, Decimal):
                        d[k] = float(v)
                rows.append(d)
            return rows


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


def fmt_rows(rows: List[Dict[str, Any]], limit: int = 6) -> str:
    if not rows:
        return "(sin filas)"
    lines = []
    for r in rows[:limit]:
        lines.append(" | ".join(f"{k}={v}" for k, v in r.items()))
    if len(rows) > limit:
        lines.append(f"... +{len(rows) - limit} más")
    return "\n".join(lines)


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


def gt_promedios(asig=None, grado=None, grupo=None):
    clauses = ["ga.iddocente=%(iddocente)s", "COALESCE(ga.estado,'on')='on'", "an.nota IS NOT NULL"]
    params: Dict[str, Any] = {"iddocente": IDDOCENTE}
    if asig:
        clauses.append("a.nombre ILIKE %(asig)s")
        params["asig"] = f"%{asig}%"
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


def gt_student(nombre_ilike: str, asig=None):
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


def gt_student_period(nombre_ilike: str, idperiodo: int):
    return q(
        """
        SELECT TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre, e.primer_apellido, e.segundo_apellido)) AS estudiante,
               a.nombre AS asignatura, p.nombre AS periodo,
               ROUND(AVG(an.nota)::numeric,2) AS promedio, COUNT(*) AS total_notas
        FROM acad_gradosasignaturas ga
        JOIN acad_asignaturas a ON a.id=ga.idasignatura
        JOIN acad_actividades act ON act.idgrados_asignatura=ga.id
        JOIN acad_periodos p ON p.id=act.idperiodo
        JOIN acad_actividades_notas an ON an.idactividad=act.id
        JOIN acad_matricula_academica ma ON ma.id=an.idmatricula
        JOIN acad_estumatricula em ON em.id=ma.idestumatricula
        JOIN estu_estudiantes e ON e.id=em.idestudiante
        WHERE ga.iddocente=%(iddocente)s AND act.idperiodo=%(pid)s AND an.nota IS NOT NULL
          AND (e.primer_nombre || ' ' || COALESCE(e.segundo_nombre,'') || ' ' || e.primer_apellido || ' ' || COALESCE(e.segundo_apellido,''))
              ILIKE %(nom)s
        GROUP BY 1,2,3 ORDER BY 2
        """,
        {"iddocente": IDDOCENTE, "pid": idperiodo, "nom": f"%{nombre_ilike}%"},
    )


def gt_attention(threshold=3.5, limit=10):
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
        ORDER BY promedio ASC LIMIT %(lim)s
        """,
        {"iddocente": IDDOCENTE, "th": threshold, "lim": limit},
    )


def gt_rank_periodo(pid: int, order="DESC", limit=5):
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
        ORDER BY promedio {order} NULLS LAST LIMIT %(lim)s
        """,
        {"iddocente": IDDOCENTE, "pid": pid, "lim": limit},
    )


def gt_top_grupo(asig, grado, grupo, limit=5):
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


def gt_best_per_subject():
    return q(
        """
        WITH ranked AS (
          SELECT a.nombre AS asignatura, g.nombre AS grado, gg.nombre AS grupo,
            TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre, e.primer_apellido, e.segundo_apellido)) AS estudiante,
            ROUND(AVG(an.nota)::numeric,2) AS promedio,
            ROW_NUMBER() OVER (PARTITION BY a.nombre, g.nombre, gg.nombre ORDER BY AVG(an.nota) DESC NULLS LAST) AS rn
          FROM acad_gradosasignaturas ga
          JOIN acad_asignaturas a ON a.id=ga.idasignatura
          JOIN acad_gradosgrupos gg ON gg.id=ga.idgrados_grupos
          JOIN acad_grados g ON g.id=gg.idgrado
          JOIN acad_actividades act ON act.idgrados_asignatura=ga.id
          JOIN acad_actividades_notas an ON an.idactividad=act.id
          JOIN acad_matricula_academica ma ON ma.id=an.idmatricula
          JOIN acad_estumatricula em ON em.id=ma.idestumatricula
          JOIN estu_estudiantes e ON e.id=em.idestudiante
          WHERE ga.iddocente=%(iddocente)s AND COALESCE(ga.estado,'on')='on' AND an.nota IS NOT NULL
          GROUP BY a.nombre, g.nombre, gg.nombre, e.id, e.primer_nombre, e.segundo_nombre, e.primer_apellido, e.segundo_apellido
        )
        SELECT asignatura, grado, grupo, estudiante, promedio FROM ranked WHERE rn=1
        ORDER BY 1,2,3
        """
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
        SELECT COUNT(*) AS registros
        FROM student_attendance sa
        JOIN acad_gradosasignaturas ga
          ON ga.idgrados_grupos=sa.course_id AND ga.idasignatura=sa.subject_id
        WHERE ga.iddocente=%(iddocente)s
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
        WHERE ga.iddocente=%(iddocente)s ORDER BY p.id
        """
    )


QUESTIONS: List[Tuple[str, Callable]] = [
    # Snapshot / listados
    ("¿Cuáles son mis asignaturas?", gt_asignaturas),
    ("¿Cuántas asignaturas/grupos dicto?", lambda: [{"total": len(gt_asignaturas())}]),
    ("Dame un resumen del avance de mis grupos", lambda: gt_promedios()),
    ("¿Cómo van las notas de Educación Artística?", lambda: gt_promedios("Artística")),
    ("¿Cómo van las notas de Religión?", lambda: gt_promedios("Religión")),
    ("¿Cómo va Ética & Valores?", lambda: gt_promedios("Ética")),
    ("¿Cuál es el promedio de Religión en 6°A?", lambda: gt_promedios("Religión", "6", "6°A")),
    ("¿Cómo va Ética & Valores en 7°A?", lambda: gt_promedios("Ética", "7", "7°A")),
    ("Promedio de Educación Artística en 8°A", lambda: gt_promedios("Artística", "8", "8°A")),
    ("¿Cuál es el promedio de Educación Artística en 11°A?", lambda: gt_promedios("Artística", "11", "11°A")),
    ("Promedio de Religión en 9°A", lambda: gt_promedios("Religión", "9", "9°A")),
    ("¿Cómo va Educación Artística en 6°B?", lambda: gt_promedios("Artística", "6", "6°B")),
    # Asistencia / PDM / periodos / global
    ("Muéstrame la asistencia de mis grupos", gt_attendance),
    ("Lista mis planes de mejoramiento (PDM)", gt_pdm),
    ("¿Qué periodos académicos aparecen en mis actividades?", gt_periodos),
    ("¿Cuál es el promedio global de todas mis notas?", gt_global),
    # Atención
    ("¿Qué estudiantes necesitan atención?", lambda: gt_attention(3.5, 10)),
    ("Estudiantes con promedio bajo 3.0 en mis materias", lambda: gt_attention(3.0, 10)),
    ("Estudiantes con promedio bajo 2.5", lambda: gt_attention(2.5, 8)),
    # Estudiante concreto
    ("¿Cómo le va a AARON ELIAS GALEANO?", lambda: gt_student("AARON%GALEANO")),
    ("¿Cómo le va a ANA SOFIA DONADO en Religión?", lambda: gt_student("ANA SOFIA%DONADO", "Religión")),
    ("¿Cómo le va a HALLYS ANDREA SALGADO?", lambda: gt_student("HALLYS%SALGADO")),
    ("¿Cómo le va a VALERY SOFIA JULIAO?", lambda: gt_student("VALERY%JULIAO")),
    ("¿Cómo le va a BIANCA VALENTINA SANCHEZ?", lambda: gt_student("BIANCA%SANCHEZ")),
    # Estudiante + periodo
    ("¿Cómo le fue a AARON ELIAS GALEANO en el primer periodo?", lambda: gt_student_period("AARON%GALEANO", 1)),
    ("¿Cómo le va a AARON ELIAS GALEANO en el Periodo 3?", lambda: gt_student_period("AARON%GALEANO", 4)),
    ("¿Cómo le fue a VALERY SOFIA JULIAO en el segundo periodo?", lambda: gt_student_period("VALERY%JULIAO", 3)),
    ("¿Cómo le fue a HALLYS ANDREA SALGADO en el Periodo 1?", lambda: gt_student_period("HALLYS%SALGADO", 1)),
    # Rankings periodo
    ("¿Quién es el mejor estudiante en el global del Periodo 3?", lambda: gt_rank_periodo(4, "DESC", 5)),
    ("¿Quiénes son los 5 estudiantes con peor promedio en el Periodo 3?", lambda: gt_rank_periodo(4, "ASC", 5)),
    ("Mejor estudiante del Periodo 1", lambda: gt_rank_periodo(1, "DESC", 5)),
    ("Peores 5 del Periodo 2", lambda: gt_rank_periodo(3, "ASC", 5)),
    # Top por grupo
    ("Top 5 estudiantes de Educación Artística en 6°A", lambda: gt_top_grupo("Artística", "6", "6°A", 5)),
    ("Top 3 de Religión en 8°A", lambda: gt_top_grupo("Religión", "8", "8°A", 3)),
    ("Top 5 de Ética & Valores en 7°A", lambda: gt_top_grupo("Ética", "7", "7°A", 5)),
    # Mejor por cada asignatura
    ("¿Cuál es el mejor estudiante de cada asignatura?", gt_best_per_subject),
    ("Mejor estudiante por cada asignatura", gt_best_per_subject),
    # Conteo
    ("¿Cuántos estudiantes con nota tengo en Religión 11°A?", lambda: gt_promedios("Religión", "11", "11°A")),
    ("¿Cuántos estudiantes con nota en Educación Artística 10°A?", lambda: gt_promedios("Artística", "10", "10°A")),
]


def score(question: str, truth: Any, answer: Optional[str]) -> str:
    """Rough automatic verdict."""
    if not answer:
        return "mal"
    a = answer.lower()
    if "error" in (truth[0] if isinstance(truth, list) and truth else {}):
        return "parcial"

    # Extract key numbers from truth
    nums = []
    names = []
    if isinstance(truth, list):
        for r in truth[:5]:
            if not isinstance(r, dict):
                continue
            for k, v in r.items():
                if isinstance(v, (int, float)) and k != "idperiodo":
                    nums.append(str(v))
                    # also without trailing .0
                    if isinstance(v, float) and v == int(v):
                        nums.append(str(int(v)))
                if k in ("estudiante", "asignatura") and v:
                    # first meaningful token of name
                    tok = str(v).split()
                    if tok:
                        names.append(tok[0].lower())

    # Empty truth cases
    if isinstance(truth, list) and (
        not truth
        or (len(truth) == 1 and truth[0].get("registros") == 0)
        or (len(truth) == 1 and truth[0].get("total_pdm") == 0)
    ):
        if any(x in a for x in ("no aparece", "no encontr", "0", "sin ")):
            return "bien"
        return "parcial"

    hit_nums = sum(1 for n in set(nums) if n in a or n.replace(".", ",") in a)
    hit_names = sum(1 for n in set(names[:8]) if n in a)

    if "mejor estudiante de cada" in question.lower() or "por cada asignatura" in question.lower():
        # need several subject lines
        if hit_names >= 3 and ("artística" in a or "artistica" in a) and "relig" in a:
            return "bien"
        return "parcial" if hit_names >= 1 else "mal"

    if hit_nums >= 1 and (hit_names >= 1 or "promedio" in a or "dictas" in a or "periodo" in a):
        return "bien"
    if hit_nums >= 1 or hit_names >= 1:
        return "parcial"
    if any(x in a for x in ("no aparece", "no encontr", "no pude")) and nums:
        return "mal"
    return "parcial"


def main():
    results = []
    counts = {"bien": 0, "parcial": 0, "mal": 0}
    print(f"Docente id={IDDOCENTE} | {len(QUESTIONS)} preguntas\n")

    for i, (question, gt_fn) in enumerate(QUESTIONS, 1):
        conv = f"mao_ext_{i}_{int(time.time())}"
        print(f"\n===== Q{i}/{len(QUESTIONS)} =====")
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
        verdict = score(question, truth, answer)
        counts[verdict] = counts.get(verdict, 0) + 1
        print(f"--- MAO ({elapsed:.1f}s) tools={tools} → {verdict.upper()} ---")
        print((answer or json.dumps(resp, ensure_ascii=False))[:900])

        results.append(
            {
                "n": i,
                "question": question,
                "verdict": verdict,
                "ground_truth": truth,
                "mao_response": answer,
                "mao_tools": tools,
                "seconds": round(elapsed, 2),
                "mao_error": resp.get("error") if isinstance(resp, dict) else None,
            }
        )

    out = "/tmp/mao_compare_extended.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"counts": counts, "results": results}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n===== RESUMEN =====")
    print(counts)
    print(f"Guardado {out}")


if __name__ == "__main__":
    main()
