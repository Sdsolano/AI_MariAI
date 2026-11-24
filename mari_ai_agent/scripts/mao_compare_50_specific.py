#!/usr/bin/env python3
"""
50+ preguntas específicas (actividad/periodo/materia/estudiante) + set previo.
Compara ground-truth SQL vs Mao actual vs multiagente.
iddocente=3 (docente con Matemáticas/Geometría/Ed. Física en aca_2).
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
import urllib.request
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras

DB = dict(host="127.0.0.1", port=5432, dbname="aca_2", user="postgres", password="123")
DB_URL = "postgresql://postgres:123@127.0.0.1:5432/aca_2"
IDDOCENTE = 3
MAO_URL = "http://127.0.0.1:8000/api/v1/chat/teacher/"
MULTI_URL = "http://127.0.0.1:8000/api/v1/chat/teacher/multiagent"
OUT = Path("/home/d/Documents/GitHub/EduApp/AI_MariAI/mari_ai_agent/scripts/mao_compare_50_report.json")


def fold(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(s or ""))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


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


def post(url: str, payload: Dict[str, Any], timeout: int = 180) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        data["_elapsed_client_ms"] = int((time.time() - t0) * 1000)
        return data
    except Exception as e:
        body = ""
        if hasattr(e, "read"):
            try:
                body = e.read().decode()
            except Exception:
                pass
        return {
            "error": str(e),
            "body": body[:500],
            "response": None,
            "_elapsed_client_ms": int((time.time() - t0) * 1000),
        }


def extract_nums(text: str) -> List[float]:
    found = []
    for m in re.finditer(r"\b(\d+(?:[.,]\d+)?)\b", text or ""):
        try:
            found.append(float(m.group(1).replace(",", ".")))
        except ValueError:
            pass
    return found


def score_answer(answer: str, expect: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns (verdict, detail) where verdict in bien|parcial|mal|error
    expect keys: kind, value (float), name (str), names (list), alt_values
    """
    if not answer:
        return "error", "sin respuesta"
    a = fold(answer)
    kind = expect.get("kind")

    if kind == "number":
        target = float(expect["value"])
        nums = extract_nums(answer)
        # tolerate 2.5 vs 2.50
        hit = any(abs(n - target) < 0.051 for n in nums)
        name = expect.get("name")
        name_ok = True
        if name:
            tokens = [t for t in fold(name).split() if len(t) > 2][:3]
            name_ok = all(t in a for t in tokens) if tokens else True
        if hit and name_ok:
            return "bien", f"número {target}" + (f" + nombre" if name else "")
        if hit or name_ok:
            return "parcial", f"hit_num={hit} hit_name={name_ok} target={target}"
        return "mal", f"esperaba {target}" + (f" / {name}" if name else "")

    if kind == "name":
        name = expect.get("name") or ""
        tokens = [t for t in fold(name).split() if len(t) > 2][:3]
        if tokens and all(t in a for t in tokens):
            return "bien", f"nombre {name}"
        # partial: apellido
        if tokens and any(t in a for t in tokens[-2:]):
            return "parcial", f"parcial {name}"
        return "mal", f"esperaba {name}"

    if kind == "contains_any":
        names = expect.get("names") or []
        hits = 0
        for n in names:
            toks = [t for t in fold(n).split() if len(t) > 2][:2]
            if toks and all(t in a for t in toks):
                hits += 1
        if hits >= max(1, len(names) // 2):
            return "bien", f"{hits}/{len(names)} nombres"
        if hits:
            return "parcial", f"{hits}/{len(names)} nombres"
        return "mal", "ningún nombre esperado"

    if kind == "text_has":
        needles = expect.get("needles") or []
        hits = sum(1 for n in needles if fold(n) in a)
        if hits == len(needles):
            return "bien", "todas las claves"
        if hits:
            return "parcial", f"{hits}/{len(needles)}"
        return "mal", f"faltan {needles}"

    return "parcial", "sin scorer"


def build_questions() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    def add(qid: str, question: str, expect: Dict[str, Any], tag: str):
        items.append({"id": qid, "q": question, "expect": expect, "tag": tag})

    # ---- Actividades: mejor/peor nota ----
    acts = q(
        """
        SELECT a.nombre AS asignatura,
               p.nombre AS periodo,
               act.idperiodo,
               act.id AS idactividad,
               COALESCE(NULLIF(TRIM(act.descripcion), ''), 'Actividad') AS actividad,
               ROUND(MAX(an.nota)::numeric, 2) AS max_nota,
               ROUND(MIN(an.nota)::numeric, 2) AS min_nota,
               ROUND(AVG(an.nota)::numeric, 2) AS avg_nota,
               COUNT(an.id)::int AS n
        FROM acad_gradosasignaturas ga
        JOIN acad_asignaturas a ON a.id = ga.idasignatura
        JOIN acad_actividades act ON act.idgrados_asignatura = ga.id
        LEFT JOIN acad_periodos p ON p.id = act.idperiodo
        JOIN acad_actividades_notas an ON an.idactividad = act.id
        WHERE ga.iddocente = %(iddocente)s
          AND COALESCE(ga.estado,'on')='on'
          AND an.nota IS NOT NULL
          AND LENGTH(COALESCE(NULLIF(TRIM(act.descripcion), ''), '')) >= 8
        GROUP BY 1,2,3,4,5
        HAVING COUNT(an.id) >= 10
        ORDER BY a.nombre, act.idperiodo, COUNT(an.id) DESC
        """
    )

    # Deduplicate by (asig, periodo, actividad lower)
    seen = set()
    uniq_acts = []
    for r in acts:
        key = (fold(r["asignatura"]), r["idperiodo"], fold(r["actividad"])[:40])
        if key in seen:
            continue
        seen.add(key)
        uniq_acts.append(r)

    for i, r in enumerate(uniq_acts[:18]):
        act = r["actividad"]
        asig = r["asignatura"]
        per = r["periodo"] or f"periodo {r['idperiodo']}"
        # quién sacó la mejor nota en esa actividad
        top = q(
            """
            SELECT TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre,
                                   e.primer_apellido, e.segundo_apellido)) AS est,
                   an.nota
            FROM acad_actividades_notas an
            JOIN acad_matricula_academica ma ON ma.id = an.idmatricula
            JOIN acad_estumatricula em ON em.id = ma.idestumatricula
            JOIN estu_estudiantes e ON e.id = em.idestudiante
            WHERE an.idactividad = %(aid)s AND an.nota IS NOT NULL
            ORDER BY an.nota DESC NULLS LAST
            LIMIT 1
            """,
            {"aid": r["idactividad"]},
        )
        worst = q(
            """
            SELECT TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre,
                                   e.primer_apellido, e.segundo_apellido)) AS est,
                   an.nota
            FROM acad_actividades_notas an
            JOIN acad_matricula_academica ma ON ma.id = an.idmatricula
            JOIN acad_estumatricula em ON em.id = ma.idestumatricula
            JOIN estu_estudiantes e ON e.id = em.idestudiante
            WHERE an.idactividad = %(aid)s AND an.nota IS NOT NULL
            ORDER BY an.nota ASC NULLS LAST
            LIMIT 1
            """,
            {"aid": r["idactividad"]},
        )
        if top:
            add(
                f"act-max-{i}",
                f"¿Cuál fue la mejor nota de la actividad «{act}» de {asig} en {per}?",
                {
                    "kind": "number",
                    "value": float(top[0]["nota"]),
                    "name": top[0]["est"],
                },
                "actividad_max",
            )
        if worst:
            add(
                f"act-min-{i}",
                f"¿Cuál fue la peor nota de la actividad «{act}» de {asig} en {per}?",
                {
                    "kind": "number",
                    "value": float(worst[0]["nota"]),
                    "name": worst[0]["est"],
                },
                "actividad_min",
            )
        add(
            f"act-avg-{i}",
            f"¿Cuál es el promedio del grupo en la actividad «{act}» ({asig}, {per})?",
            {"kind": "number", "value": float(r["avg_nota"])},
            "actividad_avg",
        )

    # ---- Rankings por asignatura + periodo ----
    pairs = q(
        """
        SELECT DISTINCT a.nombre AS asignatura, p.nombre AS periodo, act.idperiodo
        FROM acad_gradosasignaturas ga
        JOIN acad_asignaturas a ON a.id = ga.idasignatura
        JOIN acad_actividades act ON act.idgrados_asignatura = ga.id
        JOIN acad_periodos p ON p.id = act.idperiodo
        JOIN acad_actividades_notas an ON an.idactividad = act.id
        WHERE ga.iddocente = %(iddocente)s AND an.nota IS NOT NULL
        ORDER BY 1, 3
        """
    )
    for j, p in enumerate(pairs):
        rows = q(
            """
            SELECT TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre,
                                   e.primer_apellido, e.segundo_apellido)) AS est,
                   ROUND(AVG(an.nota)::numeric, 2) AS promedio,
                   COUNT(*)::int AS n
            FROM acad_gradosasignaturas ga
            JOIN acad_asignaturas a ON a.id = ga.idasignatura
            JOIN acad_actividades act ON act.idgrados_asignatura = ga.id
            JOIN acad_actividades_notas an ON an.idactividad = act.id
            JOIN acad_matricula_academica ma ON ma.id = an.idmatricula
            JOIN acad_estumatricula em ON em.id = ma.idestumatricula
            JOIN estu_estudiantes e ON e.id = em.idestudiante
            WHERE ga.iddocente = %(iddocente)s
              AND COALESCE(ga.estado,'on')='on'
              AND an.nota IS NOT NULL
              AND a.nombre ILIKE %(asig)s
              AND act.idperiodo = %(pid)s
            GROUP BY e.id, e.primer_nombre, e.segundo_nombre, e.primer_apellido, e.segundo_apellido
            HAVING COUNT(*) >= 2
            ORDER BY promedio DESC
            LIMIT 1
            """,
            {"iddocente": IDDOCENTE, "asig": p["asignatura"], "pid": p["idperiodo"]},
        )
        rows_w = q(
            """
            SELECT TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre,
                                   e.primer_apellido, e.segundo_apellido)) AS est,
                   ROUND(AVG(an.nota)::numeric, 2) AS promedio
            FROM acad_gradosasignaturas ga
            JOIN acad_asignaturas a ON a.id = ga.idasignatura
            JOIN acad_actividades act ON act.idgrados_asignatura = ga.id
            JOIN acad_actividades_notas an ON an.idactividad = act.id
            JOIN acad_matricula_academica ma ON ma.id = an.idmatricula
            JOIN acad_estumatricula em ON em.id = ma.idestumatricula
            JOIN estu_estudiantes e ON e.id = em.idestudiante
            WHERE ga.iddocente = %(iddocente)s
              AND COALESCE(ga.estado,'on')='on'
              AND an.nota IS NOT NULL
              AND a.nombre ILIKE %(asig)s
              AND act.idperiodo = %(pid)s
            GROUP BY e.id, e.primer_nombre, e.segundo_nombre, e.primer_apellido, e.segundo_apellido
            HAVING COUNT(*) >= 2
            ORDER BY promedio ASC
            LIMIT 1
            """,
            {"iddocente": IDDOCENTE, "asig": p["asignatura"], "pid": p["idperiodo"]},
        )
        if rows:
            add(
                f"rank-best-{j}",
                f"¿Quién fue el mejor estudiante de {p['asignatura']} en el {p['periodo']}?",
                {
                    "kind": "number",
                    "value": float(rows[0]["promedio"]),
                    "name": rows[0]["est"],
                },
                "ranking_mejor",
            )
        if rows_w:
            add(
                f"rank-worst-{j}",
                f"¿Quién fue el peor estudiante de {p['asignatura']} en el {p['periodo']}?",
                {
                    "kind": "number",
                    "value": float(rows_w[0]["promedio"]),
                    "name": rows_w[0]["est"],
                },
                "ranking_peor",
            )

    # ---- Estudiantes concretos: promedio / mejor nota ----
    students = q(
        """
        SELECT DISTINCT TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre,
                                        e.primer_apellido, e.segundo_apellido)) AS est,
               e.primer_nombre, e.primer_apellido
        FROM acad_gradosasignaturas ga
        JOIN acad_actividades act ON act.idgrados_asignatura = ga.id
        JOIN acad_actividades_notas an ON an.idactividad = act.id
        JOIN acad_matricula_academica ma ON ma.id = an.idmatricula
        JOIN acad_estumatricula em ON em.id = ma.idestumatricula
        JOIN estu_estudiantes e ON e.id = em.idestudiante
        WHERE ga.iddocente = %(iddocente)s AND an.nota IS NOT NULL
        ORDER BY 1
        LIMIT 8
        """
    )
    asigs = [r["asignatura"] for r in q(
        """
        SELECT DISTINCT a.nombre AS asignatura
        FROM acad_gradosasignaturas ga
        JOIN acad_asignaturas a ON a.id = ga.idasignatura
        WHERE ga.iddocente = %(iddocente)s
        ORDER BY 1
        """
    )]

    for k, st in enumerate(students[:6]):
        short = f"{st['primer_nombre']} {st['primer_apellido']}".strip()
        for asig in asigs[:3]:
            best = q(
                """
                SELECT an.nota,
                       COALESCE(NULLIF(TRIM(act.descripcion), ''), 'Actividad') AS actividad,
                       p.nombre AS periodo
                FROM acad_gradosasignaturas ga
                JOIN acad_asignaturas a ON a.id = ga.idasignatura
                JOIN acad_actividades act ON act.idgrados_asignatura = ga.id
                LEFT JOIN acad_periodos p ON p.id = act.idperiodo
                JOIN acad_actividades_notas an ON an.idactividad = act.id
                JOIN acad_matricula_academica ma ON ma.id = an.idmatricula
                JOIN acad_estumatricula em ON em.id = ma.idestumatricula
                JOIN estu_estudiantes e ON e.id = em.idestudiante
                WHERE ga.iddocente = %(iddocente)s
                  AND a.nombre ILIKE %(asig)s
                  AND an.nota IS NOT NULL
                  AND TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre,
                                      e.primer_apellido, e.segundo_apellido))
                      ILIKE %(nom)s
                ORDER BY an.nota DESC NULLS LAST
                LIMIT 1
                """,
                {
                    "iddocente": IDDOCENTE,
                    "asig": asig,
                    "nom": f"%{st['primer_nombre']}%{st['primer_apellido']}%",
                },
            )
            if best:
                add(
                    f"stu-best-{k}-{fold(asig)[:6]}",
                    f"¿Cuál es la mejor nota de {short} en {asig}?",
                    {
                        "kind": "number",
                        "value": float(best[0]["nota"]),
                        "name": short,
                    },
                    "estudiante_mejor_nota",
                )

        avg_p = q(
            """
            SELECT ROUND(AVG(an.nota)::numeric, 2) AS promedio, COUNT(*)::int AS n
            FROM acad_gradosasignaturas ga
            JOIN acad_actividades act ON act.idgrados_asignatura = ga.id
            JOIN acad_actividades_notas an ON an.idactividad = act.id
            JOIN acad_matricula_academica ma ON ma.id = an.idmatricula
            JOIN acad_estumatricula em ON em.id = ma.idestumatricula
            JOIN estu_estudiantes e ON e.id = em.idestudiante
            WHERE ga.iddocente = %(iddocente)s
              AND act.idperiodo = 3
              AND an.nota IS NOT NULL
              AND TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre,
                                  e.primer_apellido, e.segundo_apellido))
                  ILIKE %(nom)s
            """,
            {
                "iddocente": IDDOCENTE,
                "nom": f"%{st['primer_nombre']}%{st['primer_apellido']}%",
            },
        )
        if avg_p and avg_p[0].get("promedio") is not None:
            add(
                f"stu-avg-p2-{k}",
                f"¿Cómo le fue a {short} en el segundo periodo (promedio global en mis cursos)?",
                {"kind": "number", "value": float(avg_p[0]["promedio"]), "name": short},
                "estudiante_periodo",
            )

    # ---- Previas / clásicas ----
    classic = [
        (
            "classic-peor-mat-p2",
            "cuál fue el peor estudiante del segundo periodo en matemáticas",
            None,
        ),
        (
            "classic-mejor-mat-p2",
            "quién fue el mejor estudiante de Matemáticas en el periodo 2",
            None,
        ),
        (
            "classic-mejor-p1",
            "mejor estudiante del periodo 1",
            None,
        ),
        (
            "classic-peor-p2-global",
            "peor estudiante del segundo periodo",
            None,
        ),
        (
            "classic-global",
            "promedio global de todas mis notas",
            None,
        ),
        (
            "classic-atencion",
            "qué estudiantes necesitan atención bajo 3.0",
            None,
        ),
        (
            "classic-mejor-cada",
            "mejor estudiante de cada asignatura",
            None,
        ),
        (
            "classic-asignaturas",
            "qué asignaturas dicto",
            None,
        ),
    ]

    # fill classic expects from SQL
    peor_mat = q(
        """
        SELECT TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre,
                               e.primer_apellido, e.segundo_apellido)) AS est,
               ROUND(AVG(an.nota)::numeric, 2) AS promedio
        FROM acad_gradosasignaturas ga
        JOIN acad_asignaturas a ON a.id = ga.idasignatura
        JOIN acad_actividades act ON act.idgrados_asignatura = ga.id
        JOIN acad_actividades_notas an ON an.idactividad = act.id
        JOIN acad_matricula_academica ma ON ma.id = an.idmatricula
        JOIN acad_estumatricula em ON em.id = ma.idestumatricula
        JOIN estu_estudiantes e ON e.id = em.idestudiante
        WHERE ga.iddocente=%(iddocente)s AND a.nombre ILIKE '%%matem%%'
          AND act.idperiodo=3 AND an.nota IS NOT NULL
        GROUP BY 1 ORDER BY promedio ASC LIMIT 1
        """
    )
    mejor_mat = q(
        """
        SELECT TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre,
                               e.primer_apellido, e.segundo_apellido)) AS est,
               ROUND(AVG(an.nota)::numeric, 2) AS promedio
        FROM acad_gradosasignaturas ga
        JOIN acad_asignaturas a ON a.id = ga.idasignatura
        JOIN acad_actividades act ON act.idgrados_asignatura = ga.id
        JOIN acad_actividades_notas an ON an.idactividad = act.id
        JOIN acad_matricula_academica ma ON ma.id = an.idmatricula
        JOIN acad_estumatricula em ON em.id = ma.idestumatricula
        JOIN estu_estudiantes e ON e.id = em.idestudiante
        WHERE ga.iddocente=%(iddocente)s AND a.nombre ILIKE '%%matem%%'
          AND act.idperiodo=3 AND an.nota IS NOT NULL
        GROUP BY 1 ORDER BY promedio DESC LIMIT 1
        """
    )
    mejor_p1 = q(
        """
        SELECT TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre,
                               e.primer_apellido, e.segundo_apellido)) AS est,
               ROUND(AVG(an.nota)::numeric, 2) AS promedio
        FROM acad_gradosasignaturas ga
        JOIN acad_actividades act ON act.idgrados_asignatura = ga.id
        JOIN acad_actividades_notas an ON an.idactividad = act.id
        JOIN acad_matricula_academica ma ON ma.id = an.idmatricula
        JOIN acad_estumatricula em ON em.id = ma.idestumatricula
        JOIN estu_estudiantes e ON e.id = em.idestudiante
        WHERE ga.iddocente=%(iddocente)s AND act.idperiodo=1 AND an.nota IS NOT NULL
        GROUP BY 1 ORDER BY promedio DESC LIMIT 1
        """
    )
    peor_p2 = q(
        """
        SELECT TRIM(CONCAT_WS(' ', e.primer_nombre, e.segundo_nombre,
                               e.primer_apellido, e.segundo_apellido)) AS est,
               ROUND(AVG(an.nota)::numeric, 2) AS promedio
        FROM acad_gradosasignaturas ga
        JOIN acad_actividades act ON act.idgrados_asignatura = ga.id
        JOIN acad_actividades_notas an ON an.idactividad = act.id
        JOIN acad_matricula_academica ma ON ma.id = an.idmatricula
        JOIN acad_estumatricula em ON em.id = ma.idestumatricula
        JOIN estu_estudiantes e ON e.id = em.idestudiante
        WHERE ga.iddocente=%(iddocente)s AND act.idperiodo=3 AND an.nota IS NOT NULL
        GROUP BY 1 ORDER BY promedio ASC LIMIT 1
        """
    )
    glob = q(
        """
        SELECT ROUND(AVG(an.nota)::numeric, 2) AS promedio_global
        FROM acad_gradosasignaturas ga
        JOIN acad_actividades act ON act.idgrados_asignatura = ga.id
        JOIN acad_actividades_notas an ON an.idactividad = act.id
        WHERE ga.iddocente=%(iddocente)s AND an.nota IS NOT NULL
        """
    )
    asigs_list = [r["asignatura"] for r in q(
        """
        SELECT a.nombre AS asignatura FROM acad_gradosasignaturas ga
        JOIN acad_asignaturas a ON a.id=ga.idasignatura
        WHERE ga.iddocente=%(iddocente)s ORDER BY 1
        """
    )]

    classic_expect = {
        "classic-peor-mat-p2": {
            "kind": "number",
            "value": float(peor_mat[0]["promedio"]),
            "name": peor_mat[0]["est"],
        }
        if peor_mat
        else {"kind": "text_has", "needles": ["matem"]},
        "classic-mejor-mat-p2": {
            "kind": "number",
            "value": float(mejor_mat[0]["promedio"]),
            "name": mejor_mat[0]["est"],
        }
        if mejor_mat
        else {"kind": "text_has", "needles": ["matem"]},
        "classic-mejor-p1": {
            "kind": "number",
            "value": float(mejor_p1[0]["promedio"]),
            "name": mejor_p1[0]["est"],
        }
        if mejor_p1
        else {"kind": "text_has", "needles": ["periodo"]},
        "classic-peor-p2-global": {
            "kind": "number",
            "value": float(peor_p2[0]["promedio"]),
            "name": peor_p2[0]["est"],
        }
        if peor_p2
        else {"kind": "text_has", "needles": ["periodo"]},
        "classic-global": {
            "kind": "number",
            "value": float(glob[0]["promedio_global"]),
        }
        if glob
        else {"kind": "text_has", "needles": ["promedio"]},
        "classic-atencion": {"kind": "text_has", "needles": ["3"]},
        "classic-mejor-cada": {
            "kind": "contains_any",
            "names": asigs_list[:4],
        },
        "classic-asignaturas": {
            "kind": "contains_any",
            "names": asigs_list,
        },
    }
    for cid, cq, _ in classic:
        add(cid, cq, classic_expect[cid], "clasica")

    # Ensure >= 50
    if len(items) < 50:
        # pad with more activity avg questions already mostly there
        pass

    return items


def select_balanced(questions: List[Dict[str, Any]], target: int = 55) -> List[Dict[str, Any]]:
    """Toma todas las clásicas + muestra específica hasta target (>=50)."""
    by_tag: Dict[str, List[Dict[str, Any]]] = {}
    for item in questions:
        by_tag.setdefault(item["tag"], []).append(item)

    picked: List[Dict[str, Any]] = []
    # siempre todas las clásicas
    picked.extend(by_tag.get("clasica", []))

    quotas = [
        ("actividad_max", 12),
        ("actividad_min", 8),
        ("actividad_avg", 8),
        ("ranking_mejor", 8),
        ("ranking_peor", 8),
        ("estudiante_mejor_nota", 10),
        ("estudiante_periodo", 4),
    ]
    for tag, n in quotas:
        bucket = by_tag.get(tag, [])
        picked.extend(bucket[:n])

    # relleno si falta
    if len(picked) < target:
        seen = {p["id"] for p in picked}
        for item in questions:
            if item["id"] in seen:
                continue
            picked.append(item)
            if len(picked) >= target:
                break
    return picked[: max(target, 50)]


def main() -> None:
    all_q = build_questions()
    questions = select_balanced(all_q, target=55)
    print(f"Preguntas generadas: {len(all_q)} | seleccionadas: {len(questions)}")
    if len(questions) < 50:
        print("WARNING: menos de 50 preguntas")

    results = []
    tallies = {
        "mao": {"bien": 0, "parcial": 0, "mal": 0, "error": 0},
        "multi": {"bien": 0, "parcial": 0, "mal": 0, "error": 0},
    }

    for i, item in enumerate(questions, 1):
        qid, question, expect, tag = item["id"], item["q"], item["expect"], item["tag"]
        print(f"\n[{i}/{len(questions)}] ({tag}) {question[:90]}")

        payload_base = {
            "user_id": str(IDDOCENTE),
            "message": question,
            "db_url": DB_URL,
            "context": {"role": "teacher", "iddocente": IDDOCENTE},
            "conversation_history": [],
        }

        mao = post(MAO_URL, {**payload_base, "conversation_id": f"c50-mao-{qid}"})
        mao_ans = mao.get("response") or mao.get("reply") or ""
        if mao.get("error") and not mao_ans:
            mv, md = "error", mao.get("error", "")[:120]
        else:
            mv, md = score_answer(mao_ans, expect)
        tallies["mao"][mv] += 1
        print(f"  MAO   [{mv}] {mao_ans[:140].replace(chr(10),' ')}")

        multi = post(
            MULTI_URL,
            {
                **payload_base,
                "conversation_id": f"c50-multi-{qid}",
                "max_rounds": 5,
            },
            timeout=200,
        )
        multi_ans = multi.get("response") or ""
        if multi.get("error") and not multi_ans:
            xv, xd = "error", multi.get("error", "")[:120]
        else:
            xv, xd = score_answer(multi_ans, expect)
        tallies["multi"][xv] += 1
        print(
            f"  MULTI [{xv}] sat={multi.get('satisfactory')} r={multi.get('rounds_used')} "
            f"{multi_ans[:140].replace(chr(10),' ')}"
        )

        results.append(
            {
                "id": qid,
                "tag": tag,
                "question": question,
                "expect": expect,
                "mao": {
                    "verdict": mv,
                    "detail": md,
                    "answer": mao_ans,
                    "ms": mao.get("_elapsed_client_ms"),
                    "tools": mao.get("function_calls_made"),
                },
                "multi": {
                    "verdict": xv,
                    "detail": xd,
                    "answer": multi_ans,
                    "ms": multi.get("elapsed_ms") or multi.get("_elapsed_client_ms"),
                    "satisfactory": multi.get("satisfactory"),
                    "score": multi.get("score"),
                    "rounds": multi.get("rounds_used"),
                },
            }
        )

    report = {
        "iddocente": IDDOCENTE,
        "total": len(questions),
        "tallies": tallies,
        "results": results,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 72)
    print(f"TOTAL {len(questions)}  → {OUT}")
    print("MAO  ", tallies["mao"])
    print("MULTI", tallies["multi"])


if __name__ == "__main__":
    main()
