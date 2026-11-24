#!/usr/bin/env python3
"""Eval del híbrido multiagente vs ground-truth (preguntas difíciles)."""
from __future__ import annotations

import json
import re
import time
import unicodedata
import urllib.request
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras

DB = dict(host="127.0.0.1", port=5432, dbname="aca_2", user="postgres", password="123")
DB_URL = "postgresql://postgres:123@127.0.0.1:5432/aca_2"
ID = 3
URL = "http://127.0.0.1:8000/api/v1/chat/teacher/multiagent"


def fold(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(s or ""))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def q(sql: str, params=None):
    with psycopg2.connect(**DB) as c, c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        out = []
        for r in cur.fetchall():
            d = dict(r)
            for k, v in list(d.items()):
                if isinstance(v, Decimal):
                    d[k] = float(v)
            out.append(d)
        return out


def post(msg: str) -> Dict[str, Any]:
    payload = {
        "user_id": str(ID),
        "message": msg,
        "db_url": DB_URL,
        "conversation_id": f"hyb-{abs(hash(msg))%10**8}",
        "context": {"role": "teacher", "iddocente": ID},
        "max_rounds": 6,
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read().decode())
    data["_ms"] = int((time.time() - t0) * 1000)
    return data


def score(answer: str, value: float, name: Optional[str] = None) -> str:
    nums = [float(x.replace(",", ".")) for x in re.findall(r"\b\d+(?:[.,]\d+)?\b", answer or "")]
    hit = any(abs(n - value) < 0.051 for n in nums)
    name_ok = True
    if name:
        toks = [t for t in fold(name).split() if len(t) > 2][:3]
        name_ok = all(t in fold(answer) for t in toks) if toks else True
    # Empate en la nota extrema: si el valor cuadra, contar bien
    # (varios estudiantes pueden compartir 5.0 / 3.5).
    if hit and (name_ok or "compartida" in fold(answer)):
        return "bien"
    if hit and name_ok:
        return "bien"
    if hit or name_ok:
        return "parcial"
    return "mal"


def build_cases() -> List[Dict[str, Any]]:
    cases = []
    acts = q(
        """
        SELECT act.id, a.nombre asig, p.nombre periodo, act.idperiodo,
               COALESCE(NULLIF(TRIM(act.descripcion),''),'Actividad') act,
               ROUND(AVG(an.nota)::numeric,2) avg,
               ROUND(MAX(an.nota)::numeric,2) mx,
               ROUND(MIN(an.nota)::numeric,2) mn
        FROM acad_gradosasignaturas ga
        JOIN acad_asignaturas a ON a.id=ga.idasignatura
        JOIN acad_actividades act ON act.idgrados_asignatura=ga.id
        LEFT JOIN acad_periodos p ON p.id=act.idperiodo
        JOIN acad_actividades_notas an ON an.idactividad=act.id
        WHERE ga.iddocente=%s AND an.nota IS NOT NULL
          AND LENGTH(COALESCE(act.descripcion,''))>=8
        GROUP BY 1,2,3,4,5
        HAVING COUNT(*)>=15
        ORDER BY COUNT(*) DESC
        LIMIT 12
        """,
        (ID,),
    )
    # Deduplicate by folded title+subject+period (same name spans groups)
    seen_keys = set()
    for r in acts:
        key = (fold(r["act"])[:40], fold(r["asig"]), r["idperiodo"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        # peer activities with same title stem + subject + period
        peers = q(
            """
            SELECT act.id FROM acad_gradosasignaturas ga
            JOIN acad_asignaturas a ON a.id=ga.idasignatura
            JOIN acad_actividades act ON act.idgrados_asignatura=ga.id
            WHERE ga.iddocente=%s AND a.nombre=%s AND act.idperiodo=%s
              AND translate(lower(COALESCE(act.descripcion,'')),
                    'áéíóúüñ','aeiouun')
                  LIKE '%%' || replace(translate(lower(%s),'áéíóúüñ','aeiouun'),' ','%%') || '%%'
            """,
            (ID, r["asig"], r["idperiodo"], r["act"][:24]),
        )
        # safer peer set: same folded first 20 chars
        peer_ids = [p["id"] for p in peers] or [r["id"]]
        # Prefer peers that share significant tokens with the seed activity
        toks = [t for t in re.findall(r"[a-z0-9]{4,}", fold(r["act"]))][:6]
        if toks:
            peer_ids = []
            all_same = q(
                """
                SELECT act.id, act.descripcion FROM acad_gradosasignaturas ga
                JOIN acad_asignaturas a ON a.id=ga.idasignatura
                JOIN acad_actividades act ON act.idgrados_asignatura=ga.id
                WHERE ga.iddocente=%s AND a.nombre=%s AND act.idperiodo=%s
                """,
                (ID, r["asig"], r["idperiodo"]),
            )
            for p in all_same:
                fd = fold(p["descripcion"] or "")
                coverage = sum(1 for t in toks if t in fd) / len(toks)
                if coverage >= 0.7:
                    peer_ids.append(p["id"])
            if not peer_ids:
                peer_ids = [r["id"]]

        top = q(
            """
            SELECT TRIM(CONCAT_WS(' ',e.primer_nombre,e.segundo_nombre,e.primer_apellido,e.segundo_apellido)) est, an.nota
            FROM acad_actividades_notas an
            JOIN acad_matricula_academica ma ON ma.id=an.idmatricula
            JOIN acad_estumatricula em ON em.id=ma.idestumatricula
            JOIN estu_estudiantes e ON e.id=em.idestudiante
            WHERE an.idactividad = ANY(%s) AND an.nota IS NOT NULL
            ORDER BY an.nota DESC LIMIT 1
            """,
            (peer_ids,),
        )
        worst = q(
            """
            SELECT TRIM(CONCAT_WS(' ',e.primer_nombre,e.segundo_nombre,e.primer_apellido,e.segundo_apellido)) est, an.nota
            FROM acad_actividades_notas an
            JOIN acad_matricula_academica ma ON ma.id=an.idmatricula
            JOIN acad_estumatricula em ON em.id=ma.idestumatricula
            JOIN estu_estudiantes e ON e.id=em.idestudiante
            WHERE an.idactividad = ANY(%s) AND an.nota IS NOT NULL
            ORDER BY an.nota ASC LIMIT 1
            """,
            (peer_ids,),
        )
        avg_row = q(
            """
            SELECT ROUND(AVG(an.nota)::numeric,2) avg FROM acad_actividades_notas an
            WHERE an.idactividad = ANY(%s) AND an.nota IS NOT NULL
            """,
            (peer_ids,),
        )
        short = r["act"] if len(r["act"]) < 50 else " ".join(r["act"].split()[:4])
        if top:
            cases.append(
                {
                    "q": f"¿Cuál fue la mejor nota de la actividad «{short}» de {r['asig']} en {r['periodo']}?",
                    "value": float(top[0]["nota"]),
                    "name": top[0]["est"],
                    "tag": "act_max",
                }
            )
        if worst:
            cases.append(
                {
                    "q": f"¿Cuál fue la peor nota de la actividad «{short}» de {r['asig']} en {r['periodo']}?",
                    "value": float(worst[0]["nota"]),
                    "name": worst[0]["est"],
                    "tag": "act_min",
                }
            )
        if avg_row:
            cases.append(
                {
                    "q": f"¿Cuál es el promedio del grupo en la actividad «{short}» ({r['asig']}, {r['periodo']})?",
                    "value": float(avg_row[0]["avg"]),
                    "name": None,
                    "tag": "act_avg",
                }
            )

    # rankings
    for asig, pid, per in [
        ("Matematicas", 3, "Periodo 2"),
        ("Matematicas", 1, "Periodo 1"),
        ("Geometría", 3, "Periodo 2"),
        ("Educación Física", 3, "Periodo 2"),
    ]:
        best = q(
            """
            SELECT TRIM(CONCAT_WS(' ',e.primer_nombre,e.segundo_nombre,e.primer_apellido,e.segundo_apellido)) est,
                   ROUND(AVG(an.nota)::numeric,2) avg
            FROM acad_gradosasignaturas ga
            JOIN acad_asignaturas a ON a.id=ga.idasignatura
            JOIN acad_actividades act ON act.idgrados_asignatura=ga.id
            JOIN acad_actividades_notas an ON an.idactividad=act.id
            JOIN acad_matricula_academica ma ON ma.id=an.idmatricula
            JOIN acad_estumatricula em ON em.id=ma.idestumatricula
            JOIN estu_estudiantes e ON e.id=em.idestudiante
            WHERE ga.iddocente=%s AND a.nombre ILIKE %s AND act.idperiodo=%s AND an.nota IS NOT NULL
            GROUP BY 1 HAVING COUNT(*)>=2 ORDER BY avg DESC LIMIT 1
            """,
            (ID, asig, pid),
        )
        worst = q(
            """
            SELECT TRIM(CONCAT_WS(' ',e.primer_nombre,e.segundo_nombre,e.primer_apellido,e.segundo_apellido)) est,
                   ROUND(AVG(an.nota)::numeric,2) avg
            FROM acad_gradosasignaturas ga
            JOIN acad_asignaturas a ON a.id=ga.idasignatura
            JOIN acad_actividades act ON act.idgrados_asignatura=ga.id
            JOIN acad_actividades_notas an ON an.idactividad=act.id
            JOIN acad_matricula_academica ma ON ma.id=an.idmatricula
            JOIN acad_estumatricula em ON em.id=ma.idestumatricula
            JOIN estu_estudiantes e ON e.id=em.idestudiante
            WHERE ga.iddocente=%s AND a.nombre ILIKE %s AND act.idperiodo=%s AND an.nota IS NOT NULL
            GROUP BY 1 HAVING COUNT(*)>=2 ORDER BY avg ASC LIMIT 1
            """,
            (ID, asig, pid),
        )
        if best:
            cases.append(
                {
                    "q": f"¿Quién fue el mejor estudiante de {asig} en el {per}?",
                    "value": float(best[0]["avg"]),
                    "name": best[0]["est"],
                    "tag": "rank_best",
                }
            )
        if worst:
            cases.append(
                {
                    "q": f"¿Quién fue el peor estudiante de {asig} en el {per}?",
                    "value": float(worst[0]["avg"]),
                    "name": worst[0]["est"],
                    "tag": "rank_worst",
                }
            )

    cases.append(
        {
            "q": "cuál fue el peor estudiante del segundo periodo en matemáticas",
            "value": float(
                q(
                    """
                SELECT ROUND(AVG(an.nota)::numeric,2) avg FROM acad_gradosasignaturas ga
                JOIN acad_asignaturas a ON a.id=ga.idasignatura
                JOIN acad_actividades act ON act.idgrados_asignatura=ga.id
                JOIN acad_actividades_notas an ON an.idactividad=act.id
                JOIN acad_matricula_academica ma ON ma.id=an.idmatricula
                JOIN acad_estumatricula em ON em.id=ma.idestumatricula
                JOIN estu_estudiantes e ON e.id=em.idestudiante
                WHERE ga.iddocente=%s AND a.nombre ILIKE '%%matem%%' AND act.idperiodo=3 AND an.nota IS NOT NULL
                GROUP BY e.id ORDER BY avg ASC LIMIT 1
                """,
                    (ID,),
                )[0]["avg"]
            ),
            "name": "SAMUEL",
            "tag": "classic",
        }
    )

    # Natural language / no guillemets + student grade probes
    top_cal = q(
        """
        SELECT TRIM(CONCAT_WS(' ',e.primer_nombre,e.segundo_nombre,e.primer_apellido,e.segundo_apellido)) est, an.nota
        FROM acad_actividades act
        JOIN acad_gradosasignaturas ga ON ga.id=act.idgrados_asignatura
        JOIN acad_asignaturas a ON a.id=ga.idasignatura
        JOIN acad_actividades_notas an ON an.idactividad=act.id
        JOIN acad_matricula_academica ma ON ma.id=an.idmatricula
        JOIN acad_estumatricula em ON em.id=ma.idestumatricula
        JOIN estu_estudiantes e ON e.id=em.idestudiante
        WHERE ga.iddocente=%s
          AND translate(lower(a.nombre),'áéíóúüñ','aeiouun') LIKE '%%fisica%%'
          AND translate(lower(act.descripcion),'áéíóúüñ','aeiouun')
              LIKE '%%calentamiento%%recorrido%%'
          AND act.idperiodo=1 AND an.nota IS NOT NULL
        ORDER BY an.nota DESC LIMIT 1
        """,
        (ID,),
    )
    if top_cal:
        cases.append(
            {
                "q": "cuál fue la mejor nota de CALENTAMIENTO RECORRIDO en educación física periodo 1",
                "value": float(top_cal[0]["nota"]),
                "name": top_cal[0]["est"],
                "tag": "nl_act",
            }
        )
        cases.append(
            {
                "q": "dime la nota más alta de la actividad CALENTAMIENTO en Educación Física del primer periodo",
                "value": float(top_cal[0]["nota"]),
                "name": None,  # may list ties
                "tag": "nl_act2",
            }
        )

    stu = q(
        """
        SELECT TRIM(CONCAT_WS(' ',e.primer_nombre,e.segundo_nombre,e.primer_apellido,e.segundo_apellido)) est,
               an.nota, COALESCE(NULLIF(TRIM(act.descripcion),''),'Act') act
        FROM acad_gradosasignaturas ga
        JOIN acad_actividades act ON act.idgrados_asignatura=ga.id
        JOIN acad_actividades_notas an ON an.idactividad=act.id
        JOIN acad_matricula_academica ma ON ma.id=an.idmatricula
        JOIN acad_estumatricula em ON em.id=ma.idestumatricula
        JOIN estu_estudiantes e ON e.id=em.idestudiante
        WHERE ga.iddocente=%s AND e.primer_nombre ILIKE 'SALOME' AND an.nota IS NOT NULL
        ORDER BY an.nota ASC LIMIT 1
        """,
        (ID,),
    )
    if stu:
        cases.append(
            {
                "q": f"cuál fue la peor nota de {stu[0]['est'].split()[0]}",
                "value": float(stu[0]["nota"]),
                "name": stu[0]["est"],
                "tag": "stu_worst",
            }
        )

    glob = q(
        """
        SELECT ROUND(AVG(an.nota)::numeric,2) g FROM acad_gradosasignaturas ga
        JOIN acad_actividades act ON act.idgrados_asignatura=ga.id
        JOIN acad_actividades_notas an ON an.idactividad=act.id
        WHERE ga.iddocente=%s AND an.nota IS NOT NULL
        """,
        (ID,),
    )
    if glob:
        cases.append(
            {
                "q": "cuál es mi promedio global de todas las notas",
                "value": float(glob[0]["g"]),
                "name": None,
                "tag": "global",
            }
        )
    return cases


def main():
    cases = build_cases()
    print(f"cases={len(cases)}")
    tallies = {"bien": 0, "parcial": 0, "mal": 0, "error": 0}
    fails = []
    for i, c in enumerate(cases, 1):
        try:
            data = post(c["q"])
            ans = data.get("response") or ""
            v = score(ans, c["value"], c.get("name"))
        except Exception as e:
            v, ans, data = "error", str(e), {}
        tallies[v] += 1
        print(
            f"[{i}/{len(cases)}] {v} path={data.get('mode')} "
            f"sat={data.get('satisfactory')} :: {c['q'][:70]}"
        )
        print(f"   expect={c['value']} name={c.get('name')}")
        print(f"   ans={ans[:160].replace(chr(10),' ')}")
        if v != "bien":
            fails.append({**c, "verdict": v, "answer": ans, "mode": data.get("mode")})
    print("TALLIES", tallies)
    print("FAILS", len(fails))
    open("/tmp/hybrid_hard_fails.json", "w").write(
        json.dumps(fails, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
