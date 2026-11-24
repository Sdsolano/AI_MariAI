"""
Hybrid teacher reasoner (experimental) — does NOT replace Mao actual.

Pipeline (como razona un analista humano):
  1) Parse intent + entity extract (actividad «…», materia, periodo, estudiante)
  2) Atajos SQL determinísticos (ranking, actividad max/min/avg, mejor nota alumno…)
  3) Si no hay atajo o falla match → loop Query+Critic con tools
  4) Writer solo con evidencia

Endpoint: POST /api/v1/chat/teacher/multiagent
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from app.services import teacher_context_service as teacher_ctx
from app.services import teacher_sql_service as teacher_sql
from app.services.teacher_context_service import _collapse_repeats, fold

logger = logging.getLogger(__name__)

MAX_ROUNDS = 8
MODEL = "gpt-4o-mini"
SATISFACTORY_SCORE = 0.78


def _safe_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def _chat(client: OpenAI, system: str, user: str, *, temperature: float = 0.1) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=1200,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or "{}"


def _soft(s: str) -> str:
    return _collapse_repeats(fold(s or ""))


# ---------- entity extraction ----------

_SUBJECT_TOKEN_MAP = [
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


def guess_subject(question: str) -> Optional[str]:
    soft = _soft(question)
    for full in (
        "educacion fisica",
        "educacion artistica",
        "matematicas",
        "geometria",
        "religion",
        "etica",
    ):
        if full in soft:
            for tok, label in _SUBJECT_TOKEN_MAP:
                if tok in full:
                    return label
    for tok, label in _SUBJECT_TOKEN_MAP:
        if tok in soft:
            return label
    return None


def extract_activity_name(question: str) -> Optional[str]:
    q = question or ""
    for pat in (
        r"«([^»]{3,120})»",
        r'"([^"]{3,120})"',
        r"'([^']{3,120})'",
        r"actividad\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9][A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9\s\-_/()]{3,80})",
        # sin comillas: "mejor nota de CALENTAMIENTO RECORRIDO en Educación..."
        r"(?:nota|promedio)\s+(?:de\s+la\s+|del\s+)?(?:actividad\s+)?([A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑa-záéíóúüñ0-9\s\-_/()]{4,60}?)"
        r"(?=\s+(?:de|en|del|para)\s+)",
    ):
        m = re.search(pat, q, re.I)
        if m:
            name = m.group(1).strip(" .,:;")
            # cut trailing " de Educación..." if captured by loose pattern
            name = re.split(
                r"\s+de\s+(Educaci[oó]n|Matem|Geometr|Religi|Ética|Etica|Periodo|Ingles|Inglés)",
                name,
                maxsplit=1,
                flags=re.I,
            )[0].strip()
            # discard if it looks like a student name probe / too generic
            soft = _soft(name)
            if soft in {"la", "el", "grupo", "mejor", "peor", "estudiante"}:
                continue
            # Avoid treating student names as activities when "actividad" absent
            if "actividad" not in _soft(q) and not re.search(r"[«»\"']", q):
                # Heuristic: 2–4 Capitalized tokens → likely a person
                caps = re.findall(r"\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+\b", name)
                if 1 <= len(caps) <= 4 and len(name.split()) <= 4:
                    # unless it looks like an activity title keyword
                    if not re.search(
                        r"calentamiento|evaluacion|prueba|ejercicio|fundamento|actividad",
                        soft,
                    ):
                        continue
            if len(name) >= 3:
                return name
    return None


def extract_student_name(question: str) -> Optional[str]:
    return teacher_ctx.extract_student_name(question)


def classify_intent(question: str) -> str:
    soft = _soft(question)
    stu = extract_student_name(question)
    act = extract_activity_name(question)
    # "mejor nota de NOMBRE" → estudiante (no actividad)
    looks_student_grade = bool(
        stu
        and re.search(r"(mejor|peor|mas alta|mas baja)\s+nota\s+de", soft)
        and "actividad" not in soft
        and not re.search(r"[«»\"']", question or "")
    )
    if re.search(r"promedio\s+global|global\s+de\s+todas", soft):
        return "global_avg"
    if looks_student_grade and re.search(r"mejor|mas alta|maxima", soft):
        return "student_best_grade"
    if looks_student_grade and re.search(r"peor|mas baja|minima", soft):
        return "student_worst_grade"
    if re.search(r"mejor\s+nota|nota\s+mas\s+alta|maxima\s+nota", soft) and act:
        return "activity_max"
    if re.search(r"peor\s+nota|nota\s+mas\s+baja|minima\s+nota", soft) and act:
        return "activity_min"
    if re.search(r"promedio.+(actividad|grupo)|promedio\s+del\s+grupo", soft) and (
        act or "actividad" in soft
    ):
        return "activity_avg"
    if re.search(r"mejor\s+nota|nota\s+mas\s+alta", soft) and stu:
        return "student_best_grade"
    if re.search(r"peor\s+nota|nota\s+mas\s+baja", soft) and stu:
        return "student_worst_grade"
    if re.search(r"mejor\s+de\s+cada|cada\s+asignatura|peor\s+de\s+cada", soft):
        return "best_per_subject"
    if re.search(r"necesitan\s+atencion|bajo\s+\d", soft):
        return "attention"
    if re.search(r"que\s+asignaturas|materias\s+dicto", soft):
        return "subjects"
    if re.search(
        r"(?:(?:el|la)\s+)?peor\s+estudiant|(quien|quién).{0,20}peor|peor\s+del\s+(?:primer|segundo|tercer|periodo)",
        soft,
    ):
        return "rank_worst"
    if re.search(
        r"(?:(?:el|la)\s+)?mejor\s+estudiant|(quien|quién).{0,20}mejor|mejor\s+del\s+(?:primer|segundo|tercer|periodo)",
        soft,
    ):
        return "rank_best"
    if extract_activity_name(question) and re.search(r"mejor|peor|promedio|nota", soft):
        if "peor" in soft or "baja" in soft:
            return "activity_min"
        if "promedio" in soft:
            return "activity_avg"
        return "activity_max"
    return "adhoc"


# ---------- deterministic answers ----------

def _fmt_student(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip())


async def try_deterministic(
    db_url: str,
    iddocente: int,
    question: str,
) -> Optional[Dict[str, Any]]:
    intent = classify_intent(question)
    periods = await teacher_sql.list_periods(db_url, iddocente)
    period_rows = periods.get("result") or []
    pid = teacher_sql.resolve_period_id(period_rows, question)
    asig = guess_subject(question)
    act_name = extract_activity_name(question)
    soft = _soft(question)
    trace_base = {
        "intent": intent,
        "asignatura": asig,
        "idperiodo": pid,
        "actividad": act_name,
    }

    if intent == "global_avg":
        g = await teacher_sql.global_average(db_url, iddocente)
        rows = g.get("result") or []
        if not rows:
            return None
        r0 = rows[0]
        return {
            "response": (
                f"Tu promedio global es **{r0.get('promedio_global')}** "
                f"({r0.get('total_notas')} notas, "
                f"{r0.get('estudiantes_con_nota')} estudiantes con nota)."
            ),
            "satisfactory": True,
            "score": 1.0,
            "path": "deterministic",
            "trace": [{"agent": "deterministic", **trace_base, "tool": "global_average"}],
            "evidence": rows,
        }

    if intent in ("activity_max", "activity_min", "activity_avg") and act_name:
        mode = {"activity_max": "max", "activity_min": "min", "activity_avg": "avg"}[intent]
        payload = await teacher_sql.activity_extreme_grade(
            db_url,
            iddocente,
            actividad=act_name,
            asignatura=asig,
            idperiodo=pid,
            mode=mode,
        )
        rows = payload.get("result") or []
        if not rows:
            # retries are inside activity_extreme_grade; one more pass name-only
            payload = await teacher_sql.activity_extreme_grade(
                db_url,
                iddocente,
                actividad=act_name,
                asignatura=None,
                idperiodo=None,
                mode=mode,
            )
            rows = payload.get("result") or []
        if not rows:
            return {
                "response": None,
                "satisfactory": False,
                "score": 0.0,
                "path": "deterministic_miss",
                "trace": [
                    {
                        "agent": "deterministic",
                        **trace_base,
                        "tool": "activity_extreme_grade",
                        "ok": False,
                        "error": payload.get("error"),
                    }
                ],
                "continue": True,
            }
        matched = payload.get("matched_activity") or {}
        if mode == "avg":
            r0 = rows[0]
            answer = (
                f"El promedio del grupo en la actividad «{r0.get('actividad')}» "
                f"({r0.get('asignatura')}, {r0.get('periodo')}) es "
                f"**{r0.get('promedio')}** ({r0.get('total_notas')} notas)."
            )
        else:
            r0 = rows[0]
            kind = "mejor" if mode == "max" else "peor"
            names = [
                _fmt_student(str(r.get("estudiante") or ""))
                for r in rows
                if r.get("estudiante")
            ]
            # de-dupe preserving order
            seen_n: set = set()
            uniq = []
            for n in names:
                k = n.lower()
                if k in seen_n:
                    continue
                seen_n.add(k)
                uniq.append(n)
            if len(uniq) == 1:
                who = f"de **{uniq[0]}**"
            elif len(uniq) <= 4:
                who = "compartida por " + ", ".join(f"**{n}**" for n in uniq)
            else:
                who = (
                    "compartida por "
                    + ", ".join(f"**{n}**" for n in uniq[:3])
                    + f" y {len(uniq) - 3} más"
                )
            answer = (
                f"La {kind} nota de la actividad «{r0.get('actividad')}» "
                f"({r0.get('asignatura')}, {r0.get('periodo')}) fue "
                f"**{r0.get('nota')}** {who}."
            )
        return {
            "response": answer,
            "satisfactory": True,
            "score": 1.0,
            "path": "deterministic",
            "trace": [
                {
                    "agent": "deterministic",
                    **trace_base,
                    "tool": "activity_extreme_grade",
                    "mode": mode,
                    "matched": matched.get("actividad"),
                    "ok": True,
                }
            ],
            "evidence": rows,
        }

    if intent in ("rank_best", "rank_worst"):
        if pid is None and period_rows:
            # sin periodo → no inventar; dejar al loop
            if not re.search(r"periodo|período|primer|segundo|tercer", soft):
                pass
            else:
                return None
        if pid is None:
            return None
        order = "DESC" if intent == "rank_best" else "ASC"
        ranked = await teacher_sql.rank_students_by_period(
            db_url,
            iddocente,
            int(pid),
            order=order,
            limit=1,
            asignatura=asig,
        )
        rows = ranked.get("result") or []
        if not rows:
            return {
                "response": None,
                "satisfactory": False,
                "score": 0.0,
                "path": "deterministic_miss",
                "continue": True,
                "trace": [
                    {
                        "agent": "deterministic",
                        **trace_base,
                        "tool": "rank_students_by_period",
                        "ok": False,
                    }
                ],
            }
        r0 = rows[0]
        kind = "mejor" if intent == "rank_best" else "peor"
        scope = f"en **{asig}**" if asig else "en el global de tus cursos"
        pname = next(
            (
                p.get("nombre")
                for p in period_rows
                if int(p.get("idperiodo") or 0) == int(pid)
            ),
            f"periodo {pid}",
        )
        # fetch a few more to detect ties at the extreme promedio
        ranked_more = await teacher_sql.rank_students_by_period(
            db_url,
            iddocente,
            int(pid),
            order=order,
            limit=8,
            asignatura=asig,
        )
        more = ranked_more.get("result") or rows
        top_avg = float(r0.get("promedio") or 0)
        tied = [
            r
            for r in more
            if abs(float(r.get("promedio") or 0) - top_avg) < 0.011
        ] or [r0]
        if len(tied) == 1:
            who = f"**{_fmt_student(str(tied[0].get('estudiante')))}**"
        else:
            who = (
                ", ".join(
                    f"**{_fmt_student(str(r.get('estudiante')))}**" for r in tied[:4]
                )
                + ("…" if len(tied) > 4 else "")
                + " (empate)"
            )
        answer = (
            f"El {kind} estudiante {scope} en **{pname}** fue "
            f"{who}, "
            f"con promedio **{r0.get('promedio')}** "
            f"({r0.get('total_notas')} notas)."
        )
        return {
            "response": answer,
            "satisfactory": True,
            "score": 1.0,
            "path": "deterministic",
            "trace": [
                {
                    "agent": "deterministic",
                    **trace_base,
                    "tool": "rank_students_by_period",
                    "ok": True,
                }
            ],
            "evidence": tied,
        }

    if intent in ("student_best_grade", "student_worst_grade"):
        nombre = extract_student_name(question)
        if not nombre:
            return None
        payload = await teacher_sql.student_best_grade(
            db_url,
            iddocente,
            nombre=nombre,
            asignatura=asig,
            worst=(intent == "student_worst_grade"),
        )
        rows = payload.get("result") or []
        if not rows:
            return {
                "response": None,
                "satisfactory": False,
                "continue": True,
                "score": 0.0,
                "path": "deterministic_miss",
                "trace": [
                    {
                        "agent": "deterministic",
                        **trace_base,
                        "tool": "student_best_grade",
                        "ok": False,
                    }
                ],
            }
        r0 = rows[0]
        kind = "Peor" if intent == "student_worst_grade" else "Mejor"
        where = f" en **{r0.get('asignatura')}**" if r0.get("asignatura") else ""
        periodo_txt = f" ({r0.get('periodo')})" if r0.get("periodo") else ""
        answer = (
            f"{kind} nota de **{_fmt_student(str(r0.get('estudiante')))}**{where}: "
            f"**{r0.get('nota')}** en «{r0.get('actividad')}»{periodo_txt}."
        )
        return {
            "response": answer,
            "satisfactory": True,
            "score": 1.0,
            "path": "deterministic",
            "trace": [
                {
                    "agent": "deterministic",
                    **trace_base,
                    "tool": "student_best_grade",
                    "ok": True,
                }
            ],
            "evidence": rows,
        }

    if intent == "best_per_subject":
        worst = "peor" in soft
        ranked = await teacher_sql.best_student_per_subject_group(
            db_url, iddocente, order="ASC" if worst else "DESC"
        )
        rows = ranked.get("result") or []
        if not rows:
            return None
        kind = "Peor" if worst else "Mejor"
        lines = [f"{kind} estudiante de cada asignatura/grupo:", ""]
        for r in rows:
            lines.append(
                f"- **{r.get('asignatura')}** ({r.get('grado')} {r.get('grupo')}): "
                f"{_fmt_student(str(r.get('estudiante')))} — promedio **{r.get('promedio')}**"
            )
        return {
            "response": "\n".join(lines),
            "satisfactory": True,
            "score": 1.0,
            "path": "deterministic",
            "trace": [{"agent": "deterministic", "tool": "best_per_subject"}],
            "evidence": rows,
        }

    return None


# ---------- agentic fallback ----------

QUERY_SYS = """Eres QUERY AGENT. Elige UNA acción JSON:
{
  "action": "list_periods|list_activities|activity_grade|rank_period|student_grade|run_sql|global_avg",
  "asignatura": "opcional",
  "idperiodo": null,
  "actividad": "opcional nombre",
  "nombre": "opcional estudiante",
  "mode": "max|min|avg",
  "order": "ASC|DESC",
  "sql": "solo si run_sql",
  "rationale": "breve"
}
Reglas: si preguntan por actividad, usa list_activities o activity_grade (no SQL libre primero).
Filtros de materia SIN tildes (matematicas). Periodo = id de list_periods.
"""

CRITIC_SYS = """CRITIC JSON:
{"satisfactory":bool,"score":0-1,"reason":"","gaps":[],"next_hints":[]}
true solo con filas concretas que responden (actividad correcta si se pidió).
"""

WRITER_SYS = """WRITER JSON: {"answer":"markdown corto en español solo con evidencia"}"""


async def _exec_tool(
    db_url: str,
    iddocente: int,
    proposal: Dict[str, Any],
) -> Dict[str, Any]:
    action = (proposal.get("action") or "").strip().lower()
    asig = proposal.get("asignatura")
    pid = proposal.get("idperiodo")
    try:
        pid = int(pid) if pid is not None and str(pid).strip() != "" else None
    except (TypeError, ValueError):
        pid = None
    act = proposal.get("actividad")
    nombre = proposal.get("nombre")
    mode = (proposal.get("mode") or "max").lower()
    order = proposal.get("order") or "DESC"

    if action in ("list_periods", "periods"):
        return await teacher_sql.list_periods(db_url, iddocente)
    if action in ("list_activities", "activities"):
        return await teacher_sql.list_activities(
            db_url,
            iddocente,
            asignatura=asig,
            idperiodo=pid,
            actividad=act,
            limit=30,
        )
    if action in ("activity_grade", "activity_extreme"):
        if not act:
            return {"error": "falta actividad", "result": [], "count": 0}
        return await teacher_sql.activity_extreme_grade(
            db_url,
            iddocente,
            actividad=str(act),
            asignatura=asig,
            idperiodo=pid,
            mode=mode if mode in ("max", "min", "avg") else "max",
        )
    if action in ("rank_period", "rank"):
        if pid is None:
            return {"error": "falta idperiodo", "result": [], "count": 0}
        return await teacher_sql.rank_students_by_period(
            db_url,
            iddocente,
            int(pid),
            order=str(order),
            limit=5,
            asignatura=asig,
        )
    if action in ("student_grade", "student_best"):
        if not nombre:
            return {"error": "falta nombre", "result": [], "count": 0}
        return await teacher_sql.student_best_grade(
            db_url,
            iddocente,
            nombre=str(nombre),
            asignatura=asig,
            worst=(mode == "min" or str(order).upper().startswith("A")),
        )
    if action in ("global_avg", "global"):
        return await teacher_sql.global_average(db_url, iddocente)
    if action in ("run_sql", "sql"):
        sql = proposal.get("sql") or ""
        return await teacher_sql.run_teacher_sql(db_url, iddocente, str(sql))
    if action in ("describe_schema", "schema"):
        return teacher_sql.describe_teacher_schema()
    return {"error": f"acción desconocida: {action}", "result": [], "count": 0}


def _has_rows(payload: Dict[str, Any]) -> bool:
    if not payload or payload.get("error") or payload.get("rejected"):
        return False
    rows = payload.get("result")
    return isinstance(rows, list) and len(rows) > 0


async def run_agentic_loop(
    *,
    client: OpenAI,
    db_url: str,
    iddocente: int,
    question: str,
    max_rounds: int,
    prior_trace: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    trace: List[Dict[str, Any]] = list(prior_trace or [])
    evidence: List[Dict[str, Any]] = []
    periods = await teacher_sql.list_periods(db_url, iddocente)
    evidence.append({"source": "list_periods", "payload": periods})
    resolved_period = teacher_sql.resolve_period_id(
        periods.get("result") or [], question
    )
    subject_hint = guess_subject(question)
    act_hint = extract_activity_name(question)
    student_hint = extract_student_name(question)
    intent = classify_intent(question)

    # proactive seed for activity questions
    if act_hint or intent.startswith("activity_"):
        seed = await teacher_sql.list_activities(
            db_url,
            iddocente,
            asignatura=subject_hint,
            idperiodo=resolved_period,
            actividad=act_hint,
            limit=15,
        )
        evidence.append({"source": "list_activities", "payload": seed})
        trace.append(
            {
                "agent": "seed",
                "action": "list_activities",
                "ok": _has_rows(seed),
                "count": seed.get("count") or 0,
            }
        )

    last_error = None
    critic: Dict[str, Any] = {}
    satisfactory = False

    for round_idx in range(1, max_rounds + 1):
        critic = _safe_json(
            _chat(
                client,
                CRITIC_SYS,
                json.dumps(
                    {
                        "pregunta": question,
                        "intent": intent,
                        "hints": {
                            "asignatura": subject_hint,
                            "idperiodo": resolved_period,
                            "actividad": act_hint,
                            "estudiante": student_hint,
                        },
                        "evidencia": [
                            {
                                "source": e.get("source"),
                                "error": (e.get("payload") or {}).get("error"),
                                "count": (e.get("payload") or {}).get("count"),
                                "sample": (
                                    (e.get("payload") or {}).get("result") or []
                                )[:6],
                                "matched_activity": (e.get("payload") or {}).get(
                                    "matched_activity"
                                ),
                            }
                            for e in evidence
                        ],
                        "ultimo_error": last_error,
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            )
        )
        score = float(critic.get("score") or 0)
        satisfactory = bool(critic.get("satisfactory")) and score >= SATISFACTORY_SCORE
        if satisfactory and not any(
            _has_rows(e.get("payload") or {})
            for e in evidence
            if e.get("source") not in ("list_periods", "describe_schema")
        ):
            satisfactory = False
        trace.append(
            {
                "round": round_idx,
                "agent": "critic",
                "satisfactory": satisfactory,
                "score": score,
                "reason": critic.get("reason"),
                "gaps": critic.get("gaps"),
            }
        )
        if satisfactory:
            break

        proposal = _safe_json(
            _chat(
                client,
                QUERY_SYS,
                json.dumps(
                    {
                        "pregunta": question,
                        "intent": intent,
                        "hints": {
                            "asignatura": subject_hint,
                            "idperiodo": resolved_period,
                            "actividad": act_hint,
                            "estudiante": student_hint,
                            "periodos": periods.get("result") or [],
                        },
                        "gaps": critic.get("gaps") or [],
                        "next_hints": critic.get("next_hints") or [],
                        "ultimo_error": last_error,
                        "actividades_candidatas": (
                            (evidence[-1].get("payload") or {}).get("result")
                            if evidence
                            and evidence[-1].get("source") == "list_activities"
                            else None
                        ),
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            )
        )
        # fill defaults from hints if agent omitted
        if not proposal.get("actividad") and act_hint:
            proposal["actividad"] = act_hint
        if not proposal.get("asignatura") and subject_hint:
            proposal["asignatura"] = subject_hint
        if proposal.get("idperiodo") is None and resolved_period is not None:
            proposal["idperiodo"] = resolved_period
        if not proposal.get("nombre") and student_hint:
            proposal["nombre"] = student_hint
        if not proposal.get("action"):
            if act_hint:
                proposal["action"] = "activity_grade"
                proposal["mode"] = (
                    "min"
                    if "peor" in _soft(question) or "baja" in _soft(question)
                    else "avg"
                    if "promedio" in _soft(question)
                    else "max"
                )
            elif intent in ("rank_best", "rank_worst"):
                proposal["action"] = "rank_period"
                proposal["order"] = "ASC" if intent == "rank_worst" else "DESC"
            else:
                proposal["action"] = "list_activities"

        payload = await _exec_tool(db_url, iddocente, proposal)
        ok = _has_rows(payload)
        last_error = None if ok else str(
            payload.get("error") or "sin filas"
        )
        evidence.append(
            {
                "source": proposal.get("action") or "tool",
                "payload": payload,
                "proposal": proposal,
            }
        )
        trace.append(
            {
                "round": round_idx,
                "agent": "query",
                "action": proposal.get("action"),
                "ok": ok,
                "count": payload.get("count")
                if payload.get("count") is not None
                else len(payload.get("result") or [])
                if isinstance(payload.get("result"), list)
                else 0,
                "error": last_error,
                "rationale": proposal.get("rationale"),
            }
        )

    writer = _safe_json(
        _chat(
            client,
            WRITER_SYS,
            json.dumps(
                {
                    "pregunta": question,
                    "satisfactory": satisfactory,
                    "critic": critic,
                    "evidencia": [
                        {
                            "source": e.get("source"),
                            "result": (e.get("payload") or {}).get("result")[:15]
                            if isinstance((e.get("payload") or {}).get("result"), list)
                            else None,
                            "matched_activity": (e.get("payload") or {}).get(
                                "matched_activity"
                            ),
                            "error": (e.get("payload") or {}).get("error"),
                        }
                        for e in evidence
                    ],
                },
                ensure_ascii=False,
                default=str,
            ),
        )
    )
    answer = (writer.get("answer") or "").strip()
    if not answer:
        answer = (
            "No alcancé una respuesta satisfactoria con las consultas disponibles."
            if not satisfactory
            else "Encontré datos pero no pude redactar la respuesta."
        )
    return {
        "response": answer,
        "satisfactory": satisfactory,
        "score": float(critic.get("score") or 0),
        "path": "agentic",
        "trace": trace,
        "critic": critic,
        "rounds_used": len([t for t in trace if t.get("agent") == "query"]),
        "evidence_tail": evidence[-3:],
    }


async def run_multiagent_teacher_qa(
    *,
    client: OpenAI,
    db_url: str,
    iddocente: int,
    question: str,
    teacher_name: str = "",
    max_rounds: int = MAX_ROUNDS,
) -> Dict[str, Any]:
    started = datetime.now()
    # 1) Deterministic hybrid shortcuts (human-like)
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
            "plan": {"path": "deterministic", "teacher": teacher_name},
            "elapsed_ms": elapsed_ms,
            "mode": "hybrid_deterministic",
            "timestamp": datetime.now().isoformat(),
        }

    prior = (det or {}).get("trace") or []
    # 2) Agentic loop until satisfactory
    agentic = await run_agentic_loop(
        client=client,
        db_url=db_url,
        iddocente=iddocente,
        question=question,
        max_rounds=max_rounds,
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
        "plan": {"path": "hybrid_agentic", "teacher": teacher_name},
        "elapsed_ms": elapsed_ms,
        "mode": "hybrid_agentic",
        "timestamp": datetime.now().isoformat(),
    }
