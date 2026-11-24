#!/usr/bin/env python3
"""Eval Responses API endpoint vs the same 55Q report used for Mao/hybrid."""
from __future__ import annotations

import json
import re
import time
import unicodedata
import urllib.request

REP = "scripts/mao_compare_50_report.json"
URL = "http://127.0.0.1:8000/api/v1/chat/teacher/responses"
DB = "postgresql://postgres:123@127.0.0.1:5432/aca_2"
ID = 3
OUT = "/tmp/responses_hybrid_55_results.json"


def fold(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(s or ""))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def score(answer: str, expect: dict) -> str:
    value = expect.get("value")
    name = expect.get("name")
    nums = [
        float(x.replace(",", "."))
        for x in re.findall(r"\b\d+(?:[.,]\d+)?\b", answer or "")
    ]
    hit = value is None or any(abs(n - float(value)) < 0.051 for n in nums)
    name_ok = True
    if name:
        toks = [t for t in fold(name).split() if len(t) > 2][:3]
        name_ok = all(t in fold(answer) for t in toks) if toks else True
    if hit and (name_ok or "compartida" in fold(answer) or "empate" in fold(answer)):
        return "bien"
    if hit or name_ok:
        return "parcial"
    return "mal"


def post(msg: str) -> dict:
    payload = {
        "user_id": str(ID),
        "message": msg,
        "db_url": DB,
        "conversation_id": f"resp-{abs(hash(msg)) % 10**8}",
        "context": {"role": "teacher", "iddocente": ID},
        "max_rounds": 6,
        "model": "gpt-5-mini",
        "reasoning_effort": "medium",
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read().decode())
    data["_ms"] = int((time.time() - t0) * 1000)
    return data


def main():
    rep = json.load(open(REP))
    tallies = {"bien": 0, "parcial": 0, "mal": 0, "error": 0}
    by_tag = {}
    rows = []
    for i, r in enumerate(rep["results"], 1):
        q = r["question"]
        try:
            data = post(q)
            ans = data.get("response") or ""
            v = score(ans, r.get("expect") or {})
            mode = data.get("mode")
            ms = data.get("_ms") or data.get("elapsed_ms")
            rounds = data.get("rounds_used")
        except Exception as e:
            v, ans, mode, ms, rounds = "error", str(e), "", 0, 0
        tallies[v] += 1
        tag = r.get("tag") or "?"
        by_tag.setdefault(tag, {"bien": 0, "parcial": 0, "mal": 0, "error": 0})
        by_tag[tag][v] += 1
        print(f"[{i}/55] {v} rounds={rounds} ms={ms} :: {q[:65]}")
        print(f"   ans={(ans or '')[:160].replace(chr(10), ' ')}")
        rows.append(
            {
                "id": r["id"],
                "tag": tag,
                "question": q,
                "expect": r.get("expect"),
                "verdict": v,
                "answer": ans,
                "mode": mode,
                "ms": ms,
                "rounds": rounds,
                # prior baselines from report
                "mao_verdict": (r.get("mao") or {}).get("verdict"),
                "old_multi_verdict": (r.get("multi") or {}).get("verdict"),
            }
        )
    print("TALLIES", tallies)
    print("BY_TAG", json.dumps(by_tag, indent=2))
    summary = {
        "tallies": tallies,
        "by_tag": by_tag,
        "model": "gpt-5-mini",
        "reasoning_effort": "medium",
        "api": "responses",
        "baseline_report_tallies": rep.get("tallies"),
    }
    open(OUT, "w").write(
        json.dumps({"summary": summary, "results": rows}, ensure_ascii=False, indent=2)
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
