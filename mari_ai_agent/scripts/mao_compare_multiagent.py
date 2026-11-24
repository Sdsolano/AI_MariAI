#!/usr/bin/env python3
"""Compare Mao actual vs multi-agent query loop on the same questions."""
from __future__ import annotations

import json
import time
import urllib.request
from typing import Any, Dict, List

DB_URL = "postgresql://postgres:123@127.0.0.1:5432/aca_2"
IDDOCENTE = 3
MAO_URL = "http://127.0.0.1:8000/api/v1/chat/teacher/"
MULTI_URL = "http://127.0.0.1:8000/api/v1/chat/teacher/multiagent"

QUESTIONS: List[str] = [
    "cuál fue el peor estudiante del segundo periodo en matemáticas",
    "mejor estudiante del periodo 1",
    "promedio global de todas mis notas",
    "qué estudiantes necesitan atención bajo 3.0",
]


def post(url: str, payload: Dict[str, Any], timeout: int = 180) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    data["_elapsed_client_ms"] = int((time.time() - t0) * 1000)
    return data


def main() -> None:
    for i, q in enumerate(QUESTIONS, 1):
        base = {
            "user_id": str(IDDOCENTE),
            "message": q,
            "db_url": DB_URL,
            "context": {"role": "teacher", "iddocente": IDDOCENTE},
            "conversation_history": [],
        }
        print("\n" + "=" * 72)
        print(f"Q{i}: {q}")
        try:
            mao = post(
                MAO_URL,
                {**base, "conversation_id": f"cmp-mao-{i}"},
            )
            print("--- MAO actual ---")
            print(mao.get("response") or mao.get("reply"))
            print("tools:", mao.get("function_calls_made"), "ms:", mao.get("_elapsed_client_ms"))
        except Exception as e:
            print("MAO ERROR:", e)
            mao = {}

        try:
            multi = post(
                MULTI_URL,
                {
                    **base,
                    "conversation_id": f"cmp-multi-{i}",
                    "max_rounds": 6,
                },
            )
            print("--- MULTIAGENT ---")
            print(multi.get("response"))
            print(
                "satisfactory:",
                multi.get("satisfactory"),
                "score:",
                multi.get("score"),
                "rounds:",
                multi.get("rounds_used"),
                "ms:",
                multi.get("elapsed_ms") or multi.get("_elapsed_client_ms"),
            )
            for step in (multi.get("trace") or [])[-4:]:
                if step.get("agent") in ("query", "critic"):
                    print(" ", step)
        except Exception as e:
            print("MULTI ERROR:", e)


if __name__ == "__main__":
    main()
