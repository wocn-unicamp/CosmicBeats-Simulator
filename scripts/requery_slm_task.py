#!/usr/bin/env python3
"""Reenvia a API o prompt exato de uma tarefa do SLM v3 e mostra a resposta final.

Motivo (docs X.5 e X.6): o CSV guarda a origem da decisao, mas nao o texto da
resposta. Para saber o que houve numa falha de formato, o prompt e reconstruido
por replay (as decisoes gravadas reproduzem a frota do momento) e reenviado com a
configuracao do protocolo v3. Foi assim que se viu que as 5 falhas do SLM nas
regras ineditas sao respostas em lista, [{...}], cada uma apontando um satelite na
regiao exigida.

Usa uma chamada de API por execucao.

Uso:
    venv/bin/python scripts/requery_slm_task.py 0 32     # semente 0, tarefa 32
"""

import contextlib
import csv
import io
import os
import random
import sys
import tempfile
import warnings

import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
warnings.filterwarnings("ignore")

SRC_DIR = "logs/generalization/heldout_slm_v3"


def main() -> int:
    seed, task = int(sys.argv[1]), int(sys.argv[2])
    rows = {int(r["task_id"]): r for r in csv.DictReader(
        open(f"{SRC_DIR}/seed{seed}/mec_metrics_SLM.csv", newline=""))}
    os.environ.update({"MEC_ENGINE": "SLM", "MEC_SEED": str(seed),
                       "MEC_OUTDIR": tempfile.mkdtemp(prefix="requery_"),
                       "MEC_RULE_SPLIT": "heldout", "MEC_ARRIVAL_RATE": "4.0"})
    import src.ai_logic as al
    from src.schedulers.llm_scheduler import build_prompt
    from src.schedulers.slm_scheduler import MAX_OUTPUT_TOKENS, SLM_MODEL, SLMScheduler
    captured = {}

    class Replay(SLMScheduler):
        def _decide_route(self, t):
            if int(t["id"]) == task:
                captured["prompt"] = build_prompt(t, t["slm_fleet_snapshot"])
            rec = rows[int(t["id"])]
            t["decision_source"] = rec["decision_source"]
            sid = rec["decision_sat_id"].strip()
            return int(sid) if sid else None

    al.SLMScheduler = Replay
    random.seed(seed)
    from src.sim.simulator import Simulator
    with contextlib.redirect_stdout(io.StringIO()):
        Simulator("configs/config.json").execute()
    if "prompt" not in captured:
        print(f"tarefa {task} nao encontrada na semente {seed}")
        return 1

    from dotenv import load_dotenv
    load_dotenv(".env")
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{SLM_MODEL}:generateContent",
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, verify=False, timeout=120,
        json={"contents": [{"parts": [{"text": captured["prompt"]}]}],
              "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json",
                                   "maxOutputTokens": MAX_OUTPUT_TOKENS}})
    print("HTTP", r.status_code)
    if r.status_code == 200:
        cand = r.json()["candidates"][0]
        final = "".join(p.get("text", "") for p in cand["content"]["parts"] if not p.get("thought"))
        print("finish:", cand.get("finishReason"))
        print("resposta final:", repr(final[:400]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
