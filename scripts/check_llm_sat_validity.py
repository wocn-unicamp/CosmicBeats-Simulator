#!/usr/bin/env python3
"""Confere se cada satelite escolhido pelo LLM era valido no momento da decisao.

Motivo (docs X.4): o artigo dizia que o motor "re-validates the target against the
live fleet". Nem o LLMScheduler nem o simulador validam: o satellite_id do modelo e
aceito como veio. Este script mede o impacto disso.

Metodo: replay. O simulador roda com um LLM de reproducao que devolve as decisoes
gravadas no CSV; o CSV gerado precisa bater com o gravado (senao a frota nao e a
mesma). A cada decisao, o satelite escolhido e conferido contra a frota do momento:
existe, bateria > 20% e RAM livre suficiente.

Resultado em 27/09: 14 corridas identicas ao gravado; das 905 decisoes que
rotearam, 1 invalida (v2 semente 9, tarefa 68, satelite 15 com 19,6% de bateria).

Uso:
    venv/bin/python scripts/check_llm_sat_validity.py logs/mec_metrics_LLM.csv 0 seen
    venv/bin/python scripts/check_llm_sat_validity.py \
        logs/generalization/heldout_lm_v2/seed9/mec_metrics_LLM.csv 9 heldout
"""

import contextlib
import csv
import io
import json
import os
import random
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)


def main() -> int:
    src, seed, split = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    rows = list(csv.DictReader(open(src, newline="")))
    by_id = {int(r["task_id"]): r for r in rows}
    out = tempfile.mkdtemp(prefix="satvalid_")
    os.environ.update({"MEC_ENGINE": "LLM", "MEC_SEED": str(seed), "MEC_OUTDIR": out,
                       "MEC_RULE_SPLIT": split, "MEC_ARRIVAL_RATE": "4.0"})
    import src.ai_logic as al
    from src.schedulers.llm_scheduler import LLMScheduler
    bad = []

    class Replay(LLMScheduler):
        def decide(self, t, fleet, floor=20.0):
            sid = by_id[int(t["id"])]["decision_sat_id"].strip()
            if not sid:
                return None
            sid = int(sid) if sid.lstrip("-").isdigit() else sid
            sat = next((s for s in fleet if s["id"] == sid), None)
            if sat is None or sat["battery_pct"] <= floor or sat["ram_free"] < t.get("ram", 0):
                bad.append({"task": t["id"], "sat": sid, "anomaly": t.get("semantic_anomaly", ""),
                            "battery_pct": None if sat is None else sat["battery_pct"],
                            "ram_free": None if sat is None else sat["ram_free"]})
            return sid

    al.LLMScheduler = Replay
    random.seed(seed)
    from src.sim.simulator import Simulator
    with contextlib.redirect_stdout(io.StringIO()):
        Simulator("configs/config.json").execute()
    keys = ("task_id", "decision_sat_id", "success", "semantic_compliant")
    got = [tuple(r[k].strip() for k in keys)
           for r in csv.DictReader(open(os.path.join(out, "mec_metrics_LLM.csv"), newline=""))]
    same = got == [tuple(r[k].strip() for k in keys) for r in rows]
    routed = sum(1 for r in rows if r["decision_sat_id"].strip())
    print(json.dumps({"src": src, "identical": same, "routed": routed, "invalid": bad}))
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
