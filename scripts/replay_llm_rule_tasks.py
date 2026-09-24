#!/usr/bin/env python3
"""Testa se as decisoes conformes do LLM nas regras ineditas foram do modelo.

O camera-ready afirma que o LLM mantem 20/20 nas regras ineditas (sementes 0-2
da execucao antiga, logs/generalization/heldout_lm). Essa execucao nao gravava a
origem da decisao, e uma falha de API vira DROP. Numa regra de descarte, esse DROP
e conforme por acidente (IX.8). A campanha completa que resolveria isso (v2) esta
parada: a API do modelo responde 503 em ~90% das chamadas.

Este teste pergunta exatamente o que a frase precisa. Para cada tarefa com regra
das sementes 0-2 da execucao antiga, ele verifica se o MODELO, diante do mesmo
prompt, toma uma decisao conforme.

1. replay (sem API): roda o simulador com um LLM de reproducao que devolve as
   decisoes gravadas no CSV antigo. A reproducao so vale se o CSV gerado bater
   tarefa a tarefa com o gravado. Nesse caso, a frota em cada tarefa com regra e a
   que o modelo viu. Salva o prompt e a frota de cada tarefa com regra, e marca as
   que o pre-filtro do LLM resolve sem chamar a API.

2. probe (com API): envia cada prompt com a configuracao do llm_scheduler.py
   (temperatura 0, JSON, 128 tokens), insistindo ate obter HTTP 200. A resposta e
   pontuada com a mesma funcao do simulador (semantic_rules.is_compliant), com o
   satellite_id tal como o modelo o devolveu.

O prompt e a configuracao de geracao nao mudaram desde a execucao antiga: entre
55b5bec e o HEAD, o llm_scheduler.py so mudou nas retentativas e no decision_source.

Uso:
    venv/bin/python scripts/replay_llm_rule_tasks.py replay --seeds 0-2
    venv/bin/python scripts/replay_llm_rule_tasks.py probe
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import io
import json
import os
import random
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.chdir(REPO)

SRC_DIR = "logs/generalization/heldout_lm"
OUT_DIR = "logs/diagnostics/llm_rule_replay"
KEY_COLS = ("task_id", "arrival_time_s", "region", "anomaly", "decision_sat_id",
            "success", "semantic_compliant")


def recorded(seed: int):
    with open(f"{SRC_DIR}/seed{seed}/mec_metrics_LLM.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def replay(seed: int, config: str) -> dict:
    rows = recorded(seed)
    by_id = {int(r["task_id"]): r for r in rows}

    os.environ.update({"MEC_ENGINE": "LLM", "MEC_SEED": str(seed),
                       "MEC_OUTDIR": os.path.join(OUT_DIR, "replay", f"seed{seed}"),
                       "MEC_RULE_SPLIT": "heldout", "MEC_ARRIVAL_RATE": "4.0"})
    os.makedirs(os.environ["MEC_OUTDIR"], exist_ok=True)

    import src.ai_logic as al
    from src.schedulers.llm_scheduler import LLMScheduler

    captured = []

    class ReplayLLM(LLMScheduler):
        """Devolve a decisao gravada; guarda prompt e frota das tarefas com regra."""
        def decide(self, task_dict, fleet, battery_safety_pct=20.0):
            rec = by_id[int(task_dict["id"])]
            if task_dict.get("semantic_anomaly"):
                # Mesmo pre-filtro do LLMScheduler.decide: sem satelite valido na
                # regiao da tarefa, o LLM descarta sem chamar a API.
                prefilter = not any(s.get("region") == task_dict.get("region")
                                    and s.get("battery_pct", 100.0) > battery_safety_pct
                                    for s in fleet)
                captured.append({"seed": seed, "task_id": int(rec["task_id"]),
                                 "anomaly": rec["anomaly"].strip(),
                                 "region": task_dict.get("region"),
                                 "prefilter": prefilter,
                                 "old_decision": rec["decision_sat_id"].strip(),
                                 "old_compliant": int(rec["semantic_compliant"]),
                                 "fleet": [dict(s) for s in fleet],
                                 "prompt": None if prefilter else self._build_prompt(task_dict, fleet)})
            sid = rec["decision_sat_id"].strip()
            return int(sid) if sid.lstrip("-").isdigit() else (sid or None)

    al.LLMScheduler = ReplayLLM
    random.seed(seed)
    from src.sim.simulator import Simulator
    with contextlib.redirect_stdout(io.StringIO()):
        Simulator(config).execute()

    with open(os.path.join(os.environ["MEC_OUTDIR"], "mec_metrics_LLM.csv"), newline="") as fh:
        got = list(csv.DictReader(fh))
    norm = lambda rs: [tuple(r[c].strip() for c in KEY_COLS) for r in rs]
    identical = norm(got) == norm(rows)
    return {"seed": seed, "rule_tasks": len(captured), "identical": identical,
            "tasks": len(rows), "captured": captured if identical else []}


def parse_like_scheduler(raw: str):
    """Mesmo parsing do LLMScheduler.decide. Devolve (satellite_id, ok)."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*?\}', raw, re.DOTALL)
        if not m:
            return None, False
        try:
            parsed = json.loads(m.group())
        except json.JSONDecodeError:
            return None, False
    return parsed.get("satellite_id"), True


def probe(max_tries: int = 60) -> list:
    import requests
    import warnings
    from dotenv import load_dotenv
    from src.schedulers.llm_scheduler import LLM_MODEL
    from src.semantic_rules import is_compliant
    warnings.filterwarnings("ignore")
    load_dotenv(".env")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent"
    headers = {"x-goog-api-key": os.environ["GEMINI_API_KEY"]}

    items = [i for p in sorted(glob.glob(f"{OUT_DIR}/prompts_seed*.json")) for i in json.load(open(p))]
    # Primeiro as ambiguas: um DROP antigo pode ter sido falha de API. Um roteamento
    # para satelite so pode ter vindo do modelo.
    items.sort(key=lambda i: (i["old_decision"] != "", i["seed"], i["task_id"]))
    out = []
    for item in items:
        res = {k: item[k] for k in ("seed", "task_id", "anomaly", "region", "prefilter",
                                    "old_decision", "old_compliant")}
        if item["prefilter"]:
            # O pre-filtro nao chama o modelo: a decisao e DROP por construcao.
            res.update({"source": "prefilter", "new_decision": None,
                        "new_compliant": int(is_compliant(item["anomaly"], None, item["fleet"]))})
        else:
            body = {"contents": [{"parts": [{"text": item["prompt"]}]}],
                    "generationConfig": {"temperature": 0.0,
                                         "responseMimeType": "application/json",
                                         "maxOutputTokens": 128}}
            tries, r = 0, None
            while tries < max_tries:
                tries += 1
                try:
                    r = requests.post(url, headers=headers, json=body, verify=False, timeout=60)
                    if r.status_code == 200:
                        break
                except requests.RequestException:
                    r = None
                # Intervalo fixo: com ~90% de 503, recuar mais so atrasa. 4 req/min
                # fica bem abaixo do limite de 10 RPM do llm_scheduler.py.
                time.sleep(15)
            res["tries"] = tries
            if r is not None and r.status_code == 200:
                raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                sid, ok = parse_like_scheduler(raw)
                res.update({"source": "model" if ok else "parse_failure",
                            "new_decision": sid, "raw": raw[:300],
                            "new_compliant": int(ok and is_compliant(item["anomaly"], sid, item["fleet"]))})
            else:
                res.update({"source": "api_failure", "new_decision": None, "new_compliant": None})
        print(json.dumps({k: v for k, v in res.items() if k != "raw"}), flush=True)
        out.append(res)
        time.sleep(4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["replay", "probe"])
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--config", default="configs/config.json")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.phase == "replay":
        from run_experiments import parse_seeds
        seeds = parse_seeds(args.seeds)
        if len(seeds) > 1:
            # Uma semente por processo: o simulador guarda estado global.
            import subprocess
            rc = 0
            for s in seeds:
                rc |= subprocess.call([sys.executable, __file__, "replay", "--seeds", str(s),
                                       "--config", args.config])
            return rc
        res = replay(seeds[0], args.config)
        if res.get("captured"):
            with open(f"{OUT_DIR}/prompts_seed{res['seed']}.json", "w") as fh:
                json.dump(res["captured"], fh, indent=2, ensure_ascii=False)
        print(json.dumps({k: v for k, v in res.items() if k != "captured"}))
        return 0 if res["identical"] else 1

    results = probe()
    with open(f"{OUT_DIR}/probe_results.json", "w") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"[json] {OUT_DIR}/probe_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
