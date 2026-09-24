#!/usr/bin/env python3
"""Testa por que o SLM falha no parse nas regras de regiao (IX.9.2b).

Hipotese: o raciocinio da Gemma esgota maxOutputTokens=1024 nas tarefas que
exigem ler uma regra de regiao, e a resposta chega cortada. O teste direto e o
finishReason da API: MAX_TOKENS confirma, STOP refuta.

Para isso, cada falha precisa ser repetida com o prompt EXATO que o modelo
recebeu. O prompt inclui o estado da frota (bateria, RAM, carga solar), que por sua
vez depende de todas as decisoes anteriores. Por isso o script tem duas fases.

1. replay (sem API): roda o simulador com um SLM de reproducao que devolve as
   decisoes gravadas no CSV da campanha v2, com a mesma temporizacao assincrona
   do SLM real. A reproducao so vale se o CSV gerado bater tarefa a tarefa com o
   gravado; caso contrario a semente e descartada. Quando bate, a frota vista em
   cada falha e a mesma que o modelo viu. O prompt de cada tarefa com
   parse_failure e salvo.

2. probe (com API): reenvia cada prompt salvo com a configuracao original
   (temperatura 0, maxOutputTokens 1024) e com um orcamento maior (4096),
   gravando finishReason, uso de tokens e se a resposta e aproveitavel.

Nao altera o slm_scheduler.py: o ai_logic importa esse modulo mesmo quando o
motor e o LLM, e ha uma campanha do LLM em andamento.

Uso:
    venv/bin/python scripts/replay_slm_failures.py replay --seeds 0-9
    venv/bin/python scripts/replay_slm_failures.py probe
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
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.chdir(REPO)

SRC_DIR = "logs/generalization/heldout_lm_v2"
OUT_DIR = "logs/diagnostics/slm_parse_failures"
KEY_COLS = ("task_id", "arrival_time_s", "region", "anomaly", "decision_sat_id",
            "success", "semantic_compliant", "decision_source")


def recorded(seed: int):
    with open(f"{SRC_DIR}/seed{seed}/mec_metrics_SLM.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def replay(seed: int, config: str) -> dict:
    rows = recorded(seed)
    by_id = {int(r["task_id"]): r for r in rows}
    if not any(r["decision_source"].strip() == "parse_failure" for r in rows):
        return {"seed": seed, "failures": 0, "skipped": "sem parse_failure"}

    os.environ.update({"MEC_ENGINE": "SLM", "MEC_SEED": str(seed),
                       "MEC_OUTDIR": os.path.join(OUT_DIR, "replay", f"seed{seed}"),
                       "MEC_RULE_SPLIT": "heldout", "MEC_ARRIVAL_RATE": "4.0"})
    os.makedirs(os.environ["MEC_OUTDIR"], exist_ok=True)

    import src.ai_logic as al
    from src.schedulers.slm_scheduler import SLMScheduler

    captured = []

    class ReplaySLM(SLMScheduler):
        """Devolve a decisao gravada; guarda o prompt das falhas de parse."""
        def _decide_route(self, task_dict):
            fleet = task_dict.get("slm_fleet_snapshot", [])
            rec = by_id[int(task_dict["id"] if "id" in task_dict else task_dict["task_id"])]
            task_dict["decision_source"] = rec["decision_source"].strip()
            if task_dict["decision_source"] == "parse_failure":
                captured.append({"seed": seed, "task_id": int(rec["task_id"]),
                                 "anomaly": rec["anomaly"].strip(),
                                 "prompt": self._build_prompt(task_dict, fleet)})
            sid = rec["decision_sat_id"].strip()
            return int(sid) if sid.lstrip("-").isdigit() else (sid or None)

    al.SLMScheduler = ReplaySLM
    random.seed(seed)
    from src.sim.simulator import Simulator
    with contextlib.redirect_stdout(io.StringIO()):
        Simulator(config).execute()

    with open(os.path.join(os.environ["MEC_OUTDIR"], "mec_metrics_SLM.csv"), newline="") as fh:
        got = list(csv.DictReader(fh))
    norm = lambda rs: [tuple(r[c].strip() for c in KEY_COLS) for r in rs]
    identical = norm(got) == norm(rows)
    return {"seed": seed, "failures": len(captured), "identical": identical,
            "tasks": len(rows), "captured": captured if identical else []}


def probe(budgets=(1024, 4096)) -> list:
    import requests
    import warnings
    from dotenv import load_dotenv
    from src.schedulers.slm_scheduler import SLM_MODEL, _parse_gemma_json
    warnings.filterwarnings("ignore")
    load_dotenv(".env")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{SLM_MODEL}:generateContent"
    headers = {"x-goog-api-key": os.environ["GEMINI_API_KEY"]}

    out = []
    for path in sorted(glob.glob(f"{OUT_DIR}/prompts_seed*.json")):
        for item in json.load(open(path)):
            for budget in budgets:
                body = {"contents": [{"parts": [{"text": item["prompt"]}]}],
                        "generationConfig": {"temperature": 0.0, "maxOutputTokens": budget}}
                res = {"seed": item["seed"], "task_id": item["task_id"],
                       "anomaly": item["anomaly"], "max_output_tokens": budget}
                for attempt in range(6):
                    r = requests.post(url, headers=headers, json=body, verify=False, timeout=120)
                    if r.status_code == 200:
                        break
                    time.sleep(min(10 * 2 ** attempt, 120))
                res["http"] = r.status_code
                if r.status_code == 200:
                    j = r.json()
                    cand = j["candidates"][0]
                    parts = cand.get("content", {}).get("parts", [])
                    answer = [p.get("text", "") for p in parts if not p.get("thought", False)]
                    raw = "".join(answer) if answer else "".join(p.get("text", "") for p in parts)
                    combined = raw if raw.lstrip().startswith("{") else "{" + raw
                    parsed = _parse_gemma_json(combined)
                    u = j.get("usageMetadata", {})
                    res.update({"finish_reason": cand.get("finishReason"),
                                "thoughts_tokens": u.get("thoughtsTokenCount"),
                                "output_tokens": u.get("candidatesTokenCount"),
                                "parsed": parsed is not None,
                                "action": (parsed or {}).get("action"),
                                "target_region": (parsed or {}).get("target_region"),
                                "answer_tail": raw[-200:]})
                print(json.dumps({k: v for k, v in res.items() if k != "answer_tail"}), flush=True)
                out.append(res)
                time.sleep(5)   # abaixo do limite de 14 RPM do SLM
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["replay", "probe"])
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--config", default="configs/config.json")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.phase == "replay":
        from run_experiments import parse_seeds
        # Uma semente por processo: o simulador guarda estado global entre execucoes.
        seeds = parse_seeds(args.seeds)
        if len(seeds) > 1:
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
        return 0 if res.get("identical", True) else 1

    results = probe()
    with open(f"{OUT_DIR}/probe_results.json", "w") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"[json] {OUT_DIR}/probe_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
