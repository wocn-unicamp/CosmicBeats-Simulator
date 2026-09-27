#!/usr/bin/env python3
"""Tabela IV do camera-ready: conformidade nas regras ineditas, seis motores pareados.

Fontes (docs Parte X):
  deterministicos  logs/generalization/heldout/seed*/          (BASELINE, ORACLE, DRL, DRL_ONEHOT)
  SLM              logs/generalization/heldout_slm_v3/seed*/   (protocolo v3)
  LLM              logs/generalization/heldout_lm_v2/seed*/    (10 sementes)  ou
                   logs/generalization/heldout_lm/seed*/       (corrida antiga, 3 sementes)

A regra de corte (X.3) decide 10 ou 3 sementes; este script so calcula.
Metrica principal: conformidade operacional, como o simulador pontua (falha de
API ou de formato conta como nao conforme). As falhas sao contadas a parte.

Uso:
    venv/bin/python scripts/table4_unseen.py --seeds 0-9
    venv/bin/python scripts/table4_unseen.py --seeds 0-2 --llm-dir logs/generalization/heldout_lm
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from analyze_results import mcnemar_exact, wilson_ci  # noqa: E402
from run_experiments import parse_seeds  # noqa: E402
from src import semantic_rules as sr  # noqa: E402

ENGINES = ["BASELINE", "ORACLE", "DRL", "DRL_ONEHOT", "SLM", "LLM"]


def load(seeds, dirs):
    data = {e: {} for e in ENGINES}
    sources = {e: collections.Counter() for e in ENGINES}
    finish = collections.Counter()
    for s in seeds:
        for e in ENGINES:
            path = os.path.join(REPO, dirs[e], f"seed{s}", f"mec_metrics_{e}.csv")
            with open(path, newline="") as fh:
                for r in csv.DictReader(fh):
                    if e in ("SLM", "LLM"):
                        sources[e][r.get("decision_source", "model") or "model"] += 1
                        if e == "SLM":
                            finish[r.get("finish_reason", "")] += 1
                    if r["anomaly"]:
                        data[e][f"{s}:{r['task_id']}"] = (int(r["semantic_compliant"]), r["anomaly"],
                                                         r.get("decision_source", "model") or "model")
    return data, sources, finish


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--llm-dir", default="logs/generalization/heldout_lm_v2")
    ap.add_argument("--slm-dir", default="logs/generalization/heldout_slm_v3")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    dirs = {e: "logs/generalization/heldout" for e in ENGINES}
    dirs["SLM"], dirs["LLM"] = args.slm_dir, args.llm_dir

    data, sources, finish = load(seeds, dirs)
    keys = sorted(set.intersection(*[set(v) for v in data.values()]))
    n = len(keys)
    region = [k for k in keys if not sr.BY_TOKEN[data["BASELINE"][k][1]].must_drop]
    fault = [k for k in keys if k not in region]

    out = {"seeds": seeds, "n": n, "n_region": len(region), "n_fault": len(fault), "engines": {}}
    print(f"sementes {seeds} | tarefas com regra pareadas: {n} (regiao {len(region)}, falha {len(fault)})")
    print(f"{'motor':11s} {'conformes':>10s} {'%':>6s} {'Wilson 95%':>14s} {'regiao':>7s} {'falha':>6s} {'vs BASELINE':>18s}")
    for e in ENGINES:
        k = sum(data[e][x][0] for x in keys)
        lo, hi = wilson_ci(k, n)
        b01 = sum(1 for x in keys if data[e][x][0] == 1 and data["BASELINE"][x][0] == 0)
        b10 = sum(1 for x in keys if data[e][x][0] == 0 and data["BASELINE"][x][0] == 1)
        p = mcnemar_exact(b01, b10) if e != "BASELINE" else None
        reg = sum(data[e][x][0] for x in region)
        flt = sum(data[e][x][0] for x in fault)
        out["engines"][e] = {"k": k, "pct": 100 * k / n, "wilson": [lo, hi], "region": reg,
                             "fault": flt, "b01": b01, "b10": b10, "p_vs_baseline": p}
        mc = "" if p is None else f"{b01}/{b10} p={p:.2g}"
        print(f"{e:11s} {k:4d}/{n:<5d} {100*k/n:6.1f} [{100*lo:5.1f},{100*hi:5.1f}] {reg:4d}/{len(region):<3d}"
              f"{flt:3d}/{len(fault):<3d} {mc:>18s}")

    s01 = sum(1 for x in keys if data["LLM"][x][0] == 1 and data["SLM"][x][0] == 0)
    s10 = sum(1 for x in keys if data["LLM"][x][0] == 0 and data["SLM"][x][0] == 1)
    out["llm_vs_slm"] = {"b01": s01, "b10": s10, "p": mcnemar_exact(s01, s10)}
    print(f"\nLLM vs SLM: {s01}/{s10} discordantes, p={mcnemar_exact(s01, s10):.3g}")
    oracle_disc = sum(1 for x in keys if data["ORACLE"][x][0] != data["BASELINE"][x][0])
    out["oracle_vs_baseline_discordant"] = oracle_disc
    print(f"ORACLE vs BASELINE: {oracle_disc} pares discordantes")

    for e in ("SLM", "LLM"):
        fails = [x for x in keys if data[e][x][2] != "model"]
        out["engines"][e]["decision_sources_all_tasks"] = dict(sources[e])
        out["engines"][e]["rule_task_failures"] = {x: data[e][x][2] for x in fails}
        print(f"{e}: origem de todas as decisoes {dict(sources[e])}; tarefas com regra "
              f"nao decididas pelo modelo: {len(fails)} {sorted(set(data[e][x][2] for x in fails))}")
    out["slm_finish_reason"] = dict(finish)
    print(f"SLM finish_reason (todas as decisoes): {dict(finish)}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"[json] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
