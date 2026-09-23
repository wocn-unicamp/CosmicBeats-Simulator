#!/usr/bin/env python3
"""Varredura de carga: em que taxa de chegada cada recurso passa a limitar.

Primeiro passo do item 3.3, em resposta ao Revisor 3 ("the DRL problem is
essentially single-step, with weak temporal coupling and no RAM bottleneck").
Antes de desenhar um agente sequencial, e preciso saber em que regime a
sequencialidade poderia importar: um recurso que nunca se esgota nao cria
nenhum incentivo a reservar capacidade para o futuro.

Uma conta de primeira ordem sugere que a bateria (2% de SoC por tarefa aceita)
limita bem antes da RAM (8 tarefas simultaneas por satelite). Este script mede
isso no simulador em vez de confiar na conta — a recarga solar e o eclipse
entram no resultado.

Por execucao, separa dois tipos de descarte:
  por recurso  descarte de tarefa SEM restricao semantica. Nenhuma regra manda
               descartar essas tarefas, entao o descarte so pode vir de falta de
               bateria ou RAM na regiao da tarefa.
  por regra    descarte de tarefa com restricao (correto se a regra e de falha).

Uso:
    python3 scripts/load_sweep.py --rates 2 4 8 12 16 24 --seeds 0-4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiments import parse_seeds  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ENGINES = ["BASELINE", "ORACLE", "DRL", "DRL_ONEHOT"]


def fmt_rate(rate: float) -> str:
    return f"{rate:g}".replace(".", "p")


def run_one(engine, seed, rate, outroot, config):
    outdir = os.path.join(outroot, f"rate{fmt_rate(rate)}", f"seed{seed}")
    csv_path = os.path.join(outdir, f"mec_metrics_{engine}.csv")
    json_path = os.path.join(outdir, f"mec_summary_{engine}.json")
    if os.path.exists(csv_path) and os.path.exists(json_path):
        return csv_path, json_path
    os.makedirs(outdir, exist_ok=True)
    subprocess.run(
        [sys.executable, "main.py", "--engine", engine, "--seed", str(seed),
         "--arrival-rate", str(rate), "--outdir", outdir, "--config", config],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    return csv_path, json_path


def summarise(csv_path, json_path):
    if not (os.path.exists(csv_path) and os.path.exists(json_path)):
        return None
    rows = list(csv.DictReader(open(csv_path, newline="")))
    summ = json.load(open(json_path))
    plain = [r for r in rows if not r["anomaly"]]
    anom = [r for r in rows if r["anomaly"]]
    batteries = [v for k, v in summ.items() if k.startswith("sat_") and k.endswith("_battery_pct")]
    return {
        "tasks": len(rows),
        "resource_drop": sum(1 for r in plain if r["success"] == "0"),
        "plain": len(plain),
        "acr_k": sum(int(r["semantic_compliant"]) for r in anom),
        "acr_n": len(anom),
        "ram": summ.get("avg_ram_utilization_pct", 0.0),
        "min_battery": min(batteries) if batteries else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rates", nargs="+", type=float, default=[2, 4, 8, 12, 16, 24])
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--engines", nargs="+", default=DEFAULT_ENGINES)
    ap.add_argument("--outroot", default="logs/loadsweep")
    ap.add_argument("--config", default="configs/config.json")
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    outroot = args.outroot if os.path.isabs(args.outroot) else os.path.join(REPO, args.outroot)

    table = {}
    for rate in args.rates:
        for engine in args.engines:
            agg = []
            for seed in seeds:
                res = summarise(*run_one(engine, seed, rate, outroot, args.config))
                if res:
                    agg.append(res)
            if not agg:
                continue
            plain = sum(a["plain"] for a in agg)
            acr_n = sum(a["acr_n"] for a in agg)
            table[(rate, engine)] = {
                "runs": len(agg),
                "tasks_per_run": st.mean(a["tasks"] for a in agg),
                "resource_drop_pct": 100 * sum(a["resource_drop"] for a in agg) / plain if plain else 0.0,
                "acr_pct": 100 * sum(a["acr_k"] for a in agg) / acr_n if acr_n else float("nan"),
                "acr_n": acr_n,
                "ram_pct": st.mean(a["ram"] for a in agg),
                "min_battery": st.mean(a["min_battery"] for a in agg),
            }

    print("=" * 96)
    print(f"VARREDURA DE CARGA — {len(seeds)} sementes por ponto")
    print("=" * 96)
    print(f"{'lambda':>7s} {'motor':11s} {'tarefas':>8s} {'desc.recurso':>13s} "
          f"{'ACR':>8s} {'(n)':>6s} {'RAM med':>8s} {'bat.min':>8s}")
    for rate in args.rates:
        for engine in args.engines:
            r = table.get((rate, engine))
            if not r:
                continue
            print(f"{rate:>6g}/m {engine:11s} {r['tasks_per_run']:8.1f} "
                  f"{r['resource_drop_pct']:12.1f}% {r['acr_pct']:7.1f}% {r['acr_n']:>6d} "
                  f"{r['ram_pct']:7.1f}% {r['min_battery']:7.1f}%")
        print()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({f"{k[0]}|{k[1]}": v for k, v in table.items()}, fh, indent=2)
        print(f"[json] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
