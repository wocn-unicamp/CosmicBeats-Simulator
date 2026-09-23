#!/usr/bin/env python3
"""Avalia politicas no NTNMECSeqEnv, pareadas, nas sementes de teste 0-19.

Pareamento: no ambiente, o gerador aleatorio so e consumido pelas chegadas — as
decisoes nao o tocam. Com a mesma semente, TODAS as politicas veem exatamente a
mesma sequencia de tarefas, entao a comparacao e pareada tarefa a tarefa e admite
McNemar exato, como no resto do estudo.

Equivalencia com o simulador: o ambiente reproduz o simulador tarefa a tarefa
(scripts/validate_seq_env.py, 70/70). Avaliar aqui, nas mesmas sementes, equivale a
avaliar no simulador.

Metricas por politica:
  ACR          conformidade nas tarefas COM regra
  throughput   fracao das tarefas SEM regra que foram executadas
  retorno      recompensa do ambiente (depende de w)

Uso:
    python3 scripts/eval_seq.py --rates 8 12 --models models/seq/*/best_model.zip
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from analyze_results import mcnemar_exact, wilson_ci  # noqa: E402
from run_experiments import parse_seeds  # noqa: E402
from src.schedulers.drl_seq_env import NTNMECSeqEnv, oracle_action  # noqa: E402


def rollout(policy, rate: float, seed: int, w: float):
    env = NTNMECSeqEnv(arrival_rate=rate, throughput_weight=w)
    obs, _ = env.reset(seed=seed)
    ret = 0.0
    while env.task is not None:
        obs, r, *_ = env.step(policy(env, obs))
        ret += r
    return env.log, ret


def summarise(logs):
    anom = [x for x in logs if x["anomaly"]]
    plain = [x for x in logs if not x["anomaly"]]
    return {
        "acr_k": sum(x["compliant"] for x in anom), "acr_n": len(anom),
        "thr_k": sum(x["success"] for x in plain), "thr_n": len(plain),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rates", nargs="+", type=float, default=[8.0, 12.0])
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--reference", default=None,
                    help="politica de referencia para o McNemar (default: ORACLE)")
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)

    policies = {"ORACLE": lambda env, obs: oracle_action(env)}
    from stable_baselines3 import DQN
    for path in args.models:
        name = os.path.basename(os.path.dirname(path))
        model = DQN.load(path)
        policies[name] = (lambda m: (lambda env, obs: int(m.predict(obs, deterministic=True)[0])))(model)

    ref = args.reference or "ORACLE"
    out = {}
    for rate in args.rates:
        print("=" * 92)
        print(f"lambda nominal {rate:g}/min — {len(seeds)} sementes de teste, pareado")
        print("=" * 92)
        print(f"{'politica':22s} {'ACR':>18s} {'IC95%':>14s} {'throughput':>18s} {'McNemar vs ' + ref:>22s}")
        per_task = {}
        for name, pol in policies.items():
            k_a = n_a = k_t = n_t = 0
            per_task[name] = {}
            for seed in seeds:
                logs, _ = rollout(pol, rate, seed, w=0.0)
                s = summarise(logs)
                k_a += s["acr_k"]; n_a += s["acr_n"]; k_t += s["thr_k"]; n_t += s["thr_n"]
                for x in logs:
                    if x["anomaly"]:
                        per_task[name][f"{seed}:{x['task_id']}"] = x["compliant"]
            lo, hi = wilson_ci(k_a, n_a)
            out[f"{rate}|{name}"] = {"acr_k": k_a, "acr_n": n_a, "thr_k": k_t, "thr_n": n_t,
                                     "acr_wilson": [lo, hi]}
        for name in policies:
            d = out[f"{rate}|{name}"]
            mc = ""
            if name != ref:
                keys = sorted(set(per_task[name]) & set(per_task[ref]))
                b01 = sum(1 for k in keys if per_task[name][k] == 1 and per_task[ref][k] == 0)
                b10 = sum(1 for k in keys if per_task[name][k] == 0 and per_task[ref][k] == 1)
                p = mcnemar_exact(b01, b10)
                mc = f"{b01:3d}/{b10:<3d} p={p:.4f}"
                d["mcnemar_vs_ref"] = {"b01": b01, "b10": b10, "p": p}
            print(f"{name:22s} {d['acr_k']:5d}/{d['acr_n']:<5d} {100*d['acr_k']/d['acr_n']:5.1f}% "
                  f"[{100*d['acr_wilson'][0]:4.1f},{100*d['acr_wilson'][1]:5.1f}] "
                  f"{d['thr_k']:6d}/{d['thr_n']:<6d} {100*d['thr_k']/d['thr_n']:5.1f}%  {mc:>22s}")
        print()
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"[json] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
