#!/usr/bin/env python3
"""Fronteira conformidade x throughput dos agentes sequenciais, com IC por bootstrap.

A pergunta do item 3.3: gamma>0 da uma fronteira MELHOR, ou os agentes com
gamma=0,99 so ocupam outros pontos da mesma curva? Um ponto com ACR maior e
throughput menor nao responde isso sozinho.

Comparacao justa: cada agente gamma=0,99 contra o ENVELOPE CONVEXO SUPERIOR da
familia gamma=0 (+ ORACLE) no mesmo throughput. O envelope, e nao so os pontos,
porque misturar politicas aleatoriamente tambem e uma opcao disponivel para quem
usa a familia miope — o envelope e o melhor que ela consegue.

Incerteza: bootstrap por SEMENTE (as 20 sementes de teste sao a unidade
independente; tarefas dentro de uma semente nao sao). Cada reamostragem recalcula
todos os agentes, o envelope e o ganho.

Uso:
    python3 scripts/frontier_seq.py --rate 12 --boot 2000 --fig docs/fig_fronteira_l12.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from src.schedulers.drl_seq_env import NTNMECSeqEnv, oracle_action  # noqa: E402


def per_seed(policy, rate, seeds):
    """{semente: (acr_k, acr_n, thr_k, thr_n)} — contagens, nao taxas."""
    out = {}
    for s in seeds:
        env = NTNMECSeqEnv(arrival_rate=rate, throughput_weight=0.0)
        obs, _ = env.reset(seed=s)
        while env.task is not None:
            obs, *_ = env.step(policy(env, obs))
        a = [x for x in env.log if x["anomaly"]]
        p = [x for x in env.log if not x["anomaly"]]
        out[s] = (sum(x["compliant"] for x in a), len(a), sum(x["success"] for x in p), len(p))
    return out


def point(counts, seeds):
    ak = sum(counts[s][0] for s in seeds); an = sum(counts[s][1] for s in seeds)
    tk = sum(counts[s][2] for s in seeds); tn = sum(counts[s][3] for s in seeds)
    return 100 * tk / tn, 100 * ak / an


def upper_hull(pts):
    P = sorted(pts); hull = []
    for p in P:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def interp(hull, x):
    for (x1, y1), (x2, y2) in zip(hull, hull[1:]):
        if x1 <= x <= x2:
            return y1 + (y2 - y1) * (x - x1) / (x2 - x1) if x2 > x1 else max(y1, y2)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate", type=float, default=12.0)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--fig", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--train-seed", type=int, default=42,
                    help="semente de treino dos agentes a comparar (42 = modelos originais)")
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    from stable_baselines3 import DQN
    agents = {"ORACLE": ("g0", per_seed(lambda e, o: oracle_action(e), args.rate, seeds))}
    suffix = "" if args.train_seed == 42 else f"_s{args.train_seed}"
    for path in sorted(glob.glob(os.path.join(REPO, "models/seq/*/best_model.zip"))):
        name = os.path.basename(os.path.dirname(path))
        seeded = "_s" in name.rsplit("_l", 1)[-1]
        if (suffix and not name.endswith(suffix)) or (not suffix and seeded):
            continue
        fam = "g99" if name.startswith("g0p99_") else "g0"
        m = DQN.load(path)
        agents[name] = (fam, per_seed(lambda e, o, m=m: int(m.predict(o, deterministic=True)[0]),
                                      args.rate, seeds))

    def gains(sample):
        H = upper_hull([point(c, sample) for f, c in agents.values() if f == "g0"])
        res = {}
        for name, (f, c) in agents.items():
            if f != "g99":
                continue
            x, y = point(c, sample)
            h = interp(H, x)
            res[name] = (x, y, None if h is None else y - h)
        return res, H

    base, H = gains(seeds)
    rng = random.Random(12345)
    boots = {n: [] for n in base}
    for _ in range(args.boot):
        sample = [rng.choice(seeds) for _ in seeds]
        g, _ = gains(sample)
        for n, (_, _, gain) in g.items():
            if gain is not None:
                boots[n].append(gain)

    print(f"lambda nominal {args.rate:g}/min, {len(seeds)} sementes, bootstrap por semente x{args.boot}")
    print("envelope gamma=0 (+ORACLE):", [(round(x, 1), round(y, 1)) for x, y in H])
    print(f"\n{'agente gamma=0,99':18s} {'thr':>6s} {'ACR':>6s} {'ganho sobre o envelope':>24s} {'IC95% bootstrap':>18s}")
    summary = {}
    for n, (x, y, gain) in sorted(base.items(), key=lambda kv: kv[1][0]):
        b = sorted(boots[n])
        if gain is None or len(b) < 50:
            print(f"{n:18s} {x:5.1f}% {y:5.1f}%  {'fora do alcance do envelope':>24s}")
            continue
        lo, hi = b[int(0.025 * len(b))], b[int(0.975 * len(b)) - 1]
        flag = "  *" if lo > 0 else ""
        print(f"{n:18s} {x:5.1f}% {y:5.1f}% {gain:+22.1f} pp  [{lo:+5.1f}, {hi:+5.1f}]{flag}")
        summary[n] = {"throughput": x, "acr": y, "gain_pp": gain, "ci95": [lo, hi]}
    print("\n* = IC inteiramente acima de zero")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rate": args.rate, "hull_g0": H, "agents": summary}, fh, indent=2)
    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for fam, color, label in (("g0", "#b0413e", "γ = 0 (míope)"), ("g99", "#1f5fa8", "γ = 0,99")):
            pts = sorted(point(c, seeds) for n, (f, c) in agents.items() if f == fam and n != "ORACLE")
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", color=color, label=label)
        hx, hy = zip(*H)
        ax.plot(hx, hy, "--", color="#b0413e", alpha=0.6, label="envelope γ = 0")
        g99 = upper_hull([point(c, seeds) for f, c in agents.values() if f == "g99"])
        ax.plot(*zip(*g99), "-", color="#1f5fa8", alpha=0.6, label="envelope γ = 0,99")
        ox, oy = point(agents["ORACLE"][1], seeds)
        ax.plot([ox], [oy], "s", color="black", label="ORACLE")
        ax.set_xlabel("Throughput (% das tarefas sem regra executadas)")
        ax.set_ylabel("ACR (% das tarefas com regra conformes)")
        ax.set_title(f"Fronteira conformidade × throughput, λ = {args.rate:g}/min nominal")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout(); fig.savefig(args.fig, dpi=160)
        print(f"[fig] {args.fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
