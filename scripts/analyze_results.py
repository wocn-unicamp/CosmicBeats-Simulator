#!/usr/bin/env python3
"""Análise estatística dos resultados do comparativo de orquestração NTN-MEC.

Recomputa, a partir dos CSVs por tarefa (a fonte primária), todas as métricas que
entram no artigo — com intervalos de confiança e testes pareados, que a versão
submetida não reportava.

Aceita dois layouts de entrada:
  logs/mec_metrics_{ENGINE}.csv            (corrida única, seed fixo)
  logs/seed*/mec_metrics_{ENGINE}.csv      (multi-seed; agrega preservando o par)

O desenho é PAREADO: todos os engines veem a mesma sequência de tarefas dentro de
um mesmo seed. Isso permite McNemar exato, bem mais poderoso que comparar taxas
independentes — e é o que torna viável concluir algo com poucas anomalias por run.

Uso:
    python3 scripts/analyze_results.py
    python3 scripts/analyze_results.py --latex
    python3 scripts/analyze_results.py --logs-dir logs --json out.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

ENGINES = ["BASELINE", "DRL", "SLM", "LLM"]

# Parâmetros do cenário, espelhando src/ai_logic.py (TASK_DURATION_S, mec_ram_total).
TASK_DURATION_S = 60.0
RAM_TOTAL_MB = 4096.0
SIM_DURATION_S = 1200.0

# Rótulos legíveis para os três tipos de restrição semântica.
ANOMALY_LABELS = {
    "restricao_gdpr_europa": "GDPR (rotear p/ EUROPE)",
    "restricao_soberania_brasil": "Soberania (rotear p/ BRAZIL)",
    "falha_hardware_camera_esq": "Falha de hardware (descartar)",
}


# --------------------------------------------------------------------------- #
# Estatística exata, sem dependências externas (scipy não está disponível).
# --------------------------------------------------------------------------- #

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """IC de Wilson para uma proporção. Bem comportado com n pequeno e p nas bordas."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _log_binom_pmf(i: int, n: int, p: float) -> float:
    """log P(X = i). Em espaco logaritmico porque com n na casa do milhar
    C(n,i) estoura o float muito antes de ser multiplicado por p**i."""
    if p <= 0.0:
        return 0.0 if i == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if i == n else -math.inf
    log_c = (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1))
    return log_c + i * math.log(p) + (n - i) * math.log1p(-p)


def _binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) para X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(math.exp(_log_binom_pmf(i, n, p)) for i in range(k, n + 1))


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) para X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.exp(_log_binom_pmf(i, n, p)) for i in range(0, k + 1))


def _bisect(fn, target: float, lo: float = 0.0, hi: float = 1.0, iters: int = 200) -> float:
    """Resolve fn(p) = target em [lo, hi]; fn é monótona no intervalo."""
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """IC exato (Clopper-Pearson). Conservador; é o padrão para n pequeno.

    Obtido invertendo a CDF binomial por bisseção, evitando a dependência de scipy.
    """
    if n == 0:
        return (0.0, 0.0)
    # P(X>=k|p) cresce com p; P(X<=k|p) decresce, por isso o sinal trocado no upper.
    lower = 0.0 if k == 0 else _bisect(lambda p: _binom_sf(k, n, p), alpha / 2.0)
    upper = 1.0 if k == n else _bisect(lambda p: -_binom_cdf(k, n, p), -alpha / 2.0)
    return (lower, upper)


def mcnemar_exact(b01: int, b10: int) -> float:
    """p-valor bilateral do teste exato de McNemar sobre os pares discordantes.

    Sob H0 cada discordância é uma moeda honesta, então a contagem menor segue
    Binomial(b01 + b10, 0.5).
    """
    n = b01 + b10
    if n == 0:
        return 1.0
    tail = min(b01, b10)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(0, tail + 1)) / 2**n)


# --------------------------------------------------------------------------- #
# Carga de dados
# --------------------------------------------------------------------------- #

def load_runs(logs_dir: str) -> dict[str, dict[str, dict]]:
    """Retorna {engine: {chave_da_tarefa: linha}}.

    A chave é (seed, task_id) para que o pareamento entre engines sobreviva à
    agregação multi-seed — tarefa 7 do seed 3 só pode ser comparada com a tarefa 7
    do seed 3 dos demais engines.
    """
    runs: dict[str, dict[str, dict]] = {e: {} for e in ENGINES}
    seed_dirs = sorted(glob.glob(os.path.join(logs_dir, "seed*")))
    sources = [(os.path.basename(d), d) for d in seed_dirs] or [("seed0", logs_dir)]

    for seed, directory in sources:
        for engine in ENGINES:
            path = os.path.join(directory, f"mec_metrics_{engine}.csv")
            if not os.path.exists(path):
                continue
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    runs[engine][f"{seed}:{row['task_id']}"] = row
    return {e: r for e, r in runs.items() if r}


def paired_keys(runs: dict[str, dict[str, dict]]) -> list[str]:
    """Só as tarefas presentes em TODOS os engines — o pareamento exige isso."""
    common = None
    for rows in runs.values():
        common = set(rows) if common is None else common & set(rows)
    return sorted(common or [])


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #

def compliance_table(runs, keys):
    """SCR e ACR com IC de Wilson e Clopper-Pearson."""
    out = {}
    for engine, rows in runs.items():
        anomaly = [k for k in keys if rows[k]["anomaly"]]
        scr_k = sum(int(rows[k]["semantic_compliant"]) for k in keys)
        acr_k = sum(int(rows[k]["semantic_compliant"]) for k in anomaly)
        out[engine] = {
            "scr": {"k": scr_k, "n": len(keys),
                    "wilson": wilson_ci(scr_k, len(keys)),
                    "exact": clopper_pearson_ci(scr_k, len(keys))},
            "acr": {"k": acr_k, "n": len(anomaly),
                    "wilson": wilson_ci(acr_k, len(anomaly)),
                    "exact": clopper_pearson_ci(acr_k, len(anomaly))},
        }
    return out


def pairwise_mcnemar(runs, keys, anomaly_only: bool):
    """McNemar exato para cada par de engines, sobre a conformidade por tarefa."""
    engines = [e for e in ENGINES if e in runs]
    subset = [k for k in keys if runs[engines[0]][k]["anomaly"]] if anomaly_only else keys
    results = []
    for i, a in enumerate(engines):
        for b in engines[i + 1:]:
            b01 = sum(1 for k in subset
                      if runs[a][k]["semantic_compliant"] == "1"
                      and runs[b][k]["semantic_compliant"] == "0")
            b10 = sum(1 for k in subset
                      if runs[a][k]["semantic_compliant"] == "0"
                      and runs[b][k]["semantic_compliant"] == "1")
            results.append({"a": a, "b": b, "b01": b01, "b10": b10,
                            "n": len(subset), "p": mcnemar_exact(b01, b10)})
    return results


def latency_stats(runs, keys, outlier_ms: float = 10_000.0):
    """Distribuição da latência de decisão.

    Reportar só a média é enganoso quando a cauda é dominada por backoff de
    rate-limit da API: a média do LLM fica ~6x a mediana. Separamos as duas coisas.
    """
    out = {}
    for engine, rows in runs.items():
        vals = sorted(float(rows[k]["latency_ms"]) for k in keys)
        tail = [v for v in vals if v > outlier_ms]
        core = [v for v in vals if v <= outlier_ms]
        n = len(vals)
        out[engine] = {
            "n": n,
            "mean": st.mean(vals),
            "median": st.median(vals),
            "sd": st.stdev(vals) if n > 1 else 0.0,
            "p25": vals[n // 4],
            "p75": vals[(3 * n) // 4],
            "min": vals[0],
            "max": vals[-1],
            "n_outliers": len(tail),
            "outlier_pct": 100.0 * len(tail) / n if n else 0.0,
            "median_core": st.median(core) if core else float("nan"),
            "mean_core": st.mean(core) if core else float("nan"),
        }
    return out


def ram_utilisation(runs, keys, sim_duration: float = SIM_DURATION_S):
    """RAM ponderada no tempo, replicando src/ai_logic.py:372-393.

    Existe para validar os mec_summary_*.json, que contêm resíduo da métrica de
    snapshot antiga (12.2 para todos os engines) e divergem do artigo.
    """
    out = {}
    for engine, rows in runs.items():
        per_sat: dict[str, float] = defaultdict(float)
        for k in keys:
            r = rows[k]
            if r["success"] != "1":
                continue
            hold = min(TASK_DURATION_S, sim_duration - float(r["arrival_time_s"]))
            if hold > 0:
                per_sat[r["decision_sat_id"]] += hold * 500.0
        sats = sorted({rows[k]["decision_sat_id"] for k in keys if rows[k]["decision_sat_id"]})
        utils = [per_sat.get(s, 0.0) / (sim_duration * RAM_TOTAL_MB) * 100 for s in sats]
        out[engine] = {"per_sat": dict(zip(sats, utils)),
                       "avg": sum(utils) / len(utils) if utils else 0.0}
    return out


def anomaly_breakdown(runs, keys):
    """Conformidade por tipo de restrição — expõe o n minúsculo de cada célula."""
    out = {}
    for engine, rows in runs.items():
        per_type: dict[str, list[int]] = defaultdict(list)
        for k in keys:
            a = rows[k]["anomaly"]
            if a:
                per_type[a].append(int(rows[k]["semantic_compliant"]))
        out[engine] = {t: (sum(v), len(v)) for t, v in sorted(per_type.items())}
    return out


def energy_provenance(runs, keys):
    """Verifica se joules_cost é constante por engine (i.e. premissa, não medição)."""
    out = {}
    for engine, rows in runs.items():
        vals = {float(rows[k]["joules_cost"]) for k in keys}
        out[engine] = {"distinct_values": sorted(vals),
                       "is_constant": len(vals) == 1,
                       "total_j": sum(float(rows[k]["joules_cost"]) for k in keys)}
    return out


# --------------------------------------------------------------------------- #
# Saída
# --------------------------------------------------------------------------- #

def pct(x: float) -> str:
    return f"{x * 100:.1f}"


def report(runs, keys, sim_duration):
    comp = compliance_table(runs, keys)
    n_anom = comp[next(iter(comp))]["acr"]["n"]
    seeds = len({k.split(":")[0] for k in keys})

    print("=" * 78)
    print(f"ANÁLISE — {len(keys)} tarefas pareadas, {n_anom} com anomalia, {seeds} seed(s)")
    print("=" * 78)

    print("\n[1] CONFORMIDADE SEMÂNTICA (IC 95%)\n")
    print(f"{'engine':10s} {'SCR':>16s} {'Wilson':>16s} {'ACR':>10s} {'Wilson':>16s} {'exato':>16s}")
    for e in ENGINES:
        if e not in comp:
            continue
        s, a = comp[e]["scr"], comp[e]["acr"]
        print(f"{e:10s} {s['k']:3d}/{s['n']:<3d} {pct(s['k']/s['n']):>6s}% "
              f"[{pct(s['wilson'][0]):>5s},{pct(s['wilson'][1]):>6s}] "
              f"{a['k']:2d}/{a['n']:<2d} {pct(a['k']/a['n']):>5s}% "
              f"[{pct(a['wilson'][0]):>5s},{pct(a['wilson'][1]):>6s}] "
              f"[{pct(a['exact'][0]):>5s},{pct(a['exact'][1]):>6s}]")

    for label, anomaly_only in (("TAREFAS COM ANOMALIA", True), ("TODAS AS TAREFAS", False)):
        print(f"\n[2] McNEMAR EXATO PAREADO — {label}\n")
        for r in pairwise_mcnemar(runs, keys, anomaly_only):
            flag = "SIGNIFICATIVO" if r["p"] < 0.05 else "n.s."
            print(f"  {r['a']:9s} vs {r['b']:9s}  discordantes {r['b01']:3d}/{r['b10']:<3d} "
                  f"(n={r['n']:4d})  p={r['p']:.4f}  {flag}")

    print("\n[3] LATÊNCIA DE DECISÃO (ms)\n")
    print(f"{'engine':10s} {'mediana':>10s} {'média':>10s} {'dp':>10s} "
          f"{'p25':>9s} {'p75':>9s} {'máx':>10s} {'>10s':>8s} {'med. s/ cauda':>14s}")
    for e in ENGINES:
        if e not in runs:
            continue
        L = latency_stats(runs, keys)[e]
        print(f"{e:10s} {L['median']:10.1f} {L['mean']:10.1f} {L['sd']:10.1f} "
              f"{L['p25']:9.1f} {L['p75']:9.1f} {L['max']:10.1f} "
              f"{L['n_outliers']:3d} ({L['outlier_pct']:.0f}%) {L['median_core']:14.1f}")

    print("\n[4] RAM PONDERADA NO TEMPO (%)  [validação dos summaries]\n")
    for e, v in ram_utilisation(runs, keys, sim_duration).items():
        per = "  ".join(f"SAT-{s}={u:.1f}" for s, u in v["per_sat"].items())
        print(f"  {e:10s} média={v['avg']:5.1f}   {per}")

    print("\n[5] CONFORMIDADE POR TIPO DE RESTRIÇÃO\n")
    for e, v in anomaly_breakdown(runs, keys).items():
        cells = "  ".join(f"{ANOMALY_LABELS.get(t, t).split(' ')[0]}={k}/{n}" for t, (k, n) in v.items())
        print(f"  {e:10s} {cells}")

    print("\n[6] PROVENIÊNCIA DA ENERGIA\n")
    for e, v in energy_provenance(runs, keys).items():
        kind = "CONSTANTE (premissa de modelo)" if v["is_constant"] else "variável (medida)"
        print(f"  {e:10s} {v['distinct_values']} J/decisão → {kind}; total={v['total_j']:.3f} J")
    print()


def latex_tables(runs, keys):
    """Fragmentos prontos para colar no .tex (Tabela III com IC)."""
    comp = compliance_table(runs, keys)
    lat = latency_stats(runs, keys)
    print("% --- SCR/ACR com IC95% de Wilson ---")
    print("\\begin{tabular}{lcc}\n\\hline\nEngine & SCR [95\\% CI] & ACR [95\\% CI] \\\\\n\\hline")
    for e in ENGINES:
        if e not in comp:
            continue
        s, a = comp[e]["scr"], comp[e]["acr"]
        print(f"{e} & {pct(s['k']/s['n'])}\\% [{pct(s['wilson'][0])}--{pct(s['wilson'][1])}] "
              f"& {pct(a['k']/a['n'])}\\% [{pct(a['wilson'][0])}--{pct(a['wilson'][1])}] \\\\")
    print("\\hline\n\\end{tabular}\n")
    print("% --- Latência: mediana e IQR, não média ---")
    print("\\begin{tabular}{lccc}\n\\hline\nEngine & Median (ms) & IQR (ms) & Tail $>$10\\,s \\\\\n\\hline")
    for e in ENGINES:
        if e not in lat:
            continue
        L = lat[e]
        print(f"{e} & {L['median']:.1f} & {L['p25']:.1f}--{L['p75']:.1f} "
              f"& {L['n_outliers']}/{L['n']} \\\\")
    print("\\hline\n\\end{tabular}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--sim-duration", type=float, default=SIM_DURATION_S)
    ap.add_argument("--latex", action="store_true", help="emite fragmentos LaTeX")
    ap.add_argument("--json", metavar="PATH", help="grava o resultado bruto em JSON")
    args = ap.parse_args()

    runs = load_runs(args.logs_dir)
    if not runs:
        raise SystemExit(f"nenhum mec_metrics_*.csv encontrado em {args.logs_dir}/")
    keys = paired_keys(runs)
    if not keys:
        raise SystemExit("nenhuma tarefa comum a todos os engines — pareamento impossível")

    report(runs, keys, args.sim_duration)
    if args.latex:
        latex_tables(runs, keys)
    if args.json:
        payload = {
            "n_tasks": len(keys),
            "n_seeds": len({k.split(":")[0] for k in keys}),
            "compliance": compliance_table(runs, keys),
            "mcnemar_anomaly": pairwise_mcnemar(runs, keys, True),
            "mcnemar_all": pairwise_mcnemar(runs, keys, False),
            "latency": latency_stats(runs, keys),
            "ram": ram_utilisation(runs, keys, args.sim_duration),
            "anomaly_breakdown": anomaly_breakdown(runs, keys),
            "energy": energy_provenance(runs, keys),
        }
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2, default=list)
        print(f"[json] escrito em {args.json}")


if __name__ == "__main__":
    main()
