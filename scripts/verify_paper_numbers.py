#!/usr/bin/env python3
"""Confere cada valor numerico afirmado no artigo contra os dados que o geraram.

Existe porque a versao submetida tinha numeros que nao batiam com os artefatos
commitados (a utilizacao de RAM nos summaries era residuo de uma metrica antiga)
e porque a revisao introduziu dezenas de valores novos — intervalos de
confianca, p-valores, quantis de latencia. Qualquer edicao no .tex ou nova
execucao deve manter este script passando.

Cada checagem declara de onde o valor vem, entao a falha aponta a origem, nao
so o sintoma.

Uso:
    python3 scripts/verify_paper_numbers.py
    python3 scripts/verify_paper_numbers.py --tex Relatorio_ED2_241327.tex
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_results import (  # noqa: E402
    anomaly_breakdown, compliance_table, latency_stats, load_runs,
    pairwise_mcnemar, paired_keys, ram_utilisation,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Checker:
    def __init__(self, tex: str):
        self.tex = tex
        self.ok: list[str] = []
        self.bad: list[tuple[str, str, str]] = []

    def expect(self, what: str, needle: str, source: str) -> None:
        (self.ok.append(what) if needle in self.tex
         else self.bad.append((what, needle, source)))

    def report(self) -> int:
        for what, needle, source in self.bad:
            print(f"  FALHA  {what}")
            print(f"         esperava no .tex: {needle!r}")
            print(f"         origem do valor:  {source}")
        total = len(self.ok) + len(self.bad)
        status = "OK" if not self.bad else "FALHOU"
        print(f"\n[{status}] {len(self.ok)}/{total} valores conferem")
        return 1 if self.bad else 0


def pct1(x: float) -> str:
    return f"{x * 100:.1f}"


def check_single_run(c: Checker, logs_dir: str) -> None:
    """Corrida canonica de 69 tarefas que embasa o artigo."""
    runs = load_runs(logs_dir)
    keys = paired_keys(runs)
    src = f"{logs_dir}/mec_metrics_*.csv"
    comp = compliance_table(runs, keys)

    for engine, data in comp.items():
        for metric in ("scr", "acr"):
            lo, hi = data[metric]["wilson"]
            hi_s = "100" if hi >= 0.9999 else pct1(hi)
            c.expect(f"IC Wilson {engine} {metric.upper()}",
                     f"{pct1(lo)}--{hi_s}", src)

    expected_p = {("BASELINE", "LLM"): "0.031", ("DRL", "LLM"): "0.0625",
                  ("DRL", "SLM"): "0.250", ("BASELINE", "DRL"): "1.000"}
    for r in pairwise_mcnemar(runs, keys, anomaly_only=True):
        key = tuple(sorted((r["a"], r["b"])))
        if key in expected_p:
            got, want = r["p"], float(expected_p[key])
            assert abs(got - want) < 6e-4, f"McNemar {key}: script={got}, .tex={want}"
            c.expect(f"McNemar {key[0]} vs {key[1]}", f"$p={expected_p[key]}$", src)

    lat = latency_stats(runs, keys)
    c.expect("mediana LLM", r"\SI{955}{\milli\second}", src)
    c.expect("IQR LLM", r"\numrange{877}{1131}", src)
    c.expect("cauda de rate-limit", f"{lat['LLM']['n_outliers']} of {lat['LLM']['n']}", src)
    c.expect("mediana DRL", r"\SI{0.438}{\milli\second}", src)
    c.expect("mediana BASELINE", r"\SI{0.014}{\milli\second}", src)

    for engine, v in ram_utilisation(runs, keys).items():
        c.expect(f"RAM {engine}", f"{v['avg']:.1f}\\%", src)


def check_multiseed(c: Checker, logs_dir: str) -> None:
    """Replicacao multi-semente dos motores deterministicos."""
    if not os.path.isdir(logs_dir):
        print(f"  (pulando multi-semente: {logs_dir} nao existe)")
        return
    runs = load_runs(logs_dir)
    keys = paired_keys(runs)
    if not keys:
        return
    src = f"{logs_dir}/seed*/mec_metrics_*.csv"
    comp = compliance_table(runs, keys)
    seeds = len({k.split(":")[0] for k in keys})

    c.expect("n tarefas pareadas", f"{len(keys):,}".replace(",", "{,}") + " paired tasks", src)
    c.expect("n anomalias", f"{comp['BASELINE']['acr']['n']} anomaly-carrying", src)
    c.expect("n sementes", f"{seeds} independent", src)

    for engine in ("BASELINE", "DRL"):
        if engine not in comp:
            continue
        a = comp[engine]["acr"]
        c.expect(f"ACR multi-semente {engine}", f"{a['k'] / a['n'] * 100:.1f}\\%", src)
        lo, hi = a["wilson"]
        c.expect(f"IC multi-semente {engine}", f"{pct1(lo)}--{pct1(hi)}", src)

    for r in pairwise_mcnemar(runs, keys, anomaly_only=True):
        if {r["a"], r["b"]} == {"BASELINE", "DRL"}:
            c.expect("McNemar multi-semente", f"$p={r['p']:.3f}$", src)

    bd = anomaly_breakdown(runs, keys)["BASELINE"]
    total = sum(n for _, n in bd.values())
    hw = next(n for t, (_, n) in bd.items() if "hardware" in t)
    c.expect("teto da politica cega", f"{hw / total * 100:.1f}\\% ({hw} of {total})", src)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", default=os.path.join(REPO, "Relatorio_ED2_241327.tex"))
    ap.add_argument("--logs-dir", default=os.path.join(REPO, "logs"))
    ap.add_argument("--multiseed-dir", default=os.path.join(REPO, "logs/multiseed"))
    args = ap.parse_args()

    with open(args.tex) as fh:
        c = Checker(fh.read())

    print("verificando a corrida canonica (69 tarefas)...")
    check_single_run(c, args.logs_dir)
    print("verificando a replicacao multi-semente...")
    check_multiseed(c, args.multiseed_dir)
    return c.report()


if __name__ == "__main__":
    sys.exit(main())
