#!/usr/bin/env python3
"""Gera os graficos da apresentacao do WPMC 2026 a partir dos dados do repositorio.

Nenhum numero e digitado a mao: tudo sai dos mesmos arquivos que o
scripts/verify_paper_numbers.py confere contra o artigo (commit e4de793).

  fig_acr_canonical.png   corrida canonica, ACR com IC de Wilson     (slide 13)
  fig_multiseed.png       20 sementes, BASELINE vs DRL e teto de 1/3 (slide 14)
  fig_generalization.png  regras conhecidas vs ineditas, 6 motores   (slide 16)
  fig_latency.png         mediana de latencia vs janela entre tarefas (slide 18)
  fig_frontier.png        gamma>0: agentes vs envelope miope conjunto (slide 20)

Visual alinhado ao modelo de referencia (paper/slides/...reference.pdf): fundo
#F9F9F9, cinzas, fontes Golos Text. Cor so para o que importa: azul = motores
que leem a regra em linguagem natural.

Uso:
    venv/bin/python scripts/build_slide_figures.py [--out paper/slides/figs]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from analyze_results import compliance_table, latency_stats, load_runs, paired_keys  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

BG = "#F9F9F9"
INK = "#1A1A1A"
GREY = "#9A9A9A"      # motores que nao leem a regra
GREY_LIGHT = "#CFCFCF"
BLUE = "#1E4E8C"      # motores de linguagem (leem a regra)
BLUE_LIGHT = "#8FB0DA"
ACCENT = "#B0413E"    # linhas de referencia (teto, janela)


def setup_style() -> None:
    # Golos Text nao tem letras gregas (gamma); DejaVu Sans entra como fallback por glifo.
    family = (["Golos Text", "DejaVu Sans"]
              if any("Golos" in f.name for f in font_manager.fontManager.ttflist) else ["DejaVu Sans"])
    plt.rcParams.update({
        "font.family": family, "font.size": 11, "axes.facecolor": BG, "figure.facecolor": BG,
        "axes.edgecolor": "#6A6A6A", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "regular",
        "savefig.facecolor": BG, "savefig.dpi": 220,
    })


def err(k, n, wilson):
    p = 100 * k / n
    return p, [[p - 100 * wilson[0]], [100 * wilson[1] - p]]


def fig_acr_canonical(out):
    runs = load_runs(os.path.join(REPO, "logs"))
    comp = compliance_table(runs, paired_keys(runs))
    engines = ["BASELINE", "DRL", "SLM", "LLM"]
    fig, ax = plt.subplots(figsize=(385 / 72, 225 / 72))
    for i, e in enumerate(engines):
        a = comp[e]["acr"]
        p, yerr = err(a["k"], a["n"], a["wilson"])
        color = BLUE if e in ("SLM", "LLM") else GREY
        ax.bar(i, p, color=color, width=0.62, zorder=2)
        ax.errorbar(i, p, yerr=yerr, color=INK, capsize=6, lw=1.4, zorder=3)
        ax.text(i, 3, f"{a['k']}/{a['n']}", ha="center", va="bottom", color="white",
                fontsize=12, fontweight="bold", zorder=4)
    ax.set_xticks(range(4), engines)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Anomaly compliance, ACR (%)")
    ax.grid(axis="y", color="#E2E2E2", zorder=0)
    ax.set_title("Canonical run · bars = Wilson 95% CI", fontsize=10.5, color="#555555")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_acr_canonical.png")); plt.close(fig)


def fig_multiseed(out):
    runs = load_runs(os.path.join(REPO, "logs/multiseed"), ["BASELINE", "DRL"])
    keys = paired_keys(runs)
    comp = compliance_table(runs, keys)
    fig, ax = plt.subplots(figsize=(345 / 72, 225 / 72))
    for i, e in enumerate(["BASELINE", "DRL"]):
        a = comp[e]["acr"]
        p, yerr = err(a["k"], a["n"], a["wilson"])
        ax.bar(i, p, color=GREY, width=0.55, zorder=2)
        ax.errorbar(i, p, yerr=yerr, color=INK, capsize=7, lw=1.4, zorder=3)
        ax.text(i, p + yerr[1][0] + 1.5, f"{p:.1f}%", ha="center", fontsize=12, color=INK)
    ax.axhline(100 / 3, color=ACCENT, ls="--", lw=1.6, zorder=1)
    ax.text(1.36, 100 / 3 + 1.0, "blind-policy\nbound = 1/3", color=ACCENT, ha="left", va="bottom", fontsize=10.5)
    ax.set_xticks([0, 1], ["BASELINE", "DRL"])
    ax.set_xlim(-0.5, 1.95); ax.set_ylim(0, 50)
    ax.set_ylabel("ACR (%)")
    ax.grid(axis="y", color="#E2E2E2", zorder=0)
    seeds = len({k.split(":")[0] for k in keys})
    ax.set_title(f"{seeds} seeds · {comp['BASELINE']['acr']['n']} rule tasks · Wilson 95% CI",
                 fontsize=10.5, color="#555555")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_multiseed.png")); plt.close(fig)


def fig_generalization(out):
    t20 = json.load(open(os.path.join(REPO, "logs/generalization/table20.json")))["splits"]["seen"]
    unseen = json.load(open(os.path.join(REPO, "logs/generalization/table4_unseen_s0-9.json")))
    runs = load_runs(os.path.join(REPO, "logs"))
    canon = compliance_table(runs, paired_keys(runs))
    engines = ["BASELINE", "ORACLE", "DRL", "DRL_ONEHOT", "SLM", "LLM"]
    labels = ["BASELINE", "ORACLE\n(rule table)", "DRL", "DRL-OH\n(rule one-hot)", "SLM", "LLM"]
    seen = {}
    for e in engines:
        if e in ("SLM", "LLM"):
            a = canon[e]["acr"]; seen[e] = (a["k"], a["n"], a["wilson"])
        else:
            v = t20["engines"][e]; seen[e] = (v["k"], v["n"], v["wilson"])
    fig, ax = plt.subplots(figsize=(610 / 72, 265 / 72))
    w = 0.38
    for i, e in enumerate(engines):
        lm = e in ("SLM", "LLM")
        p, yerr = err(*seen[e])
        ax.bar(i - w / 2, p, w, color=BLUE_LIGHT if lm else GREY_LIGHT, zorder=2,
               label="Design-time rules" if i == 0 else None)
        ax.errorbar(i - w / 2, p, yerr=yerr, color="#555555", capsize=3, lw=1, zorder=3)
        u = unseen["engines"][e]
        p2, yerr2 = err(u["k"], unseen["n"], u["wilson"])
        ax.bar(i + w / 2, p2, w, color=BLUE if lm else GREY, zorder=2,
               label="Unseen rules" if i == 0 else None)
        ax.errorbar(i + w / 2, p2, yerr=yerr2, color=INK, capsize=3, lw=1, zorder=3)
        ax.text(i + w / 2, p2 + yerr2[1][0] + 2, f"{p2:.1f}%" if p2 < 100 else "100%",
                ha="center", fontsize=11.5, color=INK, fontweight="bold" if lm else "normal")
    ax.set_xticks(range(len(engines)), labels, fontsize=11)
    ax.set_ylim(0, 108); ax.set_yticks(range(0, 101, 25))
    ax.set_ylabel("Anomaly compliance (%)")
    ax.grid(axis="y", color="#E2E2E2", zorder=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=11, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_generalization.png")); plt.close(fig)


def fig_latency(out):
    runs = load_runs(os.path.join(REPO, "logs"))
    lat = latency_stats(runs, paired_keys(runs))
    # Janela entre tarefas: intervalo medio efetivo 5/(1-exp(-5*lambda)), lambda = 4/min.
    window_ms = 1000 * 5.0 / (1 - math.exp(-5.0 * 4.0 / 60.0))
    engines = ["BASELINE", "DRL", "SLM", "LLM"]
    notes = {"BASELINE": "conditional checks", "DRL": "MLP inference",
             "SLM": "NPU benchmark (modelled)", "LLM": "cloud round-trip (measured)"}
    fig, ax = plt.subplots(figsize=(530 / 72, 210 / 72))
    for i, e in enumerate(engines):
        m = lat[e]["median"]
        lm = e in ("SLM", "LLM")
        ax.barh(i, m, color=BLUE if lm else GREY, height=0.55, zorder=2)
        if e == "LLM":
            ax.errorbar(m, i, xerr=[[m - lat[e]["p25"]], [lat[e]["p75"] - m]], color=INK,
                        capsize=5, lw=1.2, zorder=3)
        label = f"{m:.3f} ms" if m < 1 else f"{m:.0f} ms"
        if m > 20:   # barra longa: texto dentro dela, para nao cruzar a linha da janela
            ax.text(0.007, i, f"{label}  ·  {notes[e]}", va="center", fontsize=11, color="white")
        else:
            ax.text(m * 1.6, i, f"{label}  ·  {notes[e]}", va="center", fontsize=11, color=INK)
    ax.axvline(window_ms, color=ACCENT, ls="--", lw=1.6)
    ax.text(window_ms / 1.15, -0.62, f"mean time between tasks: {window_ms / 1000:.1f} s",
            color=ACCENT, ha="right", fontsize=10.5)
    ax.set_xscale("log"); ax.set_xlim(0.005, 60000)
    ax.set_yticks(range(4), engines); ax.invert_yaxis()
    ax.set_xlabel("Median decision latency (ms, log scale)")
    ax.grid(axis="x", color="#E2E2E2", zorder=0)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_latency.png")); plt.close(fig)


def fig_frontier(out):
    runs = {t: json.load(open(os.path.join(REPO, f"docs/frontier_l12_s{t}_pooled.json")))
            for t in (42, 43, 44)}
    hull = runs[42]["hull_g0"]
    fig, ax = plt.subplots(figsize=(298 / 72, 285 / 72))
    hx, hy = zip(*hull)
    ax.plot(hx, hy, "--", color=GREY, lw=2, label="myopic envelope (24 agents, γ = 0)")
    markers = {42: "o", 43: "s", 44: "^"}
    for t, run in runs.items():
        for name, v in run["agents"].items():
            w005 = "_w0p05_" in name
            ax.plot(v["throughput"], v["acr"], markers[t], ms=9 if w005 else 6,
                    color=BLUE if w005 else BLUE_LIGHT, mec=INK if w005 else "none",
                    label=(f"γ = 0.99, w = 0.05 · seed {t}" if w005 else None))
    ax.set_xlim(55, 77); ax.set_ylim(82, 101)
    ax.set_xlabel("Throughput (%)")
    ax.set_ylabel("Anomaly compliance (%)")
    ax.grid(color="#E2E2E2")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, fontsize=10)
    ax.set_title("Heavy battery load (12 tasks/min)", fontsize=10.5, color="#555555")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_frontier.png")); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "paper/slides/figs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    setup_style()
    for f in (fig_acr_canonical, fig_multiseed, fig_generalization, fig_latency, fig_frontier):
        f(args.out)
        print(f"[ok] {f.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
