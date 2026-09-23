#!/usr/bin/env python3
"""Tabela de generalizacao: cada motor nas regras conhecidas vs. nas ineditas.

E o experimento que responde o Revisor 3. A pergunta nao e "quem acerta mais nas
tres regras do artigo" — nisso uma tabela de tres entradas ganha de todo mundo.
A pergunta e o que sobra quando chega uma restricao que ninguem enumerou em
tempo de projeto.

O que cada coluna deve mostrar, e por que:

  ORACLE      100% nas conhecidas (a tabela as contem) e, nas ineditas, colapso
              exato para o BASELINE — nao erra com excecao, erra em silencio.
  DRL         teto cego nos dois splits: so as regras de descarte sao
              recuperaveis a partir de um bit.
  DRL_ONEHOT  paridade de informacao nas conhecidas; nas ineditas o one-hot vira
              vetor nulo e ele volta ao caso cego.
  SLM / LLM   unicos capazes de aplicar uma restricao a partir da frase que
              acompanha a tarefa.

A fracao de regras de descarte e igual nos dois splits (1/3), entao o teto cego
e o mesmo e as colunas sao comparaveis. Sem isso a diferenca entre splits
mediria a composicao do corpus, nao a capacidade dos motores.

Uso:
    python3 scripts/generalization_table.py --seeds 0-9 --engines BASELINE ORACLE DRL DRL_ONEHOT
    python3 scripts/generalization_table.py --seeds 0-4 --latex
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_results import mcnemar_exact, wilson_ci  # noqa: E402
from run_experiments import parse_seeds  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = ["seen", "heldout"]
DEFAULT_ENGINES = ["BASELINE", "ORACLE", "DRL", "DRL_ONEHOT"]


def run_one(engine: str, seed: int, split: str, outroot: str, config: str) -> str:
    """Executa uma combinacao se ainda nao houver resultado, e devolve o CSV."""
    outdir = os.path.join(outroot, split, f"seed{seed}")
    csv_path = os.path.join(outdir, f"mec_metrics_{engine}.csv")
    if os.path.exists(csv_path):
        return csv_path
    os.makedirs(outdir, exist_ok=True)
    subprocess.run(
        [sys.executable, "main.py", "--engine", engine, "--seed", str(seed),
         "--rule-split", split, "--outdir", outdir, "--config", config],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    return csv_path


def collect(engines, seeds, outroot, config):
    """{split: {engine: {chave: conforme}}} apenas sobre tarefas com anomalia."""
    import csv as _csv
    data = {s: {e: {} for e in engines} for s in SPLITS}
    for split in SPLITS:
        for seed in seeds:
            for engine in engines:
                path = run_one(engine, seed, split, outroot, config)
                if not os.path.exists(path):
                    continue
                with open(path, newline="") as fh:
                    for row in _csv.DictReader(fh):
                        if row["anomaly"]:
                            data[split][engine][f"{seed}:{row['task_id']}"] = int(
                                row["semantic_compliant"])
    return data


def paired(data, split, engines):
    """Chaves presentes em todos os motores — o pareamento exige isso."""
    common = None
    for e in engines:
        ks = set(data[split][e])
        common = ks if common is None else common & ks
    return sorted(common or [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--engines", nargs="+", default=DEFAULT_ENGINES)
    ap.add_argument("--outroot", default="logs/generalization")
    ap.add_argument("--config", default="configs/config.json")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    outroot = args.outroot if os.path.isabs(args.outroot) else os.path.join(REPO, args.outroot)
    data = collect(args.engines, seeds, outroot, args.config)

    rows, payload = [], {"seeds": seeds, "engines": args.engines, "splits": {}}
    for split in SPLITS:
        keys = paired(data, split, args.engines)
        payload["splits"][split] = {"n_anomaly": len(keys), "engines": {}}
        for e in args.engines:
            k = sum(data[split][e][key] for key in keys)
            lo, hi = wilson_ci(k, len(keys)) if keys else (0.0, 0.0)
            payload["splits"][split]["engines"][e] = {
                "k": k, "n": len(keys), "wilson": [lo, hi]}
        rows.append((split, keys))

    print("=" * 74)
    print(f"GENERALIZACAO — {len(seeds)} sementes; ACR sobre tarefas com anomalia")
    print("=" * 74)
    header = f"\n{'engine':12s}" + "".join(
        f"{s + ' (n=' + str(payload['splits'][s]['n_anomaly']) + ')':>28s}" for s in SPLITS)
    print(header)
    for e in args.engines:
        line = f"{e:12s}"
        for s in SPLITS:
            d = payload["splits"][s]["engines"][e]
            pct = 100 * d["k"] / d["n"] if d["n"] else 0.0
            lo, hi = (100 * x for x in d["wilson"])
            line += f"{d['k']:>6d}/{d['n']:<4d}{pct:5.1f}% [{lo:4.1f},{hi:5.1f}]"
        print(line)

    # O contraste que carrega o argumento: o quanto cada motor PERDE ao sair das
    # regras enumeradas. Pareado dentro do split, via McNemar exato.
    print("\nqueda das conhecidas para as ineditas (pontos percentuais):")
    for e in args.engines:
        a = payload["splits"]["seen"]["engines"][e]
        b = payload["splits"]["heldout"]["engines"][e]
        pa = 100 * a["k"] / a["n"] if a["n"] else 0.0
        pb = 100 * b["k"] / b["n"] if b["n"] else 0.0
        print(f"  {e:12s} {pa:5.1f}% -> {pb:5.1f}%   delta = {pb - pa:+6.1f} pp")

    print("\nMcNemar exato, motor vs. BASELINE, dentro de cada split:")
    for split in SPLITS:
        keys = paired(data, split, args.engines)
        for e in args.engines:
            if e == "BASELINE":
                continue
            b01 = sum(1 for k in keys if data[split][e][k] == 1 and data[split]["BASELINE"][k] == 0)
            b10 = sum(1 for k in keys if data[split][e][k] == 0 and data[split]["BASELINE"][k] == 1)
            p = mcnemar_exact(b01, b10)
            flag = "SIGNIFICATIVO" if p < 0.05 else "n.s."
            print(f"  {split:8s} {e:12s} discordantes {b01:3d}/{b10:<3d} p={p:.4f}  {flag}")

    if args.latex:
        print("\n% --- tabela de generalizacao ---")
        print("\\begin{tabular}{lcc}\n\\hline")
        print("Engine & ACR, design-time rules & ACR, unseen rules \\\\\n\\hline")
        for e in args.engines:
            a = payload["splits"]["seen"]["engines"][e]
            b = payload["splits"]["heldout"]["engines"][e]
            fa = 100 * a["k"] / a["n"] if a["n"] else 0.0
            fb = 100 * b["k"] / b["n"] if b["n"] else 0.0
            name = e.replace("_", r"\_")   # Python 3.10 nao aceita barra em f-string
            print(f"{name} & {fa:.1f}\\% & {fb:.1f}\\% \\\\")
        print("\\hline\n\\end{tabular}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n[json] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
