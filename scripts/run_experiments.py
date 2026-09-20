#!/usr/bin/env python3
"""Executor multi-semente resumivel para o comparativo de orquestracao NTN-MEC.

Por que resumivel: BASELINE e DRL rodam em segundos e de graca, mas SLM e LLM
sao chamadas a API Gemini (~69 por execucao) e o gargalo do tier gratuito nao e
o limite por minuto, e sim o limite DIARIO. Uma campanha de 20 sementes precisa
de ~1.600 chamadas por engine remoto, o que nao cabe num dia. Este script deixa
rodar o que a cota do dia permite, para limpo, e retoma depois sem refazer nada.

O estado vive no proprio disco: uma combinacao (semente, engine) e considerada
concluida quando o CSV e o JSON dela existem e sao coerentes. O manifesto e so
um registro auditavel — apagar o manifesto nao faz nada ser reexecutado.

Uso:
    # planeja sem executar
    python3 scripts/run_experiments.py --seeds 0-19 --dry-run

    # motores locais (rapido, sem cota)
    python3 scripts/run_experiments.py --seeds 0-19 --engines BASELINE DRL

    # motores remotos, limitando o consumo do dia
    python3 scripts/run_experiments.py --seeds 0-19 --engines SLM LLM --max-api-runs 12
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_ENGINES = ["BASELINE", "ORACLE", "DRL"]          # deterministicos, sem rede
API_ENGINES = ["SLM", "LLM"]                 # consomem cota da API Gemini
ALL_ENGINES = LOCAL_ENGINES + API_ENGINES


def parse_seeds(spec: str) -> list[int]:
    """Aceita '0-19', '0,3,7' ou a combinacao dos dois."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))


def run_paths(outdir: str, seed: int, engine: str) -> tuple[str, str]:
    d = os.path.join(outdir, f"seed{seed}")
    return (os.path.join(d, f"mec_metrics_{engine}.csv"),
            os.path.join(d, f"mec_summary_{engine}.json"))


def is_done(outdir: str, seed: int, engine: str) -> bool:
    """Concluida = CSV com linhas, JSON valido, e contagens coerentes entre os dois.

    A checagem cruzada evita tratar como pronta uma execucao interrompida no meio
    da escrita, que deixaria um CSV truncado ao lado de um JSON ausente ou velho.
    """
    csv_path, json_path = run_paths(outdir, seed, engine)
    if not (os.path.exists(csv_path) and os.path.exists(json_path)):
        return False
    try:
        with open(csv_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        with open(json_path) as fh:
            summary = json.load(fh)
        return bool(rows) and summary.get("total_tasks") == len(rows)
    except Exception:
        return False


def execute(seed: int, engine: str, outdir: str, config: str, timeout: int) -> dict:
    """Roda uma combinacao e devolve o seu registro para o manifesto."""
    target = os.path.join(outdir, f"seed{seed}")
    os.makedirs(target, exist_ok=True)
    cmd = [sys.executable, "main.py", "--engine", engine,
           "--seed", str(seed), "--outdir", target, "--config", config]

    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
        rc, tail = proc.returncode, (proc.stdout or "")[-400:] + (proc.stderr or "")[-400:]
    except subprocess.TimeoutExpired:
        rc, tail = -1, f"timeout apos {timeout}s"

    elapsed = time.time() - started
    ok = rc == 0 and is_done(outdir, seed, engine)
    tasks = None
    if ok:
        with open(run_paths(outdir, seed, engine)[1]) as fh:
            tasks = json.load(fh).get("total_tasks")
    return {"seed": seed, "engine": engine, "ok": ok, "returncode": rc,
            "seconds": round(elapsed, 1), "total_tasks": tasks,
            "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
            "tail": None if ok else tail.strip()[-400:]}


def load_manifest(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path) as fh:
                return json.load(fh)
        except Exception:
            pass
    return {"runs": []}


def save_manifest(path: str, manifest: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0-19", help="ex.: '0-19' ou '0,3,7' (default: 0-19)")
    ap.add_argument("--engines", nargs="+", default=ALL_ENGINES, choices=ALL_ENGINES)
    ap.add_argument("--outdir", default="logs/multiseed")
    ap.add_argument("--config", default="configs/config.json")
    ap.add_argument("--timeout", type=int, default=3600, help="limite por execucao, em segundos")
    ap.add_argument("--max-api-runs", type=int, default=None,
                    help="teto de execucoes de SLM/LLM nesta invocacao (cota diaria)")
    ap.add_argument("--stop-after-failures", type=int, default=3,
                    help="para os motores remotos apos N falhas seguidas (default: 3)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(REPO, args.outdir)
    manifest_path = os.path.join(outdir, "manifest.json")

    # Ordena por semente para que cada semente fique completa em todos os engines
    # antes da proxima: uma campanha interrompida rende seeds inteiras, que e o
    # que a analise pareada consegue usar.
    pending, done = [], 0
    for seed in seeds:
        for engine in args.engines:
            if is_done(outdir, seed, engine):
                done += 1
            else:
                pending.append((seed, engine))

    api_pending = [x for x in pending if x[1] in API_ENGINES]
    print(f"sementes: {len(seeds)} | engines: {' '.join(args.engines)}")
    print(f"ja concluidas: {done} | pendentes: {len(pending)} "
          f"(remotas: {len(api_pending)}, estimadas ~{len(api_pending) * 69} chamadas de API)")
    if args.max_api_runs is not None:
        print(f"teto de execucoes remotas nesta rodada: {args.max_api_runs}")

    if args.dry_run:
        for seed, engine in pending[:40]:
            print(f"  [pendente] seed={seed} engine={engine}")
        if len(pending) > 40:
            print(f"  ... e mais {len(pending) - 40}")
        return 0
    if not pending:
        print("nada a fazer — campanha completa.")
        return 0

    manifest = load_manifest(manifest_path)
    api_used = 0
    consecutive_failures = 0

    for seed, engine in pending:
        is_api = engine in API_ENGINES
        if is_api and args.max_api_runs is not None and api_used >= args.max_api_runs:
            print(f"\nteto de execucoes remotas atingido ({args.max_api_runs}). "
                  f"Retome depois com o mesmo comando — o progresso esta no disco.")
            break

        print(f"  -> seed={seed:3d} engine={engine:9s} ... ", end="", flush=True)
        record = execute(seed, engine, outdir, args.config, args.timeout)
        manifest["runs"].append(record)
        save_manifest(manifest_path, manifest)   # grava a cada passo: queda nao perde historico

        if record["ok"]:
            print(f"ok ({record['total_tasks']} tarefas, {record['seconds']}s)")
            consecutive_failures = 0
        else:
            print(f"FALHOU (rc={record['returncode']}, {record['seconds']}s)")
            if record["tail"]:
                print(f"     {record['tail'].splitlines()[-1][:120]}")
            consecutive_failures += 1

        if is_api:
            api_used += 1
            if consecutive_failures >= args.stop_after_failures:
                print(f"\n{consecutive_failures} falhas seguidas num motor remoto — "
                      f"provavelmente a cota diaria acabou. Parando de forma limpa.")
                break

    total_ok = sum(1 for seed in seeds for e in args.engines if is_done(outdir, seed, e))
    print(f"\nconcluidas: {total_ok}/{len(seeds) * len(args.engines)} | manifesto: {manifest_path}")
    print(f"analise:    python3 scripts/analyze_results.py --logs-dir {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
