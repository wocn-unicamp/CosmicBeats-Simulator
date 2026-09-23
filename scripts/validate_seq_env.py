#!/usr/bin/env python3
"""Valida o NTNMECSeqEnv contra o simulador, tarefa a tarefa.

O NTNMECEnv original treinava o DRL numa distribuicao de estados que nao batia com
a simulacao (docs, Parte IV.4): 17,4% de descartes forcados no treino contra
praticamente nenhum na simulacao. Um ambiente sequencial com o mesmo defeito
produziria um agente otimizado para o problema errado.

Por isso a exigencia aqui e comparacao EXATA, nao estatistica. O ambiente consome
o gerador aleatorio na mesma ordem que ai_logic; com a mesma semente, o ORACLE
dentro dele deve tomar exatamente as mesmas decisoes que o OracleScheduler no
simulador — mesma tarefa, mesmo instante, mesma regiao, mesma anomalia, mesmo
satelite, mesmo desfecho de conformidade.

Fontes de referencia (ja geradas pelo simulador, motor ORACLE):
  logs/loadsweep/rate*/seed*/          6 taxas x 5 sementes, regras conhecidas
  logs/generalization/seen/seed*/      20 sementes, lambda = 4
  logs/generalization/heldout/seed*/   20 sementes, lambda = 4, regras ineditas

Uso:
    python3 scripts/validate_seq_env.py
"""

from __future__ import annotations

import csv
import glob
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.schedulers.drl_seq_env import NTNMECSeqEnv, oracle_action  # noqa: E402


def env_rows(rate: float, seed: int, split: str):
    env = NTNMECSeqEnv(arrival_rate=rate, rule_split=split)
    env.reset(seed=seed)
    while env.task is not None:
        env.step(oracle_action(env))
    return [(r["task_id"], float(r["t"]), r["region"], r["anomaly"],
             "" if r["sat"] is None else str(r["sat"]), r["success"], r["compliant"])
            for r in env.log]


def sim_rows(path: str):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append((int(r["task_id"]), float(r["arrival_time_s"]), r["region"],
                         r["anomaly"], r["decision_sat_id"], int(r["success"]),
                         int(r["semantic_compliant"])))
    return rows


def cases():
    for path in sorted(glob.glob(os.path.join(REPO, "logs/loadsweep/rate*/seed*/mec_metrics_ORACLE.csv"))):
        m = re.search(r"rate([0-9p]+)/seed(\d+)/", path)
        yield float(m.group(1).replace("p", ".")), int(m.group(2)), "seen", path
    for split in ("seen", "heldout"):
        for path in sorted(glob.glob(os.path.join(REPO, f"logs/generalization/{split}/seed*/mec_metrics_ORACLE.csv"))):
            seed = int(re.search(r"seed(\d+)/", path).group(1))
            yield 4.0, seed, split, path


def main() -> int:
    total = ok = 0
    first_bad = None
    for rate, seed, split, path in cases():
        total += 1
        a, b = env_rows(rate, seed, split), sim_rows(path)
        if a == b:
            ok += 1
        elif first_bad is None:
            diff = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            first_bad = (rate, seed, split, len(a), len(b), diff,
                         a[diff] if diff < len(a) else None, b[diff] if diff < len(b) else None)
    print(f"casos comparados: {total}   identicos tarefa a tarefa: {ok}   divergentes: {total - ok}")
    if first_bad:
        rate, seed, split, na, nb, i, x, y = first_bad
        print(f"primeira divergencia: lambda={rate} semente={seed} split={split} "
              f"(env {na} tarefas, simulador {nb})")
        print(f"  linha {i}:\n    env:        {x}\n    simulador:  {y}")
    return 0 if ok == total and total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
