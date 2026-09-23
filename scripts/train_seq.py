#!/usr/bin/env python3
"""Treina o agente sequencial (item 3.3) no NTNMECSeqEnv.

Ablacao: MESMA observacao, MESMA recompensa, MESMO ambiente — varia so gamma.
  gamma = 0     o agente otimiza cada decisao isoladamente (como o DRL do artigo)
  gamma = 0,99  horizonte efetivo de ~100 decisoes, ~a janela inteira sob carga
Se gamma>0 nao melhorar nada, a resposta ao Revisor 3 e que, neste modelo, a
decisao sequencial nao importa — tambem e resultado.

O peso de throughput w e varrido para tracar a fronteira conformidade x
throughput (ver docstring do ambiente).

Uso:
    python3 scripts/train_seq.py --gamma 0.99 --w 0.25 --arrival-rate 12
Saida: models/seq/g<gamma>_w<w>_l<lambda>/best_model.zip (+ evaluations.npz)
"""

from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from stable_baselines3 import DQN  # noqa: E402
from stable_baselines3.common.callbacks import EvalCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from src.schedulers.drl_seq_env import NTNMECSeqEnv  # noqa: E402


def run_name(gamma: float, w: float, rate: float) -> str:
    return f"g{gamma:g}_w{w:g}_l{rate:g}".replace(".", "p")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--w", type=float, required=True, help="peso de throughput")
    ap.add_argument("--arrival-rate", type=float, default=12.0, help="tarefas/min (nominal)")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = os.path.join(REPO, "models", "seq", run_name(args.gamma, args.w, args.arrival_rate))
    os.makedirs(outdir, exist_ok=True)

    def make():
        return Monitor(NTNMECSeqEnv(arrival_rate=args.arrival_rate, throughput_weight=args.w))

    # Sementes de TREINO e de AVALIACAO sao disjuntas das de TESTE (0-19): sem
    # semente explicita o ambiente sorteia sempre a partir de TRAIN_SEED_OFFSET.
    train_env, eval_env = make(), make()
    # Semente explicita tambem no ambiente de avaliacao: sem ela a escolha do
    # "melhor modelo" pelo EvalCallback nao seria reprodutivel.
    eval_env.reset(seed=args.seed + 10_000)
    eval_cb = EvalCallback(eval_env, best_model_save_path=outdir, log_path=outdir,
                           eval_freq=10_000, n_eval_episodes=20, deterministic=True,
                           verbose=0)
    # Hiperparametros do DQN do artigo, exceto gamma (a variavel da ablacao) e o
    # buffer/treino, dimensionados para episodios de ~150 decisoes.
    model = DQN("MlpPolicy", train_env,
                learning_rate=1e-3, buffer_size=100_000, learning_starts=5_000,
                batch_size=128, gamma=args.gamma, tau=1.0, target_update_interval=1_000,
                train_freq=4, exploration_fraction=0.3, exploration_final_eps=0.05,
                policy_kwargs={"net_arch": [64, 64]}, seed=args.seed, verbose=0)
    model.learn(total_timesteps=args.timesteps, callback=eval_cb)
    model.save(os.path.join(outdir, "final_model"))
    print(f"[ok] {run_name(args.gamma, args.w, args.arrival_rate)} -> {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
