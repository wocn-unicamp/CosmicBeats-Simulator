"""
Train DQN agent for NTN-MEC satellite routing.

Usage:
    python scripts/train_drl.py [--timesteps N]

Saves model to models/drl_agent.zip
"""

import sys
import os
import argparse

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from src.schedulers.drl_gym_env import NTNMECEnv


def train(timesteps: int = 50_000):
    print(f"=== DQN Training: NTN-MEC Satellite Routing ===")
    print(f"Timesteps : {timesteps:,}")
    print(f"Obs space : {NTNMECEnv().observation_space}")
    print(f"Act space : {NTNMECEnv().action_space}")
    print()

    os.makedirs("models", exist_ok=True)

    train_env = Monitor(NTNMECEnv())
    eval_env  = Monitor(NTNMECEnv())

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path="models/",
        eval_freq=5_000,
        n_eval_episodes=500,
        deterministic=True,
        verbose=1,
    )

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=1e-3,
        buffer_size=20_000,
        learning_starts=1_000,
        batch_size=128,
        gamma=0.0,                   # single-step env — no discounting
        tau=1.0,                     # hard target update
        target_update_interval=500,
        train_freq=4,
        exploration_fraction=0.30,   # 30% of training in exploration
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=1,
        seed=42,
    )

    model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=True)

    save_path = "models/drl_agent"
    model.save(save_path)
    print(f"\nModelo salvo em: {save_path}.zip")

    # Quick evaluation
    print("\n=== Avaliação final (1000 episódios, determinístico) ===")
    obs, _ = eval_env.reset()
    total_reward = 0.0
    n_eval = 1000
    for _ in range(n_eval):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, _ = eval_env.reset()

    print(f"Recompensa média: {total_reward / n_eval:.4f}")
    print("Esperado > 0.80 para agente treinado (roteamento bem-sucedido)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50_000)
    args = parser.parse_args()
    train(args.timesteps)
