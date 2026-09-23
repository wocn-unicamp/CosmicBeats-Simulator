"""
DRL Scheduler — Stable Baselines3 DQN agent for NTN-MEC routing.

Loads a pre-trained DQN model (models/best_model.zip).
Train with: python scripts/train_drl.py

Action mapping (same as NTNMECEnv):
  0 → SAT-1  (BRAZIL)
  1 → SAT-2  (EUROPE)
  2 → SAT-15 (USA)
  3 → DROP

Scientific invariant preserved: agent receives has_anomaly=1 but NOT the anomaly
type — it cannot learn GDPR/sovereignty semantics. This justifies lower
semantic_compliance vs SLM/LLM in comparative results.
"""

import os
import random
import numpy as np

from src import semantic_rules

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

# Satellite order MUST match NTNMECEnv._SATS
_SAT_ORDER = [
    {"id": 1,  "region": "BRAZIL"},
    {"id": 2,  "region": "EUROPE"},
    {"id": 15, "region": "USA"},
]
_REGION_IDX = {"USA": 0, "BRAZIL": 1, "EUROPE": 2}
_RAM_TOTAL   = 4096.0
_BAT_SAFETY  = 20.0


class DRLScheduler:
    def __init__(self, observe_rule_type: bool = False):
        """observe_rule_type=False reproduz o braco do artigo. True usa o agente
        treinado com one-hot das regras conhecidas — a paridade de informacao
        pedida pelo Revisor 3. Cada variante tem o seu proprio modelo, e o
        espaco de observacao precisa casar com o que foi treinado."""
        from stable_baselines3 import DQN
        self.observe_rule_type = observe_rule_type
        subdir = "onehot" if observe_rule_type else ""
        path = os.path.abspath(os.path.join(_MODELS_DIR, subdir, "best_model.zip"))
        if not os.path.exists(path):
            flag = " --observe-rule-type" if observe_rule_type else ""
            raise FileNotFoundError(
                f"DRL model not found at {path}. "
                f"Run: python scripts/train_drl.py{flag}"
            )
        self.model = DQN.load(path)
        expected = self.model.observation_space.shape[0]
        wanted = 13 + (len(semantic_rules.ONEHOT_ORDER) if observe_rule_type else 0)
        if expected != wanted:
            raise ValueError(
                f"modelo em {path} espera observacao de {expected} dims, mas esta "
                f"variante constroi {wanted}. Modelo e variante estao trocados.")
        # DQN.load() reseeds the global RNG via SB3's set_random_seed, which would
        # shift the Poisson arrival sequence. Restore THIS run's seed (not a fixed
        # 0), otherwise every seed would collapse onto the same task stream.
        random.seed(int(os.environ.get("MEC_SEED", "0")))
        print(f">>> [ENGINE] DRL Scheduler Initialized (SB3-DQN | {path})")
        self._step = 0

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def decide(self, task_dict, fleet, battery_safety_pct=_BAT_SAFETY):
        obs = self._build_obs(task_dict, fleet)
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)

        decision_id = self._resolve_action(action, task_dict, fleet, battery_safety_pct)

        self._step += 1
        action_names = ["PREFER_SAT1(BR)", "PREFER_SAT2(EU)", "PREFER_SAT15(US)", "DROP"]
        sat_str = f"SAT {decision_id}" if decision_id is not None else "None"
        print(f"   [DRL-DQN step={self._step}] action={action_names[action]} → {sat_str}")

        return decision_id

    # ------------------------------------------------------------------ #
    # Observation builder                                                  #
    # ------------------------------------------------------------------ #

    def _build_obs(self, task_dict, fleet):
        region_oh = [0.0, 0.0, 0.0]
        ridx = _REGION_IDX.get(task_dict.get("region", "USA"), 0)
        region_oh[ridx] = 1.0
        has_anomaly = float(bool(task_dict.get("semantic_anomaly")))
        # Vetor nulo para uma restricao fora do conjunto conhecido: o agente
        # sabe que ha uma (has_anomaly=1) mas nao qual — por construcao.
        rule_oh = (semantic_rules.onehot(task_dict.get("semantic_anomaly"))
                   if self.observe_rule_type else [])

        fleet_by_id = {s["id"]: s for s in fleet}
        sat_features = []
        for sat_def in _SAT_ORDER:
            sat = fleet_by_id.get(sat_def["id"], {})
            sat_features += [
                sat.get("battery_pct", 0.0) / 100.0,
                sat.get("ram_free",    0.0) / _RAM_TOTAL,
                float(sat.get("solar_charging", True)),
            ]

        return np.array(region_oh + [has_anomaly] + rule_oh + sat_features,
                        dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Action → satellite mapping                                          #
    # ------------------------------------------------------------------ #

    def _resolve_action(self, action, task_dict, fleet, battery_safety_pct):
        """Revalida a acao escolhida contra o estado real da frota.

        A regiao exigida nem sempre e a da tarefa: GDPR manda para EUROPE e
        soberania para BRAZIL, independentemente de onde a tarefa surgiu. A
        versao anterior comparava sempre com a regiao da propria tarefa e, com
        isso, anulava justamente as decisoes corretas de roteamento cruzado —
        a politica escolhia o satelite certo e a validacao o descartava.

        Isso nao altera os resultados publicados: o agente cego de 13 dims
        aprendeu a sempre emitir DROP diante de uma anomalia, e DROP retorna
        antes desta checagem; fora das anomalias, a regiao exigida e a da tarefa.
        O bug so se manifesta quando o agente consegue ver qual e a regra.
        """
        rule = semantic_rules.get(task_dict.get("semantic_anomaly"))
        if rule is not None and rule.must_drop:
            # Descartar e a unica acao conforme; rotear nunca e valido aqui.
            return None
        required_region = (rule.required_region if rule is not None
                           else task_dict.get("region"))
        task_ram = task_dict.get("ram", 0)

        if action == 3:
            return None  # explicit DROP

        if action >= len(_SAT_ORDER):
            return None

        target_def = _SAT_ORDER[action]
        fleet_by_id = {s["id"]: s for s in fleet}
        sat = fleet_by_id.get(target_def["id"])

        if sat is None:
            return None
        if sat.get("region") != required_region:
            return None
        if sat.get("battery_pct", 0.0) <= battery_safety_pct:
            return None
        if sat.get("ram_free", 0.0) < task_ram:
            return None

        return sat["id"]
