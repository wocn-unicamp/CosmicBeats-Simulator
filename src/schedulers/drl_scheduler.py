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
        """Executa a acao da politica se ela for fisicamente viavel.

        O validador so pode usar informacao que a politica tambem tem no momento
        da decisao. Ele checa viabilidade fisica (bateria acima do piso, RAM
        suficiente) e a regiao: a da propria tarefa, salvo no braco one-hot
        diante de uma anomalia, em que a politica pode legitimamente rotear para
        outra regiao porque observa o tipo da regra.

        Historico, porque os dois erros anteriores foram reais:
        1. A primeira versao exigia sempre a regiao da tarefa, e anulava as
           decisoes corretas de roteamento cruzado (GDPR -> EUROPE a partir de
           uma tarefa de outra regiao).
        2. A correcao seguinte consultava semantic_rules pelo token da tarefa —
           forcava DROP em regra de descarte e exigia a regiao da regra. Isso
           vazava ao validador a identidade da regra, que o braco cego nao
           observa e que NENHUM braco observa numa regra inedita: 3 dos 5
           descartes "conformes" do DRL_ONEHOT nas regras ineditas eram
           forcados pelo validador, nao escolhidos pela politica.
        3. Uma versao intermediaria aceitava qualquer regiao viavel para toda
           tarefa com anomalia, nos dois bracos. Deixava de vazar a regra, mas
           mudava o braco cego publicado: a politica cega as vezes escolhe outra
           regiao numa anomalia, e o validador original anulava essa escolha.
        A conformidade continua julgada por semantic_rules, na metrica — que e
        onde o conhecimento da regra deve ficar.
        """
        if action == 3:
            return None  # DROP explicito

        if action >= len(_SAT_ORDER):
            return None

        target_def = _SAT_ORDER[action]
        fleet_by_id = {s["id"]: s for s in fleet}
        sat = fleet_by_id.get(target_def["id"])
        if sat is None:
            return None

        # Rotear para fora da propria regiao so e permitido a quem consegue
        # distinguir o tipo de regra (o braco one-hot) e so em tarefa com
        # anomalia. O braco cego fica preso a regiao da tarefa, exatamente como
        # no artigo. O que decide e o desenho de observacao do braco — nunca a
        # regra oculta da tarefa.
        cross_region_ok = self.observe_rule_type and bool(task_dict.get("semantic_anomaly"))
        if not cross_region_ok and sat.get("region") != task_dict.get("region"):
            return None
        if sat.get("battery_pct", 0.0) <= battery_safety_pct:
            return None
        if sat.get("ram_free", 0.0) < task_dict.get("ram", 0):
            return None

        return sat["id"]
