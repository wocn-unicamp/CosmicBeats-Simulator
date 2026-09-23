"""NTNMECSeqEnv — ambiente SEQUENCIAL para o item 3.3 (resposta ao Revisor 3).

O `NTNMECEnv` original e um bandit de passo unico: cada estado e sorteado i.i.d.,
toda decisao termina o episodio e gamma=0. O revisor apontou exatamente isso —
"weak temporal coupling" — e pediu um cenario em que a decisao sequencial importe.

Aqui o estado persiste entre decisoes. Aceitar uma tarefa consome 2% de SoC do
satelite e ocupa 500 MB por 60 s; o episodio cobre a janela inteira de 1.200 s.
A varredura de carga (docs, Parte IX.2) mostrou que, sob pressao, a BATERIA e o
recurso que se esgota, e que ate o ORACLE perde conformidade porque uma tarefa com
regra de regiao chega quando o satelite exigido ja esta abaixo do piso. Uma
politica com gamma>0 pode aprender a reservar bateria na regiao que as regras vao
exigir — e isso que este ambiente permite medir.

FIDELIDADE AO SIMULADOR (obrigatoria, e verificada tarefa a tarefa):
  - mesma ordem de eventos por passo de 5 s: libera RAM, atualiza bateria,
    verifica chegada;
  - mesmo gerador de chegadas: no maximo uma por passo, proxima cronometrada a
    partir do passo que serviu a anterior (Parte IX.3);
  - MESMA ORDEM DE CONSUMO do gerador aleatorio (expovariate, regiao, anomalia,
    escolha da regra). Com a mesma semente, a sequencia de tarefas e identica a
    do simulador — o que permite validar o ambiente por comparacao exata, e nao
    apenas estatistica. O descompasso treino/simulacao encontrado no NTNMECEnv
    original (Parte IV.4) nao pode se repetir aqui;
  - mesma dinamica de bateria em pontos percentuais (capacidade nominal, IX.4);
  - mesmo validador do braco one-hot (IX.5): rota para outra regiao so em tarefa
    com anomalia, sem nunca consultar a regra oculta.

RECOMPENSA. Pela metrica atual, descartar uma tarefa SEM regra nao custa nada em
conformidade; um agente com visao de futuro poderia descartar todo o trafego comum
para guardar bateria para as tarefas com regra. Por isso a recompensa tem um peso
explicito de throughput `throughput_weight` (w):
    tarefa com regra   +1 se conforme, -1 se nao
    tarefa sem regra   +w se executada, 0 se descartada
    acao inviavel      -invalid_penalty (a tarefa e tratada como descartada)
Quanto vale w e decisao de pesquisa, nao de implementacao: o experimento varre w
e reporta a fronteira conformidade x throughput.
"""

from __future__ import annotations

import math
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src import semantic_rules

# Constantes espelhadas de src/ai_logic.py. Nao sao importadas de la porque
# ai_logic importa os schedulers remotos (e o dotenv) — e a validacao exata contra
# o simulador e o que garante que continuam em sincronia.
STEP_S = 5.0
HORIZON_S = 1200.0
N_STEPS = int(HORIZON_S / STEP_S)          # 240 passos, t = 0 ... 1195
ORBITAL_PERIOD_S = 5400.0
ECLIPSE_THRESHOLD = -0.10
SOLAR_CHARGE_PCT = 0.083
ECLIPSE_DRAIN_PCT = 0.040
TASK_ENERGY_COST_PCT = 2.0
BATTERY_SAFETY_PCT = 20.0
RAM_TOTAL_MB = 4096.0
TASK_RAM_MB = 500.0
TASK_DURATION_S = 60.0
ANOMALY_RATE = 0.10

# Ordem das acoes = ordem do DRLScheduler (_SAT_ORDER). SoC inicial e fase orbital
# por regiao, como atribuidos em ai_logic.setup_nodes (nodeID % 3).
SATS = (
    {"id": 1,  "region": "BRAZIL", "soc0": 90.0, "phase": 0.0},
    {"id": 2,  "region": "EUROPE", "soc0": 75.0, "phase": 1800.0},
    {"id": 15, "region": "USA",    "soc0": 85.0, "phase": 3600.0},
)
REGION_IDX = {"USA": 0, "BRAZIL": 1, "EUROPE": 2}       # one-hot da regiao (= NTNMECEnv)
SAT_OF_REGION = {s["region"]: i for i, s in enumerate(SATS)}
DROP = 3


class NTNMECSeqEnv(gym.Env):
    """Uma decisao por chegada; o episodio e a janela de 1.200 s inteira."""

    metadata = {"render_modes": []}

    def __init__(self, arrival_rate: float = 12.0, throughput_weight: float = 0.25,
                 rule_split: str = "seen", invalid_penalty: float = 0.1):
        super().__init__()
        self.lam_per_s = arrival_rate / 60.0
        self.w = throughput_weight
        self.invalid_penalty = invalid_penalty
        self.pool = semantic_rules.rules(None if rule_split == "all" else rule_split)
        n_rules = len(semantic_rules.ONEHOT_ORDER)
        # regiao(3) + has_anomaly(1) + one-hot(3) + por satelite
        # [SoC, RAM livre, sol, sin fase, cos fase](5x3) + fracao da janela(1)
        self.n_obs = 3 + 1 + n_rules + 5 * len(SATS) + 1
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_obs,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(4)   # 0=SAT-1, 1=SAT-2, 2=SAT-15, 3=DROP

    # ------------------------------------------------------------------ #
    # Dinamica — espelho de ai_logic.step()                                #
    # ------------------------------------------------------------------ #

    def _release(self, t: float) -> None:
        keep = []
        for comp_t, i, ram in self.active:
            if comp_t <= t:
                self.ram[i] = min(RAM_TOTAL_MB, self.ram[i] + ram)
            else:
                keep.append((comp_t, i, ram))
        self.active = keep

    def _battery(self, t: float) -> None:
        for i, sat in enumerate(SATS):
            v = math.sin(2 * math.pi * (t + sat["phase"]) / ORBITAL_PERIOD_S)
            eclipse = v <= ECLIPSE_THRESHOLD
            if eclipse:
                self.soc[i] = max(0.0, self.soc[i] - ECLIPSE_DRAIN_PCT)
            else:
                self.soc[i] = min(100.0, self.soc[i] + SOLAR_CHARGE_PCT)
            self.solar[i] = not eclipse

    def _advance(self):
        """Avanca passo a passo ate a proxima chegada; None ao fim da janela."""
        while self.k < N_STEPS:
            t = self.k * STEP_S
            self._release(t)
            self._battery(t)
            if self.next_t < 0:
                self.next_t = t + self.rng.expovariate(self.lam_per_s)
            if t < self.next_t:
                self.k += 1
                continue
            # Mesma ordem de consumo do RNG que ai_logic: regiao, depois anomalia.
            rv = self.rng.random()
            region = "BRAZIL" if rv < 0.33 else ("USA" if rv < 0.66 else "EUROPE")
            token = None
            if self.rng.random() < ANOMALY_RATE:
                token = self.rng.choice(self.pool).token
            self.t = t
            self.task_id += 1
            return {"id": self.task_id, "region": region, "token": token, "t": t}
        return None

    # ------------------------------------------------------------------ #
    # Observacao                                                           #
    # ------------------------------------------------------------------ #

    def _obs(self) -> np.ndarray:
        task = self.task
        region_oh = [0.0] * 3
        region_oh[REGION_IDX[task["region"]]] = 1.0
        has_anomaly = float(bool(task["token"]))
        # Vetor nulo para regra fora do conjunto de projeto: o agente sabe que ha
        # restricao (has_anomaly) mas nao qual — como o DRL_ONEHOT.
        rule_oh = semantic_rules.onehot(task["token"])
        sat_feats = []
        for i, sat in enumerate(SATS):
            ang = 2 * math.pi * (self.t + sat["phase"]) / ORBITAL_PERIOD_S
            # A fase orbital deixa o agente antecipar eclipse — sem ela, "reservar
            # bateria para depois" nao teria base observavel.
            sat_feats += [self.view_soc(i) / 100.0, self.ram[i] / RAM_TOTAL_MB,
                          float(self.solar[i]), math.sin(ang), math.cos(ang)]
        return np.array(region_oh + [has_anomaly] + rule_oh + sat_feats
                        + [self.t / HORIZON_S], dtype=np.float32)

    def view_soc(self, i: int) -> float:
        """SoC como os schedulers o VEEM: arredondado a 1 casa.

        ai_logic._build_fleet entrega round(SoC, 1). Perto do piso isso decide a
        viabilidade: com SoC real de 20,03% o scheduler ve 20,0 e a checagem
        "> 20" falha. A dinamica continua sobre o valor bruto; so a decisao e a
        observacao usam o arredondado. Sem isso, 14 de 70 corridas de validacao
        divergiam do simulador, todas sob carga alta.
        """
        return round(self.soc[i], 1)

    def fleet(self) -> list[dict]:
        """Estado da frota no formato dos schedulers do simulador."""
        return [{"id": s["id"], "region": s["region"], "battery_pct": self.view_soc(i),
                 "ram_free": self.ram[i], "solar_charging": self.solar[i]}
                for i, s in enumerate(SATS)]

    # ------------------------------------------------------------------ #
    # Gym API                                                              #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # random.Random(s) gera a mesma sequencia que random.seed(s) no simulador.
        s = seed if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        self.rng = random.Random(s)
        self.soc = [sat["soc0"] for sat in SATS]
        self.ram = [RAM_TOTAL_MB] * len(SATS)
        self.solar = [True] * len(SATS)
        self.active = []
        self.k = 0
        self.next_t = -1.0
        self.task_id = 0
        self.t = 0.0
        self.log = []
        self.task = self._advance()
        return self._obs(), {}

    def resolve(self, action: int):
        """Validador do braco one-hot (IX.5): so viabilidade fisica e regiao.

        Rota para outra regiao so em tarefa com anomalia; a regra oculta nunca e
        consultada. Devolve o indice do satelite ou None (descarte).
        """
        if action == DROP or action >= len(SATS):
            return None
        task = self.task
        cross_ok = bool(task["token"])
        if not cross_ok and SATS[action]["region"] != task["region"]:
            return None
        if self.view_soc(action) <= BATTERY_SAFETY_PCT or self.ram[action] < TASK_RAM_MB:
            return None
        return action

    def step(self, action):
        action = int(action)
        task = self.task
        fleet = self.fleet()
        idx = self.resolve(action)
        sat_id = SATS[idx]["id"] if idx is not None else None
        compliant = semantic_rules.is_compliant(task["token"], sat_id, fleet)

        if idx is not None:
            self.ram[idx] = max(0.0, self.ram[idx] - TASK_RAM_MB)
            self.soc[idx] = max(0.0, self.soc[idx] - TASK_ENERGY_COST_PCT)
            self.active.append((self.t + TASK_DURATION_S, idx, TASK_RAM_MB))

        if task["token"]:
            reward = 1.0 if compliant else -1.0
        else:
            reward = self.w if idx is not None else 0.0
        if action != DROP and idx is None:
            reward -= self.invalid_penalty

        self.log.append({"task_id": task["id"], "t": task["t"], "region": task["region"],
                         "anomaly": task["token"] or "", "sat": sat_id,
                         "success": int(idx is not None), "compliant": int(compliant)})

        # Proxima chegada cronometrada a partir DESTE passo, como no simulador.
        self.next_t = self.t + self.rng.expovariate(self.lam_per_s)
        self.k += 1
        self.task = self._advance()
        terminated = self.task is None
        obs = np.zeros(self.n_obs, dtype=np.float32) if terminated else self._obs()
        return obs, reward, terminated, False, {"compliant": compliant, "sat": sat_id}


# ---------------------------------------------------------------------- #
# Politicas de referencia (para validacao e comparacao)                   #
# ---------------------------------------------------------------------- #

def oracle_action(env: NTNMECSeqEnv) -> int:
    """Espelho do OracleScheduler: tabela das regras de projeto + guloso regional."""
    task = env.task
    rule = (semantic_rules.BY_TOKEN.get(task["token"])
            if task["token"] in semantic_rules.ONEHOT_ORDER else None)
    if rule is not None and rule.must_drop:
        return DROP
    region = rule.required_region if rule is not None else task["region"]
    i = SAT_OF_REGION[region]
    if env.view_soc(i) > BATTERY_SAFETY_PCT and env.ram[i] >= TASK_RAM_MB:
        return i
    return DROP
