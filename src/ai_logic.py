# src/ai_logic.py
import csv
import json
import math
import os
import random
import time
from src import semantic_rules

from src.schedulers.llm_scheduler import LLMScheduler
from src.schedulers.baseline_scheduler import BaselineScheduler
from src.schedulers.slm_scheduler import SLMScheduler, NPU_LATENCY_MS as SLM_NPU_LATENCY_MS
from src.schedulers.drl_scheduler import DRLScheduler
from src.schedulers.oracle_scheduler import OracleScheduler

# Custo energético por decisão — parâmetros calibrados por tipo de hardware
JOULES_PER_DECISION = {
    "BASELINE": 0.001,   # CPU if/else, ~1μs @ 1W
    "DRL":      0.005,   # Q-table lookup + update, ~2ms @ 2W
    "SLM":      0.100,   # NPU inference, 50ms @ 2W
    "LLM":      5.000,   # LEO→GS TX, ~1s @ 5W
}

# Atraso de propagação LEO↔GS round-trip (550km orbit: 2×550000/3e8 ≈ 3.67ms)
LLM_PROPAGATION_MS = 3.67

# Duração de processamento de uma tarefa — após este período a RAM é liberada
TASK_DURATION_S = 60.0

# Modelo de eclipse orbital — bateria drena quando sem luz solar (premissa ISL sempre disponível)
ORBITAL_PERIOD_S      = 5400.0  # período orbital LEO (90 min)
ECLIPSE_THRESHOLD     = -0.10   # sin < -0.1 → eclipse (~53% do ciclo ≈ 46 min/órbita)
SOLAR_CHARGE_RATE_PCT = 0.083   # +0.083%/step em luz solar (+1%/min)
ECLIPSE_DRAIN_PCT     = 0.040   # -0.040%/step em eclipse (consumo de housekeeping)
TASK_ENERGY_COST_PCT  = 2.0     # -2% SOC por tarefa aceita (consumo de processamento MEC)
BATTERY_SAFETY_PCT    = 20.0    # abaixo disto, satélite recusa novas tarefas


class MECOrchestrator:
    def __init__(self, brain=None, anomaly_rate=0.10, *args, **kwargs):
        self._engine = os.environ.get("MEC_ENGINE", "BASELINE")
        print(f">>> [MECOrchestrator] Inicializando com o motor: {self._engine}")

        if self._engine == "LLM":
            self.brain = LLMScheduler()
        elif self._engine == "SLM":
            self.brain = SLMScheduler()
        elif self._engine == "DRL":
            self.brain = DRLScheduler()
        elif self._engine == "ORACLE":
            self.brain = OracleScheduler()
        elif self._engine == "BASELINE":
            self.brain = BaselineScheduler()
        else:
            print(f">>> [MECOrchestrator] Motor '{self._engine}' desconhecido, usando BASELINE.")
            self.brain = BaselineScheduler()
            self._engine = "BASELINE"

        self.anomaly_rate  = anomaly_rate
        self.task_id       = 0
        self.mec_satellites = []
        self.mec_tasks_dropped = 0
        self.next_task_time = -1.0
        self.lambda_rate   = 4.0 / 60.0   # 4 tarefas/minuto

        # MEC_RULE_SPLIT: "seen" (padrão, reproduz o artigo), "heldout" (só
        # restrições inéditas) ou "all" (mistura as duas).
        split = os.environ.get("MEC_RULE_SPLIT", "seen")
        self._rule_pool = semantic_rules.rules(None if split == "all" else split)
        if not self._rule_pool:
            raise ValueError(f"MEC_RULE_SPLIT inválido: {split!r}")

        self.task_log    = []   # uma entrada por tarefa finalizada
        self.active_tasks = []  # (completion_time, sat_id, ram_used) — para liberar RAM
        self._last_step_time = 0.0
        self._step_delta     = 0.0   # inferido do loop; usado para fechar a janela

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def setup_nodes(self, all_nodes):
        print(f">>> [MEC] Setup Nodes called with {len(all_nodes)} nodes.")
        region_map = {0: "USA", 1: "BRAZIL", 2: "EUROPE"}

        for node in all_nodes:
            name = getattr(node, 'iName', str(node))
            if "Satellite" in name or "SAT" in name:
                nid = getattr(node, 'nodeID', 0)
                region = region_map[nid % 3]

                node.mec_ram_total  = 4096.0
                node.mec_ram_free   = node.mec_ram_total
                node.mec_region     = region
                node.mec_tasks_completed = 0
                node.mec_tasks_dropped   = 0

                # Bateria: capacidade e SOC inicial variam por hardware (nodeID % 3)
                capacity_map      = {0: 100.0, 1: 75.0,  2: 50.0}
                soc_map           = {0: 85.0,  1: 90.0,  2: 75.0}
                orbital_phase_map = {0: 3600.0, 1: 0.0, 2: 1800.0}
                node_idx = nid % 3
                node.mec_battery_capacity_wh = capacity_map[node_idx]
                node.mec_battery_soc_pct     = soc_map[node_idx]
                node.mec_orbital_phase       = orbital_phase_map[node_idx]
                node.mec_solar_charging      = True

                self.mec_satellites.append(node)
                print(f">>> [MEC] Upgraded {name} (ID {nid}) -> {region} | "
                      f"battery={node.mec_battery_soc_pct:.0f}% "
                      f"({node.mec_battery_capacity_wh:.0f}Wh)")

    # ------------------------------------------------------------------ #
    # Modelo de eclipse orbital / bateria                                  #
    # ------------------------------------------------------------------ #

    def _update_battery(self, current_time_sec):
        """Atualiza SOC de todos os satélites a cada tick (5s): carrega em sol, drena em eclipse."""
        for s in self.mec_satellites:
            phase     = getattr(s, 'mec_orbital_phase', 0.0)
            solar_val = math.sin(2 * math.pi * (current_time_sec + phase) / ORBITAL_PERIOD_S)
            in_eclipse = (solar_val <= ECLIPSE_THRESHOLD)
            if in_eclipse:
                s.mec_battery_soc_pct = max(0.0, s.mec_battery_soc_pct - ECLIPSE_DRAIN_PCT)
            else:
                s.mec_battery_soc_pct = min(100.0, s.mec_battery_soc_pct + SOLAR_CHARGE_RATE_PCT)
            s.mec_solar_charging = not in_eclipse

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _build_fleet(self, current_time_sec):
        fleet = []
        for s in self.mec_satellites:
            fleet.append({
                "id":             getattr(s, 'nodeID', 0),
                "region":         getattr(s, 'mec_region', 'UNK'),
                "battery_pct":    round(getattr(s, 'mec_battery_soc_pct', 100.0), 1),
                "solar_charging": getattr(s, 'mec_solar_charging', True),
                "ram_free":       getattr(s, 'mec_ram_free', 0),
            })
        return fleet

    def _release_completed_tasks(self, current_time_sec):
        """Libera RAM das tarefas que concluíram processamento (após TASK_DURATION_S)."""
        still_active = []
        for comp_t, sid, ram in self.active_tasks:
            if comp_t <= current_time_sec:
                for s in self.mec_satellites:
                    if getattr(s, 'nodeID') == sid:
                        s.mec_ram_free = min(s.mec_ram_total, s.mec_ram_free + ram)
                        break
            else:
                still_active.append((comp_t, sid, ram))
        self.active_tasks = still_active

    @staticmethod
    def _semantic_compliant(task, decision_id, fleet):
        """Verifica se a decisão de roteamento respeita a restrição semântica.

        Delegado a src/semantic_rules para que o mesmo ground truth valha tanto
        para as regras conhecidas em tempo de projeto quanto para as held-out.
        Equivalência com a versão anterior (if/elif de três ramos) verificada
        sobre as 2.950 decisões já registradas.
        """
        return semantic_rules.is_compliant(
            task.get('semantic_anomaly'), decision_id, fleet)

    def _apply_decision(self, task, decision_id, current_time_sec,
                        latency_ms, joules_cost, fleet):
        anomalia    = task.get('semantic_anomaly', '')
        anomalia_str = f" [anomalia={anomalia}]" if anomalia else ""
        compliant   = self._semantic_compliant(task, decision_id, fleet)

        if decision_id is not None:
            print(f"   ---> [{self._engine}] Assign Task {task['id']} to SAT {decision_id}"
                  f"{anomalia_str} | {latency_ms:.0f}ms | {joules_cost:.3f}J"
                  f"{' ✗semantic' if not compliant else ''}")
            for s in self.mec_satellites:
                if getattr(s, 'nodeID') == decision_id:
                    s.mec_tasks_completed += 1
                    s.mec_ram_free        = max(0,   s.mec_ram_free        - task.get('ram', 0))
                    s.mec_battery_soc_pct = max(0.0, s.mec_battery_soc_pct - TASK_ENERGY_COST_PCT)
                    break
            self.active_tasks.append(
                (current_time_sec + TASK_DURATION_S, decision_id, task.get('ram', 0))
            )
        else:
            print(f"   ---> [{self._engine}] NO ROUTE Task {task['id']}"
                  f"{anomalia_str} | {latency_ms:.0f}ms | {joules_cost:.3f}J")
            self.mec_tasks_dropped += 1

        self.task_log.append({
            "task_id":            task['id'],
            "region":             task.get('region', ''),
            "anomaly":            anomalia,
            "arrival_time_s":     task.get('arrival_time', current_time_sec),
            "decision_time_s":    current_time_sec,
            "latency_ms":         round(latency_ms, 3),
            "joules_cost":        round(joules_cost, 4),
            "decision_sat_id":    decision_id if decision_id is not None else "",
            "success":            1 if decision_id is not None else 0,
            "semantic_compliant": 1 if compliant else 0,
            "engine":             self._engine,
        })

    # ------------------------------------------------------------------ #
    # Main loop step                                                       #
    # ------------------------------------------------------------------ #

    def step(self, current_time_sec):
        # O loop chama step() em t = 0, delta, 2*delta, ... (N-1)*delta, de modo que
        # o timestamp do ultimo passo e (N-1)*delta e nao a duracao da janela (N*delta).
        # Guardamos o delta para fechar a janela corretamente em save_metrics().
        if current_time_sec > self._last_step_time:
            self._step_delta = current_time_sec - self._last_step_time
        self._last_step_time = current_time_sec

        # 1. Libera RAM de tarefas concluídas
        self._release_completed_tasks(current_time_sec)

        # 2. Atualiza bateria (drena eclipse / carrega sol a cada tick de 5s)
        self._update_battery(current_time_sec)

        # 3. Processa inferências SLM concluídas neste tick
        if hasattr(self.brain, 'check_completed_inferences'):
            for task, decision_id in self.brain.check_completed_inferences(current_time_sec):
                latency_ms  = SLM_NPU_LATENCY_MS   # latência NPU simulada (não wall-clock da API)
                joules_cost = JOULES_PER_DECISION.get("SLM", 0.1)
                fleet       = task.get('slm_fleet_snapshot', [])
                self._apply_decision(task, decision_id, current_time_sec,
                                     latency_ms, joules_cost, fleet)

        # 4. Poisson: inicializa ou verifica chegada de nova tarefa
        if self.next_task_time < 0:
            self.next_task_time = current_time_sec + random.expovariate(self.lambda_rate)

        if current_time_sec < self.next_task_time:
            return

        # 5. Gera nova tarefa
        self.task_id += 1
        rand_val = random.random()
        if rand_val < 0.33:
            region = "BRAZIL"
        elif rand_val < 0.66:
            region = "USA"
        else:
            region = "EUROPE"

        task = {"id": self.task_id, "region": region, "ram": 500,
                "arrival_time": current_time_sec}

        if random.random() < self.anomaly_rate:
            rule = random.choice(self._rule_pool)
            task["semantic_anomaly"] = rule.token
            # A frase só acompanha regras held-out: para elas é o único canal de
            # informação, já que não estão na lista de regras do prompt. As regras
            # seen continuam chegando pelo prompt, exatamente como no artigo — o
            # que mantém a corrida canônica reproduzível.
            if rule.split != "seen":
                task["semantic_statement"] = rule.statement

        fleet = self._build_fleet(current_time_sec)
        anomalia_str = f" [anomalia={task['semantic_anomaly']}]" if task.get('semantic_anomaly') else ""
        region_bat   = next((f"{s['battery_pct']:.0f}%" for s in fleet if s['region'] == region), '?')
        solar_icon   = next(("☀" if s['solar_charging'] else "◑" for s in fleet if s['region'] == region), '')
        print(f"\n[MEC T+{current_time_sec:.1f}] New Task {self.task_id} ({region} BAT={region_bat}{solar_icon}){anomalia_str}")

        if hasattr(self.brain, 'receive_task'):
            # SLM: async
            self.brain.receive_task(task, current_time_sec, fleet)
        else:
            # LLM / BASELINE / DRL: sync
            t_wall      = time.perf_counter()
            decision_id = self.brain.decide(task, fleet, BATTERY_SAFETY_PCT)
            wall_ms     = (time.perf_counter() - t_wall) * 1000

            latency_ms  = wall_ms + LLM_PROPAGATION_MS if self._engine == "LLM" else wall_ms
            joules_cost = JOULES_PER_DECISION.get(self._engine, 0.001)
            self._apply_decision(task, decision_id, current_time_sec,
                                 latency_ms, joules_cost, fleet)

        self.next_task_time = current_time_sec + random.expovariate(self.lambda_rate)

    # ------------------------------------------------------------------ #
    # Relatórios                                                           #
    # ------------------------------------------------------------------ #

    def print_stats(self):
        print("\n" + "=" * 44)
        print("         MEC ORCHESTRATOR REPORT")
        print("=" * 44)
        total_completed = 0

        for s in self.mec_satellites:
            completed  = getattr(s, 'mec_tasks_completed', 0)
            region     = getattr(s, 'mec_region', 'UNK')
            soc        = getattr(s, 'mec_battery_soc_pct', 0.0)
            cap        = getattr(s, 'mec_battery_capacity_wh', 0.0)
            solar_str  = "Solar" if getattr(s, 'mec_solar_charging', True) else "Eclipse"
            total_completed += completed

            print(f"SAT {getattr(s, 'nodeID')} ({region})")
            print(f"  Battery SOC:     {soc:.1f}% ({cap:.0f}Wh) | {solar_str}")
            print(f"  RAM Free:        {getattr(s, 'mec_ram_free', 0):.0f} MB")
            print(f"  Tasks Completed: {completed}")
            print("-" * 22)

        n = len(self.task_log)
        print(f"\nTOTAL TASKS GENERATED:    {self.task_id}")
        print(f"TOTAL COMPLETED:          {total_completed}")
        print(f"TOTAL REJECTED:           {self.mec_tasks_dropped}")
        if n > 0:
            success_rate  = total_completed / self.task_id * 100
            avg_lat       = sum(r['latency_ms']  for r in self.task_log) / n
            total_joules  = sum(r['joules_cost'] for r in self.task_log)
            avg_joules    = total_joules / n
            compliant_n   = sum(r['semantic_compliant'] for r in self.task_log)
            anomaly_tasks = [r for r in self.task_log if r['anomaly']]

            effective_n   = sum(1 for r in self.task_log if r['success'] == 1 and r['semantic_compliant'] == 1)
            correct_drops = sum(1 for r in self.task_log if r['success'] == 0 and r['semantic_compliant'] == 1)
            print(f"SUCCESS RATE:             {success_rate:.1f}%  (throughput — tarefas roteadas)")
            print(f"EFFECTIVE SUCCESS:        {effective_n}/{n} ({effective_n/n*100:.1f}%)  (roteadas + semanticamente corretas)")
            print(f"CORRECT DROPS:            {correct_drops}  (drops semanticamente válidos, ex: hardware failure)")
            print(f"AVG LATENCY:              {avg_lat:.1f} ms")
            print(f"TOTAL ENERGY:             {total_joules:.3f} J")
            print(f"JOULES/DECISION:          {avg_joules:.4f} J")
            print(f"SEMANTIC COMPLIANCE:      {compliant_n}/{n} ({compliant_n/n*100:.1f}%)")
            if anomaly_tasks:
                anom_ok = sum(r['semantic_compliant'] for r in anomaly_tasks)
                print(f"ANOMALY COMPLIANCE:       {anom_ok}/{len(anomaly_tasks)}")

        if hasattr(self.brain, 'print_qtable'):
            self.brain.print_qtable()

        print("=" * 44 + "\n")

    def save_metrics(self):
        outdir = os.environ.get("MEC_OUTDIR", "logs")
        os.makedirs(outdir, exist_ok=True)
        engine = self._engine

        csv_path   = os.path.join(outdir, f"mec_metrics_{engine}.csv")
        fieldnames = [
            "task_id", "region", "anomaly",
            "arrival_time_s", "decision_time_s", "latency_ms",
            "joules_cost", "decision_sat_id",
            "success", "semantic_compliant", "engine",
        ]
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.task_log)

        n       = len(self.task_log)
        summary = {"engine": engine, "total_tasks": n}
        if n > 0:
            completed      = sum(r['success']            for r in self.task_log)
            anomaly_tasks  = [r for r in self.task_log   if r['anomaly']]
            compliant      = sum(r['semantic_compliant'] for r in self.task_log)
            anom_compliant = sum(r['semantic_compliant'] for r in anomaly_tasks)

            summary.update({
                "success_count":            completed,
                "success_rate":             round(completed / n, 4),
                "drop_count":               self.mec_tasks_dropped,
                "drop_rate":                round(self.mec_tasks_dropped / n, 4),
                "avg_latency_ms":           round(sum(r['latency_ms']  for r in self.task_log) / n, 3),
                "total_joules":             round(sum(r['joules_cost'] for r in self.task_log), 4),
                "joules_per_decision":      round(JOULES_PER_DECISION.get(engine, 0.001), 4),
                "anomaly_task_count":       len(anomaly_tasks),
                "semantic_compliance_rate": round(compliant / n, 4),
                "anomaly_compliance_rate":  round(anom_compliant / len(anomaly_tasks), 4) if anomaly_tasks else None,
                "effective_success_count":  sum(1 for r in self.task_log if r['success'] == 1 and r['semantic_compliant'] == 1),
                "effective_success_rate":   round(sum(1 for r in self.task_log if r['success'] == 1 and r['semantic_compliant'] == 1) / n, 4),
                "correct_drop_count":       sum(1 for r in self.task_log if r['success'] == 0 and r['semantic_compliant'] == 1),
            })
            # RAM utilization ponderada pelo tempo (integral de RAM_usada×dt / sim_total×RAM_total).
            # Métrica de snapshot (final tick) era idêntica em todos os engines pois os últimos
            # 60s têm as mesmas tarefas aceitas — este cálculo usa o task_log completo.
            # Janela completa = instante do ultimo passo + um delta (ver step()).
            sim_duration = (self._last_step_time + self._step_delta) or 1.0
            sat_ram_seconds: dict = {}
            for r in self.task_log:
                if r['success'] != 1:
                    continue
                t_arrive = r['arrival_time_s']
                hold = min(TASK_DURATION_S, sim_duration - t_arrive)
                if hold > 0:
                    sid = str(r['decision_sat_id'])
                    sat_ram_seconds[sid] = sat_ram_seconds.get(sid, 0.0) + hold * r.get('ram', 500)
            if self.mec_satellites and sim_duration > 0:
                tw_utils = [
                    sat_ram_seconds.get(str(getattr(s, 'nodeID')), 0.0)
                    / (sim_duration * s.mec_ram_total) * 100
                    for s in self.mec_satellites
                ]
                summary["avg_ram_utilization_pct"] = round(sum(tw_utils) / len(tw_utils), 1)
            else:
                summary["avg_ram_utilization_pct"] = 0.0
            # Estado final de bateria por satélite
            for s in self.mec_satellites:
                summary[f"sat_{getattr(s, 'nodeID')}_final_battery_pct"] = round(
                    getattr(s, 'mec_battery_soc_pct', 0.0), 1
                )

        json_path = os.path.join(outdir, f"mec_summary_{engine}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        print(f"[MEC] Metricas salvas: {csv_path} | {json_path}")
