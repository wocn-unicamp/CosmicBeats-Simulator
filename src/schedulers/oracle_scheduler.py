"""Oráculo de tabela de regras — o braço de controle que o Revisor 3 pediu.

A crítica do revisor é que, com três restrições fixas, um dicionário de três
entradas resolve o problema, de modo que o experimento não justificava um
modelo de linguagem. Este motor é exatamente esse dicionário, implementado com
honestidade: ele conhece, em tempo de projeto, as três regras do conjunto
`seen` e aplica cada uma perfeitamente.

O ponto do experimento é o que acontece depois. A tabela é montada uma vez, a
partir das regras enumeradas quando o sistema foi escrito. Uma restrição que
aparece em operação e não está na tabela não produz erro nem exceção — produz
silêncio: o motor simplesmente não percebe que havia uma obrigação e decide
pelo critério numérico de sempre. É assim que uma tabela de regras falha no
mundo real, e é isso que o braço mede.

Por isso a tabela é construída a partir de `ONEHOT_ORDER` (só as `seen`) e não
do registro completo: consultar o registro inteiro daria ao oráculo
conhecimento das regras held-out que, por construção, ele não deveria ter.
"""

from src import semantic_rules
from src.schedulers.baseline_scheduler import BaselineScheduler


class OracleScheduler:
    """Tabela de regras fixa + heurística gulosa como comportamento padrão."""

    def __init__(self):
        # Congela em tempo de construção apenas o que era conhecido em tempo de
        # projeto. A cópia é deliberada: nada que apareça depois entra aqui.
        self._table = {
            token: semantic_rules.BY_TOKEN[token]
            for token in semantic_rules.ONEHOT_ORDER
        }
        self._fallback = BaselineScheduler()
        print(f">>> [ENGINE] Oracle Scheduler Initialized "
              f"(tabela fixa de {len(self._table)} regras + fallback guloso)")

    def decide(self, task_dict, fleet, battery_safety_pct=20.0):
        rule = self._table.get(task_dict.get("semantic_anomaly"))

        if rule is None:
            # Sem restrição, ou restrição fora da tabela. Nos dois casos o motor
            # não tem como distinguir: decide pelo critério numérico.
            return self._fallback.decide(task_dict, fleet, battery_safety_pct)

        if rule.must_drop:
            return None

        # Regra de região: só satisfaz quem estiver na região exigida e couber.
        candidates = [
            s for s in fleet
            if s.get("region") == rule.required_region
            and s.get("battery_pct", 100.0) > battery_safety_pct
            and s.get("ram_free", 0) >= task_dict.get("ram", 0)
        ]
        if not candidates:
            # Descartar é o único desfecho disponível; não é conforme para uma
            # regra de região, e a métrica registra isso como tal.
            return None

        best = max(candidates, key=lambda s: (s.get("battery_pct", 0), s.get("ram_free", 0)))
        return best.get("id")
