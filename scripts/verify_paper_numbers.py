#!/usr/bin/env python3
"""Confere cada valor numerico afirmado no artigo contra os dados que o geraram.

Existe porque a versao submetida tinha numeros que nao batiam com os artefatos
commitados (a utilizacao de RAM nos summaries era residuo de uma metrica antiga)
e porque a revisao introduziu dezenas de valores novos — intervalos de
confianca, p-valores, quantis de latencia. Qualquer edicao no .tex ou nova
execucao deve manter este script passando.

Cada checagem declara de onde o valor vem, entao a falha aponta a origem, nao
so o sintoma.

Uso:
    python3 scripts/verify_paper_numbers.py
    python3 scripts/verify_paper_numbers.py --tex Relatorio_ED2_241327.tex
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_results import (  # noqa: E402
    anomaly_breakdown, compliance_table, latency_stats, load_runs,
    pairwise_mcnemar, paired_keys, ram_utilisation,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Sementes da coluna "regras ineditas" da Tabela IV do camera-ready.
PAPER_HELDOUT_SEEDS = [0, 1, 2]


class Checker:
    def __init__(self, tex: str):
        # O LaTeX trata quebra de linha como espaco; o verificador tambem precisa,
        # ou um valor dividido entre duas linhas do fonte parece ausente.
        import re as _re
        self._ws = _re.compile(r"\s+")
        self.tex = self._ws.sub(" ", tex)
        self.ok: list[str] = []
        self.bad: list[tuple[str, str, str]] = []

    def expect(self, what: str, needle: str, source: str) -> None:
        needle = self._ws.sub(" ", needle)
        (self.ok.append(what) if needle in self.tex
         else self.bad.append((what, needle, source)))

    def report(self) -> int:
        for what, needle, source in self.bad:
            print(f"  FALHA  {what}")
            print(f"         esperava no .tex: {needle!r}")
            print(f"         origem do valor:  {source}")
        total = len(self.ok) + len(self.bad)
        status = "OK" if not self.bad else "FALHOU"
        print(f"\n[{status}] {len(self.ok)}/{total} valores conferem")
        return 1 if self.bad else 0


def pct1(x: float) -> str:
    return f"{x * 100:.1f}"


def check_single_run(c: Checker, logs_dir: str) -> None:
    """Corrida canonica de 69 tarefas que embasa o artigo."""
    runs = load_runs(logs_dir)
    keys = paired_keys(runs)
    src = f"{logs_dir}/mec_metrics_*.csv"
    comp = compliance_table(runs, keys)

    for engine, data in comp.items():
        for metric in ("scr", "acr"):
            lo, hi = data[metric]["wilson"]
            hi_s = "100" if hi >= 0.9999 else pct1(hi)
            c.expect(f"IC Wilson {engine} {metric.upper()}",
                     f"{pct1(lo)}--{hi_s}", src)

    expected_p = {("BASELINE", "LLM"): "0.031", ("DRL", "LLM"): "0.0625",
                  ("DRL", "SLM"): "0.250", ("BASELINE", "DRL"): "1.000"}
    for r in pairwise_mcnemar(runs, keys, anomaly_only=True):
        key = tuple(sorted((r["a"], r["b"])))
        if key in expected_p:
            got, want = r["p"], float(expected_p[key])
            assert abs(got - want) < 6e-4, f"McNemar {key}: script={got}, .tex={want}"
            c.expect(f"McNemar {key[0]} vs {key[1]}", f"$p={expected_p[key]}$", src)

    lat = latency_stats(runs, keys)
    c.expect("mediana LLM", r"\SI{955}{\milli\second}", src)
    c.expect("IQR LLM", r"\numrange{877}{1131}", src)
    c.expect("cauda de rate-limit", f"{lat['LLM']['n_outliers']} of {lat['LLM']['n']}", src)
    c.expect("mediana DRL", r"\SI{0.438}{\milli\second}", src)
    c.expect("mediana BASELINE", r"\SI{0.014}{\milli\second}", src)

    for engine, v in ram_utilisation(runs, keys).items():
        c.expect(f"RAM {engine}", f"{v['avg']:.1f}\\%", src)


def check_multiseed(c: Checker, logs_dir: str) -> None:
    """Replicacao multi-semente dos motores deterministicos."""
    if not os.path.isdir(logs_dir):
        print(f"  (pulando multi-semente: {logs_dir} nao existe)")
        return
    # O paragrafo multi-semente do artigo fala especificamente dos motores
    # deterministicos. Fixa-los aqui evita que a verificacao mude de resultado
    # conforme a campanha remota vai depositando sementes parciais de SLM/LLM.
    det = ["BASELINE", "DRL"]
    runs = load_runs(logs_dir, det)
    keys = paired_keys(runs)
    if not keys:
        return
    src = f"{logs_dir}/seed*/mec_metrics_{{{','.join(det)}}}.csv"
    comp = compliance_table(runs, keys)
    seeds = len({k.split(":")[0] for k in keys})

    c.expect("n tarefas pareadas", f"{len(keys):,}".replace(",", "{,}") + " paired tasks", src)
    c.expect("n anomalias", f"{comp['BASELINE']['acr']['n']} anomaly-carrying", src)
    c.expect("n sementes", f"{seeds} independent", src)

    for engine in ("BASELINE", "DRL"):
        if engine not in comp:
            continue
        a = comp[engine]["acr"]
        c.expect(f"ACR multi-semente {engine}", f"{a['k'] / a['n'] * 100:.1f}\\%", src)
        lo, hi = a["wilson"]
        c.expect(f"IC multi-semente {engine}", f"{pct1(lo)}--{pct1(hi)}", src)

    for r in pairwise_mcnemar(runs, keys, anomaly_only=True):
        if {r["a"], r["b"]} == {"BASELINE", "DRL"}:
            c.expect("McNemar multi-semente", f"$p={r['p']:.3f}$", src)

    # O teto de uma politica cega e 1/3 em ESPERANCA (tres tipos de regra
    # equiprovaveis, so o de descarte e identificavel a partir de um bit). Antes
    # este check usava a fracao de descarte REALIZADA na amostra como se fosse o
    # teto; e uma estimativa ruidosa dele, e a distincao passou a importar quando
    # a correcao do _resolve_action levou o DRL acima da fracao realizada.
    bd = anomaly_breakdown(runs, keys)["BASELINE"]
    total = sum(n for _, n in bd.values())
    hw = next(n for t, (_, n) in bd.items() if "hardware" in t)
    assert 0.20 < hw / total < 0.45, (
        f"fracao de regras de descarte ({hw}/{total}) longe de 1/3 — o corpus "
        f"deixou de ser equilibrado e o argumento do teto nao vale")
    c.expect("teto cego declarado como 1/3", r"bound is $1/3$", src)


def check_generalization(c: Checker, repo: str) -> None:
    """Tabela de generalizacao (Secao V-D): conhecidas vs. ineditas.

    Coluna das ineditas recomputada sobre as tarefas pareadas entre os SEIS
    motores (os deterministicos vivem em heldout/, os remotos em heldout_lm/).
    E o pareamento que torna o McNemar valido, entao o n do texto tem de ser o
    da intersecao, nao o de cada motor isolado.
    """
    import csv as _csv
    from analyze_results import mcnemar_exact, wilson_ci
    sys.path.insert(0, repo)
    from src import semantic_rules as sr

    engines = ["BASELINE", "ORACLE", "DRL", "DRL_ONEHOT", "SLM", "LLM"]
    lm_dir = os.path.join(repo, "logs/generalization/heldout_lm")
    det_dir = os.path.join(repo, "logs/generalization/heldout")
    if not os.path.isdir(lm_dir):
        print("  (pulando generalizacao: sem dados de SLM/LLM nas ineditas)")
        return
    # As sementes que o ARTIGO usa, fixadas. Ler "tudo o que houver no diretorio"
    # faria o resultado mudar sozinho quando a campanha da versao estendida
    # depositar sementes novas — o mesmo erro que ja colapsou o pareamento uma vez.
    seeds = PAPER_HELDOUT_SEEDS
    data = {e: {} for e in engines}
    for s_ in seeds:
        for e in engines:
            base = lm_dir if e in ("SLM", "LLM") else det_dir
            path = os.path.join(base, f"seed{s_}", f"mec_metrics_{e}.csv")
            if not os.path.exists(path):
                continue
            with open(path, newline="") as fh:
                for r in _csv.DictReader(fh):
                    if r["anomaly"]:
                        data[e][f"{s_}:{r['task_id']}"] = (int(r["semantic_compliant"]), r["anomaly"])
    keys = sorted(set.intersection(*[set(v) for v in data.values()]))
    src = "logs/generalization/heldout{,_lm}/seed*/"
    n = len(keys)
    c.expect("n pareado nas ineditas", f"$n={n}$, paired across all", src)
    c.expect("sementes das ineditas", f"Unseen: {len(seeds)} seeds", src)
    unseen = {e: 100 * sum(data[e][x][0] for x in keys) / n for e in engines}

    b01 = sum(1 for x in keys if data["LLM"][x][0] == 1 and data["BASELINE"][x][0] == 0)
    b10 = sum(1 for x in keys if data["LLM"][x][0] == 0 and data["BASELINE"][x][0] == 1)
    c.expect("McNemar LLM vs BASELINE (ineditas)", f"{b01}/{b10}", src)
    assert mcnemar_exact(b01, b10) < 0.00015, mcnemar_exact(b01, b10)
    c.expect("p do LLM nas ineditas", "$p=0.0001$", src)
    oracle_disc = sum(1 for x in keys if data["ORACLE"][x][0] != data["BASELINE"][x][0])
    assert oracle_disc == 0, f"ORACLE deixou de ser identico ao BASELINE ({oracle_disc})"
    c.expect("ORACLE identico ao BASELINE", "0~discordant pairs", src)

    # Decomposicao: o texto afirma que, nas regras de REGIAO, DRL-OH e SLM
    # resolvem 5 cada, contra 6 do BASELINE e 15 do LLM.
    region = [x for x in keys if not sr.BY_TOKEN[data["BASELINE"][x][1]].must_drop]
    tot = {e: sum(data[e][x][0] for x in region) for e in engines}
    words = {15: "fifteen", 11: "eleven", 14: "fourteen", 16: "sixteen"}
    c.expect("n de tarefas de regiao", f"on the {words.get(len(region), len(region))} that", src)
    c.expect("SLM nas de regiao", f"it solves {tot['SLM']}, against", src)
    # DRL-OH volta ao nivel da heuristica nas ineditas: pares discordantes vs BASELINE.
    d01 = sum(1 for x in keys if data["DRL_ONEHOT"][x][0] == 1 and data["BASELINE"][x][0] == 0)
    d10 = sum(1 for x in keys if data["DRL_ONEHOT"][x][0] == 0 and data["BASELINE"][x][0] == 1)
    assert mcnemar_exact(d01, d10) > 0.05, "DRL-OH deixou de ser indistinguivel do BASELINE"
    c.expect("DRL-OH vs BASELINE nas ineditas", f"{d01}/{d10} discordant pairs", src)
    slm_drop = sum(data["SLM"][x][0] for x in keys if x not in region)
    assert slm_drop == len(keys) - len(region), "SLM deixou de descartar todas as falhas"
    c.expect("BASELINE nas de regiao", f"heuristic's {tot['BASELINE']}", src)
    c.expect("LLM nas de regiao", f"the LLM's {tot['LLM']}", src)

    # Coluna das conhecidas: deterministicos em 20 sementes.
    import json as _json
    t20 = os.path.join(repo, "logs/generalization/table20.json")
    seen_tab = _json.load(open(t20))["splits"]["seen"]
    seen = {e: 100 * v["k"] / v["n"] for e, v in seen_tab["engines"].items()}
    lo, hi = seen_tab["engines"]["ORACLE"]["wilson"]
    c.expect("IC do ORACLE nas conhecidas", f"{100 * lo:.1f}--100", t20)
    # SLM e LLM na coluna das conhecidas vem da corrida canonica (n=8, com adaga).
    canon = compliance_table(load_runs(os.path.join(repo, "logs")),
                             paired_keys(load_runs(os.path.join(repo, "logs"))))
    for e in ("SLM", "LLM"):
        a = canon[e]["acr"]; seen[e] = 100 * a["k"] / a["n"]

    # LINHA INTEIRA de cada motor. Checar numeros soltos deixava passar uma tabela
    # errada sempre que o mesmo valor aparecia na prosa — o teste negativo que
    # reintroduziu o 50,0% antigo do DRL-OH na tabela passou despercebido assim.
    label = {"BASELINE": "BASELINE", "ORACLE": "ORACLE", "DRL": "DRL",
             "DRL_ONEHOT": "DRL-OH", "SLM": "SLM", "LLM": "LLM"}
    for e in engines:
        dag = "$^{\\dagger}$" if e in ("SLM", "LLM") else ""
        u = f"{unseen[e]:.1f}\\%"
        u = f"\\textbf{{{u}}}" if e == "LLM" else u
        c.expect(f"Tabela IV, linha {label[e]}",
                 f"{label[e]} & {seen[e]:.1f}\\%{dag} & {u} \\\\", src)


def check_reviewer2_budgets(c: Checker) -> None:
    """Respostas ao Revisor 2: janela de latencia e adequacao energetica.

    Recomputadas das constantes do proprio simulador, para que o texto nao
    diverja silenciosamente se alguma delas mudar.
    """
    # Taxa EFETIVA, nao a nominal: o gerador resolve chegadas no passo de 5 s, no
    # maximo uma por passo, e cronometra a proxima a partir do passo que serviu a
    # anterior. O intervalo e 5*ceil(X/5), X~Exp(lambda), cuja media e
    # 5/(1-exp(-5*lambda)). Com lambda=4/min isso da 17,6 s (3,4/min), e nao 15 s
    # — erro que chegou a entrar no texto e foi pego pela varredura de carga.
    import math
    step_s, lam_nominal_per_s = 5.0, 4.0 / 60.0
    mean_gap_s = step_s / (1 - math.exp(-step_s * lam_nominal_per_s))
    lam_per_s = 1.0 / mean_gap_s
    llm_median_s = 0.955                     # mediana medida (Secao V-C)
    smallest_bat_J = 50 * 3600.0             # 50 Wh (SAT-2)
    llm_run_J = 5.0 * 69                     # JOULES_PER_DECISION['LLM'] x 69
    task_J = 0.02 * smallest_bat_J           # TASK_ENERGY_COST_PCT = 2%
    src = "src/ai_logic.py (constantes do modelo)"
    c.expect("intervalo medio entre chegadas (efetivo)", f"{mean_gap_s:.1f}~s", src)
    c.expect("taxa efetiva", f"$\\approx${60 * lam_per_s:.1f}~tasks/min", src)
    little = 100 * (60 * lam_per_s) * 500 / (3 * 4096)
    c.expect("Little com a taxa efetiva", f"{little:.1f}\\%", src)
    c.expect("fracao da janela usada pelo LLM", f"{100 * llm_median_s * lam_per_s:.1f}\\%", src)
    c.expect("energia do LLM vs menor bateria", f"{100 * llm_run_J / smallest_bat_J:.2f}\\%", src)
    c.expect("tarefa vs decisao do LLM", f"about {task_J / 5.0:.0f} LLM decisions", src)


def check_battery(c: Checker, repo: str) -> None:
    """SoC inicial (constantes do simulador) e final (corrida canonica, BASELINE).

    Os valores finais estavam no artigo desde a submissao sem nenhuma checagem.
    Os iniciais entraram quando se constatou que as capacidades em Wh sao
    nominais — a dinamica e em pontos percentuais e ignora a capacidade — e que
    o que de fato distingue os satelites e o SoC de partida.
    """
    import json as _json, re as _re
    src_code = os.path.join(repo, "src/ai_logic.py")
    code = open(src_code).read()
    soc_map = eval(_re.search(r"soc_map\s*=\s*(\{[^}]*\})", code).group(1))
    region_map = eval(_re.search(r"region_map\s*=\s*(\{[^}]*\})", code).group(1))
    by_region = {region_map[k]: v for k, v in soc_map.items()}
    order = ("BRAZIL", "EUROPE", "USA")          # SAT-1, SAT-2, SAT-15 no texto
    NAME = {"BRAZIL": "Brazil", "EUROPE": "Europe", "USA": "USA"}  # grafia do artigo
    c.expect("SoC inicial por regiao",
             "/".join(f"{by_region[r]:.0f}" for r in order) + "\\%", src_code)
    summ = _json.load(open(os.path.join(repo, "logs/mec_summary_BASELINE.json")))
    for sat, reg in (("1", "BRAZIL"), ("2", "EUROPE"), ("15", "USA")):
        final = summ[f"sat_{sat}_final_battery_pct"]
        c.expect(f"SoC final SAT-{sat}",
                 f"{final:.1f}\\% (SAT-{sat}, {NAME[reg]}, from {by_region[reg]:.0f}\\%",
                 "logs/mec_summary_BASELINE.json")
    # A bateria de SAT-2 termina 1,8 pp acima do piso de 20% — afirmado no texto.
    margin = summ["sat_2_final_battery_pct"] - 20.0
    c.expect("margem do SAT-2 sobre o piso", f"{margin:.1f}~percentage points", "idem")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", default=os.path.join(REPO, "Relatorio_ED2_241327.tex"))
    ap.add_argument("--logs-dir", default=os.path.join(REPO, "logs"))
    ap.add_argument("--multiseed-dir", default=os.path.join(REPO, "logs/multiseed"))
    args = ap.parse_args()

    with open(args.tex) as fh:
        c = Checker(fh.read())

    print("verificando a corrida canonica (69 tarefas)...")
    check_single_run(c, args.logs_dir)
    print("verificando a replicacao multi-semente...")
    check_multiseed(c, args.multiseed_dir)
    print("verificando a tabela de generalizacao...")
    check_generalization(c, REPO)
    print("verificando as respostas ao Revisor 2...")
    check_reviewer2_budgets(c)
    print("verificando os estados de bateria...")
    check_battery(c, REPO)
    return c.report()


if __name__ == "__main__":
    sys.exit(main())
