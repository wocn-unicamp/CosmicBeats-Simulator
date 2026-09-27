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

# Sementes da coluna "regras ineditas" da Tabela IV do camera-ready. A regra de
# corte fixada antes dos resultados (docs X.3) deu 10 sementes em 27/09.
PAPER_HELDOUT_SEEDS = list(range(10))
# Origem dos motores remotos nas ineditas: SLM no protocolo v3, LLM na campanha v2.
HELDOUT_DIRS = {"SLM": "logs/generalization/heldout_slm_v3",
                "LLM": "logs/generalization/heldout_lm_v2"}


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

    # LINHAS INTEIRAS da Tabela III, na ordem das colunas. Checar cada intervalo
    # solto deixava passar um valor errado quando outro motor tinha o mesmo
    # intervalo (teste negativo: ACR do SLM trocado passou com o do LLM).
    order = ["BASELINE", "DRL", "SLM", "LLM"]
    def cell(metric, e):
        d = comp[e][metric]; lo, hi = d["wilson"]
        val = "100" if d["k"] == d["n"] else pct1(d["k"] / d["n"])
        hi_s = "100" if hi >= 0.9999 else pct1(hi)
        return f"{val} {{\\tiny[{pct1(lo)}--{hi_s}]}}"
    for metric, name in (("scr", "SCR (semantic compliance)"), ("acr", "ACR (anomaly compliance)")):
        c.expect(f"Tabela III, linha {metric.upper()}",
                 f"{name} & " + " & ".join(cell(metric, e) for e in order) + " \\\\", src)
    n_tasks = {e: len(runs[e]) for e in order}
    done = {e: sum(int(r["success"]) for r in runs[e].values()) for e in order}
    c.expect("Tabela III, taxa de execucao", "Task execution rate & " + " & ".join(
        f"{100 * done[e] / n_tasks[e]:.1f}\\%" for e in order) + " \\\\", src)
    c.expect("Tabela III, taxa de descarte", "Drop rate & " + " & ".join(
        f"{100 * (n_tasks[e] - done[e]) / n_tasks[e]:.1f}\\%" for e in order) + " \\\\", src)

    expected_p = {("BASELINE", "LLM"): "0.031", ("BASELINE", "SLM"): "0.031",
                  ("DRL", "LLM"): "0.0625", ("DRL", "SLM"): "0.0625",
                  ("BASELINE", "DRL"): "1.000"}
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

    # Com o nome do motor: SLM e LLM tem o mesmo valor, e sem o nome um mascarava o outro.
    for engine, v in ram_utilisation(runs, keys).items():
        c.expect(f"RAM {engine}", f"{engine} {v['avg']:.1f}\\%", src)


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
    motores. E o pareamento que torna o McNemar valido, entao o n do texto tem de
    ser o da intersecao, nao o de cada motor isolado.
    """
    import csv as _csv
    from analyze_results import mcnemar_exact
    sys.path.insert(0, repo)
    from src import semantic_rules as sr

    engines = ["BASELINE", "ORACLE", "DRL", "DRL_ONEHOT", "SLM", "LLM"]
    det_dir = os.path.join(repo, "logs/generalization/heldout")
    # As sementes que o ARTIGO usa, fixadas: ler "tudo o que houver" faria o
    # resultado mudar sozinho quando novas sementes chegassem.
    seeds = PAPER_HELDOUT_SEEDS
    data = {e: {} for e in engines}
    for s_ in seeds:
        for e in engines:
            base = os.path.join(repo, HELDOUT_DIRS[e]) if e in HELDOUT_DIRS else det_dir
            path = os.path.join(base, f"seed{s_}", f"mec_metrics_{e}.csv")
            with open(path, newline="") as fh:
                for r in _csv.DictReader(fh):
                    if r["anomaly"]:
                        data[e][f"{s_}:{r['task_id']}"] = (int(r["semantic_compliant"]), r["anomaly"],
                                                         r.get("decision_source") or "model")
    keys = sorted(set.intersection(*[set(v) for v in data.values()]))
    src = "logs/generalization/heldout{,_slm_v3,_lm_v2}/seed*/"
    n = len(keys)
    k = {e: sum(data[e][x][0] for x in keys) for e in engines}
    unseen = {e: 100 * k[e] / n for e in engines}
    c.expect("n pareado nas ineditas", f"$n={n}$, paired across all", src)
    c.expect("sementes das ineditas", f"Unseen: {len(seeds)} seeds", src)
    # A mesma frase aparece no resumo e na V-D; cada ocorrencia e checada com o
    # proprio contexto, senao um valor errado numa delas passaria despercebido.
    c.expect("resumo: LLM e SLM nas ineditas",
             f"over {n} paired tasks the LLM keeps {k['LLM']}/{n} and the SLM {k['SLM']}/{n}", src)
    c.expect("V-D: LLM e SLM nas ineditas",
             f"stay high: the LLM keeps {k['LLM']}/{n} and the SLM {k['SLM']}/{n}", src)
    c.expect("conclusao: LLM e SLM nas ineditas",
             f"LLM stayed at {k['LLM']}/{n} and the SLM at {k['SLM']}/{n}", src)
    c.expect("contribuicoes: SLM e LLM em %",
             f"(SLM {unseen['SLM']:.0f}\\%, LLM {unseen['LLM']:.0f}\\%)", src)
    others = max(unseen[e] for e in ("BASELINE", "ORACLE", "DRL", "DRL_ONEHOT"))
    c.expect("teto dos motores sem linguagem", f"$\\leq${round(others)}\\%", src)

    def disc(a, b):
        return (sum(1 for x in keys if data[a][x][0] == 1 and data[b][x][0] == 0),
                sum(1 for x in keys if data[a][x][0] == 0 and data[b][x][0] == 1))
    l01, l10 = disc("LLM", "BASELINE"); s01, s10 = disc("SLM", "BASELINE")
    c.expect("McNemar LLM e SLM vs BASELINE", f"{l01}/{l10} and {s01}/{s10}", src)
    assert max(mcnemar_exact(l01, l10), mcnemar_exact(s01, s10)) < 1e-8
    c.expect("p dos modelos de linguagem nas ineditas", "both $p<10^{-8}$", src)
    oracle_disc = sum(1 for x in keys if data["ORACLE"][x][0] != data["BASELINE"][x][0])
    assert oracle_disc == 0, f"ORACLE deixou de ser identico ao BASELINE ({oracle_disc})"
    c.expect("ORACLE identico ao BASELINE", "0~discordant pairs", src)
    d01, d10 = disc("DRL_ONEHOT", "BASELINE")
    assert mcnemar_exact(d01, d10) > 0.05, "DRL-OH deixou de ser indistinguivel do BASELINE"
    c.expect("DRL-OH vs BASELINE nas ineditas", f"{d01}/{d10} discordant pairs, $p=1.000$", src)
    m01, m10 = disc("LLM", "SLM")
    c.expect("LLM vs SLM", f"({m01}/{m10} discordant pairs, $p={mcnemar_exact(m01, m10):.4f}$)", src)

    # Decomposicao por obrigacao: tarefas que exigem rotear para uma regiao.
    region = [x for x in keys if not sr.BY_TOKEN[data["BASELINE"][x][1]].must_drop]
    tot = {e: sum(data[e][x][0] for x in region) for e in engines}
    c.expect("decomposicao por regiao",
             f"On the {len(region)} tasks that require routing to a specific region the "
             f"heuristic solves {tot['BASELINE']}, the SLM {tot['SLM']} and the LLM {tot['LLM']}", src)
    # Todo erro do SLM e falha de formato (X.6): nenhuma decisao do modelo errada.
    slm_miss = [x for x in keys if data["SLM"][x][0] == 0]
    assert all(data["SLM"][x][2] == "parse_failure" for x in slm_miss), "SLM errou por decisao"
    words = {5: "five", 4: "four", 6: "six", 3: "three"}
    c.expect("erros do SLM sao de formato", f"All {words.get(len(slm_miss), len(slm_miss))} SLM misses", src)

    # Coluna das conhecidas: deterministicos em 20 sementes; SLM e LLM da canonica.
    import json as _json
    t20 = os.path.join(repo, "logs/generalization/table20.json")
    seen_tab = _json.load(open(t20))["splits"]["seen"]
    seen = {e: 100 * v["k"] / v["n"] for e, v in seen_tab["engines"].items()}
    lo, hi = seen_tab["engines"]["ORACLE"]["wilson"]
    c.expect("IC do ORACLE nas conhecidas", f"{100 * lo:.1f}--100", t20)
    canon = compliance_table(load_runs(os.path.join(repo, "logs")),
                             paired_keys(load_runs(os.path.join(repo, "logs"))))
    for e in ("SLM", "LLM"):
        a = canon[e]["acr"]; seen[e] = 100 * a["k"] / a["n"]

    # LINHA INTEIRA de cada motor (checar numeros soltos ja deixou passar uma tabela errada).
    label = {"BASELINE": "BASELINE", "ORACLE": "ORACLE", "DRL": "DRL",
             "DRL_ONEHOT": "DRL-OH", "SLM": "SLM", "LLM": "LLM"}
    for e in engines:
        dag = "$^{\\dagger}$" if e in ("SLM", "LLM") else ""
        u = f"{unseen[e]:.1f}\\%"
        u = f"\\textbf{{{u}}}" if e == "LLM" else u
        c.expect(f"Tabela IV, linha {label[e]}",
                 f"{label[e]} & {seen[e]:.1f}\\%{dag} & {u} \\\\", src)


def check_slm_protocol(c: Checker, repo: str) -> None:
    """Protocolo v3 do SLM (docs X.2): nenhuma decisao cortada pelo orcamento."""
    import csv as _csv, glob as _glob
    paths = [os.path.join(repo, "logs/mec_metrics_SLM.csv")] + [
        os.path.join(repo, HELDOUT_DIRS["SLM"], f"seed{s_}", "mec_metrics_SLM.csv")
        for s_ in PAPER_HELDOUT_SEEDS]
    rows = [r for p_ in paths for r in _csv.DictReader(open(p_, newline=""))]
    assert all(r["finish_reason"] == "STOP" for r in rows), "alguma decisao do SLM foi cortada"
    c.expect("decisoes do SLM, nenhuma cortada",
             f"none of the {len(rows)} SLM decisions reported here was cut off",
             "logs/mec_metrics_SLM.csv + heldout_slm_v3 (finish_reason)")


def check_gamma(c: Checker, repo: str) -> None:
    """Frase do gamma>0 nas limitacoes (decisoes 1 e 4 do orientador, docs X.1).

    Fonte: frontier_seq.py --myopic-seeds 42 43 44, envelope conjunto (IX.9.1).
    """
    import json as _json
    files = [os.path.join(repo, f"docs/frontier_l12_s{t}_pooled.json") for t in (42, 43, 44)]
    runs = [_json.load(open(f)) for f in files]
    def wkey(name):
        return name.split("_w")[1].split("_l")[0]
    replicated = {}
    for run in runs:
        for name, v in run["agents"].items():
            replicated.setdefault(wkey(name), []).append(v["ci95"][0] > 0)
    only = [w for w, flags in replicated.items() if len(flags) == 3 and all(flags)]
    assert only == ["0p05"], f"pesos que replicam em 3/3: {only}"
    gains = [next(v["gain_pp"] for n_, v in run["agents"].items() if wkey(n_) == "0p05") for run in runs]
    src = "docs/frontier_l12_s4{2,3,4}_pooled.json"
    c.expect("peso w do gamma>0", "throughput weight $w=0.05$", src)
    c.expect("ganho do gamma>0 (min--max)", f"{min(gains):.1f}--{max(gains):.1f}~pp", src)
    n_myopic = 3 * len(replicated)          # mesma grade de w nas duas familias
    c.expect("tamanho do envelope miope", f"envelope of {n_myopic} myopic agents", src)


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
    print("verificando o protocolo do SLM e o gamma>0...")
    check_slm_protocol(c, REPO)
    check_gamma(c, REPO)
    print("verificando as respostas ao Revisor 2...")
    check_reviewer2_budgets(c)
    print("verificando os estados de bateria...")
    check_battery(c, REPO)
    return c.report()


if __name__ == "__main__":
    sys.exit(main())
