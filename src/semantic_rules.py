"""Registro de restricoes semanticas, com separacao treino/teste.

Motivacao (Revisor 3): na versao submetida as restricoes eram tres literais
snake_case fixos e o ground truth era um if/elif de tres ramos. Nessa escala um
dicionario de tres entradas atinge 100%, entao o experimento nao sustentava a
tese de que um modelo de linguagem e necessario. A critica procede.

A resposta nao e defender o desenho e sim mudar o que se mede. Uma tabela de
regras e um DRL com one-hot resolvem perfeitamente as regras ENUMERADAS em
tempo de projeto; o que so um modelo de linguagem consegue e aplicar uma
restricao que ninguem enumerou. Por isso o registro tem dois conjuntos:

  SEEN    as tres regras originais, conhecidas em tempo de projeto. Todo motor
          pode codifica-las: a tabela tem entrada, o DRL tem dimensao no one-hot,
          o prompt as lista explicitamente.

  HELDOUT restricoes novas, que chegam à tarefa como uma FRASE em linguagem
          natural e nao aparecem em lugar nenhum do prompt, da tabela ou do
          espaco de observacao. Tabela e DRL nao tem como representa-las; um
          modelo de linguagem, em principio, tem.

O ground truth e deliberadamente objetivo — rotear para uma regiao ou descartar
— para que a conformidade continue sendo medida por um predicado verificavel, e
nao por julgamento sobre a qualidade da resposta.
"""

from __future__ import annotations

from dataclasses import dataclass

DROP = "DROP"


@dataclass(frozen=True)
class SemanticRule:
    """Uma restricao e a acao que ela torna obrigatoria.

    token     identificador anexado a tarefa (o que a versao original usava).
    statement frase em linguagem natural; e o que vai ao modelo quando a regra
              e held-out, ja que nesse caso o token sozinho nao carrega o
              significado para quem nunca o viu.
    required  DROP, ou "REGION:<NOME>" para exigir roteamento aquela regiao.
    split     "seen" ou "heldout".
    family    agrupa parafrases/variantes para analise (ex.: privacidade).
    """

    token: str
    statement: str
    required: str
    split: str
    family: str

    @property
    def required_region(self) -> str | None:
        if self.required == DROP:
            return None
        return self.required.split(":", 1)[1]

    @property
    def must_drop(self) -> bool:
        return self.required == DROP


# --------------------------------------------------------------------------- #
# SEEN — as tres regras do artigo submetido. Tokens preservados exatamente para
# que as corridas antigas continuem sendo reproduzidas e comparaveis.
# --------------------------------------------------------------------------- #

_SEEN = [
    SemanticRule(
        token="restricao_gdpr_europa",
        statement="This task carries EU personal data under GDPR and must be "
                  "processed on a satellite in the EUROPE region.",
        required="REGION:EUROPE",
        split="seen",
        family="privacy",
    ),
    SemanticRule(
        token="restricao_soberania_brasil",
        statement="This task is subject to Brazilian data sovereignty rules and "
                  "must be processed on a satellite in the BRAZIL region.",
        required="REGION:BRAZIL",
        split="seen",
        family="sovereignty",
    ),
    SemanticRule(
        token="falha_hardware_camera_esq",
        statement="The left camera that produced this task has failed; the task "
                  "is unusable and must be discarded.",
        required=DROP,
        split="seen",
        family="hardware",
    ),
]

# --------------------------------------------------------------------------- #
# HELDOUT — nunca aparecem no prompt, na tabela de regras nem na observacao do
# DRL. Cobrem tres tipos de generalizacao distintos, para que a analise consiga
# separar "parafraseou" de "raciocinou":
#
#   parafrase        mesma obrigacao das SEEN, redacao e rotulo diferentes.
#   nova jurisdicao  mesma ESTRUTURA (rotear para regiao) com regiao inedita.
#   nova falha       mesma estrutura (descartar) com causa inedita.
#
# A proporcao de regras de DESCARTE e mantida igual a do conjunto seen (1/3:
# duas de seis aqui, uma de tres la). Isso nao e cosmetico. O teto de uma
# politica cega "anomalia -> descarta" e exatamente a fracao de regras de
# descarte, entao um conjunto held-out com proporcao diferente faria o DRL
# parecer melhor ou pior por composicao do corpus, e nao por capacidade. Com as
# proporcoes iguais, o teto cego e o mesmo nos dois splits e a comparacao mede
# o que deveria medir.
#
# Uma tabela de regras erra todas por construcao; o DRL com one-hot nao tem
# dimensao para representa-las. Um modelo de linguagem, em principio, pode
# aplica-las a partir da frase que acompanha a tarefa.
# --------------------------------------------------------------------------- #

_HELDOUT = [
    SemanticRule(
        token="restricao_residencia_dados_ue",
        statement="Data-residency policy requires that this payload never leave "
                  "European jurisdiction; process it in the EUROPE region.",
        required="REGION:EUROPE",
        split="heldout",
        family="privacy",           # parafrase da GDPR, redacao independente
    ),
    SemanticRule(
        token="restricao_hipaa_eua",
        statement="This task contains protected health information governed by "
                  "US HIPAA and must be processed in the USA region.",
        required="REGION:USA",
        split="heldout",
        family="health",            # jurisdicao inedita (USA)
    ),
    SemanticRule(
        token="restricao_itar_eua",
        statement="Export-control (ITAR) rules apply to this payload; it may "
                  "only be processed on United States territory, i.e. the USA "
                  "region.",
        required="REGION:USA",
        split="heldout",
        family="export",            # jurisdicao inedita, motivacao distinta
    ),
    SemanticRule(
        token="falha_bateria_celula_critica",
        statement="A critical battery cell fault corrupted the acquisition for "
                  "this task; the data cannot be trusted and must be discarded.",
        required=DROP,
        split="heldout",
        family="hardware",          # causa de descarte inedita
    ),
    SemanticRule(
        token="restricao_lgpd_brasil",
        statement="Brazilian LGPD governs the personal data in this task; it "
                  "must be processed on a satellite in the BRAZIL region.",
        required="REGION:BRAZIL",
        split="heldout",
        family="privacy",           # parafrase estrutural da soberania, base legal distinta
    ),
    SemanticRule(
        token="falha_calibracao_sensor_termico",
        statement="The thermal sensor lost calibration before this acquisition, "
                  "so the product is invalid and must be discarded.",
        required=DROP,
        split="heldout",
        family="hardware",
    ),
]

ALL_RULES = _SEEN + _HELDOUT
BY_TOKEN = {r.token: r for r in ALL_RULES}

# Ordem canonica do one-hot do DRL. Contem SOMENTE as regras seen: e exatamente
# esse o ponto do experimento — o agente recebe paridade de informacao nas
# regras conhecidas em tempo de projeto, e nao tem como representar as demais.
ONEHOT_ORDER = [r.token for r in _SEEN]


def rules(split: str | None = None) -> list[SemanticRule]:
    if split is None:
        return list(ALL_RULES)
    return [r for r in ALL_RULES if r.split == split]


def get(token: str | None) -> SemanticRule | None:
    return BY_TOKEN.get(token) if token else None


def onehot(token: str | None) -> list[float]:
    """Codificacao one-hot das regras seen; vetor nulo para ausencia ou held-out.

    Held-out virar vetor nulo nao e descuido: e a limitacao que o experimento
    quer expor. Um espaco de observacao fixado em tempo de projeto nao tem onde
    colocar uma restricao que so apareceu depois.
    """
    vec = [0.0] * len(ONEHOT_ORDER)
    if token in ONEHOT_ORDER:
        vec[ONEHOT_ORDER.index(token)] = 1.0
    return vec


def is_compliant(token: str | None, decision_sat_id, fleet) -> bool:
    """Ground truth da conformidade, identico para todos os motores.

    Sem restricao, qualquer roteamento valido serve. Com restricao, so a acao
    que ela prescreve conta — inclusive quando essa acao e descartar.
    """
    rule = get(token)
    if rule is None:
        return True
    if rule.must_drop:
        return decision_sat_id is None
    if decision_sat_id is None:
        return False
    sat = next((s for s in fleet if s["id"] == decision_sat_id), {})
    return sat.get("region") == rule.required_region


def prompt_rule_block(split: str = "seen") -> str:
    """Bloco de regras listado no prompt — apenas as seen, por desenho.

    As held-out chegam ao modelo pela frase anexada a propria tarefa; se
    entrassem aqui deixariam de ser held-out.
    """
    lines = []
    for i, r in enumerate(rules(split), 1):
        action = ("drop the task entirely" if r.must_drop
                  else f"route exclusively to {r.required_region}; drop if unavailable")
        lines.append(f" {i}. {r.token}: {action}.")
    return "\n".join(lines)
