# Achados da revisão — WPMC 2026, paper #1571341762

**Artigo:** *Semantic Task Orchestration in LEO-NTN Multi-Access Edge Computing: A Comparative Study*
**Situação:** aceito (weak accept ×2, accept if room ×1; Contribuições 3/3/2)
**Período coberto:** 19 a 22 de setembro de 2026
**Código e dados:** `wocn-unicamp/CosmicBeats-Simulator`, branches `camera-ready/wpmc2026-reviewer-response` e `faixa2/multiseed-infra`

Este documento registra tudo o que foi encontrado ao auditar o artigo submetido e ao
executar a revisão. Inclui deliberadamente os achados que **enfraquecem** afirmações
nossas, porque são justamente os que precisam de decisão antes do camera-ready ir
para o IEEE Xplore.

---

## Sumário executivo

Sete achados mudam o que podemos afirmar. Em ordem de gravidade:

1. **A energia por decisão nunca foi medida.** São quatro constantes fixas no código.
   O "5.000×" do abstract é a razão entre duas delas, não um resultado experimental.
2. **A latência de 6112 ms é, em sua maior parte, artefato de rate-limit da API**, não
   custo de inferência. A mediana real é 955 ms.
3. **Com n=8 anomalias, só uma das seis comparações é estatisticamente significativa.**
   A ordenação de quatro vias que o artigo afirmava não se sustentava.
4. **A vantagem do DRL sobre o BASELINE não replica** em 20 sementes.
5. **Dois bugs no código**, um deles anulando exatamente o experimento que o Revisor 3
   pediu. Nenhum afeta os resultados originais publicados; um afetou um resultado que
   nós mesmos acrescentamos na revisão.
6. **O Revisor 3 está certo, e agora temos a medida disso:** com as regras conhecidas em
   tempo de projeto, uma tabela de três entradas e um DRL com paridade de informação
   chegam ambos a 100%. Um modelo de linguagem não se justifica nessa escala.
7. **Os números de RAM do artigo estavam corretos**; eram os artefatos commitados que
   estavam errados. Não houve número inventado — houve lacuna de reprodutibilidade.

---

## Parte I — Auditoria do artigo submetido

### I.1 Achados que agravam as críticas dos revisores

#### A energia por decisão é premissa, não medição

`src/ai_logic.py` define quatro constantes fixas, aplicadas uniformemente a toda
decisão, sem contagem de tokens, tempo medido ou modelo de rádio:

| motor | J/decisão | justificativa no comentário do código |
|---|---|---|
| BASELINE | 0,001 | CPU if/else, ~1 µs @ 1 W |
| DRL | 0,005 | "Q-table lookup" — **errado**, o agente é um MLP [64,64] em PyTorch |
| SLM | 0,100 | inferência NPU, 50 ms @ 2 W |
| LLM | 5,000 | transmissão LEO→GS, ~1 s @ 5 W |

Verificação: `total_joules` é literalmente `constante × 69` nos quatro motores. Logo o
**"5.000× do abstract é 5,0 / 0,001"** — a razão entre dois números escolhidos à mão.
Os Revisores 1 e 2 pediram a metodologia de calibração; a resposta honesta é que não
havia calibração. A constante do LLM ainda cobra apenas o *uplink* e omite a inferência
no datacenter, ou seja, **subestima** o custo real da nuvem.

#### A latência de 6112 ms é dominada por backoff de quota

A distribuição por tarefa (que já estava gravada, mas nunca foi analisada) é bimodal:

| estatística | valor |
|---|---|
| mediana | **954,7 ms** |
| média | 6111,9 ms |
| desvio-padrão | 13784,7 ms (2,25× a média) |
| IQR | 876,6 – 1130,6 ms |
| chamadas > 10 s | 10 de 69 (14,5%) |
| caudas | 18000, 35026, 40147, 40669, 47318, 51491, 51713, 52001 ms |

Os valores de 30 a 52 s casam com o backoff do próprio código (`sleep(30×(tentativa+1))`
em HTTP 429). A média é **6,4× a mediana**. O §III-E afirmava que 6112 ms era o
*"full ground-station round-trip (empirically measured)"*, confundindo backoff de cota
com propagação.

Corrigir isso **melhora** o artigo: o LLM opera em ~1 s, não em 6 s.

#### Com n=8, a ordenação central não se sustentava

Intervalos de Wilson 95% e teste exato de McNemar pareado (os quatro motores veem a
mesma sequência de tarefas, o que permite o teste pareado, bem mais potente):

| motor | ACR | IC 95% |
|---|---|---|
| BASELINE | 2/8 = 25,0% | [7,1 – 59,1] |
| DRL | 3/8 = 37,5% | [13,7 – 69,4] |
| SLM | 6/8 = 75,0% | [40,9 – 92,9] |
| LLM | 8/8 = 100,0% | [67,6 – 100,0] |

| comparação pareada | discordantes | p | veredito |
|---|---|---|---|
| LLM vs BASELINE | 6 / 0 | **0,031** | significativo |
| LLM vs DRL | 5 / 0 | 0,0625 | não significativo |
| SLM vs DRL | 3 / 0 | 0,250 | não significativo |
| SLM vs BASELINE | 5 / 1 | 0,219 | não significativo |
| DRL vs BASELINE | 3 / 2 | 1,000 | não significativo |

**Só LLM vs BASELINE é significativo** — e é justamente o contraste que o abstract
destaca (100% vs 25%). Sorte. Todas as demais afirmações de ordenação, inclusive
"o DRL supera o BASELINE", eram indistinguíveis de ruído.

### I.2 Achado que inocenta o artigo

O `.tex` reportava utilização de RAM de 13,7 / 12,1 / 12,3 / 13,1%. **Nenhum**
`mec_summary_*.json` do histórico do git continha esses valores — todos tinham 12,2.
Isso parecia, à primeira vista, número sem lastro.

Recalculando a integral ponderada no tempo a partir dos CSVs por tarefa, obtém-se
**exatamente 13,7 / 12,1 / 12,3 / 13,1**. Os JSONs é que eram resíduo de uma métrica de
*snapshot* antiga: o código foi corrigido para a integral, mas os logs nunca foram
regerados — e depois foram deletados no commit `93ac69e`.

Não houve número inventado. Houve lacuna de reprodutibilidade, agora fechada.

### I.3 Confirmações diretas das críticas

- **Assimetria de informação (Rev. 3).** A observação do DRL tem **1 bit** `has_anomaly`;
  o subtipo existe apenas como variável oculta de *reward shaping*. Nos dados, o DRL
  descartou **todas as 8** tarefas com anomalia — política degenerada "anomalia → DROP".
- **Cenário semântico trivial (Rev. 3).** As restrições eram 3 literais snake_case fixos
  e o ground truth um `if/elif` de 3 ramos. Um dicionário de 3 entradas atinge 100%.
  A crítica procede integralmente.
- **Sem gargalo de recurso (Rev. 3).** RAM ocupa 12–14% de 4096 MB; a restrição nunca
  vincula. `gamma=0` e `step()` sempre terminal: é um bandit de passo único.
- **O "SLM on-device" é remoto.** Chama a mesma API Gemini (`gemma-4-26b-a4b-it`). A
  latência de 50 ms é constante; a latência real de API **é medida** e depois descartada.
  Está declarado na Tabela I como *benchmark*, mas precisa ficar inequívoco no texto.

### I.4 Bloqueios de reprodutibilidade encontrados

`requirements.txt` sem `stable-baselines3`, `gymnasium` e `torch`; `venv/` sem o
diretório `bin/`; `scripts/train_drl.py` deletado mas ainda referenciado pelo código;
`logs/` inteiro deletado; `.env` ausente. Em outras palavras, **nenhum resultado do
artigo era reproduzível a partir do repositório como ele estava**. Todos corrigidos.

---

## Parte II — Correções aplicadas ao camera-ready

Onze itens, todos verificados contra o PDF compilado. O artigo permanece em **6 páginas**
(limite confirmado), sem erros de compilação nem referências pendentes.

| # | Correção | Revisor |
|---|---|---|
| 1 | Latência passa a mediana 955 ms (IQR 877–1131); cauda atribuída a rate-limit | R1 |
| 2 | Energia reenquadrada como modelo paramétrico declarado, com derivação de cada constante | R1, R2 |
| 3 | IC de Wilson na Tabela III e os quatro McNemar no texto | R1 |
| 4 | Abstract, contribuições e conclusão suavizados | R1 |
| 5 | Subseção *Limitations* assumindo a assimetria do DRL e o vocabulário fechado | R3 |
| 6 | 8 siglas definidas; formato das restrições e `NTNMECEnv` explicitados | R2 |
| 7 | Fig. 2 convertida de `verbatim` para `lstlisting` | R1 |
| 8 | Diacrítico do autor corrigido (faltava `fontenc[T1]`) | R1 |
| 9 | Impacto diferencial do eclipse de 53% por motor | R1 |
| 10 | Caveat de saturação para drop-rate e RAM | R1 |
| 11 | `logs/` regenerado e commitado | — |

Sobre o item 8: o PDF submetido emitia o acento de "Octávio" como glifo solto. Provado
no nível de bytes — submetido `4f 63 74 61 cc81 76 69 6f` (Octa + U+0301 + vio), revisado
`4f 63 74 c3a1 76 69 6f` (á precomposto).

Sobre o item 11: os motores SLM e LLM **não** foram re-executados, deliberadamente. Eles
dependem de API e re-executá-los substituiria o registro experimental que embasa o
artigo. Verificou-se, campo a campo, que os summaries arquivados divergiam dos CSVs em
**exatamente um** campo (`avg_ram_utilization_pct`), corrigido cirurgicamente.

---

## Parte III — Bugs encontrados no código

### III.1 Janela da integral de RAM (baixo impacto)

`_last_step_time` para em 1195 s (timestamp do último passo), mas a janela simulada é
240 × 5 = 1200 s. Faltava um passo no denominador. Corrigido; o código agora produz
exatamente os valores do artigo.

### III.2 Revalidação do DRL anulava roteamento cruzado (alto impacto)

`_resolve_action` comparava o satélite escolhido com a região da **própria tarefa**.
A premissa é falsa sempre que uma regra exige outra região — GDPR manda para EUROPE e
soberania para BRAZIL, venha a tarefa de onde vier. A política escolhia o satélite
correto e a validação posterior o descartava.

**Como foi isolado**, porque o caminho importa e quase parei na primeira hipótese:

1. O braço de paridade dava 5/8. Parecia teto de capacidade.
2. Treinei **8× mais** (400 mil passos). Deu **exatamente** os mesmos 5/8 — subtreino
   descartado.
3. Sondei a política com frota saudável: **9/9** em todas as combinações de regra ×
   região. A política estava perfeita, logo o erro só podia estar *depois* dela.

Corrigido, o braço de paridade vai a **8/8**.

**Impacto nos resultados:**

- **Resultados originais do artigo: intactos.** O agente cego sempre emite DROP, que
  retorna antes dessa checagem. Verificado bit a bit na corrida canônica.
- **Resultado que nós acrescentamos na revisão: afetado.** Em 4 das 20 sementes o agente
  cego escolhia rotear numa tarefa com anomalia. O DRL multi-semente passou de
  40/138 (29,0%) para **47/138 (34,1%)**, e o McNemar de p=0,818 para p=0,320.
  A conclusão do parágrafo não muda — a vantagem sobre o BASELINE continua sem replicar —
  mas os números foram corrigidos no `.tex`.

### III.3 Erros de método na análise, corrigidos

- **Pareamento colapsando.** O analisador exigia a tarefa presente em *todos* os motores
  carregados. Quando o primeiro run remoto entrou no diretório multi-semente, a
  agregação colapsou para a única semente onde todos existiam, e os números
  "multi-semente" passaram a descrever 1 semente em vez de 20. Corrigido com fixação
  explícita do conjunto de motores.
- **Overflow numérico.** As somas binomiais exatas estouravam o float com n na casa do
  milhar (`C(n,i) × pⁱ`). Passadas para espaço logarítmico, preservando os valores de n
  pequeno já validados.

---

## Parte IV — Resultados novos

### IV.1 Replicação multi-semente (20 sementes, 1.337 tarefas, 138 com anomalia)

| motor | ACR | IC 95% |
|---|---|---|
| BASELINE | 37/138 = 26,8% | [20,1 – 34,8] |
| DRL | 47/138 = 34,1% | [26,7 – 42,3] |

McNemar pareado: **p = 0,320**. A vantagem do DRL sobre o BASELINE observada na corrida
única (37,5% vs 25,0%) **não replica**.

O que replica, e mais nitidamente, é o argumento information-theoretic do artigo: uma
política que não consegue ler a regra não pode fazer melhor, em esperança, do que a
probabilidade de uma ação fixa ser a certa. Com os três tipos de regra equiprováveis,
esse limite é **1/3**. Os 34,1% [26,7 – 42,3] estão sobre ele.

> **Nota de método.** Na primeira versão deste texto eu tratava a fração de descartes
> *realizada* na amostra (28,3%) como se fosse o teto. É uma estimativa ruidosa dele, e a
> distinção passou a importar quando a correção do III.2 levou o DRL acima daquela
> fração. A formulação correta é 1/3 em esperança.

### IV.2 Tabela de generalização (20 sementes, 138 tarefas com anomalia por split)

O experimento que responde o Revisor 3. Dois conjuntos de restrições:

- **conhecidas** — as três do artigo, enumeradas em tempo de projeto. Todo motor pode
  codificá-las: a tabela tem entrada, o DRL tem dimensão no one-hot, o prompt as lista.
- **inéditas** — seis restrições novas (HIPAA/EUA, ITAR/EUA, LGPD/Brasil, residência de
  dados UE, e duas falhas de hardware novas), que chegam à tarefa como **frase em
  linguagem natural** e não aparecem no prompt, na tabela nem no espaço de observação.

A proporção de regras de descarte é igual nos dois conjuntos (1/3). Isso não é cosmético:
o teto da política cega *é* essa fração, e um corpus desbalanceado faria o DRL parecer
melhor por composição, não por capacidade. Na primeira versão o held-out tinha 2/5 de
descarte e o DRL apareceu à frente do oráculo — artefato, corrigido antes de qualquer
conclusão.

| motor | conhecidas | inéditas | queda |
|---|---|---|---|
| BASELINE | 26,8% [20,1–34,8] | 24,6% [18,2–32,4] | −2,2 pp |
| **ORACLE** (tabela de regras) | **100,0%** [97,3–100] | 24,6% [18,2–32,4] | **−75,4 pp** |
| DRL (cego) | 34,1% [26,7–42,3] | 37,7% [30,0–46,0] | +3,6 pp |
| **DRL_ONEHOT** (paridade) | **100,0%** [97,3–100] | 47,8% [39,7–56,1] | −52,2 pp |

**Duas leituras:**

1. **O Revisor 3 está certo.** Com as regras enumeradas em tempo de projeto, a tabela de
   três entradas e o DRL com paridade chegam ambos a 100%. Nessa escala um modelo de
   linguagem não se justifica, e precisamos publicar isso em vez de tentar refutá-lo.
2. **A tabela de regras falha em silêncio.** Nas inéditas o ORACLE tem **0/0 pares
   discordantes** contra o BASELINE em 138 tarefas — não acerta pouco, toma *exatamente
   as mesmas decisões*. Não levanta exceção, não sinaliza: simplesmente deixa de
   perceber que havia uma obrigação.

### IV.3 Caveat obrigatório sobre o DRL com paridade

Os 47,8% do DRL_ONEHOT nas inéditas são estatisticamente significativos contra o
BASELINE (p = 0,0003). **Reportar só isso seria enganoso.** Decompondo:

| grupo de regras inéditas | DRL_ONEHOT | BASELINE |
|---|---|---|
| exigem DESCARTE | 49/49 (100,0%) | 0/49 (0,0%) |
| exigem EUROPE | 16/26 (61,5%) | 12/26 (46,2%) |
| exigem BRAZIL/USA | **1/63 (1,6%)** | 22/63 (34,9%) |

Sondando a política num estado fora da distribuição de treino — `has_anomaly=1` com
one-hot todo zero, que é exatamente o que uma regra inédita produz — o comportamento é
**arbitrário**: descarta para BRAZIL e USA, roteia para EUROPE. O ganho agregado vem
inteiro do reflexo de descartar. Nas regras que exigem uma região específica diferente de
EUROPE, o agente é **20× pior que a heurística ingênua**.

Não é generalização parcial. É reflexo que por acaso se alinha com parte do corpus.

### IV.4 Descompasso entre distribuição de treino e simulação

No treino, bateria ~ U(15,100) e RAM livre ~ U(0,4096) tornam um satélite específico
indisponível em **17,4%** dos casos. Na simulação real, a RAM fica em 13,7% de ocupação e
as baterias acima do limiar, de modo que isso praticamente não ocorre. O agente é
recompensado por descartar em situações que quase não acontecem.

Afeta os dois braços do DRL, inclusive o do artigo. Com o bug III.2 corrigido deixou de
causar dano observável (o braço de paridade já atinge 100% nas conhecidas), mas continua
sendo fraqueza metodológica e é candidato natural a crítica de revisor de periódico.

### IV.5 Custo real da campanha remota

Medido: **979 s por execução de SLM** — bem acima dos ~8 min que a estimativa por limite
de RPM sugeria. A campanha completa de replicação (40 execuções, ~2.760 chamadas)
demanda cerca de 10 h de parede, distribuídas em vários dias de cota gratuita. Por isso o
executor é resumível: consome a cota do dia, para limpo e retoma sem refazer nada.

Uma execução independente do SLM na semente 0 reproduziu **SCR 97,1% e ACR 75,0%**,
idênticos ao artigo, diferindo apenas em drops por infraestrutura (4 contra 7) — o que
reforça a leitura de que aqueles drops eram instabilidade de API, não decisão do modelo.

### IV.6 Regras inéditas: os modelos de linguagem (resultado decisivo)

Três sementes, 20 tarefas com anomalia pareadas entre os **seis** motores, todas sob
restrições que nenhum deles viu em tempo de projeto.

| motor | ACR | IC 95% | McNemar vs BASELINE |
|---|---|---|---|
| BASELINE | 6/20 = 30,0% | [14,5 – 51,9] | — |
| ORACLE | 6/20 = 30,0% | [14,5 – 51,9] | 0/0, p = 1,000 |
| DRL (cego) | 6/20 = 30,0% | [14,5 – 51,9] | 5/5, p = 1,000 |
| DRL_ONEHOT | 10/20 = 50,0% | [29,9 – 70,1] | 6/2, p = 0,289 |
| SLM | 10/20 = 50,0% | [29,9 – 70,1] | 5/1, p = 0,219 |
| **LLM** | **20/20 = 100,0%** | [83,9 – 100,0] | **14/0, p = 0,0001** |

O agregado já separa o LLM de todo o resto, mas é a decomposição por tipo de obrigação
que distingue raciocínio de reflexo:

| obrigação | BASELINE | ORACLE | DRL | DRL_ONEHOT | SLM | **LLM** |
|---|---|---|---|---|---|---|
| descartar (n=5) | 0/5 | 0/5 | 5/5 | 5/5 | 5/5 | **5/5** |
| rotear a EUROPE (n=9) | 4/9 | 4/9 | 1/9 | 5/9 | 4/9 | **9/9** |
| rotear a BRAZIL/USA (n=6) | 2/6 | 2/6 | 0/6 | 0/6 | 1/6 | **6/6** |

**O LLM é o único motor que acerta as regras de região** — 9/9 e 6/6. Todos os demais
ficam no nível do BASELINE ou abaixo dele nesse grupo. Descartar diante de uma anomalia é
barato e qualquer um faz; aplicar a obrigação correta exige ler a regra.

**O SLM não generaliza.** Seus 50% são inteiramente explicados pelo reflexo de descartar
(5/5) somado a acertos de região no nível do acaso (4/9 e 1/6) — numericamente idêntico
ao DRL_ONEHOT, cujo comportamento já sabemos ser arbitrário. Isso é um resultado novo e
relevante: o artigo apresenta o SLM como meio-termo capaz (75% de ACR nas regras
conhecidas), e nas inéditas ele se comporta como reflexo. A distinção que importa não é
"modelo de linguagem vs. otimizador numérico", é **qual** modelo de linguagem.

Ressalva de tamanho: n=20 e três sementes. O IC do LLM é [83,9 – 100], e o contraste
contra o BASELINE é sólido (p=0,0001), mas a separação entre SLM e DRL_ONEHOT não é
resolvível nesse n. Ampliar as sementes é o próximo passo.

---

## Parte V — Reformulação proposta da tese

A tese do artigo submetido — *"modelos de linguagem superam DRL em conformidade
semântica"* — é atacável e, nos dados, não é significativa exceto contra a heurística.

O experimento que está tomando forma sustenta algo mais defensável:

> Nas restrições enumeradas em tempo de projeto, **tabela de regras, DRL com paridade de
> informação e LLM empatam em 100%**. Um modelo de linguagem não se justifica ali. O que
> só ele pode sustentar é a restrição que ninguém enumerou — e é isso que o experimento
> de generalização mede.

Isso converte a crítica mais dura do Revisor 3 em contribuição, em vez de tentar
refutá-la.

---

## Parte VI — O que falta e riscos abertos

### Falta

1. **Mais sementes na coluna das regras inéditas.** As três atuais (n=20) já separam o
   LLM de todo o resto com p=0,0001, mas não separam o SLM do reflexo do DRL_ONEHOT.
   Custo: ~20 min por execução remota.
2. **Medição real de energia** (item 3.2 do plano) — exige `sudo` para ler os contadores
   RAPL; não iniciado.
3. **Pressão de recurso e formulação sequencial** (item 3.3, resposta ao Rev. 3 sobre
   γ=0) — não iniciado.

### Riscos

- ~~O maior: se SLM e LLM também colapsarem nas regras inéditas...~~ **Risco resolvido**
  (ver IV.6). O LLM sustenta 20/20; o SLM, não. A tese de generalização se sustenta, mas
  com um refinamento que não estava previsto: ela vale para o LLM e **não** para o SLM, o
  que obriga a rever como o artigo posiciona o SLM como meio-termo.
- O corpus held-out tem seis regras. É pouco para sustentar afirmação forte de
  generalização; convém ampliá-lo antes da submissão do periódico.
- A composição do corpus influencia os agregados (ver IV.3). Qualquer regra nova deve
  preservar a proporção de 1/3 de descartes.

---

## Apêndice — Como verificar

Tudo abaixo roda a partir do repositório, sem chave de API (exceto os motores remotos).

```bash
# Confere cada valor numérico do .tex contra o arquivo de dados que o originou.
# Retorna exit 1 em divergência; testado com adulteração proposital.
venv/bin/python scripts/verify_paper_numbers.py

# Estatística da corrida canônica: Wilson, Clopper-Pearson e McNemar exatos,
# distribuição de latência, RAM ponderada no tempo, proveniência da energia.
venv/bin/python scripts/analyze_results.py --logs-dir logs

# Replicação multi-semente (motores determinísticos).
venv/bin/python scripts/analyze_results.py --logs-dir logs/multiseed --engines BASELINE DRL

# Tabela de generalização.
venv/bin/python scripts/generalization_table.py --seeds 0-19 \
    --engines BASELINE ORACLE DRL DRL_ONEHOT

# Campanha remota, resumível e limitada por cota diária.
venv/bin/python scripts/run_experiments.py --seeds 0-19 --engines SLM LLM \
    --outdir logs/multiseed --max-api-runs N
```

A estatística é implementada sem dependências externas (`scipy` não estava disponível):
Wilson, Clopper-Pearson por inversão da CDF binomial e McNemar exato. O Clopper-Pearson
foi validado contra o valor fechado conhecido — 8/8 → [63,1 – 100], e 0,025^(1/8) = 0,631.
