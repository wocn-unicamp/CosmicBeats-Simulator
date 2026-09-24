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
5. **Bugs no código — e um erro meu ao corrigi-los.** Um bug anulava exatamente o
   experimento que o Revisor 3 pediu. A minha primeira correção fez o validador do DRL
   consultar a regra oculta, o que vazava informação e inflou números que cheguei a
   publicar no camera-ready. Encontrado e corrigido (Parte IX.5); os resultados
   originais do artigo nunca foram afetados.
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
> ⚠ **Corrigido depois (Parte IX.5).** Esta "correção" (29,0% → 34,1%) veio do validador que vazava. O valor honesto é o original: **40/138 = 29,0%, p = 0,818**.


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
> ⚠ **Corrigido depois (Parte IX.5).** Com o validador corrigido: DRL **40/138 = 29,0% [22,1–37,0], p = 0,818**. A conclusão (não replica) se mantém; 29,0% está dentro do limite de 1/3.


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
> ⚠ **Corrigido depois (Parte IX.5).** Com o validador corrigido: DRL 29,0% (conhecidas) e 36,2% (inéditas); **DRL_ONEHOT 23,2% nas inéditas** — abaixo do BASELINE (24,6%), McNemar 19/21, p = 0,87. As demais linhas não mudam.


**Duas leituras:**

1. **O Revisor 3 está certo.** Com as regras enumeradas em tempo de projeto, a tabela de
   três entradas e o DRL com paridade chegam ambos a 100%. Nessa escala um modelo de
   linguagem não se justifica, e precisamos publicar isso em vez de tentar refutá-lo.
2. **A tabela de regras falha em silêncio.** Nas inéditas o ORACLE tem **0/0 pares
   discordantes** contra o BASELINE em 138 tarefas — não acerta pouco, toma *exatamente
   as mesmas decisões*. Não levanta exceção, não sinaliza: simplesmente deixa de
   perceber que havia uma obrigação.

### IV.3 Caveat obrigatório sobre o DRL com paridade
> ⚠ **Corrigido depois (Parte IX.5).** Esta seção inteira está superada. O "reflexo de descarte" do DRL_ONEHOT nas regras inéditas era, em boa parte, o validador forçando o descarte. Sem o vazamento, o DRL_ONEHOT fica no nível do BASELINE nas inéditas.


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
> ⚠ **Corrigido depois (Parte IX.5).** Com o validador corrigido: **DRL_ONEHOT 7/20 = 35,0%**, McNemar vs BASELINE 3/2, p = 1,0; descarte 2/5 (não 5/5). O reflexo de descarte é real no **SLM** (5/5), que nunca passou por validador.

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

## Parte VII — Auditoria final de cobertura (23/09)

Conferência item a item dos pedidos dos revisores contra o `.tex`. Revelou duas
perguntas do Revisor 2 sem resposta — e as duas atingem o enquadramento do artigo.

### VII.1 "Qual a faixa realista de latência neste cenário?" — sem resposta

O abstract promete *"millisecond-range decision windows"*. O cenário simulado não tem
janela de milissegundos: com chegadas Poisson de λ = 4/min, o intervalo médio entre
tarefas é **15 s**, e a latência mediana do LLM (0,955 s) ocupa **6,4%** dele.

Duas consequências. A frase do abstract não é sustentada pelo próprio cenário. E, pior,
ela desqualificaria o LLM: se a janela fosse de milissegundos, um motor de ~1 s estaria
fora. A resposta honesta é que a janela operante é de segundos, e nela o LLM é viável.

### VII.2 "A energia por decisão é adequada para uso real?" — sem resposta, e a resposta enfraquece a tese

Relacionando a energia de decisão com o orçamento de bateria que o próprio artigo declara
(75 / 50 / 100 Wh), sob as constantes do próprio modelo:

| motor | 69 decisões | fração da menor bateria (50 Wh = 180.000 J) |
|---|---|---|
| BASELINE | 0,069 J | 0,00004% |
| DRL | 0,345 J | 0,00019% |
| SLM | 6,9 J | 0,00383% |
| LLM | 345 J | **0,19%** |

E **executar uma única tarefa** (2% de SoC no código) custa 3.600 J na menor bateria —
**720×** o custo de uma decisão do LLM.

Ou seja: sob as premissas do próprio artigo, **o custo energético de decidir é desprezível
frente ao custo de executar**, mesmo para o LLM. O "trade-off conformidade vs.
eficiência" que o artigo apresenta como resultado central é quantitativamente vazio no
orçamento do satélite. A pergunta do revisor responde-se com "sim, é adequada" — e
justamente por isso a energia por decisão não é o eixo que o artigo diz ser.

Ressalva: as duas grandezas (5 J por decisão e 2% por tarefa) são premissas de modelo,
não medições. A comparação é consistência interna do modelo, não fato físico. Ainda
assim, é o modelo que o artigo usa para afirmar o trade-off.

### VII.3 Branch desatualizada

A versão final do artigo (multi-semente + correção do teto) está apenas em
`faixa2/multiseed-infra`. A branch `camera-ready/wpmc2026-reviewer-response` ficou
26 linhas atrás. Submeter a partir dela entregaria a versão antiga.

---

## Parte VIII — Segunda rodada do camera-ready (23/09)

Prazo efetivo: **30/09/2026** (registro de *Symposium Papers*, categoria deste artigo).
O artigo segue em **6 páginas**, sem erros de compilação, e o verificador passou de 30
para **54 valores** conferidos contra os dados.

### VIII.1 O que entrou no artigo

- **Nova Seção V-D e Tabela IV** — paridade de informação e restrições inéditas. É a
  resposta direta ao Revisor 3, antes presente só como "trabalho em andamento".
> ⚠ **Corrigido depois (Parte IX.5).** Os valores da Tabela IV foram depois corrigidos: DRL 29,0% nas conhecidas; DRL-OH 35,0% nas inéditas.

- **Resposta ao Revisor 2 sobre latência** (§V-C): a janela operante é o intervalo entre
  chegadas, 15 s; o LLM usa 6,4% dela. Orçamentos de milissegundos — típicos de
  escalonamento de enlace, não de alocação de tarefa — excluiriam o LLM. O abstract
  deixou de prometer *"millisecond-range decision windows"*.
- **Resposta ao Revisor 2 sobre adequação energética** (§V-B): a energia de decisão do LLM
  é 0,19% da menor bateria, e executar uma tarefa custa cerca de 720 decisões do LLM. O
  abstract e a conclusão foram reescritos: o custo operante da conformidade é latência e
  dependência de enlace com a estação terrena, não energia.
- **Tese reformulada** no abstract, na contribuição 3 e na conclusão: nas regras de
  projeto, tabela e DRL informado bastam; o LLM se distingue na generalização.
- *Limitations* reescrita: os dois primeiros itens foram resolvidos pela Seção V-D e
  saíram; entraram o tamanho do corpus inédito, a energia como modelo e o SLM emulado.

Para caber, foram comprimidos os parágrafos de resolução estatística, replicação
multi-semente, eclipse, Lei de Little e trabalho futuro, e removida a quebra por anomalia
da §V-A. Nenhum número verificado foi perdido.

### VIII.2 Duas afirmações do artigo que estavam erradas

**"Os 2 erros do SLM foram instabilidade de API, não raciocínio."** Desmentida pela
reexecução independente da semente 0. Ela reproduz os mesmos números (97,1% / 75,0%),
mas não os mesmos erros: a tarefa 44 passou a acertar e a 26 passou a errar, o que é
consistente com instabilidade — porém a tarefa 43 errou nas duas execuções, e na
reexecução o modelo **respondeu, e respondeu errado**: roteou uma tarefa de soberania
brasileira para EUROPE. É erro de raciocínio. Texto corrigido.

**"Oito vezes mais treino não muda o número"** — afirmação que eu mesmo havia escrito no
parágrafo multi-semente, aplicada ao DRL cego. O experimento de treinar 8× mais foi feito
no DRL_ONEHOT, e o número não mudou por causa do bug do `_resolve_action`, não por
limitação do espaço de observação. Aplicado ao DRL cego, não tinha lastro. Removida. A
versão da contribuição 2 permanece, porque lá o argumento é analítico: nenhum treino
supera o limite de 1/3 de uma política que não lê a regra.

### VIII.3 Situação das lacunas da Parte VII

- VII.1 (latência) e VII.2 (energia): **resolvidas** no texto.
- VII.3 (branch desatualizada): **resolvida** — as branches foram consolidadas.

---

## Parte IX — Versão estendida (a partir de 23/09)

Trabalho na branch `extended/journal-version`, criada a partir do camera-ready, que fica
intocado para a revisão do orientador. Correções que atingem o camera-ready são feitas
nele (via worktree separado, sem interferir nas campanhas em curso) e depois trazidas
para a branch estendida por merge.

### IX.1 Campanha de sementes adicionais nas regras inéditas

Objetivo: testar o resultado central do camera-ready, apoiado em apenas 3 sementes
(n=20). Sementes 3 a 9 para SLM e LLM — 14 execuções, ~966 chamadas de API, ~20 min
cada. Rodando destacada da sessão (`setsid nohup systemd-inhibit`), com o notebook na
tomada. O verificador do artigo fixa as sementes da Tabela IV (0, 1, 2), para não mudar
de resultado quando estas chegarem.

### IX.2 Varredura de carga (item 3.3, primeiro passo)

`scripts/load_sweep.py`, 4 motores determinísticos × 6 taxas × 5 sementes. Separa o
descarte por **recurso** (tarefa sem regra — nenhuma regra manda descartá-la) do
descarte por regra.

| λ nominal | λ efetiva | tarefas/execução | descarte por recurso | ACR ORACLE | RAM média | bateria mín. |
|---|---|---|---|---|---|---|
| 2 | 1,8 | 31,2 | 0,0% | 100,0% | 6,2% | 55,2% |
| 4 | 3,4 | 65,2 | 0,0% | 100,0% | 12,6% | 31,2% |
| 8 | 5,8 | 114,8 | 13,8% | 88,5% | 19,4% | 17,6% |
| 12 | 7,6 | 150,6 | 25,7% | 79,3% | 22,0% | 16,3% |
| 16 | 8,8 | 176,2 | 36,3% | 74,5% | 22,3% | 15,8% |
| 24 | 10,4 | 206,4 | 44,4% | 67,2% | 22,7% | 14,6% |

**A bateria é o gargalo, não a RAM.** A RAM nunca passa de 23%: a bateria (2% de SoC por
tarefa aceita) esgota antes, os satélites param de aceitar, e a RAM nunca enche. Isso
torna mais preciso o "no RAM bottleneck" do Revisor 3 — neste modelo a RAM *não consegue*
ser gargalo.

**Onde o comportamento sequencial pode importar:** sob pressão até o ORACLE perde
conformidade (100% → 88,5% em λ=8 nominal), porque uma tarefa com regra de região chega
quando o satélite exigido já está abaixo do piso de 20%. Uma política com visão de futuro
poderia reservar bateria na região que as regras vão exigir. É o regime em que γ>0 tem
algo a aprender.

### IX.3 A taxa de chegada efetiva não é a declarada (corrigido no camera-ready)

A varredura mostrou que o número de tarefas não acompanha o λ pedido (206 em λ=24/min,
contra ~480 esperadas; 65,2 em λ=4 contra 80, desvio sistemático de 3,7σ).

Causa no gerador: no máximo uma chegada por passo de 5 s, e a próxima é cronometrada a
partir do passo que serviu a anterior. O intervalo real é 5·⌈X/5⌉ s, X~Exp(λ), com média
5/(1−e^(−5λ)). O modelo prevê as contagens com erro abaixo de 1,5% de λ=8 a 24. Em λ=4:
intervalo médio de **17,6 s**, taxa efetiva **≈3,4/min**, teto absoluto de 12/min.

O código está correto; o artigo é que descrevia o processo errado. Consequências
corrigidas no camera-ready (commit `07bbc16`):

- a descrição do processo de chegada (§II-B e Tabela I);
- a Lei de Little: com a taxa efetiva prevê **13,8%**, e o BASELINE (que aceita todas as
  tarefas) observa 13,7%. A distância em relação aos "~16%" que o artigo tratava como
  "próxima" vinha toda do λ nominal;
- **os números que eu havia posto na resposta ao Revisor 2** ("15 s", "6,4%") usavam o λ
  nominal. Os corretos são 17,6 s e 5,4%. A conclusão não muda.

A varredura acima usa λ nominal no rótulo; a coluna de λ efetiva é a que vale para
interpretação.

### IX.4 As capacidades de bateria são nominais (corrigido no camera-ready)

Encontrado ao ler a dinâmica do simulador para construir o ambiente sequencial.
`mec_battery_capacity_wh` é só atribuída e impressa. Toda a dinâmica está em pontos
percentuais — 2% de SoC por tarefa, +0,083%/passo no sol, −0,040%/passo no eclipse —
independentemente da capacidade. A "heterogeneidade de hardware" da Tabela I
(75/50/100 Wh) não tem efeito nenhum: os satélites diferem só no SoC inicial
(BRAZIL 90%, EUROPE 75%, USA 85%) e na fase orbital.

Explica melhor um dado do próprio artigo: o SAT-2 terminou a 21,8% porque **começou**
em 75%, e não por ter a menor bateria. Também qualifica um número que eu havia
acrescentado: os "~720 decisões do LLM por tarefa" convertem 2% de SoC em joules pela
bateria de 50 Wh — é interpretação, e a mais conservadora. Corrigido no texto (§II-A,
Tabela I, §V-B), commit `5bf737c`. O verificador ganhou checagem dos SoC inicial e final;
os finais estavam no artigo desde a submissão sem nenhuma conferência.

### IX.5 O validador do DRL vazava a identidade da regra — erro meu, com efeito no camera-ready

Este é o achado mais importante desta fase, e o erro é meu.

Ao corrigir o `_resolve_action` (Parte III.2), fiz o validador consultar o registro de
regras pelo token da tarefa para (a) forçar DROP em regra de descarte e (b) exigir a
região da regra. Mas **o validador passou a saber a regra mesmo quando a política não
sabe**. Medido por instrumentação, sementes 0–2:

| braço | split | tarefas de descarte | política descartou | forçadas pelo validador |
|---|---|---|---|---|
| DRL (cego) | conhecidas | 4 | 3 | 1 |
| DRL (cego) | inéditas | 5 | 4 | 1 |
| DRL_ONEHOT | conhecidas | 4 | 4 | 0 |
| DRL_ONEHOT | inéditas | 5 | 2 | **3** |

Nas regras inéditas, 3 dos 5 descartes "conformes" do DRL_ONEHOT eram do validador.
A afirmação do camera-ready *"DRL-OH e SLM descartam todas as cinco tarefas de falha —
um reflexo de descarte"* estava, para o DRL-OH, parcialmente errada.

Há um segundo efeito, mais sutil. Para o braço cego, aceitar uma rota cruzada **porque
ela coincide com a região da regra oculta** também é vazamento. Então a "correção" que
fiz no número multi-semente do DRL (29,0% → 34,1%, Parte III.2) **veio do validador que
vazava**, e o valor honesto provavelmente é o original.

**Três versões do validador, e por que a terceira é a certa:**

1. *Original* — exige sempre a região da tarefa. Honesto para o braço cego (usa só a
   região, que é observada) e é o que gerou os resultados publicados. Defeito: impede o
   DRL_ONEHOT de rotear para outra região.
2. *Minha primeira correção* — consulta a regra. Vaza, como mostrado acima.
3. *Intermediária* — aceita qualquer região viável em tarefa com anomalia, nos dois
   braços. Não consulta a regra, mas **muda o braço cego publicado**: a política cega às
   vezes escolhe outra região numa anomalia, e o validador original anulava essa escolha.
   Verificado — a corrida canônica divergiu.
4. *Final* — **rotear para outra região só é permitido a quem consegue distinguir o tipo
   de regra** (o braço one-hot), e só em tarefa com anomalia. O braço cego volta a ser
   exatamente o do artigo (bit a bit na corrida canônica) e o DRL_ONEHOT continua em 8/8
   nas conhecidas. O que decide é o desenho de observação do braço, nunca a regra oculta.
   O registro de regras fica só na métrica de conformidade, que é onde deve estar.

Todos os resultados de DRL e DRL_ONEHOT (multi-semente, generalização, varredura de carga)
foram apagados e regerados com o validador final.

**Resultados corrigidos** (20 sementes; as duas últimas linhas, 3 sementes pareadas entre
os seis motores, são as da Tabela IV do camera-ready):

| medida | com vazamento | sem vazamento |
|---|---|---|
| DRL multi-semente, conhecidas | 34,1%, p=0,320 | **29,0% [22,1–37,0], p=0,818** |
| DRL_ONEHOT, conhecidas | 100% | 100% |
| DRL, inéditas | 37,7% | 36,2% |
| DRL_ONEHOT, inéditas | 47,8% | **23,2%** (BASELINE: 24,6%; p=0,87) |
| Tabela IV — DRL-OH, inéditas (n=20) | 50,0% | **35,0%** (vs BASELINE 3/2, p=1,0) |
| Tabela IV — descartes do DRL-OH | 5/5 | **2/5** |

O número do DRL multi-semente volta **exatamente** ao da era do artigo, o que confirma o
diagnóstico. Com isso, a tese fica mais nítida, não mais fraca: nas regras inéditas,
tabela de regras, DRL cego e DRL com paridade ficam **todos no nível da heurística**; o
SLM fica no reflexo de descarte; **só o LLM lê a regra**.

Corrigido no camera-ready (commits `c64a197`, `b8d8ca5`): contribuição 2, parágrafo
multi-semente, Tabela IV, Seção V-D, conclusão e *Limitations*.

**Lição de método sobre o próprio verificador.** Ao corrigir a Tabela IV, fiz um teste
negativo reintroduzindo o valor antigo (50,0%) na tabela — e o verificador **não
percebeu**. Ele conferia cada número em qualquer ponto do texto, e "35,0%" continuava na
prosa. A Tabela IV passou a ser conferida **linha a linha**; a mesma adulteração agora é
detectada. É um limite do verificador que vale registrar: checagem por substring garante
que o número existe no texto, não que está no lugar certo.

### IX.6 Ambiente sequencial validado contra o simulador, tarefa a tarefa

`src/schedulers/drl_seq_env.py` (`NTNMECSeqEnv`) — o estado persiste entre decisões:
aceitar uma tarefa consome 2% de SoC e ocupa 500 MB por 60 s, e o episódio cobre a
janela inteira. A observação inclui a fase orbital de cada satélite (sin/cos), sem a qual
"reservar bateria para depois" não teria base observável, e a fração da janela.

**Critério de validação: comparação exata, não estatística.** O ambiente consome o
gerador aleatório na mesma ordem que o simulador, então com a mesma semente o ORACLE
dentro dele deve tomar exatamente as mesmas decisões que o `OracleScheduler`
(`scripts/validate_seq_env.py`).

Primeira execução: 56 de 70 casos idênticos. Os 14 divergentes eram todos sob carga
alta, e a sequência de tarefas batia exatamente — só a decisão diferia. Causa: o
simulador entrega aos schedulers `round(SoC, 1)`. Com SoC real de 20,03% o scheduler vê
20,0 e a checagem "> 20" falha; o ambiente usava o valor bruto. Corrigido separando as
duas coisas — dinâmica sobre o SoC bruto, decisão e observação sobre o arredondado.

Resultado final: **70 de 70 idênticos** — 6 níveis de carga × 5 sementes, mais 20
sementes em λ=4 nas regras conhecidas e 20 nas inéditas. O descompasso entre
distribuição de treino e simulação encontrado no DRL original (IV.4) não se repete.

### IX.7 Agentes sequenciais: primeiros resultados

Treino em λ=12 nominal (7,6 efetiva, regime em que a bateria limita), 300 mil passos,
DQN com os hiperparâmetros do artigo exceto γ. Avaliação pareada nas sementes de teste
0–19 (disjuntas do treino), idêntica ao simulador pela validação da IX.6.

**λ = 12 nominal — 313 tarefas com regra, 2.704 sem:**

| política | ACR | IC 95% | throughput | McNemar vs ORACLE |
|---|---|---|---|---|
| ORACLE | 84,3% | [79,9–88,0] | 74,9% | — |
| γ=0,99, w=0,25 | **93,9%** | [90,7–96,1] | 72,7% | 30/0, p<0,0001 |
| γ=0, w=0,25 | 88,5% | [84,5–91,6] | 74,1% | 13/0, p=0,0002 |
| γ=0,99, w=1 | 85,6% | [81,3–89,1] | 74,6% | 4/0, p=0,125 |
| γ=0, w=1 | 84,3% | [79,9–88,0] | 74,9% | 0/0 (≡ ORACLE) |
| γ=0,99, w=0 | 100% | — | 22,1% | degenerado |
| γ=0, w=0 | 99,4% | — | 33,1% | degenerado |

Em λ=4 todos empatam em 100% de ACR (sem pressão, não há o que planejar); em λ=8 as
diferenças já aparecem, menores.

**w = 0 é degenerado, como previsto na IX.6:** sem valor para o throughput, o agente
descarta a maior parte do tráfego comum para guardar bateria. Isso confirma que a
recompensa precisa de peso explícito de throughput e que o valor de w é escolha de
pesquisa.

**Por que o γ=0 também supera o ORACLE — artefato, não planejamento.** Um agente míope
não tem motivo para economizar bateria. Medindo a taxa de descarte de tarefas comuns
**viáveis** por faixa de SoC do satélite da região (sementes 0–19, w=0,25):

| SoC | γ=0 descarta | γ=0,99 descarta |
|---|---|---|
| 20–22% | 63,3% | 66,5% |
| 22–25% | **0,0%** | 61,8% |
| 25–30% | **0,0%** | 41,8% |
| 30–40% | **0,0%** | 8,0% |
| > 40% | 0,0% | 0,5% |

O γ=0 descarta **só** na faixa colada ao piso. É efeito da aproximação de função na
fronteira de viabilidade: 20,1% é viável e 19,9% é inviável (penalidade −0,1), e a rede
não separa bem os dois. O agente "erra para o lado seguro" e poupa bateria por acidente,
justamente onde ela é escassa. Isso explica o ganho sobre o ORACLE e **não** deve ser lido
como previsão.

**O γ=0,99 aprende uma política de reserva graduada:** retém tráfego comum de forma
progressiva conforme a bateria cai, começando bem acima do piso (42% de descarte já em
25–30%, onde o γ=0 não descarta nada). É o comportamento sequencial que o Revisor 3
pediu — visível na política aprendida, e ausente no agente míope.

**O que ainda não se pode afirmar.** Com w=0,25, o γ=0,99 ganha +5,4 pp de ACR sobre o
γ=0 e perde 1,4 pp de throughput. Isso pode ser uma curva melhor ou só outro ponto da
mesma curva. Para decidir, é preciso a **fronteira** conformidade × throughput de cada γ;
uma segunda grade (w ∈ {0,05; 0,1; 0,15; 0,4; 0,6}) está em treino.

### IX.8 Falha de API se passava por decisão do modelo

Ao ampliar a coluna das regras inéditas com as sementes 3–4, o LLM caiu de 20/20 para
**23/28 (82,1%)**. Antes de corrigir o texto, investiguei.

Os 5 "erros" eram todos **descartes**, e todos em tarefas cuja região **já era** a exigida
(tarefa europeia com residência de dados na UE; tarefa brasileira com LGPD) — bastava
rotear para a própria região. Reproduzindo as decisões gravadas no ambiente sequencial,
o satélite exigido estava **viável** nos cinco casos (SoC entre 74% e 97%, RAM livre).
Refazendo as chamadas com o prompt exato, a API respondeu **HTTP 503** — *"This model is
currently experiencing high demand"*.

**Causa:** os dois schedulers remotos só refaziam a tentativa em 429 (o SLM também em
500). Um 503 fazia desistir na hora, a tarefa virava DROP, e o CSV registrava isso como
se fosse decisão do modelo. Três consequências:

1. os 5 "erros" do LLM nas sementes 3–4 são, muito provavelmente, falhas de API;
2. explica por que essas execuções rodaram quase 2× mais rápido (514–529 s contra
   870–960 s): as chamadas falhavam em vez de esperar resposta;
3. **numa regra de descarte, uma falha de API produz um DROP "conforme" por acidente.**
   Então até o 5/5 de descartes do LLM e do SLM nas sementes 0–2 — os números do
   camera-ready — pode conter falhas que acertaram sem querer. Não dá para saber a partir
   dos CSVs antigos.

**Correção** (commit `94b5131`): retentativa em 429/500/502/503/504 e erro de conexão
(até 6 tentativas, backoff limitado a 120 s), e uma coluna nova `decision_source` no CSV
(`model` / `prefilter` / `api_failure` / `parse_failure`). Falha de infraestrutura
deixa de se passar por escolha do modelo. A partir daqui é possível reportar duas
métricas distintas: conformidade **do modelo** (só decisões `model`) e conformidade
**operacional** (tudo, porque em operação uma falha de API também deixa a tarefa sem
atendimento).

**Nova campanha, instrumentada** (`logs/generalization/heldout_lm_v2`, sementes 0–9,
os dados antigos preservados para comparação): o SLM já está rodando; o LLM aguarda a API
sair do 503, com um vigia que testa a cada 10 min. Ao fim, as sementes 0–2 dirão se os
números do camera-ready se sustentam sem a ambiguidade.

**Achado lateral:** o pré-filtro do LLM decide se chama a API olhando só o satélite da
**própria região da tarefa**. Numa tarefa GDPR vinda do Brasil, é a bateria do satélite
brasileiro — irrelevante para a regra — que decide. Em λ=4 quase nunca dispara; a
coluna `decision_source` vai medir quanto.

### IX.9 A decisão sequencial ajuda — mas só sob pressão forte de bateria

Dezesseis agentes: γ ∈ {0; 0,99} × w ∈ {0; 0,05; 0,1; 0,15; 0,25; 0,4; 0,6; 1}, todos
treinados em λ=12 nominal. Avaliação pareada nas sementes de teste 0–19
(`scripts/frontier_seq.py`).

**Método.** Um ponto com ACR maior e throughput menor não prova nada sozinho — pode ser
outro ponto da mesma curva. A comparação justa é cada agente γ=0,99 contra o **envelope
convexo superior** da família γ=0 (+ ORACLE) no mesmo throughput. Usa-se o envelope, e não
só os pontos, porque misturar políticas aleatoriamente também está disponível para quem
usa a família míope. Incerteza por **bootstrap por semente** (2.000 reamostragens): a
semente é a unidade independente, e cada reamostragem recalcula agentes, envelope e ganho.

**λ = 12 nominal (a carga de treino, bateria sob pressão forte):**

| agente γ=0,99 | throughput | ACR | ganho sobre o envelope γ=0 | IC 95% |
|---|---|---|---|---|
| w=0,05 | 59,2% | 99,7% | **+4,4 pp** | [+2,4; +6,3] |
| w=0,1 | 69,8% | 98,4% | **+4,7 pp** | [+1,7; +7,6] |
| w=0,15 | 70,9% | 96,5% | **+3,0 pp** | [+0,6; +5,3] |
| w=0,25 | 72,7% | 93,9% | +1,0 pp | [−1,1; +3,1] |
| w=0,6 | 73,8% | 90,1% | +0,3 pp | [−1,9; +2,1] |
| w=0,4 | 74,6% | 85,3% | −0,8 pp | [−2,8; +0,5] |
| w=1 | 74,6% | 85,6% | −0,3 pp | [−1,6; +0,4] |

Figura: `docs/fig_fronteira_l12.png`.

**Leitura.** Na faixa de 59–71% de throughput, o agente com visão de futuro obtém de 3 a
5 pp a mais de conformidade do que qualquer mistura de agentes míopes, e os três
intervalos ficam inteiramente acima de zero. Perto do throughput máximo (≥ 72,7%) as duas
famílias são indistinguíveis. Dito de outro modo: para quem aceita abrir mão de uma fração
do tráfego comum, planejar torna a conformidade muito mais barata. No segmento do envelope
míope entre 33% e 72% de throughput, cada ponto de ACR custa cerca de 6,4 pp de
throughput; entre os agentes γ=0,99 de 69,8% e 72,7%, custa cerca de 0,6 pp. Essa razão é
descritiva; a afirmação apoiada pelo intervalo é o ganho da tabela.

**λ = 8 nominal (pressão moderada, fora da carga de treino): sem vantagem.** Ganhos entre
−1,7 e +1,6 pp, intervalos cobrindo zero (um ponto, w=0,4, marginalmente abaixo:
[−3,1; −0,2]). O envelope míope já está em 98,3% de ACR a 89% de throughput — sob pressão
moderada sobra pouco para planejar. Figura: `docs/fig_fronteira_l8.png`.

**O que isto responde ao Revisor 3.** O pedido era um cenário "com restrições de recurso
mais apertadas ou objetivos de longo prazo em que a decisão sequencial importe". A
resposta tem três partes, todas com dado:
1. neste modelo a RAM não chega a limitar — a bateria esgota antes (IX.2);
2. com a bateria sob pressão forte, a decisão sequencial melhora a conformidade de forma
   estatisticamente sustentada, e a política aprendida é interpretável: reserva graduada de
   bateria, começando bem acima do piso (IX.7);
3. o benefício depende do regime e desaparece sob pressão moderada — o γ=0 do artigo não
   estava errado no regime do artigo, que é de pressão leve.

**Limites (antes da replicação abaixo).** Uma semente de treino por configuração (a variância entre treinos não está
medida); agentes treinados só em λ=12; o valor de w é escolha de pesquisa.

#### IX.9.1 Replicação em três sementes de treino (23/09)

Os 16 agentes foram retreinados com as sementes 43 e 44 (`models/seq/*_s43`, `*_s44`;
mesmos hiperparâmetros, mesmas 300 mil interações). A avaliação continua nas mesmas
sementes de teste 0–19.

**Dois testes, do mais brando ao mais exigente:**
- *envelope da mesma semente:* cada agente γ=0,99 contra o envelope dos γ=0 treinados
  com a **mesma** semente (é o teste da IX.9);
- *envelope conjunto (o teste que vale):* cada agente γ=0,99 contra o envelope dos γ=0 das
  **três** sementes juntas (24 agentes míopes + ORACLE). O teste favorece a família míope
  de propósito, porque ela passa a ter o melhor de três treinos contra um treino só. Um
  ganho que sobrevive a isso não é sorte de treino.

Ganho sobre o envelope conjunto, λ=12 nominal, bootstrap por semente de teste ×2.000
(* = IC 95% inteiramente acima de zero; † = inteiramente abaixo):

| w | semente 42 | semente 43 | semente 44 |
|---|---|---|---|
| 0,05 | **+3,1** [+1,6; +4,5] * (thr 59%) | **+3,0** [+0,1; +6,0] * (thr 70%) | **+4,1** [+2,0; +6,1] * (thr 67%) |
| 0,1 | **+4,5** [+1,6; +7,2] * | **+5,3** [+3,0; +7,6] * | −0,3 [−2,4; +1,2] |
| 0,15 | **+2,8** [+0,4; +5,1] * | −0,2 [−3,2; +2,3] | −0,3 [−2,1; +1,2] |
| 0,25 | +1,0 [−1,1; +3,1] | −0,2 [−2,0; +1,6] | −1,2 [−2,7; +0,6] |
| 0,4 | −1,2 [−3,4; +0,2] | −0,7 [−2,8; +1,4] | **+2,6** [+0,7; +4,3] * |
| 0,6 | −0,1 [−2,5; +2,0] | +0,3 [−0,9; +1,2] | −1,8 [−4,0; −0,6] † |
| 1 | −0,7 [−1,9; +0,2] | −2,2 [−4,9; +0,0] | −2,3 [−4,3; −0,9] † |

(w=0 fica fora da comparação: esses agentes descartam quase tudo, 10–22% de throughput,
e o ganho é zero por construção. Pelo envelope da mesma semente, os resultados das
sementes 43 e 44 estão em `docs/frontier_l12_s43.txt` e `_s44.txt`; o conjunto, em
`docs/frontier_l12_s4?_pooled.txt`.)

**O que se sustenta.**
1. **w = 0,05 dá ganho significativo nas três sementes de treino, mesmo contra o envelope
   conjunto:** +3,0 a +4,1 pp de conformidade. Essa é a afirmação que a versão estendida
   pode fazer. Não é o melhor de oito agentes escolhido depois: é o mesmo w nas três.
2. Em toda semente existe pelo menos um agente com ganho de +4 a +5 pp na faixa de 59–70%
   de throughput (o máximo por semente é +4,5, +5,3 e +4,1).
3. Perto do throughput máximo (w ≥ 0,25) o planejamento não ajuda. Na semente 44 chega a
   **atrapalhar** de forma significativa (w=0,6 e w=1, −1,8 e −2,3 pp).

**O que não se sustenta, e muda a IX.9.** O ganho em w=0,1 e w=0,15 dependia da semente
de treino (2 de 3 e 1 de 3). O mesmo w também leva a throughputs diferentes conforme o
treino: com w=0,05, 59%, 70% e 67%. Logo, w **não** controla a posição na fronteira de
forma reprodutível. A frase da IX.9, "de 3 a 5 pp na faixa de 59–71%", vale como
"existe um agente γ=0,99 que faz isso em cada treino", e não como "γ=0,99 com w entre
0,05 e 0,15".

**Formulação recomendada para a versão estendida:** "Sob pressão forte de bateria, o
agente com γ = 0,99 e w = 0,05 alcança de 3 a 4 pp a mais de conformidade do que o melhor
envelope de 24 agentes míopes com o mesmo throughput, em três sementes de treino
independentes. Perto do throughput máximo o benefício desaparece."

**Limites que restam:** três sementes de treino é o mínimo para falar de variância, não uma
estimativa dela; os agentes continuam treinados só em λ=12; o bootstrap reamostra
sementes de teste, não de treino. Por isso a robustez vem da replicação (3 de 3), e não de
um intervalo que inclua a variância de treino.

### IX.9.2 SLM instrumentado — parcial, sementes 0–3 (23/09, 14h50)

A campanha v2 grava `decision_source`. Com isso dá para separar a decisão do modelo das
falhas, que nos dados antigos apareciam como DROP (IX.8). Resultados das sementes 0–3,
com o mesmo cenário das execuções antigas (sequência de tarefas idêntica, conferida
semente a semente):

| origem da decisão | tarefas sem regra | tarefas com regra |
|---|---|---|
| `model` | 227 | 20 |
| `parse_failure` | 0 | **5** |
| `api_failure` | 5 | 0 |

**Reprodutibilidade:** 24 das 25 tarefas com regra têm o mesmo desfecho de conformidade
que na execução antiga. A exceção é uma tarefa HIPAA da semente 2: antes DROP (não
conforme), agora roteada para o satélite certo. Mesmo com temperatura 0, a API não é
bit-a-bit determinística. O efeito é pequeno, mas existe.

**Achado:** todas as 5 falhas de parse caíram em tarefas **com regra** (5 de 25), e
nenhuma nas 232 sem regra. Cada falha de parse significa 6 tentativas sem JSON aproveitável,
e com temperatura 0 repetir não ajuda. Hipótese a testar: o raciocínio da Gemma consome o
orçamento de `maxOutputTokens = 1024` nas tarefas que exigem ler a regra, e a resposta vem
cortada. O teste é gravar o `finishReason` da API (`MAX_TOKENS` confirmaria). **Isso não foi
alterado agora**, porque o módulo é recarregado a cada semente e mudá-lo no meio da
campanha misturaria duas configurações.

**Consequência para a leitura do SLM:**

| métrica, sementes 0–3 | valor |
|---|---|
| ACR operacional (falha conta como não conforme) | 14/25 = 56% |
| ACR só com decisões do modelo | 14/20 = 70% |
| execução antiga, mesmas tarefas | 13/25 = 52% |

Parte do que a IX.5 chamou de "reflexo de descarte" do SLM é falha de formato, não escolha
do modelo. As duas leituras devem ir para a versão estendida. A operacional mede o sistema
como ele é; a outra mede a capacidade do modelo. Esperar as sementes 4–9 antes de
concluir: com n = 25 os intervalos são largos.

#### IX.9.2b SLM completo, sementes 0–9 (23/09, 18h30)

| origem da decisão | tarefas sem regra | tarefas com regra |
|---|---|---|
| `model` | 599 | 44 |
| `parse_failure` | 1 | **12** |
| `api_failure` | 5 | 0 |

As falhas de parse se concentram nas tarefas com regra: 12 de 56, contra 1 de 605 nas
tarefas sem regra. Pelo teste exato de Fisher unilateral, p = 5·10⁻¹³. Não é ruído de API.

**Por tipo de regra** (todas inéditas):

| regra | tarefas | falhas de parse | conformes (decisões do modelo) |
|---|---|---|---|
| falha de hardware: bateria | 7 | 0 | 7 |
| falha de hardware: sensor térmico | 7 | 0 | 7 |
| HIPAA (EUA) | 9 | 3 | 2 |
| ITAR (EUA) | 12 | 5 | 5 |
| LGPD (Brasil) | 7 | 0 | 3 |
| residência de dados (UE) | 14 | 4 | 6 |

| ACR do SLM, 10 sementes | valor | IC 95% Wilson |
|---|---|---|
| operacional (falha conta como não conforme) | 30/56 = 53,6% | [40,7; 66,0] |
| só decisões do modelo | 30/44 = 68,2% | [53,4; 80,0] |

Nas tarefas que também existem na execução antiga, 31 de 33 têm o mesmo desfecho.

**Leitura.** As regras de falha de hardware, cuja ação certa é descartar, o SLM acerta
todas (14/14). O problema está todo nas regras de **região**, que exigem rotear a tarefa
para outra jurisdição: são 16 acertos em 42 tarefas, e todas as falhas de formato caem aí.
Isso refina a IX.5. O SLM não tem um "reflexo de descarte" genérico: ele erra, ou não
chega a responder, justamente nas regras que exigem raciocinar sobre regiões. A hipótese
do `maxOutputTokens` (IX.9.2) fica mais plausível, porque essas são as regras que pedem mais
raciocínio. O teste, gravando o `finishReason`, pode ser feito agora que o SLM terminou.

**LLM, parcial (sementes 0–1 de 10):** 11 de 11 decisões do modelo conformes nas tarefas
com regra, e uma tarefa com regra perdida por `api_failure`. A API está lenta (7–10 s por
requisição trivial; a semente 1 levou 6.986 s), por isso a campanha foi reiniciada às
18h20 com `--timeout 21600`, mantendo as sementes 0 e 1. A frase do camera-ready ("o LLM
mantém 20/20", sementes 0–2) é sobre decisões do modelo. Até aqui ela se sustenta nessa
leitura; a semente 2 decide.

#### IX.9.3 Por que o SLM falha: orçamento de tokens, confirmado (24/09)

**Método.** Para testar a hipótese da IX.9.2, cada falha foi repetida com o prompt
**exato** que o modelo recebeu (`scripts/replay_slm_failures.py`). O prompt inclui o
estado da frota, que depende de todas as decisões anteriores, então o script tem duas fases:

1. **Reprodução, sem API.** Um SLM de reprodução devolve as decisões gravadas no CSV da
   campanha, com a mesma temporização assíncrona. O CSV gerado bateu tarefa a tarefa com
   o gravado nas 8 sementes que tinham falhas (colunas conferidas: tarefa, instante,
   região, anomalia, satélite, sucesso, conformidade, origem da decisão). Logo, a frota em
   cada falha é a mesma que o modelo viu. Os 13 prompts estão em
   `logs/diagnostics/slm_parse_failures/prompts_seed*.json`.
2. **Sondagem, com API.** Cada prompt foi reenviado com a configuração original
   (temperatura 0, `maxOutputTokens = 1024`) e com 4.096, gravando `finishReason` e o uso
   de tokens (`probe_results.json`).

**Resultado.**

| orçamento | `finishReason` | resposta aproveitável |
|---|---|---|
| 1.024 (o da campanha) | **MAX_TOKENS em 13/13**, sempre com 1.021 tokens de raciocínio | 0/13 |
| 4.096 | STOP em 11/13; MAX_TOKENS em 2/13 | 11/13 |

**A hipótese se confirma.** Nas regras de região, a Gemma gasta todo o orçamento
raciocinando e não chega a escrever a resposta. Com 4.096 tokens, as que terminam usam
entre 1.399 e 4.018 tokens de raciocínio. A falha também é reproduzível: com temperatura 0,
os 13 prompts falharam de novo, do mesmo jeito.

**Achado secundário 1: o parser lê o raciocínio.** Nos 2 casos que estouram até 4.096,
o parser devolve uma ação mesmo assim. Quando não há parte final, o
`slm_scheduler.py` concatena o texto do raciocínio, e o regex acha um `"action"` escrito
no meio dele. Esses casos foram contados como falha aqui. Na campanha, com 1.024, isso
não aconteceu (as 13 falharam no parse), mas a leniência existe e precisa ser fechada
antes de qualquer nova campanha do SLM.

**Achado secundário 2: respostas contraditórias e assimetria de interface.** Das 11
respostas completas, 5 dizem `action = process` com `target_region` apontando para
**outra** região. O prompt define `process` como processar na região da própria tarefa,
então a resposta se contradiz. O `_resolve_action` do SLM obedece à ação e ignora o alvo.
Conformidade das 12 tarefas com regra, com 4.096 tokens:

| leitura | conformes |
|---|---|
| como o código resolve hoje (a ação manda) | 6/12 |
| se o alvo mandasse | 11/12 |

Isso **não** autoriza trocar a regra agora: escolher o resolvedor depois de ver o resultado
seria ajustar o método ao dado. Mas expõe uma assimetria entre os motores remotos. O
LLM responde direto com `satellite_id`; o SLM responde com ação + região, que o código
traduz em satélite. Um erro de rótulo só pode prejudicar o SLM. O orçamento também é
diferente: 1.024 tokens no SLM, com raciocínio, e 128 no LLM, sem raciocínio.

**Consequência para o artigo e para a versão estendida.**
- O desempenho do SLM nas regras de região (16/42, IX.9.2b) mede **a configuração**,
  e não só o modelo: parte dos erros vem do orçamento de tokens, e parte da interface
  de saída. Isso refina de novo a IX.5: o "reflexo de descarte" era, em boa parte,
  resposta cortada.
- Para a versão estendida, a comparação justa pede **a mesma interface de saída** para
  os dois modelos (ambos com `satellite_id`, ou ambos com ação + região) e um orçamento
  que não corte o raciocínio, **fixados antes** de uma nova campanha e aplicados aos dois.
  Decisão a tomar com o orientador.
- **O camera-ready é afetado (conferido em 24/09).** A Tabela IV traz SLM = 50% nas
  regras inéditas (sementes 0–2), e o texto interpreta esse número como "a discard
  reflex rather than rule reading", na discussão da tabela e de novo na conclusão
  ("the SLM fell back on a discard reflex"). **Os números continuam corretos; a
  interpretação não se sustenta mais inteira.** Nas sementes 0–2 da campanha v2 (mesmas
  tarefas; o SLM acerta 6/15 regras de região, contra 5/15 no registro do artigo),
  3 dos 9 erros são respostas cortadas pelo orçamento de tokens (IX.9.2), não
  escolhas de descarte. Nas 10 sementes são 12 de 26 erros em regras de região. Proposta para o camera-ready, a confirmar com o autor e sem
  mudar o número de páginas: trocar "discard reflex" por uma frase que atribua o
  resultado à configuração. Por exemplo: "with a 1,024-token output budget the SLM
  often exhausts it reasoning about region constraints, so this figure reflects the
  configuration as much as the model." A mudança entra pela worktree da branch do
  camera-ready, e o verificador de números precisa continuar 58/58.

**Limites.** São 13 tarefas e uma sondagem por orçamento. A conformidade na leitura "se o
alvo mandasse" é contrafactual e foi calculada com a frota do prompt, em que a bateria está
arredondada a inteiro; nenhuma bateria estava a menos de 1 ponto do piso de 20% nesses
casos com regra.

### IX.10 Estado ao encerrar a sessão de 23/09 e como retomar

> **ESTADO AO DESLIGAR O COMPUTADOR — 23/09/2026, 19h. Comece por aqui.**
>
> **Nada está rodando.** As campanhas foram paradas de propósito para desligar a máquina.
>
> | frente | estado | dados |
> |---|---|---|
> | SLM instrumentado, regras inéditas, sementes 0–9 | **concluído** (IX.9.2b) | `logs/generalization/heldout_lm_v2/seed*/mec_metrics_SLM.csv` |
> | LLM instrumentado, regras inéditas | **sementes 0–1 concluídas; 2–9 pendentes** (a semente 2 foi interrompida aos 41 min e será refeita do zero) | `.../seed0`, `.../seed1` |
> | réplicas de treino (item 3.3) | **concluído** (IX.9.1) | `models/seq/*_s43`, `*_s44` |
>
> Os logs de progresso ficam fora do repositório: `~/cosmicbeats-slm-v2.log` e `~/cosmicbeats-llm-v2.log`.
>
> **1. Retomar o LLM**. O runner pula as sementes 0 e 1 porque os arquivos delas existem.
> A API está lenta (7–10 s por requisição trivial; a semente 1 levou 6.986 s), por isso o
> limite é de 6 h por semente. Com o notebook na tomada:
>
> ```bash
> cd /media/octaviob/HD/Projetos/CosmicBeats-Simulator
> setsid nohup systemd-inhibit --what=sleep:idle --why="campanha LLM v2" \
>   venv/bin/python -u scripts/run_experiments.py --seeds 0-9 --engines LLM \
>   --rule-split heldout --outdir logs/generalization/heldout_lm_v2 \
>   --max-api-runs 10 --timeout 21600 >> ~/cosmicbeats-llm-v2.log 2>&1 < /dev/null & disown
> tail -f ~/cosmicbeats-llm-v2.log      # acompanhar
> ```
>
> Antes, teste a API. Se ela voltar ao HTTP 503, espere. O vigia antigo ficava no
> scratchpad da sessão e **se perde ao desligar**, então é preciso repetir o teste à mão ou
> recriar o vigia.
>
> **2. Decisão do camera-ready (prazo 30/09).** Quando a semente 2 do LLM terminar, as
> sementes 0–2 permitem decidir sobre a frase "o LLM mantém 20/20". Contar só
> `decision_source = model`. Até aqui: 11/11 conformes (IX.9.2b). Se a semente 2 não sair a
> tempo, suavizar a frase por precaução.
>
> **Atualização de 24/09:** o LLM está sendo vigiado por `scripts/wait_and_run_llm.sh`
> (versionado; sobrevive a desligar a máquina). O item 3 abaixo foi **feito**: veja a IX.9.3.
> Ele gerou uma pendência nova para o camera-ready, a frase "discard reflex" do SLM, que
> depende da decisão do autor.
>
> **3. ~~Pendência aberta: o teste do `finishReason` do SLM.~~ Feito (IX.9.3).** Registrar o `finishReason` da
> API no `slm_scheduler.py` e reexecutar só as 13 tarefas com `parse_failure`, para testar se
> o `maxOutputTokens = 1024` corta a resposta nas regras de região. **Fazer só com a
> campanha do LLM parada ou concluída**, e conferir antes se o `main.py` importa o módulo do
> SLM mesmo quando o motor é o LLM.
>
> **4. Depois:** tabela pareada dos seis motores nas regras inéditas (duas leituras:
> operacional e só decisões do modelo); agentes sequenciais em λ=8; corpus de regras
> inéditas maior; RAPL (baixa prioridade).
>
> **Para a reunião de sexta:** IX.9.1 (só w=0,05 replica), IX.9.2b (SLM falha nas regras
> de região; 53,6% operacional e 68,2% só-modelo), e as decisões listadas no fim desta seção.
>
> *Não versionado, e de propósito:* `logs/generalization/heldout_lm/`, a campanha antiga sem
> instrumentação (sementes 3–6 e o manifesto), fica só para comparação.
>
> O texto abaixo é o registro das 12h45 e foi mantido como histórico.

**Rodando em segundo plano (destacado da sessão; notebook precisa ficar na tomada):**

| frente | saída | log |
|---|---|---|
| SLM instrumentado, regras inéditas, sementes 0–9 | `logs/generalization/heldout_lm_v2/` | `~/cosmicbeats-slm-v2.log` |
| Vigia do LLM: testa a API a cada 10 min e roda as sementes 0–9 quando sair do 503 | idem | `~/cosmicbeats-llm-v2.log` |
| ~~32 réplicas de treino~~ — **concluídas em 23/09; resultado na IX.9.1** | `models/seq/*_s43`, `*_s44` | — |

**Atualização de 23/09, 12h45.** A semente 0 do SLM estourou o limite de 3.600 s por
execução do runner e foi descartada. A API da Gemma respondia 200, mas levava ~4 s até
para um prompt trivial; o SLM antigo levava ~16 min por semente. É lentidão externa,
não travamento. Latência não altera a decisão; ela só é registrada. Por isso a campanha
foi reiniciada com `--timeout 10800` (3 h). O vigia do LLM foi editado para usar o mesmo
limite. O runner decide o que está concluído pelos arquivos de saída, então nada que
tinha terminado foi perdido; perderam-se só os 35 min em andamento da semente 1.

Se algo cair, retomar sem refazer nada do que completou (**usar `--timeout 10800` no SLM e `--timeout 21600` no LLM**):

```bash
venv/bin/python scripts/run_experiments.py --seeds 0-9 --engines SLM --rule-split heldout --outdir logs/generalization/heldout_lm_v2 --timeout 10800
venv/bin/python scripts/run_experiments.py --seeds 0-9 --engines LLM --rule-split heldout --outdir logs/generalization/heldout_lm_v2 --timeout 21600
```

**Próximos passos, em ordem:**

1. **Quando o v2 terminar:** tabela pareada dos seis motores nas regras inéditas usando só
   decisões com `decision_source = model`, e também a versão operacional (com as falhas).
   Comparar as sementes 0–2 do v2 com as antigas. Decide se a frase "o LLM mantém 20/20"
   do camera-ready se sustenta. **Se a API do LLM não voltar até quinta à noite,
   suavizar essa frase no camera-ready por precaução** — o resultado central (o LLM é o
   único que lê as regras de região) se sustenta de qualquer modo.
2. ~~Réplicas de treino~~ **feito (IX.9.1):** o ganho em w=0,05 se confirma em 3 de 3
   sementes contra o envelope conjunto; os ganhos em w=0,1 e w=0,15 dependem do treino.
   Para a reunião: levar a tabela da IX.9.1 e a formulação recomendada.
   *Depois que a campanha do SLM terminar:* gravar `finishReason` no `slm_scheduler.py` e
   repetir as tarefas com `parse_failure` para testar a hipótese da IX.9.2.
3. Para a versão estendida: treinar os agentes sequenciais também em λ=8; decidir com o
   orientador o valor de w; ampliar o corpus de regras inéditas (6 é pouco); item 3.2
   (energia medida por RAPL, exige sudo — perdeu prioridade pela VII.2).

**Decisões que dependem do orientador (reunião de sexta):** a tese reformulada (VIII.1,
IX.5); o posicionamento do SLM, que nas regras inéditas se comporta como reflexo de
descarte; quanto do item 3.3 entra na versão estendida; e o valor de w (a IX.9.1 mostra
que só w=0,05 replica, o que favorece fixar esse valor).

**Estado das branches:** `camera-ready/wpmc2026-reviewer-response` contém a versão para
submissão (6 páginas, verificador 58/58, PDF em `paper/revised-camera-ready.pdf`, prazo
30/09). `extended/journal-version` contém tudo isso mais o trabalho estendido.

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
