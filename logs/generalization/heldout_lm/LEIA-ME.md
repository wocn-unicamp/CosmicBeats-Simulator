# Campanha antiga das regras inéditas (SLM e LLM, protocolo v1)

Dados de 22–23/09/2026, gerados **antes** da instrumentação de `decision_source`
(docs IX.8). Neles, uma falha de API aparece como descarte, indistinguível de uma decisão
do modelo. O SLM ainda usava o protocolo v1: prompt próprio, formato ação + região e
1.024 tokens de saída.

Uso atual:
- **Sementes 0–2 do LLM:** foram a base do "20/20" do camera-ready até 27/09. O replay
  direcionado confirmou que as 20 decisões com regra foram do modelo (IX.9.4).
- **Sementes 3–5 e o SLM:** servem só de histórico e comparação.

A versão final do artigo usa `heldout_slm_v3/` (SLM) e `heldout_lm_v2/` (LLM), com 10
sementes (docs Parte X).
