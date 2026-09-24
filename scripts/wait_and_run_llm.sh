#!/bin/bash
# Espera a API do LLM sair do HTTP 503 e so entao retoma a campanha v2 do LLM
# nas regras ineditas. Rodar com 503 transformaria cada tarefa em api_failure
# (IX.8), e o runner contaria a semente como concluida.
#
# O runner pula as sementes cujos arquivos ja existem, entao reexecutar este
# script e seguro. Limite de 6 h por semente: com a API lenta, a semente 1 levou
# 6986 s (IX.10).
#
# Uso (notebook na tomada):
#   setsid nohup systemd-inhibit --what=sleep:idle --why="campanha LLM v2" \
#     bash scripts/wait_and_run_llm.sh >> ~/cosmicbeats-llm-v2.log 2>&1 < /dev/null & disown

cd "$(dirname "$0")/.." || exit 1
MODEL="${LLM_MODEL:-gemini-3.1-flash-lite}"

probe() {
  venv/bin/python - "$MODEL" <<'EOF' 2>/dev/null
import os, sys, warnings, requests
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
load_dotenv(".env")
url = f"https://generativelanguage.googleapis.com/v1beta/models/{sys.argv[1]}:generateContent"
r = requests.post(url, headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
                  json={"contents": [{"parts": [{"text": "ok"}]}],
                        "generationConfig": {"maxOutputTokens": 4}},
                  verify=False, timeout=30)
print(r.status_code)
EOF
}

# Exige dois 200 seguidos: um 200 isolado no meio de uma onda de 503 dispararia
# a campanha cedo demais.
ok=0
while [ "$ok" -lt 2 ]; do
  code=$(probe)
  echo "$(date '+%F %T') LLM API HTTP ${code:-erro}"
  if [ "$code" = "200" ]; then ok=$((ok + 1)); sleep 60; else ok=0; sleep 600; fi
done

echo "$(date '+%F %T') API disponivel — retomando a campanha do LLM"
venv/bin/python -u scripts/run_experiments.py --seeds 0-9 --engines LLM --rule-split heldout \
  --outdir logs/generalization/heldout_lm_v2 --max-api-runs 10 --timeout 21600
echo "$(date '+%F %T') campanha do LLM encerrada"
