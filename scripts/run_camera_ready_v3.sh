#!/bin/bash
# Campanhas do camera-ready final (docs Parte X). Protocolo v3 do SLM + refazer as
# sementes do LLM v2 com mais de 5% de api_failure. Registrado ANTES dos resultados.
#
# Uso (notebook na tomada; cada frente em seu processo, porque sao modelos diferentes):
#   setsid nohup systemd-inhibit --what=sleep:idle --why="SLM v3" \
#     bash scripts/run_camera_ready_v3.sh slm >> ~/cosmicbeats-slm-v3.log 2>&1 < /dev/null & disown
#   setsid nohup systemd-inhibit --what=sleep:idle --why="LLM v2 refazer" \
#     bash scripts/run_camera_ready_v3.sh llm >> ~/cosmicbeats-llm-v2-redo.log 2>&1 < /dev/null & disown
# O runner pula sementes ja concluidas: reexecutar e seguro.

cd "$(dirname "$0")/.." || exit 1
case "$1" in
  slm)
    # 1. Corrida canonica (semente 0, regras conhecidas): Tabela III e a coluna
    #    de regras conhecidas da Tabela IV. O dado v1 esta em logs/archive/slm_protocol_v1/.
    if [ ! -f logs/mec_metrics_SLM.csv ]; then
      echo "$(date '+%F %T') SLM canonica (semente 0, regras conhecidas)"
      venv/bin/python -u main.py --engine SLM --seed 0 --outdir logs
    fi
    # 2. Regras ineditas, sementes 0-9 (0-2 primeiro: e o minimo da regra de corte).
    venv/bin/python -u scripts/run_experiments.py --seeds 0-9 --engines SLM \
      --rule-split heldout --outdir logs/generalization/heldout_slm_v3 \
      --max-api-runs 10 --timeout 10800
    ;;
  llm)
    venv/bin/python -u scripts/run_experiments.py --seeds 1,4,5 --engines LLM \
      --rule-split heldout --outdir logs/generalization/heldout_lm_v2 \
      --max-api-runs 3 --timeout 21600
    ;;
  *) echo "uso: $0 slm|llm"; exit 2 ;;
esac
echo "$(date '+%F %T') $1 encerrado"
