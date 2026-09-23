'''
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
'''
import sys
import time
import random
import os
import argparse  # <-- A biblioteca mágica para terminais
from src.sim.simulator import Simulator

if __name__ == "__main__":
    
    # =================================================================
    # CONFIGURAÇÃO DO TERMINAL (CLI)
    # =================================================================
    parser = argparse.ArgumentParser(description="CosmicBeats - Simulador NTN-MEC")
    
    # Argumento 1: O Motor (com opções restritas para evitar erros de digitação)
    parser.add_argument('--engine', type=str, default="LLM", 
                        choices=["LLM", "BASELINE", "SLM", "DRL", "DRL_ONEHOT", "ORACLE"], 
                        help="Escolha o cérebro: LLM, BASELINE, SLM, DRL ou ORACLE")
    
    # Argumento 2: O arquivo de configuração JSON
    parser.add_argument('--config', type=str, default="configs/config.json", 
                        help="Caminho para o arquivo config.json")

    # Argumento 3: A semente. Todos os motores precisam receber a MESMA semente
    # para que vejam a mesma sequencia de tarefas — e o pareamento entre eles
    # e justamente o que da poder estatistico na analise.
    parser.add_argument('--seed', type=int, default=0,
                        help="Semente do gerador de numeros aleatorios (default: 0)")

    # Argumento 4: Onde gravar as metricas. Sem isso cada execucao sobrescreve
    # a anterior, o que inviabiliza rodar varias sementes.
    # Argumento 5: qual conjunto de restricoes semanticas usar. "seen" reproduz
    # o artigo; "heldout" usa restricoes que nenhum motor viu em tempo de projeto.
    parser.add_argument('--rule-split', type=str, default="seen",
                        choices=["seen", "heldout", "all"],
                        help="Conjunto de restricoes semanticas (default: seen)")

    parser.add_argument('--outdir', type=str, default="logs",
                        help="Diretorio de saida das metricas (default: logs)")

    args = parser.parse_args()

    # A semente precisa ser aplicada antes de qualquer coisa: as chegadas
    # Poisson, as regioes e as anomalias saem todas do modulo random global.
    random.seed(args.seed)

    # =================================================================
    # INJEÇÃO DA CHAVE SELETORA
    # =================================================================
    os.environ["MEC_ENGINE"] = args.engine
    os.environ["MEC_SEED"]   = str(args.seed)
    os.environ["MEC_OUTDIR"] = args.outdir
    os.environ["MEC_RULE_SPLIT"] = args.rule_split
    
    print("\n" + "="*50)
    print(f"🚀 INICIANDO COSMICBEATS")
    print(f"🧠 Motor Cognitivo : {args.engine}")
    print(f"📂 Arquivo Config  : {args.config}")
    print(f"🎲 Semente         : {args.seed}")
    print(f"💾 Saída           : {args.outdir}")
    print(f"📜 Regras          : {args.rule_split}")
    print("="*50 + "\n")

    _sim = Simulator(args.config)

    _startTime = time.perf_counter()
    _sim.execute()
    _endTime = time.perf_counter()

    print(f"\n[Simulator Info] Tempo real de execução: {_endTime-_startTime:.4f} segundos.")