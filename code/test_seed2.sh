#!/bin/bash
# test_seed2_v5.sh
# Teste rápido com seed 2, 50 gerações, janela 13x13
# Objetivo: verificar se há wins antes de lançar todas as seeds

echo "======================================"
echo " NIAI - Teste Rápido v5"
echo " Seed 2 | 50 gerações | janela 13x13"
echo "======================================"

# Correr evolução seed 2
echo "[$(date '+%H:%M:%S')] A iniciar evolução seed 2..."
python evolution.py 2
echo "[$(date '+%H:%M:%S')] Evolução concluída."

# Encontrar melhor pkl
BEST_PKL=$(ls data/mlp_best_agents/es_seed_2_*.pkl 2>/dev/null | \
    awk -F'_' '{print $NF, $0}' | sort -n | tail -1 | awk '{print $2}')

if [ -z "$BEST_PKL" ]; then
    echo "ERRO: nenhum .pkl encontrado!"
    exit 1
fi

echo ""
echo "[$(date '+%H:%M:%S')] A avaliar: $BEST_PKL"
python evaluate_best_agent.py "$BEST_PKL"
echo "[$(date '+%H:%M:%S')] Avaliação concluída."
