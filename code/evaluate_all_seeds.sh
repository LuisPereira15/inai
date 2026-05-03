#!/bin/bash
# evaluate_all_seeds_v3.sh
# Avalia diretamente os melhores .pkl v3 por seed.
# Correr dentro da pasta code com servidor Java ativo.

echo "======================================"
echo " NIAI - Mario Evaluation Runner v3"
echo "======================================"

# Verificar pasta atual
echo "A correr em: $(pwd)"
echo ""

# Melhores .pkl v3 identificados manualmente
declare -A BEST_PKLS
BEST_PKLS[1]="data/mlp_best_agents/es_seed_1_6531.000.pkl"
BEST_PKLS[2]="data/mlp_best_agents/es_seed_2_7951.160.pkl"
BEST_PKLS[3]="data/mlp_best_agents/es_seed_3_6008.827.pkl"
BEST_PKLS[4]="data/mlp_best_agents/es_seed_4_5682.982.pkl"
BEST_PKLS[5]="data/mlp_best_agents/es_seed_5_7830.689.pkl"
BEST_PKLS[11]="data/mlp_best_agents/es_seed_11_5672.532.pkl"

for SEED in 1 2 3 4 5 11; do
    PKL="${BEST_PKLS[$SEED]}"

    if [ ! -f "$PKL" ]; then
        echo "[$(date '+%H:%M:%S')] Seed $SEED: ficheiro não encontrado: $PKL"
        echo "  Verifica se estás na pasta code/"
    else
        echo ""
        echo "[$(date '+%H:%M:%S')] A avaliar seed $SEED: $PKL"
        python evaluate_best_agent.py "$PKL"
        echo "[$(date '+%H:%M:%S')] Seed $SEED concluída."
    fi
done

echo ""
echo "======================================"
echo " Avaliação completa!"
echo "======================================"