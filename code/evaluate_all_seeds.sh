#!/bin/bash
# evaluate_all_seeds_v4.sh
# Avalia os melhores agentes v4 (150 gerações, fitness corrigida)
# Requer servidor Java a correr (localhost:3000 → bash /server/run-server.sh)
#
# Uso:
#   chmod +x evaluate_all_seeds_v4.sh
#   ./evaluate_all_seeds_v4.sh

echo "======================================"
echo " NIAI - Mario Evaluation Runner v4"
echo " Seeds: 1, 2, 3, 4, 5, 11"
echo "======================================"
echo "A correr em: $(pwd)"
echo ""

declare -A BEST_PKLS
BEST_PKLS[1]="data/mlp_best_agents/es_seed_1_9046.918.pkl"
BEST_PKLS[2]="data/mlp_best_agents/es_seed_2_10910.738.pkl"
BEST_PKLS[3]="data/mlp_best_agents/es_seed_3_8870.498.pkl"
BEST_PKLS[4]="data/mlp_best_agents/es_seed_4_8423.338.pkl"
BEST_PKLS[5]="data/mlp_best_agents/es_seed_5_8826.070.pkl"
BEST_PKLS[11]="data/mlp_best_agents/es_seed_11_8825.820.pkl"

for SEED in 1 2 3 4 5 11; do
    PKL="${BEST_PKLS[$SEED]}"
    if [ ! -f "$PKL" ]; then
        echo "[$(date '+%H:%M:%S')] Seed $SEED: ficheiro não encontrado: $PKL"
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