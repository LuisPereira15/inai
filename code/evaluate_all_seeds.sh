#!/bin/bash
# evaluate_all_seeds.sh
# Avalia os melhores agentes gerados pelo run_all_seeds.sh
# Só correr DEPOIS do run_all_seeds.sh ter terminado.
#
# Uso:
#   chmod +x evaluate_all_seeds.sh
#   ./evaluate_all_seeds.sh

echo "======================================"
echo " NIAI - Mario Evaluation Runner"
echo "======================================"

# Procura automaticamente os melhores .pkl gerados para cada seed
for SEED in 1 2 3 4 5 11; do
    # Encontra o ficheiro .pkl correspondente à seed (pode ter qualquer fitness no nome)
    PKL_FILE=$(ls data/mlp_best_agents/es_seed_${SEED}_*.pkl 2>/dev/null | head -1)

    if [ -z "$PKL_FILE" ]; then
        echo "[$(date '+%H:%M:%S')] Seed $SEED: ficheiro .pkl não encontrado, a saltar..."
    else
        echo ""
        echo "[$(date '+%H:%M:%S')] A avaliar seed $SEED: $PKL_FILE"
        python evaluate_best_agent.py "$PKL_FILE"
        echo "[$(date '+%H:%M:%S')] Seed $SEED concluída."
    fi
done

echo ""
echo "======================================"
echo " Avaliação completa!"
echo " Resultados em: data/results/"
echo "======================================"
