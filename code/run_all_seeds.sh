#!/bin/bash
# run_all_seeds.sh
# Corre todas as seeds em falta, sequencialmente.
# Cada run demora várias horas — deixar o computador ligado.
#
# Uso:
#   chmod +x run_all_seeds.sh   (só é preciso fazer uma vez)
#   ./run_all_seeds.sh

echo "======================================"
echo " NIAI - Mario Evolution Runner"
echo " Seeds a correr: 1, 2, 3, 4, 5, 11"
echo "======================================"

for SEED in 1 2 3 4 5 11; do
    echo ""
    echo "[$(date '+%H:%M:%S')] A iniciar seed $SEED..."
    python evolution.py $SEED
    echo "[$(date '+%H:%M:%S')] Seed $SEED concluída."
done

echo ""
echo "======================================"
echo " Todas as seeds concluídas!"
echo "======================================"
