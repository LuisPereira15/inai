# NIAI Project — Registo de Progresso e Alterações
**Repositório:** beatrizpmonteiro/inai  
**Última atualização:** Sessão 1

---

## Estado Inicial (ponto de partida do colega)

O colega tinha implementado:
- `agents/mlp_agent.py` — MLPAgent com janela 7×7, separação landscape/enemies, 102 inputs (resolve os challenges 2 e 3 do enunciado)
- `agents/code_agent.py` — CodeAgent base funcional
- `evolution.py` — EA (µ+λ) com tournament selection, BLX-α crossover, mutação gaussiana
- `tasks/move_forward.py` — Reward function Stage 1
- `tasks/hunter.py` — Reward function Stage 2
- `evaluation.py` — Pool de 5 workers paralelos com portas TCP
- `evaluate_best_agent.py` — Avaliação final com CSV de resultados
- `mario_random_search_mlp.py` — Random Search baseline

**Dados já existentes:**
- `log_seed_1.csv` — 100 gerações completas, melhor fitness = 2740.85
- `log_seed_5.csv` — apenas 12 gerações
- `log_seed_10.csv` — apenas 74 gerações
- `eval_es_seed_1_2740.850.csv` — avaliação em 4 dificuldades, 30 runs cada
- `summary_es_seed_1_2740.850.csv` — resumo: 1 win em 120 episódios, ~97% timeout

**Diagnóstico:** o agente da seed 1 tem resultados fracos (quase só timeouts). Há margem clara de melhoria na fitness function e nos hiperparâmetros.

---

## Sessão 1 — Alterações ao Código

### Ficheiro 1: `tasks/move_forward.py` (Stage 1 Fitness)

**Problema identificado:** A penalização de morte (-50) era muito baixa, pelo que o agente não aprendia a evitar morrer. O threshold de stuck (10 frames) era demasiado agressivo, penalizando saltos normais.

**Alterações feitas:**
| Parâmetro | Antes | Depois | Justificação |
|---|---|---|---|
| Penalização de morte | -50 | -500 | Torna a sobrevivência crítica para o fitness |
| Stuck threshold | 10 frames | 15 frames | Reduz falsos positivos durante saltos |
| Stuck penalty | -0.5 | -0.3 | Menos agressiva, mais tolerante |
| Tick penalty | -0.05 | -0.02 | Não penaliza tanto exploração lenta |
| Bónus de progresso máximo | não existia | +0.5 × novo_território | Incentiva explorar novo terreno |

**Ficheiro:** `tasks/move_forward.py`

---

### Ficheiro 2: `tasks/hunter.py` (Stage 2 Fitness)

**Problema identificado:** O kill reward (25) era baixo face ao forward reward (0.5 × dx), pelo que o agente podia aprender a ignorar inimigos. Penalização de morte era mínima (-50).

**Alterações feitas:**
| Parâmetro | Antes | Depois | Justificação |
|---|---|---|---|
| Kill reward | +25 por kill | +40 por kill | Foco ainda maior em combat |
| Forward weight | 0.5 × dx | 0.3 × dx | Movimento é secundário no Stage 2 |
| Penalização de morte | -50 | -300 | Mais severa sem eliminar risco |
| Combo system | não existia | +15 × (combo-1) | Incentiva kills consecutivos |

**Ficheiro:** `tasks/hunter.py`

---

### Ficheiro 3: `evolution.py` (Algoritmo Evolutivo)

**Problema identificado:** A mutação com sigma=0.1 era demasiado grosseira para refinamento nas gerações avançadas (50+). Todos os filhos deveriam sofrer mutação (MUTATION_PROB=1.0).

**Alterações feitas:**
| Parâmetro | Antes | Depois | Justificação |
|---|---|---|---|
| SIGMA_MUT | 0.1 | 0.05 | Mutação mais fina, melhor convergência |
| MUTATION_PROB | 0.9 | 1.0 | Todos os filhos sofrem mutação |
| Adaptive sigma | não existia | 0.02 após geração 50 | Refinamento nas gerações tardias |
| ETA no logging | não existia | sim | Estimar tempo restante para cada run |

**Ficheiro:** `evolution.py`

---

## Plano de Runs a Executar

### Fase 1: Runs evolutivas (Stage 1 — MoveForwardTask)
Lançar sequencialmente ou em paralelo (se houver recursos):

```bash
# Seed 2 (nova)
python evolution.py 2

# Seed 3 (nova)
python evolution.py 3

# Seed 4 (nova)
python evolution.py 4

# Seed 5 (retomar do zero com código melhorado)
python evolution.py 5

# Seed 10 (retomar do zero com código melhorado)
python evolution.py 10
```

**Objetivo:** 5 ficheiros `.pkl` com os melhores agentes + 5 CSVs de log

---

### Fase 2: Random Search Baseline

```bash
# Correr o baseline para as mesmas seeds
python mario_random_search_mlp.py 1
python mario_random_search_mlp.py 2
python mario_random_search_mlp.py 3
python mario_random_search_mlp.py 4
python mario_random_search_mlp.py 5
```

**Nota:** o random search tem um bug intencional (o professor refere no enunciado) — o param_vector nunca é atualizado entre iterações. Manter esse comportamento para servir como baseline fraco e justo para comparação.

---

### Fase 3: Avaliação dos melhores agentes

Para cada `.pkl` gerado:
```bash
python evaluate_best_agent.py data/mlp_best_agents/es_seed_X_YYYY.pkl
```

Para os agentes de random search:
```bash
python evaluate_best_agent.py data/mlp_best_agents/random_search_seed_X_YYYY.pkl
```

---

### Fase 4: Stage 2 (HunterTask)

Alterar em `evaluation.py` a linha:
```python
TASK_TO_SOLVE = MoveForwardTask
```
para:
```python
TASK_TO_SOLVE = HunterTask
```

Depois lançar pelo menos 3 seeds:
```bash
python evolution.py 1
python evolution.py 2
python evolution.py 3
```

---

## Ficheiros Modificados Nesta Sessão

| Ficheiro | Versão | Estado |
|---|---|---|
| `tasks/move_forward.py` | v2 | ✅ Pronto para substituir |
| `tasks/hunter.py` | v2 | ✅ Pronto para substituir |
| `evolution.py` | v2 | ✅ Pronto para substituir |

---

## Sessão 2 — Hyperparameter Search e Otimização

### Resultados das 6 seeds (v2)

| Seed | Best Fitness | Notas |
|---|---|---|
| 1 | 4885 | +78% vs código original |
| 2 | 6622 | Melhor resultado |
| 3 | 5618 | Bom |
| 4 | 4823 | std baixo nas últimas gens |
| 5 | 6082 | Bom |
| 11 | 499 | ❌ Colapsou — std=0 desde gen ~20 |

**Win rate = 0% em todas as seeds** — agente não termina o nível. Distância média ~300-550px de ~3000px totais.

### Hyperparameter Search

Corridas 36 configurações × 30 gerações com seed=42 fixa.

**Conclusões:**
- MU=30 é claramente melhor (média 3308 vs 1526 para MU=80)
- LAMBDA=50 supera LAMBDA=30
- TOURNAMENT_K=5 é melhor que K=3
- SIGMA=0.1 é mais consistente; SIGMA=0.05 tem melhor máximo mas mais collapsos
- MU=80 + LAMBDA=30 é catastrófico (3 configs colapsaram para 499)

**Melhor config encontrada:** MU=30, LAMBDA=50, SIGMA=0.1→0.05, K=5 → best=6060

### Ficheiros Modificados

| Ficheiro | Versão | Alterações |
|---|---|---|
| `evolution.py` | v3 | MU=30, K=5, GENS=150, SIGMA=0.1→0.05 |
| `hyperparameter_search.py` | v1 | Novo — script de grid search |

---

## Próxima Sessão

- Analisar resultados das seeds v3 (150 gerações com novos hiperparâmetros)
- Correr Random Search baseline para as mesmas seeds
- Fazer plots de convergência para o relatório
- Correr Stage 2 (HunterTask)

---

## Sessão 3 — Análise de Resultados v3 e Correções v4

### Diagnóstico dos resultados v3

| Problema | Observação | Causa |
|---|---|---|
| Win rate = 0% | Todas as seeds, todas as dificuldades | Timeout + fitness function errada |
| Distância média = 469px | De ~3000px totais (15% do nível) | Igual à v2 apesar de fitness maior |
| Fitness alto mas inútil | 6500-7950 vs 2740 original | Agente aprende a oscilar, não a avançar |
| Timeout 750 steps | 21/30 runs chegam ao limite | Matematicamente impossível completar o nível |
| Agente oscila 57-68% | Velocidade real vs teórica | dx incremental incentiva movimento repetitivo |

### 4 Correções Aplicadas (v4)

**1. move_forward.py — Fitness function**
- Removida recompensa de `dx` por step (incentivava oscilação)
- Substituída por recompensa APENAS de progresso máximo novo (+2.0 × novo_território)
- Removida stuck penalty (impedia saltos)
- Adicionada penalização de regressão (-0.5 se andar para trás)
- Adicionada recompensa vertical para saltos (+0.5 × novo_y)
- Win bonus: +1000 → **+5000**
- Tick penalty: -0.02 → **-0.01** (horizonte mais longo)

**2. evaluation.py — Timeout**
- MAX_STEPS: 750 → **2000**
- A 1.7px/step, 750 steps = ~1275px de ~3000px (impossível completar)
- Com 2000 steps há margem suficiente para completar o nível

**3. evaluate_best_agent.py — Timeout**
- MAX_STEPS: 750 → **2000** (consistente com evaluation.py)

### Ficheiros Modificados

| Ficheiro | Versão | Alterações principais |
|---|---|---|
| `tasks/move_forward.py` | v4 | dx→max_x, sem stuck, win=5000, steps=2000 |
| `evaluation.py` | v4 | MAX_STEPS=2000 |
| `evaluate_best_agent.py` | v4 | MAX_STEPS=2000 |

### Próxima Sessão
- Analisar resultados das seeds v4
- Correr Random Search baseline
- Iniciar Stage 2 (HunterTask)
- Fazer plots para o relatório
