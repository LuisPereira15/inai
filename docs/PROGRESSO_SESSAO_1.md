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

| Parâmetro                  | Antes        | Depois                   | Justificação                                 |
| --------------------------- | ------------ | ------------------------ | ---------------------------------------------- |
| Penalização de morte      | -50          | -500                     | Torna a sobrevivência crítica para o fitness |
| Stuck threshold             | 10 frames    | 15 frames                | Reduz falsos positivos durante saltos          |
| Stuck penalty               | -0.5         | -0.3                     | Menos agressiva, mais tolerante                |
| Tick penalty                | -0.05        | -0.02                    | Não penaliza tanto exploração lenta         |
| Bónus de progresso máximo | não existia | +0.5 × novo_território | Incentiva explorar novo terreno                |

**Ficheiro:** `tasks/move_forward.py`

---

### Ficheiro 2: `tasks/hunter.py` (Stage 2 Fitness)

**Problema identificado:** O kill reward (25) era baixo face ao forward reward (0.5 × dx), pelo que o agente podia aprender a ignorar inimigos. Penalização de morte era mínima (-50).

**Alterações feitas:**

| Parâmetro             | Antes        | Depois           | Justificação                      |
| ---------------------- | ------------ | ---------------- | ----------------------------------- |
| Kill reward            | +25 por kill | +40 por kill     | Foco ainda maior em combat          |
| Forward weight         | 0.5 × dx    | 0.3 × dx        | Movimento é secundário no Stage 2 |
| Penalização de morte | -50          | -300             | Mais severa sem eliminar risco      |
| Combo system           | não existia | +15 × (combo-1) | Incentiva kills consecutivos        |

**Ficheiro:** `tasks/hunter.py`

---

### Ficheiro 3: `evolution.py` (Algoritmo Evolutivo)

**Problema identificado:** A mutação com sigma=0.1 era demasiado grosseira para refinamento nas gerações avançadas (50+). Todos os filhos deveriam sofrer mutação (MUTATION_PROB=1.0).

**Alterações feitas:**

| Parâmetro     | Antes        | Depois                  | Justificação                            |
| -------------- | ------------ | ----------------------- | ----------------------------------------- |
| SIGMA_MUT      | 0.1          | 0.05                    | Mutação mais fina, melhor convergência |
| MUTATION_PROB  | 0.9          | 1.0                     | Todos os filhos sofrem mutação          |
| Adaptive sigma | não existia | 0.02 após geração 50 | Refinamento nas gerações tardias        |
| ETA no logging | não existia | sim                     | Estimar tempo restante para cada run      |

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

| Ficheiro                  | Versão | Estado         |
| ------------------------- | ------- | -------------- |
| `tasks/move_forward.py` | v2      | ✅ Substituido |
| `tasks/hunter.py`       | v2      | ✅ Substituido |
| `evolution.py`          | v2      | ✅ Substituido |

## Próxima Sessão

- Verificar resultados das primeiras runs
- Ajustar hiperparâmetros se necessário
- Iniciar análise de resultados e plots para o relatório
