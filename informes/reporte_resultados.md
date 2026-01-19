# Reporte de resultados — Baselines (1X2)

Este reporte evalúa baselines en validación y test. Métrica principal: **logloss**.

Clases: `L` (local), `E` (empate), `V` (visitante)

## Baseline 1 — Siempre Local

| Split | Logloss | Accuracy | Balanced Acc | Macro F1 |
|---|---:|---:|---:|---:|
| Val | 6.8473 | 0.5044 | 0.3333 | 0.2235 |
| Test | 6.7211 | 0.5135 | 0.3333 | 0.2262 |

**Matriz de confusión (Val)**

| real\pred | E | L | V |
|---|---:|---:|---:|
| E | 0 | 74 | 0 |
| L | 0 | 173 | 0 |
| V | 0 | 96 | 0 |

**Matriz de confusión (Test)**

| real\pred | E | L | V |
|---|---:|---:|---:|
| E | 0 | 77 | 0 |
| L | 0 | 152 | 0 |
| V | 0 | 67 | 0 |

## Baseline 2 — Frecuencias por fase (train → val/test)

| Split | Logloss | Accuracy | Balanced Acc | Macro F1 |
|---|---:|---:|---:|---:|
| Val | 1.0417 | 0.5044 | 0.3333 | 0.2235 |
| Test | 1.0391 | 0.5135 | 0.3333 | 0.2262 |

**Matriz de confusión (Val)**

| real\pred | E | L | V |
|---|---:|---:|---:|
| E | 0 | 74 | 0 |
| L | 0 | 173 | 0 |
| V | 0 | 96 | 0 |

**Matriz de confusión (Test)**

| real\pred | E | L | V |
|---|---:|---:|---:|
| E | 0 | 77 | 0 |
| L | 0 | 152 | 0 |
| V | 0 | 67 | 0 |

## Baseline 3 — Elo simple (elos aprendidos en train)

| Split | Logloss | Accuracy | Balanced Acc | Macro F1 |
|---|---:|---:|---:|---:|
| Val | 0.9908 | 0.5364 | 0.4225 | 0.3747 |
| Test | 0.9994 | 0.5507 | 0.4437 | 0.3794 |

**Matriz de confusión (Val)**

| real\pred | E | L | V |
|---|---:|---:|---:|
| E | 0 | 54 | 20 |
| L | 0 | 140 | 33 |
| V | 0 | 52 | 44 |

**Matriz de confusión (Test)**

| real\pred | E | L | V |
|---|---:|---:|---:|
| E | 0 | 47 | 30 |
| L | 0 | 132 | 20 |
| V | 0 | 36 | 31 |
