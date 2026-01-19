# Reporte — Modelo ML (Regresión logística multinomial)

Features: rolling (forma GF/GC + puntos), neutralidad y fase.

| Split | Logloss | Accuracy | Balanced Acc | Macro F1 |
|---|---:|---:|---:|---:|
| Val | 1.0063 | 0.5102 | 0.3475 | 0.2636 |
| Test | 1.0051 | 0.5203 | 0.3461 | 0.2549 |

## Matriz de confusión (Val)

| real\pred | E | L | V |
|---|---:|---:|---:|
| E | 1 | 72 | 1 |
| L | 1 | 169 | 3 |
| V | 0 | 91 | 5 |

## Matriz de confusión (Test)

| real\pred | E | L | V |
|---|---:|---:|---:|
| E | 0 | 77 | 0 |
| L | 0 | 151 | 1 |
| V | 0 | 64 | 3 |
