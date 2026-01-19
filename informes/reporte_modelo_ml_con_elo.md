# Reporte — Modelo ML (Regresión logística multinomial)

Features: rolling (forma GF/GC + puntos), neutralidad y fase.

| Split | Logloss | Accuracy | Balanced Acc | Macro F1 |
|---|---:|---:|---:|---:|
| Val | 0.9645 | 0.5598 | 0.4251 | 0.3850 |
| Test | 0.9683 | 0.5541 | 0.4153 | 0.3577 |

## Matriz de confusión (Val)

| real\pred | E | L | V |
|---|---:|---:|---:|
| E | 1 | 63 | 10 |
| L | 1 | 157 | 15 |
| V | 0 | 62 | 34 |

## Matriz de confusión (Test)

| real\pred | E | L | V |
|---|---:|---:|---:|
| E | 0 | 61 | 16 |
| L | 0 | 144 | 8 |
| V | 0 | 47 | 20 |
