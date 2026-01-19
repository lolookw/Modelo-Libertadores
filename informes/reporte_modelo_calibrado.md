# Reporte — Calibración (Temperature Scaling)

Modelo base: `modelos\modelo_logistico_1x2_con_elo_y_dif_forma.joblib`

Temperatura aprendida en **validación**: **T = 1.1023**

## Métricas (antes vs después)

| Split | Variante | Logloss | Accuracy | Balanced Acc | Macro F1 |
|---|---|---:|---:|---:|---:|
| Val | Base | 0.9651 | 0.5598 | 0.4251 | 0.3850 |
| Val | Calibrado | 0.9638 | 0.5598 | 0.4251 | 0.3850 |
| Test | Base | 0.9700 | 0.5541 | 0.4153 | 0.3577 |
| Test | Calibrado | 0.9683 | 0.5541 | 0.4153 | 0.3577 |

## Matriz de confusión (Test — Base)

| real\pred | E | L | V |
|---|---:|---:|---:|
| E | 0 | 61 | 16 |
| L | 0 | 144 | 8 |
| V | 0 | 47 | 20 |

## Matriz de confusión (Test — Calibrado)

| real\pred | E | L | V |
|---|---:|---:|---:|
| E | 0 | 61 | 16 |
| L | 0 | 144 | 8 |
| V | 0 | 47 | 20 |
