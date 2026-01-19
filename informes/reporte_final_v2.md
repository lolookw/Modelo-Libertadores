# Reporte final — Proyecto 2 (Predicción 1X2 Libertadores)

Objetivo: predecir el resultado de un partido (E/L/V) usando solo información **pre-partido**, evitando leakage.

Split temporal: train 1996–2018, val 2019–2021, test 2022–2024.

## Resumen ejecutivo

- Baselines simples (localía / frecuencias) sirven como piso.
- Elo mejora fuerte al capturar fuerza relativa.
- El mejor modelo (Logística + Elo + forma rolling + fase + neutral) mejora logloss y accuracy en val/test.

## Baselines

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
| Val | 1.0095 | 0.5277 | 0.4265 | 0.4051 |
| Test | 1.0033 | 0.5372 | 0.4392 | 0.3913 |

**Matriz de confusión (Val)**

| real\pred | E | L | V |
|---|---:|---:|---:|
| E | 5 | 52 | 17 |
| L | 6 | 134 | 33 |
| V | 6 | 48 | 42 |

**Matriz de confusión (Test)**

| real\pred | E | L | V |
|---|---:|---:|---:|
| E | 2 | 46 | 29 |
| L | 7 | 126 | 19 |
| V | 2 | 34 | 31 |


## Modelo ML v1 (sin Elo)

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


## Modelo ML v2 (con Elo)

# Reporte — Modelo ML (Regresión logística multinomial)

Features: rolling (forma GF/GC + puntos), neutralidad y fase.

| Split | Logloss | Accuracy | Balanced Acc | Macro F1 |
|---|---:|---:|---:|---:|
| Val | 0.9651 | 0.5598 | 0.4251 | 0.3850 |
| Test | 0.9700 | 0.5541 | 0.4153 | 0.3577 |

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


## Ablation / Experimento adicional — features diferenciales de forma (dif_forma)

Se probó una variante del modelo (Logística + Elo + rolling) agregando features que capturan la **diferencia de forma** entre equipos:

- `dif_puntos_ultN = puntos_ultN_local - puntos_ultN_visitante`
- `dif_gf_ultN = gf_ultN_local - gf_ultN_visitante`
- `dif_gc_ultN = gc_ultN_local - gc_ultN_visitante`

**Resultado:** las métricas en validación y test quedaron prácticamente **idénticas** a la versión sin `dif_forma` (cambios marginales en logloss, sin mejoras consistentes en accuracy/balanced accuracy/macro F1).

**Interpretación:** la señal de “forma relativa” ya estaba capturada en gran parte por `dif_elo` y/o por las features rolling individuales, por lo que `dif_forma` no aportó información incremental en este setup lineal.

**Decisión:** mantener como modelo principal el enfoque **Logística + Elo + rolling** (pre-partido, sin leakage) y enfocar los próximos esfuerzos en mejoras con mayor retorno: **calibración de probabilidades** y/o **modelos no lineales** para capturar interacciones (especialmente para empates).


## Próximos pasos

- Mejorar predicción de empates: features específicas (diferencia de forma, travel proxy por países, etc.) y/o modelo más flexible.
- Calibración de probabilidades (temperature scaling / isotonic) y evaluación con Brier score.
- Demo Streamlit para consultar predicciones por matchup.

## Calibración de probabilidades — Temperature Scaling

Como el objetivo incluye probabilidades útiles (no solo el argmax), se aplicó **temperature scaling** para calibrar las probabilidades del modelo final.  
La temperatura **T** se aprendió **solo en validación** (sin tocar test), y luego se evaluó el impacto en test.

- Temperatura estimada: **T = 1.1023** (suaviza probabilidades; el modelo base estaba levemente overconfident).
- Resultado: mejora consistente en **logloss** sin cambios en métricas de clasificación dura (accuracy/F1), lo cual es esperable.

**Métricas (antes vs después)**

| Split | Variante | Logloss | Accuracy | Balanced Acc | Macro F1 |
|---|---|---:|---:|---:|---:|
| Val | Base | 0.9651 | 0.5598 | 0.4251 | 0.3850 |
| Val | Calibrado | 0.9638 | 0.5598 | 0.4251 | 0.3850 |
| Test | Base | 0.9700 | 0.5541 | 0.4153 | 0.3577 |
| Test | Calibrado | 0.9683 | 0.5541 | 0.4153 | 0.3577 |

**Conclusión:** se adopta la versión **calibrada** como salida final para scoring probabilístico.
