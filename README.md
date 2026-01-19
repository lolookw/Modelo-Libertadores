# Proyecto 2 — Modelo predictivo 1X2 (Libertadores 1996–2024)

Este repo construye un pipeline reproducible para **predecir el resultado de un partido** de Copa Libertadores:
- **Local / Empate / Visitante (1X2)**
- Usando **solo información disponible antes del partido** (sin leakage)
- Con enfoque de portfolio: **baseline → mejora**, evaluación sólida y reporte de errores.

---

## Objetivo

Entrenar y evaluar modelos que predigan probabilidades para:
- `L` (gana local)
- `E` (empate)
- `V` (gana visitante)

y demostrar mejoras respecto a baselines simples (localía, frecuencias históricas y Elo simple).

---

## Dataset

Base: **Proyecto 1 (Libertadores 1996–2024)**  
Ruta local del usuario (referencia):  
`C:\Users\Lorenzo\Desktop\coding\Libertadores-1996-2024`

Este repo **no sube los datos crudos** a GitHub.  
Se documenta cómo copiarlos a `data/raw/` y cómo generar `data/processed/`.

---

## Reglas anti-leakage (importante)

Features permitidas: **solo pre-partido**, calculadas con historial **hasta el día anterior** al partido.  
Prohibido usar:
- goles del mismo partido / resultado del partido
- info “post” (campeón final, ronda alcanzada final del torneo, etc.)
- cualquier feature que use partidos futuros por error

Ver: `informes/definicion_experimento.md`

---

## Split temporal (por tiempo)

Propuesta:
- Train: 1996–2018
- Validación: 2019–2021
- Test: 2022–2024

---

## Métricas

Principales:
- **Logloss (multiclase)** (calidad probabilística)
- **Balanced accuracy**

Secundarias:
- Macro F1
- Matriz de confusión por fase
- Calibración (reliability)

---

## Baselines obligatorios

1. **Naive localía**: siempre predice `Local`
2. **Frecuencias por fase**: predice usando distribución histórica por fase
3. **Elo simple**: probabilidades desde diferencia de Elo (`dif_elo`)

---

## Estructura del repo

- `data/`  
  - `raw/` (no versionado)  
  - `processed/` (no versionado)
- `src/` scripts del pipeline
- `modelos/` (no versionado)
- `informes/` reporte y figuras (figuras no versionadas por defecto)
- `notebooks/` exploración
- `app/` demo opcional (Streamlit)

---

## Cómo correr (resumen)

1) Crear entorno e instalar dependencias  
2) Preparar datos (raw → processed)  
3) Entrenar baselines y modelos  
4) Evaluar y generar informe

> Se completa cuando estén listos los scripts del pipeline.

---

## Próximos pasos

- Inventario de columnas del dataset Proyecto 1
- Definir features rolling + Elo
- Implementar baselines
- Entrenar modelo mejorado + calibración
- Reporte final + (opcional) Streamlit
