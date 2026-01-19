# Definición del experimento — Proyecto 2 (1X2)

## Target
`resultado_1x2`:
- `L` si goles_local > goles_visitante
- `E` si goles_local = goles_visitante
- `V` si goles_local < goles_visitante

## Features permitidas (solo pre-partido)
- Ratings tipo Elo: `elo_local`, `elo_visitante`, `dif_elo`
- Forma reciente rolling (últimos N partidos): puntos, GF, GC
- Rendimiento local/visita rolling
- Contexto: `fase`, `es_eliminatoria`, `anio`, `mes` (si hay fecha)
- Opcional: head-to-head histórico (hasta antes del partido)

## Prohibido (leakage)
- Cualquier dato del partido objetivo: goles, resultado, penales, eventos
- Estadísticas calculadas usando partidos futuros
- Variables post-torneo (campeón, ronda final alcanzada, etc.)

## Split temporal
- Train: 1996–2018
- Validación: 2019–2021
- Test: 2022–2024

## Métricas
- Principal: logloss multiclase
- Secundarias: balanced accuracy, macro F1, calibración

## Criterio de éxito
- Mejorar logloss vs baseline Elo ≥ 3–5% relativo (validación) y sostener en test
- Mantener calibración razonable (reliability) y explicar errores típicos
