# Proyecto 2 — Predicción 1X2 en Copa Libertadores (1996–2024)

Este repo es el **Proyecto 2** de mi portfolio (8 semanas, 4–6 proyectos). El objetivo fue construir un pipeline **publicable y reproducible** de **modelado + evaluación + storytelling**, usando como base el dataset armado en el **Proyecto 1** (Libertadores 1996–2024).

La idea del proyecto es mostrar un proceso realista de trabajo:
- definir un problema de predicción
- evitar leakage (muy común en deportes)
- establecer baselines fuertes
- mejorar incrementalmente
- evaluar con métricas correctas (no solo accuracy)
- documentar aprendizajes (incluyendo experimentos que no mejoran)

---

## Problema de predicción (1X2)

**Tarea:** predecir el resultado de un partido:
- `L` = gana el local
- `E` = empate
- `V` = gana el visitante

**Restricción clave:** usar **solo información disponible antes del partido** (pre-partido).  
Nada de features que “miran” el resultado del partido o estadísticas posteriores.

---

## Datos

**Fuente base:** dataset procesado del Proyecto 1.

Archivo de entrada (local, fuera de este repo):
- `C:\Users\Lorenzo\Desktop\coding\Libertadores-1996-2024\datos\procesados\partidos_rsssf1_enriquecido.csv`

Columnas relevantes (ejemplo):
- `fecha_parseada`, `temporada`, `fase`, `equipo_local`, `equipo_visitante`, `pais_local`, `pais_visitante`
- `goles_local`, `goles_visitante`, `resultado_norm`
- `pais_sede` (para neutralidad / finales)

**Dataset final “model-ready” (generado en este repo):**
- `data/processed/partidos_modelo_1x2.csv`
- 3,149 partidos (1996-03-13 → 2024-06-04)

Distribución global aproximada:
- `L`: ~55%
- `E`: ~23%
- `V`: ~22%

---

## Split temporal (anti-leakage)

Para simular un uso real (predicción futura), el split es **por tiempo**:

- **Train:** histórico (mayor parte del período)
- **Validation:** años recientes intermedios (para decisiones de modelo)
- **Test:** años más recientes (evaluación final)

> Nota: el split exacto se implementa en `src/preparacion_datos.py` y queda guardado en la columna `split`. Lo importante es que el test representa “futuro” respecto a train/val.

---

## Métricas

La métrica principal elegida fue **Logloss (cross-entropy)**, porque el deliverable final son **probabilidades** (útiles para ranking/decisiones), no solo un “acierto/no acierto”.

Además se reportan:
- Accuracy
- Balanced Accuracy (importante por desbalance de clases)
- Macro F1
- Matriz de confusión (para entender errores típicos)

---

# Enfoque y evolución del modelo (la historia del proyecto)

## Paso 1 — Baselines simples (pisos)

Arranqué con baselines deliberadamente simples para tener un “piso”:

### Baseline 1: “Siempre Local”
Predice `L` para todo.
- Sirve para cuantificar cuánto explica solo la localía y el desbalance.
- Da una accuracy alrededor de ~0.50–0.52, pero métricas por clase muy flojas.
- Logloss alto (probabilidades malas / demasiado seguras).

### Baseline 2: “Frecuencias por fase”
Estima probabilidades por `fase` usando train y las aplica a val/test.
- Mejora fuerte logloss vs “siempre local” (porque ya produce probabilidades razonables).
- Pero sigue tendiendo a predecir `L` como argmax, y no capta fuerza relativa de equipos.

**Aprendizaje:** hace falta una señal de “fuerza” / calidad relativa para separar `L` de `V` de forma consistente.

---

## Paso 2 — Baseline fuerte: Elo pre-partido (sin leakage)

Implementé un baseline **Elo** porque:
- es el estándar mínimo serio para fuerza relativa en deportes
- es **pre-partido** por construcción (si se calcula bien)
- es interpretable y fácil de explicar

### Elo (Baseline 3)
- Cada equipo arranca con Elo=1500
- Se recorre train **en orden temporal**
- Para cada partido se predice con Elo pre-partido y luego se actualiza Elo con el resultado real
- En val/test se predice usando Elo aprendido en train (sin actualizar)

#### Neutralidad (finales / sede)
Como en copas hay partidos “neutrales” (ej. Final o sede distinta), se incorpora una regla:
- si el partido es neutral ⇒ ventaja_local = 0
- caso normal ⇒ ventaja_local > 0

#### Empates en Elo
Un Elo binario clásico no modela empates naturalmente. Se probó un esquema simple para repartir probabilidad de empate en función de la “parejidad” (dif Elo cerca de 0 ⇒ más empate). Aun así, el empate es una clase difícil y suele no ganar como argmax.

**Resultado:** Elo mejora claramente vs baselines simples y se convierte en un baseline competitivo.

**Aprendizaje:** Elo es una gran base, pero:
- no captura “forma reciente” explícita
- y predecir empates como clase hard sigue siendo difícil

---

## Paso 3 — Primer modelo ML: “forma + fase” (y por qué NO alcanzó)

Se generaron features rolling pre-partido (ventana N=5) por equipo:

- puntos promedio últimos N
- GF promedio últimos N
- GC promedio últimos N  
para local y visitante (incluye partidos home+away, con `shift(1)` para evitar leakage)

Además:
- `neutral`
- one-hot de `fase`

Se entrenó una **Regresión Logística multinomial** (pipeline con imputación y escalado).

**Resultado:** este modelo fue flojo: tendía a predecir `L` casi siempre y no superó a Elo.

**Aprendizaje clave:** la “forma” sola es ruidosa. Falta la señal base de fuerza global.

---

## Paso 4 — Modelo ML ganador: Logística + Elo + forma (pre-partido)

El salto real fue incluir Elo como feature (pre-partido), junto con forma y fase:

Features numéricas principales:
- `dif_elo`, `elo_local`, `elo_visitante`
- `neutral`
- rollings (puntos/GF/GC local y visitante)

Feature categórica:
- `fase` (one-hot)

**Resultado:** mejora clara respecto a Elo baseline y a ML sin Elo.  
En test, el modelo logra aproximadamente:
- **logloss ~0.97**
- **accuracy ~0.55**

**Interpretación:** el modelo combina:
- fuerza relativa estructural (Elo)
- señales recientes (forma rolling)
- contexto competitivo (fase)
- y ajusta probabilidades con un modelo simple e interpretable

---

## Paso 5 — Ablation: “dif_forma” (experimento que NO mejoró)

Se probó agregar features diferenciales:
- `dif_puntos_ultN`, `dif_gf_ultN`, `dif_gc_ultN`

**Resultado:** métricas prácticamente idénticas.  
**Conclusión:** esa señal ya estaba capturada por `dif_elo` y/o rollings individuales.

Esto se documenta como parte del proceso: no todo “feature engineering razonable” mejora.

---

## Paso 6 — Calibración: Temperature Scaling (mejora final en logloss)

Como el objetivo incluye probabilidades útiles, se aplicó **temperature scaling**:

- se aprende `T` **solo en validación**
- se evalúa el impacto en test sin tocar el split

Resultado:
- `T ≈ 1.10` (modelo ligeramente sobreconfiado)
- logloss baja un poco en val y test
- accuracy y matrices casi no cambian (esperable)

**Modelo final recomendado:** ML + Elo + forma + fase + neutralidad **calibrado**.

---

## Estructura del repo

- `data/`
  - `processed/` (datasets generados para modelado)
- `src/`
  - `preparacion_datos.py` → crea `partidos_modelo_1x2.csv` con split temporal
  - `features_prepartido.py` → features rolling pre-partido
  - `features_elo.py` → Elo pre-partido como features (sin leakage)
  - `entrenamiento.py` → entrena regresión logística (pipeline reproducible)
  - `evaluacion.py` → baselines
  - `evaluacion_modelo.py` → evalúa modelo ML
  - `calibracion_temperature.py` → calibración por temperature scaling
  - `reporte_consolidado.py` → genera reporte final
- `informes/`
  - reportes `.md` con resultados
- `modelos/`
  - `.joblib` del modelo y parámetros de calibración

---

## Cómo reproducir (pipeline)

> Asume Python 3.11 en Windows y entorno virtual activo.

1) Generar dataset model-ready:
```powershell
python .\src\preparacion_datos.py --ruta_entrada "C:\Users\Lorenzo\Desktop\coding\Libertadores-1996-2024\datos\procesados\partidos_rsssf1_enriquecido.csv"
```
2) Baselines:
```powershell
python .\src\evaluacion.py
```

3) Features rolling:
```powershell
python .\src\features_prepartido.py --ventana 5
```

4) Features Elo:
```powershell
python .\src\features_elo.py
```

5) Entrenar modelo (ejemplo):
```powershell
python .\src\entrenamiento.py --ruta_features .\data\processed\partidos_features_ml_con_elo.csv --ruta_modelo_salida modelos\modelo_logistico_1x2_final.joblib
```

6) Evaluar modelo:
```powershell
python .\src\evaluacion_modelo.py --ruta_features .\data\processed\partidos_features_ml_con_elo.csv --ruta_modelo modelos\modelo_logistico_1x2_final.joblib --ruta_reporte informes\reporte_modelo_final.md
```

7) Calibración (temperature scaling):
```powershell
python .\src\calibracion_temperature.py --ruta_features .\data\processed\partidos_features_ml_con_elo.csv --ruta_modelo modelos\modelo_logistico_1x2_final.joblib
```

8) Reporte consolidado:
```powershell
python .\src\reporte_consolidado.py --ruta_ml_con_elo informes\reporte_modelo_final.md --ruta_salida informes\reporte_final.md
```

## 🧠 Principales Aprendizajes

* **El impacto del Baseline:** Un baseline "serio" basado en **Elo** cambia completamente el juego; entender la fuerza relativa de los equipos es un factor crítico para el éxito del modelo.
* **Insuficiencia de la "forma":** La métrica de *forma rolling* por sí sola no alcanza; es excesivamente ruidosa y el modelo tiende a colapsar simplificando todo a la localía.
* **Sinergia de Features:** Agregar **Elo como feature** y entrenar un modelo simple (como Regresión Logística) produce mejoras reales, consistentes y reproducibles.
* **Calibración de Probabilidades:** La calibración representa una mejora sustancial en la "calidad probabilística" (**logloss**), independientemente de si el *accuracy* se mantiene estable.
* **El desafío del empate:** Los empates siguen siendo una "clase hard" muy difícil de predecir; es el punto de fricción más claro para el desarrollo futuro.

---

## 🚀 Limitaciones y Próximos Pasos

### ⚖️ Gestión de Empates
El modelo actual rara vez predice el empate como la opción más probable (*argmax*). Para mitigar esto, se planea:
* Implementar **modelos no lineales** más robustos como `HistGradientBoosting` o `XGBoost`.
* Diseñar **features específicas de empate** (paridad extrema de fuerzas, contextos de eliminación directa, etc.).

### ⏳ Generalización por Épocas
La Copa Libertadores ha evolucionado significativamente en formato y nivel competitivo. Las estrategias a probar incluyen:
* Aplicar técnicas de **rolling por ventanas de tiempo** específicas.
* Crear **features por era/torneo** para capturar la dinámica de cada época del fútbol sudamericano.

