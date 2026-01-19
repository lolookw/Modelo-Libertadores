# src/configuracion.py

from pathlib import Path

# Carpetas del repo
RUTA_REPO = Path(__file__).resolve().parents[1]
RUTA_DATA = RUTA_REPO / "data"
RUTA_RAW = RUTA_DATA / "raw"
RUTA_PROCESSED = RUTA_DATA / "processed"
RUTA_INFORMES = RUTA_REPO / "informes"

# Archivo de salida estándar del proyecto
ARCHIVO_SALIDA_PROCESSED = RUTA_PROCESSED / "partidos_modelo_1x2.csv"

# Split temporal (congelado)
ANIO_TRAIN_HASTA = 2018
ANIO_VAL_HASTA = 2021
# test: el resto

# Columnas mínimas que vamos a usar para el modelo
COLUMNAS_MODELO = [
    "fecha_parseada",
    "temporada",
    "competicion",
    "fase",
    "instancia",
    "pais_sede",
    "equipo_local",
    "equipo_visitante",
    "pais_local",
    "pais_visitante",
    "goles_local",
    "goles_visitante",
    "resultado_norm",
]