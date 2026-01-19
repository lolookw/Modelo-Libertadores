# src/preparacion_datos.py

import argparse
from pathlib import Path

import pandas as pd

from configuracion import (
    ARCHIVO_SALIDA_PROCESSED,
    COLUMNAS_MODELO,
    ANIO_TRAIN_HASTA,
    ANIO_VAL_HASTA,
    RUTA_PROCESSED,
)
from utilidades import asegurar_directorio, imprimir_resumen


def asignar_split(temporada: int) -> str:
    if temporada <= ANIO_TRAIN_HASTA:
        return "train"
    if temporada <= ANIO_VAL_HASTA:
        return "val"
    return "test"


def preparar_dataset(ruta_entrada: Path) -> pd.DataFrame:
    df = pd.read_csv(ruta_entrada)

    # Chequeos básicos
    faltantes = [c for c in COLUMNAS_MODELO if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el CSV de entrada: {faltantes}")

    # Nos quedamos con columnas mínimas
    df = df[COLUMNAS_MODELO].copy()

    # Parse fecha (viene como string YYYY-MM-DD según tu ejemplo)
    df["fecha_parseada"] = pd.to_datetime(df["fecha_parseada"], errors="coerce")

    # Tipos
    df["temporada"] = pd.to_numeric(df["temporada"], errors="coerce").astype("Int64")

    # Limpiar target
    df["resultado_norm"] = df["resultado_norm"].astype(str).str.strip().str.upper()

    # Filtros de calidad mínimos
    df = df.dropna(subset=["fecha_parseada", "temporada", "equipo_local", "equipo_visitante", "resultado_norm"])
    df = df[df["resultado_norm"].isin(["L", "E", "V"])]

    # Orden cronológico (importante para features posteriores tipo Elo/rolling)
    df = df.sort_values(["fecha_parseada", "equipo_local", "equipo_visitante"]).reset_index(drop=True)

    # Split temporal congelado
    df["split"] = df["temporada"].apply(lambda x: asignar_split(int(x)))

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepara dataset model-ready (1X2) desde el CSV enriquecido y asigna splits temporales."
    )
    parser.add_argument(
        "--ruta_entrada",
        type=str,
        required=True,
        help="Ruta al CSV fuente (partidos_rsssf1_enriquecido.csv).",
    )
    parser.add_argument(
        "--ruta_salida",
        type=str,
        default=str(ARCHIVO_SALIDA_PROCESSED),
        help="Ruta de salida del CSV processed (default: data/processed/partidos_modelo_1x2.csv).",
    )
    args = parser.parse_args()

    ruta_entrada = Path(args.ruta_entrada)
    ruta_salida = Path(args.ruta_salida)

    if not ruta_entrada.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {ruta_entrada}")

    asegurar_directorio(ruta_salida.parent)
    df = preparar_dataset(ruta_entrada)

    df.to_csv(ruta_salida, index=False, encoding="utf-8")
    print(f"\nOK ✅ Archivo generado: {ruta_salida}")

    imprimir_resumen(df)


if __name__ == "__main__":
    main()
