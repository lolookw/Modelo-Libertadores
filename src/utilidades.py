# src/utilidades.py

from pathlib import Path
import pandas as pd


def asegurar_directorio(ruta: Path) -> None:
    ruta.mkdir(parents=True, exist_ok=True)


def imprimir_resumen(df: pd.DataFrame) -> None:
    print("\n=== Resumen dataset model-ready ===")
    print(f"Filas: {len(df):,}")
    print("Rango fechas:", df["fecha_parseada"].min().date(), "→", df["fecha_parseada"].max().date())
    print("\nFilas por split:")
    print(df["split"].value_counts(dropna=False).to_string())

    print("\nDistribución resultado_norm (global):")
    print((df["resultado_norm"].value_counts(normalize=True) * 100).round(2).astype(str) + "%")

    print("\nDistribución resultado_norm por split:")
    tabla = (
        df.groupby("split")["resultado_norm"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .rename("porcentaje")
        .reset_index()
        .sort_values(["split", "resultado_norm"])
    )
    print(tabla.to_string(index=False))
