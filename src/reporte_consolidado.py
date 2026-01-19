# src/reporte_consolidado.py

from __future__ import annotations

import argparse
from pathlib import Path

from configuracion import RUTA_INFORMES
from utilidades import asegurar_directorio


def leer_archivo(ruta: Path) -> str:
    if not ruta.exists():
        return ""
    return ruta.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Arma un reporte final consolidado (baselines + modelos).")
    parser.add_argument(
        "--ruta_baselines",
        type=str,
        default=str(RUTA_INFORMES / "reporte_resultados.md"),
        help="Reporte de baselines.",
    )
    parser.add_argument(
        "--ruta_ml_sin_elo",
        type=str,
        default=str(RUTA_INFORMES / "reporte_modelo_ml.md"),
        help="Reporte modelo ML sin Elo.",
    )
    parser.add_argument(
        "--ruta_ml_con_elo",
        type=str,
        default=str(RUTA_INFORMES / "reporte_modelo_ml_con_elo.md"),
        help="Reporte modelo ML con Elo.",
    )
    parser.add_argument(
        "--ruta_salida",
        type=str,
        default=str(RUTA_INFORMES / "reporte_final.md"),
        help="Salida del reporte final.",
    )
    args = parser.parse_args()

    ruta_baselines = Path(args.ruta_baselines)
    ruta_ml_sin_elo = Path(args.ruta_ml_sin_elo)
    ruta_ml_con_elo = Path(args.ruta_ml_con_elo)
    ruta_salida = Path(args.ruta_salida)

    asegurar_directorio(ruta_salida.parent)

    txt_baselines = leer_archivo(ruta_baselines)
    txt_ml1 = leer_archivo(ruta_ml_sin_elo)
    txt_ml2 = leer_archivo(ruta_ml_con_elo)

    rep = "# Reporte final — Proyecto 2 (Predicción 1X2 Libertadores)\n\n"
    rep += "Objetivo: predecir el resultado de un partido (E/L/V) usando solo información **pre-partido**, evitando leakage.\n\n"
    rep += "Split temporal: train 1996–2018, val 2019–2021, test 2022–2024.\n\n"

    rep += "## Resumen ejecutivo\n\n"
    rep += "- Baselines simples (localía / frecuencias) sirven como piso.\n"
    rep += "- Elo mejora fuerte al capturar fuerza relativa.\n"
    rep += "- El mejor modelo (Logística + Elo + forma rolling + fase + neutral) mejora logloss y accuracy en val/test.\n\n"

    rep += "## Baselines\n\n"
    rep += txt_baselines if txt_baselines else "_No se encontró el reporte de baselines._\n"

    rep += "\n\n## Modelo ML v1 (sin Elo)\n\n"
    rep += txt_ml1 if txt_ml1 else "_No se encontró el reporte del modelo ML sin Elo._\n"

    rep += "\n\n## Modelo ML v2 (con Elo)\n\n"
    rep += txt_ml2 if txt_ml2 else "_No se encontró el reporte del modelo ML con Elo._\n"

    rep += "\n\n## Próximos pasos\n\n"
    rep += "- Mejorar predicción de empates: features específicas (diferencia de forma, travel proxy por países, etc.) y/o modelo más flexible.\n"
    rep += "- Calibración de probabilidades (temperature scaling / isotonic) y evaluación con Brier score.\n"
    rep += "- Demo Streamlit para consultar predicciones por matchup.\n"

    ruta_salida.write_text(rep, encoding="utf-8")
    print(f"OK ✅ Reporte final generado: {ruta_salida}")


if __name__ == "__main__":
    main()
