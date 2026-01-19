# src/features_elo.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from configuracion import RUTA_PROCESSED
from utilidades import asegurar_directorio


def ventaja_por_partido(fila: pd.Series, ventaja_local: float = 60.0) -> float:
    fase = str(fila.get("fase", "")).strip().lower()
    if fase == "final":
        return 0.0

    pais_sede = str(fila.get("pais_sede", "")).strip()
    pais_local = str(fila.get("pais_local", "")).strip()
    if pais_sede and pais_local and pais_sede.lower() != pais_local.lower():
        return 0.0

    return float(ventaja_local)


def construir_elo_features(
    df: pd.DataFrame,
    elo_inicial: float = 1500.0,
    k: float = 20.0,
    escala: float = 400.0,
    ventaja_local: float = 60.0,
) -> pd.DataFrame:
    """
    Crea features Elo pre-partido para todo df.
    - Actualiza elos SOLO recorriendo train.
    - Para val/test no actualiza: usa elos al final de train.
    """
    df = df.sort_values(["fecha_parseada", "equipo_local", "equipo_visitante"]).reset_index(drop=True)
    df = df.copy()

    elos: Dict[str, float] = {}
    elo_local_pre = np.zeros(len(df), dtype=float)
    elo_visit_pre = np.zeros(len(df), dtype=float)
    ventaja_usada = np.zeros(len(df), dtype=float)

    # 1) Primer pasada: asignar Elo pre-partido para todos, usando el diccionario actual
    for i, fila in df.iterrows():
        local = str(fila["equipo_local"])
        visita = str(fila["equipo_visitante"])

        elo_l = elos.get(local, elo_inicial)
        elo_v = elos.get(visita, elo_inicial)

        elo_local_pre[i] = elo_l
        elo_visit_pre[i] = elo_v
        ventaja_usada[i] = ventaja_por_partido(fila, ventaja_local=ventaja_local)

        # Solo actualizamos durante train
        if str(fila["split"]) != "train":
            continue

        res = str(fila["resultado_norm"]).upper().strip()
        s = 0.5
        if res == "L":
            s = 1.0
        elif res == "V":
            s = 0.0

        dif = (elo_l + ventaja_usada[i]) - elo_v
        e = 1.0 / (1.0 + 10 ** (-dif / escala))

        elo_l_nuevo = elo_l + k * (s - e)
        elo_v_nuevo = elo_v + k * ((1.0 - s) - (1.0 - e))

        elos[local] = elo_l_nuevo
        elos[visita] = elo_v_nuevo

    df["elo_local"] = elo_local_pre
    df["elo_visitante"] = elo_visit_pre
    df["dif_elo"] = (df["elo_local"] + ventaja_usada) - df["elo_visitante"]
    df["ventaja_usada"] = ventaja_usada

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Agrega features Elo pre-partido al dataset de features ML.")
    parser.add_argument(
        "--ruta_features",
        type=str,
        default=str(RUTA_PROCESSED / "partidos_features_ml.csv"),
        help="Ruta al CSV con features rolling.",
    )
    parser.add_argument(
        "--ruta_salida",
        type=str,
        default=str(RUTA_PROCESSED / "partidos_features_ml_con_elo.csv"),
        help="Ruta de salida con Elo agregado.",
    )
    args = parser.parse_args()

    ruta_in = Path(args.ruta_features)
    ruta_out = Path(args.ruta_salida)

    if not ruta_in.exists():
        raise FileNotFoundError(f"No existe: {ruta_in}")

    df = pd.read_csv(ruta_in, parse_dates=["fecha_parseada"])

    necesarias = {"split", "resultado_norm", "equipo_local", "equipo_visitante", "fase", "pais_sede", "pais_local"}
    faltan = necesarias - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas para Elo: {sorted(list(faltan))}")

    df_out = construir_elo_features(df)

    asegurar_directorio(ruta_out.parent)
    df_out.to_csv(ruta_out, index=False, encoding="utf-8")
    print(f"OK ✅ Features Elo agregadas: {ruta_out}")


if __name__ == "__main__":
    main()
