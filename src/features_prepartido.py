# src/features_prepartido.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from configuracion import RUTA_PROCESSED
from utilidades import asegurar_directorio


def _puntos_por_resultado(resultado_norm: str) -> int:
    if resultado_norm == "L":
        return 3
    if resultado_norm == "E":
        return 1
    return 0  # V


def _neutral_por_regla(fase: str, pais_sede: str, pais_local: str) -> int:
    """
    Regla simple:
    - Final => neutral
    - Si hay pais_sede y difiere de pais_local => neutral
    """
    if str(fase).strip().lower() == "final":
        return 1
    ps = str(pais_sede).strip()
    pl = str(pais_local).strip()
    if ps and pl and ps.lower() != pl.lower():
        return 1
    return 0


def construir_features_rolling(
    df: pd.DataFrame,
    ventana: int = 5,
) -> pd.DataFrame:
    """
    Construye features rolling por equipo (forma) sin leakage:
    - usamos shift(1) para que el partido actual NO entre en su propia historia.
    Genera:
    - para local: puntos_ultN_local, gf_ultN_local, gc_ultN_local
    - para visitante: puntos_ultN_visita, gf_ultN_visita, gc_ultN_visita

    Nota: Rolling se calcula sobre "partidos del equipo" (home+away combinados) en orden temporal.
    """
    df = df.sort_values(["fecha_parseada", "equipo_local", "equipo_visitante"]).reset_index(drop=True)

    # Construimos una tabla "larga" por equipo-partido
    registros = []

    for idx, fila in df.iterrows():
        res = str(fila["resultado_norm"]).upper().strip()
        gl = int(fila["goles_local"]) if "goles_local" in df.columns else None
        gv = int(fila["goles_visitante"]) if "goles_visitante" in df.columns else None

        # local
        puntos_local = _puntos_por_resultado(res)
        registros.append(
            {
                "idx_partido": idx,
                "equipo": fila["equipo_local"],
                "rol": "local",
                "puntos": puntos_local,
                "gf": gl,
                "gc": gv,
                "fecha_parseada": fila["fecha_parseada"],
            }
        )

        # visitante (puntos invertidos)
        if res == "L":
            puntos_visita = 0
        elif res == "E":
            puntos_visita = 1
        else:
            puntos_visita = 3

        registros.append(
            {
                "idx_partido": idx,
                "equipo": fila["equipo_visitante"],
                "rol": "visitante",
                "puntos": puntos_visita,
                "gf": gv,
                "gc": gl,
                "fecha_parseada": fila["fecha_parseada"],
            }
        )

    largo = pd.DataFrame(registros)
    largo = largo.sort_values(["equipo", "fecha_parseada", "idx_partido"]).reset_index(drop=True)

    # Rolling sin leakage: shift(1) para excluir el partido actual
    largo["puntos_ultN"] = (
        largo.groupby("equipo")["puntos"]
        .apply(lambda s: s.shift(1).rolling(ventana, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    # GF/GC pueden tener nulos si no incluimos goles; acá asumimos que están
    largo["gf_ultN"] = (
        largo.groupby("equipo")["gf"]
        .apply(lambda s: s.shift(1).rolling(ventana, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    largo["gc_ultN"] = (
        largo.groupby("equipo")["gc"]
        .apply(lambda s: s.shift(1).rolling(ventana, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    # Separamos rol local/visitante y mergeamos al df original
    largo_local = largo[largo["rol"] == "local"][["idx_partido", "puntos_ultN", "gf_ultN", "gc_ultN"]].copy()
    largo_local = largo_local.rename(
        columns={
            "puntos_ultN": "puntos_ultN_local",
            "gf_ultN": "gf_ultN_local",
            "gc_ultN": "gc_ultN_local",
        }
    )

    largo_visita = largo[largo["rol"] == "visitante"][["idx_partido", "puntos_ultN", "gf_ultN", "gc_ultN"]].copy()
    largo_visita = largo_visita.rename(
        columns={
            "puntos_ultN": "puntos_ultN_visitante",
            "gf_ultN": "gf_ultN_visitante",
            "gc_ultN": "gc_ultN_visitante",
        }
    )

    out = df.copy()
    out["idx_partido"] = np.arange(len(out))

    out = out.merge(largo_local, on="idx_partido", how="left")
    out = out.merge(largo_visita, on="idx_partido", how="left")

    out = out.drop(columns=["idx_partido"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera features pre-partido (rolling) para modelo ML.")
    parser.add_argument(
        "--ruta_processed",
        type=str,
        default=str(RUTA_PROCESSED / "partidos_modelo_1x2.csv"),
        help="Ruta al CSV processed.",
    )
    parser.add_argument(
        "--ventana",
        type=int,
        default=5,
        help="Ventana de rolling (ultimos N partidos).",
    )
    parser.add_argument(
        "--ruta_salida",
        type=str,
        default=str(RUTA_PROCESSED / "partidos_features_ml.csv"),
        help="Ruta de salida con features.",
    )
    args = parser.parse_args()

    ruta_in = Path(args.ruta_processed)
    ruta_out = Path(args.ruta_salida)

    if not ruta_in.exists():
        raise FileNotFoundError(f"No existe: {ruta_in}")

    df = pd.read_csv(ruta_in, parse_dates=["fecha_parseada"])

    # Necesitamos goles para rolling GF/GC y target
    # Si no están, hay que regenerar processed incluyendo goles.
    necesarias = {"goles_local", "goles_visitante", "resultado_norm", "equipo_local", "equipo_visitante", "fase", "pais_sede", "pais_local", "split"}
    faltan = necesarias - set(df.columns)
    if faltan:
        raise ValueError(
            "Faltan columnas en processed para construir features. "
            f"Faltan: {sorted(list(faltan))}. "
            "Solución: regenerar processed incluyendo goles_local/goles_visitante/pais_sede."
        )

    # Feature neutral
    df["neutral"] = df.apply(lambda r: _neutral_por_regla(r["fase"], r["pais_sede"], r["pais_local"]), axis=1)

    # Features rolling
    df_feat = construir_features_rolling(df, ventana=args.ventana)

    asegurar_directorio(ruta_out.parent)
    df_feat.to_csv(ruta_out, index=False, encoding="utf-8")
    print(f"OK ✅ Features generadas: {ruta_out}")


if __name__ == "__main__":
    main()
