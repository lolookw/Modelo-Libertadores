# src/entrenamiento.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from configuracion import RUTA_PROCESSED
from utilidades import asegurar_directorio
from baselines import CLASES_1X2
from sklearn.impute import SimpleImputer



FEATURES_NUMERICAS = [
    "neutral",
    "puntos_ultN_local",
    "gf_ultN_local",
    "gc_ultN_local",
    "puntos_ultN_visitante",
    "gf_ultN_visitante",
    "gc_ultN_visitante",
]

FEATURES_CATEGORICAS = [
    "fase",
]


def cargar_features(ruta: Path) -> pd.DataFrame:
    df = pd.read_csv(ruta, parse_dates=["fecha_parseada"])
    necesarias = set(FEATURES_NUMERICAS + FEATURES_CATEGORICAS + ["resultado_norm", "split"])
    faltan = necesarias - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en partidos_features_ml.csv: {sorted(list(faltan))}")

    df["resultado_norm"] = df["resultado_norm"].astype(str).str.strip().str.upper()
    df = df[df["resultado_norm"].isin(["L", "E", "V"])].copy()
    return df


def construir_pipeline() -> Pipeline:
    # Para que el modelo no explote con NaNs (rolling al inicio, etc.)
    transformador_num = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformador_cat = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocesador = ColumnTransformer(
        transformers=[
            ("num", transformador_num, FEATURES_NUMERICAS),
            ("cat", transformador_cat, FEATURES_CATEGORICAS),
        ]
    )

    modelo = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        C=1.0,
    )

    return Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("modelo", modelo),
        ]
    )



def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena modelo ML (logística multinomial) para 1X2.")
    parser.add_argument(
        "--ruta_features",
        type=str,
        default=str(RUTA_PROCESSED / "partidos_features_ml.csv"),
        help="Ruta al CSV con features.",
    )
    parser.add_argument(
        "--ruta_modelo_salida",
        type=str,
        default="modelos/modelo_logistico_1x2.joblib",
        help="Ruta donde guardar el modelo entrenado.",
    )
    args = parser.parse_args()

    ruta_features = Path(args.ruta_features)
    ruta_modelo = Path(args.ruta_modelo_salida)

    if not ruta_features.exists():
        raise FileNotFoundError(f"No existe: {ruta_features}")

    df = cargar_features(ruta_features)

    df_train = df[df["split"] == "train"].copy()
    X_train = df_train[FEATURES_NUMERICAS + FEATURES_CATEGORICAS]
    y_train = df_train["resultado_norm"].to_numpy()

    # Alineamos el orden de clases con CLASES_1X2 = ["E","L","V"]
    # sklearn usa clases_ internamente; luego en evaluación vamos a mapear.
    pipe = construir_pipeline()
    pipe.fit(X_train, y_train)

    asegurar_directorio(ruta_modelo.parent)
    joblib.dump(pipe, ruta_modelo)

    print(f"OK ✅ Modelo guardado: {ruta_modelo}")
    print("Features numéricas:", FEATURES_NUMERICAS)
    print("Features categóricas:", FEATURES_CATEGORICAS)


if __name__ == "__main__":
    main()
