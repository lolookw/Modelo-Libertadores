# src/evaluacion_modelo.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
)

from baselines import CLASES_1X2
from configuracion import RUTA_PROCESSED, RUTA_INFORMES
from utilidades import asegurar_directorio


FEATURES_NUMERICAS = [
    "neutral",
    "puntos_ultN_local",
    "gf_ultN_local",
    "gc_ultN_local",
    "puntos_ultN_visitante",
    "gf_ultN_visitante",
    "gc_ultN_visitante",
]

FEATURES_CATEGORICAS = ["fase"]


def cargar_features(ruta: Path) -> pd.DataFrame:
    df = pd.read_csv(ruta, parse_dates=["fecha_parseada"])
    df["resultado_norm"] = df["resultado_norm"].astype(str).str.strip().str.upper()
    return df


def _evaluar(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, object]:
    # pred hard
    y_pred = np.array([CLASES_1X2[i] for i in prob.argmax(axis=1)], dtype=object)

    return {
        "logloss": float(log_loss(y_true, prob, labels=CLASES_1X2)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=CLASES_1X2)),
        "matriz_confusion": confusion_matrix(y_true, y_pred, labels=CLASES_1X2),
    }


def _formatear_matriz(cm: np.ndarray) -> str:
    etiquetas = CLASES_1X2
    encabezado = "| real\\pred | " + " | ".join(etiquetas) + " |"
    separador = "|---|" + "|".join(["---:"] * len(etiquetas)) + "|"
    filas = [encabezado, separador]
    for i, er in enumerate(etiquetas):
        vals = " | ".join(str(int(cm[i, j])) for j in range(len(etiquetas)))
        filas.append(f"| {er} | {vals} |")
    return "\n".join(filas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa modelo ML entrenado (1X2).")
    parser.add_argument(
        "--ruta_features",
        type=str,
        default=str(RUTA_PROCESSED / "partidos_features_ml.csv"),
        help="Ruta al CSV con features.",
    )
    parser.add_argument(
        "--ruta_modelo",
        type=str,
        default="modelos/modelo_logistico_1x2.joblib",
        help="Ruta al modelo joblib.",
    )
    parser.add_argument(
        "--ruta_reporte",
        type=str,
        default=str(RUTA_INFORMES / "reporte_modelo_ml.md"),
        help="Ruta del reporte de salida.",
    )
    args = parser.parse_args()

    ruta_features = Path(args.ruta_features)
    ruta_modelo = Path(args.ruta_modelo)
    ruta_reporte = Path(args.ruta_reporte)

    if not ruta_features.exists():
        raise FileNotFoundError(f"No existe: {ruta_features}")
    if not ruta_modelo.exists():
        raise FileNotFoundError(f"No existe: {ruta_modelo}")

    asegurar_directorio(ruta_reporte.parent)

    df = cargar_features(ruta_features)

    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    X_val = df_val[FEATURES_NUMERICAS + FEATURES_CATEGORICAS]
    y_val = df_val["resultado_norm"].to_numpy()

    X_test = df_test[FEATURES_NUMERICAS + FEATURES_CATEGORICAS]
    y_test = df_test["resultado_norm"].to_numpy()

    pipe = joblib.load(ruta_modelo)

    # Probabilidades: sklearn devuelve columnas según pipe.classes_
    clases_pipe = list(pipe.classes_)
    # Mapeamos a orden CLASES_1X2 = ["E","L","V"]
    idx = [clases_pipe.index(c) for c in CLASES_1X2]

    prob_val = pipe.predict_proba(X_val)[:, idx]
    prob_test = pipe.predict_proba(X_test)[:, idx]

    met_val = _evaluar(y_val, prob_val)
    met_test = _evaluar(y_test, prob_test)

    rep = "# Reporte — Modelo ML (Regresión logística multinomial)\n\n"
    rep += "Features: rolling (forma GF/GC + puntos), neutralidad y fase.\n\n"
    rep += "| Split | Logloss | Accuracy | Balanced Acc | Macro F1 |\n"
    rep += "|---|---:|---:|---:|---:|\n"
    rep += f"| Val | {met_val['logloss']:.4f} | {met_val['accuracy']:.4f} | {met_val['balanced_accuracy']:.4f} | {met_val['macro_f1']:.4f} |\n"
    rep += f"| Test | {met_test['logloss']:.4f} | {met_test['accuracy']:.4f} | {met_test['balanced_accuracy']:.4f} | {met_test['macro_f1']:.4f} |\n"

    rep += "\n## Matriz de confusión (Val)\n\n"
    rep += _formatear_matriz(met_val["matriz_confusion"]) + "\n"

    rep += "\n## Matriz de confusión (Test)\n\n"
    rep += _formatear_matriz(met_test["matriz_confusion"]) + "\n"

    ruta_reporte.write_text(rep, encoding="utf-8")
    print(f"OK ✅ Reporte generado: {ruta_reporte}")


if __name__ == "__main__":
    main()
