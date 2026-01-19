# src/calibracion_temperature.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from scipy.optimize import minimize

from baselines import CLASES_1X2
from configuracion import RUTA_PROCESSED, RUTA_INFORMES
from utilidades import asegurar_directorio


FEATURES_NUMERICAS = [
    "neutral",
    "dif_elo",
    "elo_local",
    "elo_visitante",
    "puntos_ultN_local",
    "gf_ultN_local",
    "gc_ultN_local",
    "puntos_ultN_visitante",
    "gf_ultN_visitante",
    "gc_ultN_visitante",
    "dif_puntos_ultN",
    "dif_gf_ultN",
    "dif_gc_ultN",
]

FEATURES_CATEGORICAS = ["fase"]


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / expz.sum(axis=1, keepdims=True)


def metricas(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, object]:
    y_pred = np.array([CLASES_1X2[i] for i in prob.argmax(axis=1)], dtype=object)
    return {
        "logloss": float(log_loss(y_true, prob, labels=CLASES_1X2)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=CLASES_1X2)),
        "matriz_confusion": confusion_matrix(y_true, y_pred, labels=CLASES_1X2),
    }


def formatear_matriz(cm: np.ndarray) -> str:
    etiquetas = CLASES_1X2
    encabezado = "| real\\pred | " + " | ".join(etiquetas) + " |"
    separador = "|---|" + "|".join(["---:"] * len(etiquetas)) + "|"
    filas = [encabezado, separador]
    for i, er in enumerate(etiquetas):
        vals = " | ".join(str(int(cm[i, j])) for j in range(len(etiquetas)))
        filas.append(f"| {er} | {vals} |")
    return "\n".join(filas)


def calibrar_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """
    Aprende T minimizando logloss en validación:
      prob = softmax(logits / T)
    con T > 0.
    """

    # y_true a índices según CLASES_1X2
    mapa = {c: i for i, c in enumerate(CLASES_1X2)}
    y_idx = np.array([mapa[str(y)] for y in y_true], dtype=int)

    def objetivo(logT: np.ndarray) -> float:
        T = float(np.exp(logT[0]))
        prob = softmax(logits / T)
        # log_loss necesita prob (n, k)
        return float(log_loss(y_idx, prob, labels=list(range(len(CLASES_1X2)))))

    res = minimize(objetivo, x0=np.array([0.0]), method="Nelder-Mead")
    T = float(np.exp(res.x[0]))
    return T


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibración por temperature scaling (multiclase).")
    parser.add_argument(
        "--ruta_features",
        type=str,
        default=str(RUTA_PROCESSED / "partidos_features_ml_con_elo.csv"),
        help="CSV con features (incluye Elo).",
    )
    parser.add_argument(
        "--ruta_modelo",
        type=str,
        default="modelos/modelo_logistico_1x2_con_elo_y_dif_forma.joblib",
        help="Modelo entrenado (pipeline sklearn).",
    )
    parser.add_argument(
        "--ruta_parametros",
        type=str,
        default="modelos/temperatura_calibracion.json",
        help="Dónde guardar T.",
    )
    parser.add_argument(
        "--ruta_reporte",
        type=str,
        default=str(RUTA_INFORMES / "reporte_modelo_calibrado.md"),
        help="Reporte markdown.",
    )
    args = parser.parse_args()

    ruta_features = Path(args.ruta_features)
    ruta_modelo = Path(args.ruta_modelo)
    ruta_param = Path(args.ruta_parametros)
    ruta_reporte = Path(args.ruta_reporte)

    if not ruta_features.exists():
        raise FileNotFoundError(f"No existe: {ruta_features}")
    if not ruta_modelo.exists():
        raise FileNotFoundError(f"No existe: {ruta_modelo}")

    asegurar_directorio(ruta_param.parent)
    asegurar_directorio(ruta_reporte.parent)

    df = pd.read_csv(ruta_features, parse_dates=["fecha_parseada"])
    df["resultado_norm"] = df["resultado_norm"].astype(str).str.strip().str.upper()

    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    X_val = df_val[FEATURES_NUMERICAS + FEATURES_CATEGORICAS]
    y_val = df_val["resultado_norm"].to_numpy()

    X_test = df_test[FEATURES_NUMERICAS + FEATURES_CATEGORICAS]
    y_test = df_test["resultado_norm"].to_numpy()

    pipe = joblib.load(ruta_modelo)

    # logits (decision_function) -> (n, k)
    logits_val = pipe.decision_function(X_val)
    logits_test = pipe.decision_function(X_test)

    # Orden de clases del pipe
    clases_pipe = list(pipe.classes_)
    idx = [clases_pipe.index(c) for c in CLASES_1X2]

    # Reordenamos logits al orden CLASES_1X2
    logits_val = logits_val[:, idx]
    logits_test = logits_test[:, idx]

    # Prob sin calibrar
    prob_val_base = softmax(logits_val)
    prob_test_base = softmax(logits_test)

    # Calibrar T en val
    T = calibrar_temperature(logits_val, y_val)

    # Prob calibradas
    prob_val_cal = softmax(logits_val / T)
    prob_test_cal = softmax(logits_test / T)

    met_val_base = metricas(y_val, prob_val_base)
    met_test_base = metricas(y_test, prob_test_base)
    met_val_cal = metricas(y_val, prob_val_cal)
    met_test_cal = metricas(y_test, prob_test_cal)

    ruta_param.write_text(json.dumps({"T": T}, indent=2), encoding="utf-8")

    rep = "# Reporte — Calibración (Temperature Scaling)\n\n"
    rep += f"Modelo base: `{ruta_modelo}`\n\n"
    rep += f"Temperatura aprendida en **validación**: **T = {T:.4f}**\n\n"

    rep += "## Métricas (antes vs después)\n\n"
    rep += "| Split | Variante | Logloss | Accuracy | Balanced Acc | Macro F1 |\n"
    rep += "|---|---|---:|---:|---:|---:|\n"
    rep += f"| Val | Base | {met_val_base['logloss']:.4f} | {met_val_base['accuracy']:.4f} | {met_val_base['balanced_accuracy']:.4f} | {met_val_base['macro_f1']:.4f} |\n"
    rep += f"| Val | Calibrado | {met_val_cal['logloss']:.4f} | {met_val_cal['accuracy']:.4f} | {met_val_cal['balanced_accuracy']:.4f} | {met_val_cal['macro_f1']:.4f} |\n"
    rep += f"| Test | Base | {met_test_base['logloss']:.4f} | {met_test_base['accuracy']:.4f} | {met_test_base['balanced_accuracy']:.4f} | {met_test_base['macro_f1']:.4f} |\n"
    rep += f"| Test | Calibrado | {met_test_cal['logloss']:.4f} | {met_test_cal['accuracy']:.4f} | {met_test_cal['balanced_accuracy']:.4f} | {met_test_cal['macro_f1']:.4f} |\n"

    rep += "\n## Matriz de confusión (Test — Base)\n\n"
    rep += formatear_matriz(met_test_base["matriz_confusion"]) + "\n"
    rep += "\n## Matriz de confusión (Test — Calibrado)\n\n"
    rep += formatear_matriz(met_test_cal["matriz_confusion"]) + "\n"

    ruta_reporte.write_text(rep, encoding="utf-8")
    print(f"OK ✅ Temperatura guardada: {ruta_param}")
    print(f"OK ✅ Reporte generado: {ruta_reporte}")


if __name__ == "__main__":
    main()
