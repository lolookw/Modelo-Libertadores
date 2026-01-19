# src/evaluacion.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
)


from baselines import (
    CLASES_1X2,
    BaselineSiempreLocal,
    BaselineFrecuenciasPorFase,
    probabilidades_a_clase,
)
from configuracion import RUTA_PROCESSED, RUTA_INFORMES
from utilidades import asegurar_directorio
from elo import BaselineEloSimple



def cargar_dataset_modelo(ruta: Path) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    # chequeos mínimos
    necesarias = {"split", "fase", "resultado_norm"}
    faltan = necesarias - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en el dataset processed: {sorted(list(faltan))}")
    df["resultado_norm"] = df["resultado_norm"].astype(str).str.strip().str.upper()
    return df


def _evaluar_predicciones(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, object]:
    y_pred = probabilidades_a_clase(prob)

    metricas = {
        "logloss": float(log_loss(y_true, prob, labels=CLASES_1X2)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=CLASES_1X2)),
        "matriz_confusion": confusion_matrix(y_true, y_pred, labels=CLASES_1X2),
    }

    return metricas


def _formatear_matriz_confusion(cm: np.ndarray) -> str:
    etiquetas = CLASES_1X2
    encabezado = "| real\\pred | " + " | ".join(etiquetas) + " |"
    separador = "|---|" + "|".join(["---:"] * len(etiquetas)) + "|"

    filas = [encabezado, separador]
    for i, etiqueta_real in enumerate(etiquetas):
        valores = " | ".join(str(int(cm[i, j])) for j in range(len(etiquetas)))
        filas.append(f"| {etiqueta_real} | {valores} |")

    return "\n".join(filas)



def _agregar_bloque_reporte(texto: str, titulo: str, metricas_val: Dict, metricas_test: Dict) -> str:
    texto += f"\n## {titulo}\n\n"
    texto += "| Split | Logloss | Accuracy | Balanced Acc | Macro F1 |\n"
    texto += "|---|---:|---:|---:|---:|\n"
    texto += f"| Val | {metricas_val['logloss']:.4f} | {metricas_val['accuracy']:.4f} | {metricas_val['balanced_accuracy']:.4f} | {metricas_val['macro_f1']:.4f} |\n"
    texto += f"| Test | {metricas_test['logloss']:.4f} | {metricas_test['accuracy']:.4f} | {metricas_test['balanced_accuracy']:.4f} | {metricas_test['macro_f1']:.4f} |\n"

    texto += "\n**Matriz de confusión (Val)**\n\n"
    texto += _formatear_matriz_confusion(metricas_val["matriz_confusion"]) + "\n"

    texto += "\n**Matriz de confusión (Test)**\n\n"
    texto += _formatear_matriz_confusion(metricas_test["matriz_confusion"]) + "\n"

    return texto


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa baselines para 1X2 (L/E/V).")
    parser.add_argument(
        "--ruta_processed",
        type=str,
        default=str(RUTA_PROCESSED / "partidos_modelo_1x2.csv"),
        help="Ruta al CSV processed.",
    )
    parser.add_argument(
        "--ruta_reporte",
        type=str,
        default=str(RUTA_INFORMES / "reporte_resultados.md"),
        help="Ruta del reporte markdown de salida.",
    )
    args = parser.parse_args()

    ruta_processed = Path(args.ruta_processed)
    ruta_reporte = Path(args.ruta_reporte)

    if not ruta_processed.exists():
        raise FileNotFoundError(f"No existe el processed: {ruta_processed}")

    asegurar_directorio(ruta_reporte.parent)

    df = cargar_dataset_modelo(ruta_processed)

    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    y_val = df_val["resultado_norm"].to_numpy()
    y_test = df_test["resultado_norm"].to_numpy()

    reporte = "# Reporte de resultados — Baselines (1X2)\n"
    reporte += "\nEste reporte evalúa baselines en validación y test. Métrica principal: **logloss**.\n"
    reporte += "\nClases: `L` (local), `E` (empate), `V` (visitante)\n"

    # ===== Baseline 1: Siempre Local =====
    bl1 = BaselineSiempreLocal(epsilon=1e-6)
    prob_val_bl1 = bl1.predecir_probabilidades(df_val)
    prob_test_bl1 = bl1.predecir_probabilidades(df_test)

    metricas_val_bl1 = _evaluar_predicciones(y_val, prob_val_bl1)
    metricas_test_bl1 = _evaluar_predicciones(y_test, prob_test_bl1)

    reporte = _agregar_bloque_reporte(
        reporte,
        "Baseline 1 — Siempre Local",
        metricas_val_bl1,
        metricas_test_bl1,
    )

    # ===== Baseline 2: Frecuencias por fase =====
    bl2 = BaselineFrecuenciasPorFase(epsilon=1e-6).entrenar(df_train)
    prob_val_bl2 = bl2.predecir_probabilidades(df_val)
    prob_test_bl2 = bl2.predecir_probabilidades(df_test)

    metricas_val_bl2 = _evaluar_predicciones(y_val, prob_val_bl2)
    metricas_test_bl2 = _evaluar_predicciones(y_test, prob_test_bl2)

    reporte = _agregar_bloque_reporte(
        reporte,
        "Baseline 2 — Frecuencias por fase (train → val/test)",
        metricas_val_bl2,
        metricas_test_bl2,
    )

        # ===== Baseline 3: Elo simple =====
    bl3 = BaselineEloSimple(
        elo_inicial=1500.0,
        ventaja_local=60.0,
        k=20.0,
        escala=400.0,
        prob_empate=0.25,
        epsilon=1e-6,
    )

    # Entrenamos elos recorriendo train en orden temporal (sin leakage)
    _, elos_finales = bl3.predecir_y_actualizar_en_train(df_train)

    prob_val_bl3 = bl3.predecir_con_elos_fijos(df_val, elos_finales)
    prob_test_bl3 = bl3.predecir_con_elos_fijos(df_test, elos_finales)

    metricas_val_bl3 = _evaluar_predicciones(y_val, prob_val_bl3)
    metricas_test_bl3 = _evaluar_predicciones(y_test, prob_test_bl3)

    reporte = _agregar_bloque_reporte(
        reporte,
        "Baseline 3 — Elo simple (elos aprendidos en train)",
        metricas_val_bl3,
        metricas_test_bl3,
    )


    ruta_reporte.write_text(reporte, encoding="utf-8")
    print(f"OK ✅ Reporte generado: {ruta_reporte}")


if __name__ == "__main__":
    main()
