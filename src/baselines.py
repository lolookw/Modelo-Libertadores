# src/baselines.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# Usamos orden lexicográfico para alinear con sklearn: E < L < V
CLASES_1X2: List[str] = ["E", "L", "V"]


def _probabilidades_suavizadas(prob: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Evita ceros exactos para que la logloss no explote.
    Aplica clipping y renormaliza para que sumen 1.
    """
    prob = np.clip(prob, epsilon, 1.0)
    prob = prob / prob.sum(axis=1, keepdims=True)
    return prob


@dataclass
class BaselineSiempreLocal:
    """
    Baseline 1: siempre predice 'L' (gana local) con suavizado para logloss.
    """
    epsilon: float = 1e-6

    def predecir_probabilidades(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        prob = np.zeros((n, 3), dtype=float)
        # Orden CLASES_1X2 = ["E","L","V"]
        prob[:, 0] = self.epsilon
        prob[:, 1] = 1.0 - 2 * self.epsilon
        prob[:, 2] = self.epsilon
        return _probabilidades_suavizadas(prob, self.epsilon)



@dataclass
class BaselineFrecuenciasPorFase:
    """
    Baseline 2: aprende P(L/E/V) por 'fase' usando SOLO train.
    Si una fase no existe en train, hace fallback al prior global de train.
    """
    epsilon: float = 1e-6
    tabla_fase_a_prob: Dict[str, np.ndarray] | None = None
    prior_global: np.ndarray | None = None

    def entrenar(self, df_train: pd.DataFrame) -> "BaselineFrecuenciasPorFase":
        if "fase" not in df_train.columns or "resultado_norm" not in df_train.columns:
            raise ValueError("df_train debe contener columnas: 'fase' y 'resultado_norm'")

        # Prior global en train
        conteo_global = df_train["resultado_norm"].value_counts()
        prior = np.array([conteo_global.get(c, 0) for c in CLASES_1X2], dtype=float)
        prior = prior / max(prior.sum(), 1.0)
        self.prior_global = prior

        # Distribución por fase
        tabla: Dict[str, np.ndarray] = {}
        for fase, grupo in df_train.groupby("fase"):
            conteo = grupo["resultado_norm"].value_counts()
            prob = np.array([conteo.get(c, 0) for c in CLASES_1X2], dtype=float)
            prob = prob / max(prob.sum(), 1.0)
            tabla[str(fase)] = prob

        self.tabla_fase_a_prob = tabla
        return self

    def predecir_probabilidades(self, df: pd.DataFrame) -> np.ndarray:
        if self.tabla_fase_a_prob is None or self.prior_global is None:
            raise RuntimeError("Primero tenés que llamar a .entrenar(df_train).")

        n = len(df)
        prob = np.zeros((n, 3), dtype=float)

        for i, fase in enumerate(df["fase"].astype(str).tolist()):
            if fase in self.tabla_fase_a_prob:
                prob[i, :] = self.tabla_fase_a_prob[fase]
            else:
                prob[i, :] = self.prior_global

        return _probabilidades_suavizadas(prob, self.epsilon)


def probabilidades_a_clase(prob: np.ndarray) -> np.ndarray:
    """
    Toma matriz (n,3) y devuelve clase argmax en CLASES_1X2.
    """
    idx = prob.argmax(axis=1)
    return np.array([CLASES_1X2[i] for i in idx], dtype=object)
