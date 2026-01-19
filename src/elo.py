# src/elo.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from baselines import CLASES_1X2


def _logistica(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class BaselineEloSimple:
    """
    Elo simple con ventaja local + reparto de empate.

    - Elo inicial: 1500
    - ventaja_local: suma Elo al local antes de calcular probabilidades
    - k: velocidad de actualización
    - escala: convierte diferencia Elo a prob (más grande => curva más suave)
    - prob_empate: prob fija de empate (baseline simple pero sorprendentemente útil)

    Probabilidades:
      p_empate = prob_empate
      p_no_empate = 1 - p_empate
      p_local_cond = logistic(dif_elo / escala)
      p_local = p_no_empate * p_local_cond
      p_visitante = p_no_empate * (1 - p_local_cond)

    Orden de columnas: CLASES_1X2 = ["E","L","V"] (importante)
    """
    elo_inicial: float = 1500.0
    ventaja_local: float = 60.0
    k: float = 20.0
    escala: float = 400.0
    prob_empate: float = 0.25
    epsilon: float = 1e-6

    def predecir_y_actualizar_en_train(
        self, df_train: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Recorre train en orden temporal:
        - genera probabilidades pre-partido
        - actualiza el Elo con el resultado real
        Devuelve df_train con columnas prob_* y el diccionario final de elos.
        """
        elos: Dict[str, float] = {}
        prob_list = []

        for _, fila in df_train.iterrows():
            local = str(fila["equipo_local"])
            visita = str(fila["equipo_visitante"])
            res = str(fila["resultado_norm"]).upper().strip()

            elo_l = elos.get(local, self.elo_inicial)
            elo_v = elos.get(visita, self.elo_inicial)

            prob = self._predecir_probabilidades_desde_elos(elo_l, elo_v)
            prob_list.append(prob)

            # Actualización Elo usando resultado real
            # Resultado binario para Elo base: L=1, E=0.5, V=0
            s = 0.5
            if res == "L":
                s = 1.0
            elif res == "V":
                s = 0.0

            # Prob esperada de "gana local" (sin empates) para update estándar
            dif = (elo_l + self.ventaja_local) - elo_v
            e = 1.0 / (1.0 + 10 ** (-dif / self.escala))

            elo_l = elo_l + self.k * (s - e)
            elo_v = elo_v + self.k * ((1.0 - s) - (1.0 - e))

            elos[local] = elo_l
            elos[visita] = elo_v

        prob_arr = np.vstack(prob_list)
        out = df_train.copy()
        out["prob_E"] = prob_arr[:, CLASES_1X2.index("E")]
        out["prob_L"] = prob_arr[:, CLASES_1X2.index("L")]
        out["prob_V"] = prob_arr[:, CLASES_1X2.index("V")]

        return out, elos

    def predecir_con_elos_fijos(self, df: pd.DataFrame, elos: Dict[str, float]) -> np.ndarray:
        """
        Predice probabilidades para df usando el diccionario de elos ya aprendido en train.
        No actualiza elos (para val/test).
        """
        prob_list = []
        for _, fila in df.iterrows():
            local = str(fila["equipo_local"])
            visita = str(fila["equipo_visitante"])

            elo_l = elos.get(local, self.elo_inicial)
            elo_v = elos.get(visita, self.elo_inicial)

            prob_list.append(self._predecir_probabilidades_desde_elos(elo_l, elo_v))

        return np.vstack(prob_list)

    def _predecir_probabilidades_desde_elos(self, elo_local: float, elo_visitante: float) -> np.ndarray:
        dif = (elo_local + self.ventaja_local) - elo_visitante

        # prob de local condicional a "no empate"
        p_local_cond = 1.0 / (1.0 + 10 ** (-dif / self.escala))

        p_e = float(self.prob_empate)
        p_no_e = 1.0 - p_e
        p_l = p_no_e * p_local_cond
        p_v = p_no_e * (1.0 - p_local_cond)

        prob = np.array([p_e, p_l, p_v], dtype=float)  # ["E","L","V"]

        # suavizado anti logloss infinita
        prob = np.clip(prob, self.epsilon, 1.0)
        prob = prob / prob.sum()

        return prob.reshape(1, -1)
