# src/elo.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from baselines import CLASES_1X2


@dataclass
class BaselineEloSimple:
    """
    Elo simple con ventaja local que puede apagarse en partidos neutrales + empate dependiente de parejidad.

    - Elo inicial: 1500
    - ventaja_local: suma Elo al local antes de calcular probabilidades (si no es neutral)
    - k: velocidad de actualización
    - escala: convierte diferencia Elo a prob (logística clásica Elo)
    - Empate: p_e = base + extra * exp(-(dif/escala_empate)^2)
      (más empate cuando dif_elo ~ 0)

    Orden de columnas: CLASES_1X2 = ["E","L","V"]
    """

    elo_inicial: float = 1500.0
    ventaja_local: float = 60.0
    k: float = 20.0
    escala: float = 400.0

    # Empate dependiente de parejidad
    prob_empate_base: float = 0.22
    prob_empate_extra: float = 0.16
    escala_empate: float = 230.0

    epsilon: float = 1e-6

    # Reglas de neutralidad
    apagar_ventaja_en_final: bool = True
    apagar_ventaja_si_sede_distinta: bool = True

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

            ventaja = self._ventaja_por_partido(fila)

            prob, p_local_cond, dif_elo = self._predecir_interno(elo_l, elo_v, ventaja)
            prob_list.append(prob)

            # Actualización Elo estándar (usa expectativa binaria de victoria local)
            # Map resultado a score: L=1, E=0.5, V=0
            s = 0.5
            if res == "L":
                s = 1.0
            elif res == "V":
                s = 0.0

            # Expectativa de victoria local (Elo clásico)
            e = 1.0 / (1.0 + 10 ** (-dif_elo / self.escala))

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
        Predice probabilidades para df usando el diccionario de elos aprendido en train.
        No actualiza elos (para val/test).
        """
        prob_list = []
        for _, fila in df.iterrows():
            local = str(fila["equipo_local"])
            visita = str(fila["equipo_visitante"])

            elo_l = elos.get(local, self.elo_inicial)
            elo_v = elos.get(visita, self.elo_inicial)

            ventaja = self._ventaja_por_partido(fila)

            prob, _, _ = self._predecir_interno(elo_l, elo_v, ventaja)
            prob_list.append(prob)

        return np.vstack(prob_list)

    def _ventaja_por_partido(self, fila: pd.Series) -> float:
        """
        Devuelve ventaja_local efectiva del partido.
        Reglas:
        - Si fase == Final y apagar_ventaja_en_final => 0
        - Si hay pais_sede y pais_local y son distintos, y apagar_ventaja_si_sede_distinta => 0
        Caso contrario => self.ventaja_local
        """
        try:
            fase = str(fila.get("fase", "")).strip()
        except Exception:
            fase = ""

        if self.apagar_ventaja_en_final and fase.lower() == "final":
            return 0.0

        if self.apagar_ventaja_si_sede_distinta:
            pais_sede = str(fila.get("pais_sede", "")).strip()
            pais_local = str(fila.get("pais_local", "")).strip()
            if pais_sede and pais_local and (pais_sede.lower() != pais_local.lower()):
                return 0.0

        return float(self.ventaja_local)

    def _predecir_interno(
        self, elo_local: float, elo_visitante: float, ventaja: float
    ) -> Tuple[np.ndarray, float, float]:
        """
        Devuelve:
        - prob (1,3) en orden ["E","L","V"]
        - p_local_cond (condicional a no empate)
        - dif_elo efectivo (incluye ventaja)
        """
        dif_elo = (elo_local + ventaja) - elo_visitante

        # Prob de victoria local condicional a "no empate"
        p_local_cond = 1.0 / (1.0 + 10 ** (-dif_elo / self.escala))

        # Empate depende de parejidad (dif cerca de 0 => más empate)
        p_e = self.prob_empate_base + self.prob_empate_extra * float(np.exp(- (dif_elo / self.escala_empate) ** 2))
        p_e = min(max(p_e, self.epsilon), 0.55)

        p_no_e = 1.0 - p_e
        p_l = p_no_e * p_local_cond
        p_v = p_no_e * (1.0 - p_local_cond)

        prob = np.array([p_e, p_l, p_v], dtype=float)  # ["E","L","V"]

        prob = np.clip(prob, self.epsilon, 1.0)
        prob = prob / prob.sum()

        return prob.reshape(1, -1), float(p_local_cond), float(dif_elo)
