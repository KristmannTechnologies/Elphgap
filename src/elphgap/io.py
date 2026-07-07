"""Loader for the BETE-NET training database (github.com/henniggroup/BETE-NET).

The file is a column-oriented pandas JSON dump: each column maps row-index
strings to values. Relevant columns: comp, comp_name, Freq_meV (alpha^2F
frequency grid), a2F, and the stored aggregates lambda, w_log, w_sq (meV).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class Material:
    index: int
    comp: str
    comp_name: str
    omega: np.ndarray  # meV, ascending
    a2f: np.ndarray
    lambda_ref: float
    wlog_ref: float  # meV
    wsq_ref: float  # meV, sqrt(<omega^2>)


def load_database(path: str) -> list[Material]:
    with open(path) as f:
        db = json.load(f)
    materials = []
    for i in db["comp"]:
        omega = np.asarray(db["Freq_meV"][i], dtype=np.float64)
        a2f = np.asarray(db["a2F"][i], dtype=np.float64)
        # Guard against non-physical grid points (omega <= 0 would blow up 1/omega moments).
        mask = omega > 0
        materials.append(
            Material(
                index=int(i),
                comp=db["comp"][i],
                comp_name=db["comp_name"][i],
                omega=omega[mask],
                a2f=a2f[mask],
                lambda_ref=float(db["lambda"][i]),
                wlog_ref=float(db["w_log"][i]),
                wsq_ref=float(db["w_sq"][i]),
            )
        )
    materials.sort(key=lambda m: m.index)
    return materials
