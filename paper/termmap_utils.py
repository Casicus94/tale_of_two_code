"""Shared helpers for Pareto-front term-presence maps (best-equation-first method)."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

FUNCS = {"sqrt", "log", "exp", "abs", "sin", "cos", "tan", "square", "cube", "inv"}
DATA = Path(__file__).resolve().parents[2] / "Data" / "processed" / "colombia"


def load_front(path):
    d = pd.read_csv(path)
    return d.sort_values("complexity").reset_index(drop=True)


def equation_variables(eq: str):
    toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", eq))
    return {t for t in toks if t not in FUNCS}


def term_presence(front: pd.DataFrame, variables=None):
    """variables x complexity 0/1 matrix (DataFrame), rows ordered by first appearance."""
    if variables is None:
        variables = sorted(set().union(*[equation_variables(e) for e in front["equation"]]))
    mat = pd.DataFrame(0, index=variables, columns=front["complexity"].astype(int).tolist())
    for _, r in front.iterrows():
        vs = equation_variables(r["equation"])
        for v in variables:
            if v in vs:
                mat.loc[v, int(r["complexity"])] = 1
    first = mat.apply(lambda row: (row.to_numpy().nonzero()[0][0] if row.sum() else 10**6), axis=1)
    return mat.loc[first.sort_values(kind="stable").index]


def term_summary(mat: pd.DataFrame):
    """Per-variable: first complexity, levels present, and whether present at EVERY
    complexity from the first appearance to the top of the front ('continuous')."""
    cols = mat.columns.to_numpy()
    rows = []
    for v, r in mat.iterrows():
        idx = np.nonzero(r.to_numpy())[0]
        if len(idx) == 0:
            continue
        span = r.to_numpy()[idx[0]:]
        rows.append(dict(variable=v, first_complexity=int(cols[idx[0]]), last_complexity=int(cols[idx[-1]]),
                         n_levels_present=int(r.sum()), n_levels_from_first=int(len(span)),
                         frac_present_from_first=float(span.mean()), continuous_to_top=bool(span.all())))
    return pd.DataFrame(rows)
