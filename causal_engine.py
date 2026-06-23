"""causal_engine.py — CausalForestDML 기반 ATE/CATE 추정"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def run_ate_cate(df: pd.DataFrame, cfg: Dict) -> Dict[str, Any]:
    """
    CausalForestDML로 ATE / CATE 추정.

    Returns
    -------
    dict with keys: estimator, ate, ate_ci, cate, W_cols, n
    """
    from econml.dml import CausalForestDML
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression

    T_col = cfg["T"]
    W_cols = [c for c in cfg["W"] if c in df.columns]
    Y_col = cfg["Y"]

    sub = df[[T_col] + W_cols + [Y_col]].dropna()

    T = sub[T_col].values.astype(float)
    W = sub[W_cols].values.astype(float)
    Y = sub[Y_col].values.astype(float)

    t_binary = cfg.get("T_binary", False) or set(np.unique(T)).issubset({0.0, 1.0})

    model_t = LogisticRegression(max_iter=500) if t_binary else GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_y = GradientBoostingRegressor(n_estimators=100, random_state=42)

    est = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=200,
        min_samples_leaf=10,
        cv=3,
        random_state=42,
        discrete_treatment=t_binary,
    )
    est.fit(Y, T, X=W)

    ate = float(est.ate(W))
    lb, ub = est.ate_interval(W, alpha=0.05)
    cate = est.effect(W)

    return {
        "estimator": est,
        "ate": ate,
        "ate_ci": (float(lb), float(ub)),
        "cate": cate,
        "W_cols": W_cols,
        "n": len(sub),
    }


def point_cate(estimator, w_values: np.ndarray) -> float:
    """단일 포인트 X에 대한 CATE 반환."""
    return float(estimator.effect(w_values.reshape(1, -1))[0])
