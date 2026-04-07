"""
Random Forest link-quality estimator with isotonic calibration.

Trains one RF regressor per link class (satellite, uav_terrestrial,
sea_surface, terrestrial), calibrates outputs to [0,1] with isotonic
regression, and validates with blocked K-fold splits over time.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, r2_score

FEATURE_COLS = [
    "RSSI", "SNR", "SINR", "LQI",
    "mu_SINR", "sigma_SINR", "mu_RSSI", "sigma_RSSI",
    "doppler", "type_pair", "dwell",
]
LABEL_COL = "prr_emp"
LINK_CLASSES = ["satellite", "uav_terrestrial", "sea_surface", "terrestrial"]


class LinkQualityEstimator:
    """Per-class RF regressor + isotonic calibration -> Q_ij = PRRhat."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 12,
                 random_state: int = 42):
        self._rf: Dict[str, RandomForestRegressor] = {}
        self._iso: Dict[str, IsotonicRegression] = {}
        self._n_est = n_estimators
        self._max_depth = max_depth
        self._rs = random_state
        self._trained = False

    # ─── training ─────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Train one RF per link class and calibrate with isotonic regression.

        Returns per-class validation metrics {class: {R2, MAE}}.
        """
        metrics: Dict[str, Dict[str, float]] = {}
        for lc in LINK_CLASSES:
            sub = df[df["link_class"] == lc].copy()
            if len(sub) < 50:
                metrics[lc] = {"R2": 0.0, "MAE": 1.0, "n_samples": len(sub)}
                continue

            X = sub[FEATURE_COLS].values.astype(np.float64)
            y = sub[LABEL_COL].values.astype(np.float64)
            np.nan_to_num(X, copy=False)

            val_metrics = self._blocked_kfold_train(lc, X, y, n_folds=5)
            metrics[lc] = val_metrics

        self._trained = True
        return metrics

    def _blocked_kfold_train(self, lc: str, X: np.ndarray, y: np.ndarray,
                             n_folds: int) -> Dict[str, float]:
        """Blocked K-fold (temporal ordering preserved) for validation,
        then retrain on full data for deployment."""
        n = len(y)
        fold_size = n // n_folds
        all_pred, all_true = [], []

        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n
            mask = np.ones(n, dtype=bool)
            mask[val_start:val_end] = False

            X_tr, y_tr = X[mask], y[mask]
            X_val, y_val = X[~mask], y[~mask]

            rf = RandomForestRegressor(
                n_estimators=self._n_est, max_depth=self._max_depth,
                random_state=self._rs, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            raw_pred = rf.predict(X_val)

            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            iso.fit(rf.predict(X_tr), y_tr)
            cal_pred = iso.predict(raw_pred)

            all_pred.extend(cal_pred.tolist())
            all_true.extend(y_val.tolist())

        r2 = r2_score(all_true, all_pred)
        mae = mean_absolute_error(all_true, all_pred)

        rf_full = RandomForestRegressor(
            n_estimators=self._n_est, max_depth=self._max_depth,
            random_state=self._rs, n_jobs=-1)
        rf_full.fit(X, y)
        iso_full = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso_full.fit(rf_full.predict(X), y)

        self._rf[lc] = rf_full
        self._iso[lc] = iso_full

        return {"R2": float(r2), "MAE": float(mae),
                "n_samples": len(y),
                "pred": np.array(all_pred),
                "true": np.array(all_true)}

    # ─── prediction ───────────────────────────────────────────────────

    def predict_prr(self, link_class: str,
                    features: np.ndarray) -> np.ndarray:
        """Predict calibrated PRRhat for a batch of feature vectors.

        *features* shape: (n_samples, 11) matching FEATURE_COLS order.
        Returns shape: (n_samples,) in [0, 1].
        """
        if not self._trained or link_class not in self._rf:
            return np.full(features.shape[0], 0.5)
        raw = self._rf[link_class].predict(features)
        return self._iso[link_class].predict(raw)

    def predict_single(self, link_class: str,
                       rssi: float, snr: float, sinr: float, lqi: int,
                       mu_sinr: float, sigma_sinr: float,
                       mu_rssi: float, sigma_rssi: float,
                       doppler: float, type_pair: int,
                       dwell: int) -> float:
        """Predict Q_ij for a single link."""
        x = np.array([[rssi, snr, sinr, lqi, mu_sinr, sigma_sinr,
                        mu_rssi, sigma_rssi, doppler, type_pair, dwell]],
                     dtype=np.float64)
        return float(self.predict_prr(link_class, x)[0])

    # ─── persistence ──────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"rf": self._rf, "iso": self._iso,
                         "trained": self._trained}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._rf = data["rf"]
        self._iso = data["iso"]
        self._trained = data["trained"]

    @property
    def is_trained(self) -> bool:
        return self._trained

    def feature_importances(self, link_class: str) -> Optional[np.ndarray]:
        rf = self._rf.get(link_class)
        if rf is None:
            return None
        return rf.feature_importances_
