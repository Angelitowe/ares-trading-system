"""
Regime Engine — Hidden Markov Model for market state detection.

States
------
0 = Low-vol / Risk-on      (typical bull market)
1 = High-vol / Risk-off    (crisis / bear market)

Features used for regime detection
-----------------------------------
- 22-day realised vol
- 5-day vol change (momentum)
- VIX level
- Skew (semivariance up/dn ratio)
- Term-structure slope

Uses hmmlearn's GaussianHMM if available, otherwise a threshold-based
classifier built on rolling vol rank (percentile).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegimeState:
    date: pd.Timestamp
    state: int                   # 0 = low-vol, 1 = high-vol
    prob_low: float              # posterior probability P(state=0 | obs)
    prob_high: float             # posterior probability P(state=1 | obs)
    vol_percentile: float        # rolling vol percentile [0, 1]
    label: str                   # "low_vol" | "high_vol"


class RegimeEngine:
    """
    Detect market regime (low-vol vs high-vol) using HMM.

    Parameters
    ----------
    n_states   : number of HMM states (default 2)
    n_iter     : EM iterations for HMM
    roll_window: rolling window for percentile fallback
    """

    def __init__(
        self,
        n_states: int = 2,
        n_iter: int = 100,
        roll_window: int = 252,
    ) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.roll_window = roll_window
        self._model = None
        self._backend = "none"
        self._high_vol_state = 1    # determined post-fit
        self._mid_vol_state: Optional[int] = None  # only used when n_states == 3
        self._low_vol_state: int = 0   # determined post-fit
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None

    def fit(
        self,
        rv_series: pd.Series,
        iv_series: Optional[pd.Series] = None,
        vix_slope: Optional[pd.Series] = None,
    ) -> "RegimeEngine":
        """
        Fit the regime model.

        Parameters
        ----------
        rv_series  : daily realised vol (annualised, decimal)
        iv_series  : implied vol (optional)
        vix_slope  : VIX term structure slope (optional)
        """
        features = self._build_features(rv_series, iv_series, vix_slope)
        if features.empty:
            raise ValueError("Not enough data to fit regime model")

        # Robust standardisation improves HMM stability over long horizons.
        X = features.values.astype(float)
        self._feature_mean = np.nanmean(X, axis=0)
        self._feature_std = np.nanstd(X, axis=0)
        self._feature_std = np.where(self._feature_std < 1e-8, 1.0, self._feature_std)
        X_scaled = (X - self._feature_mean) / self._feature_std

        try:
            self._fit_hmm(features.index, X_scaled)
            self._backend = "hmm"
        except Exception as exc:
            logger.info("HMM unavailable (%s), using threshold fallback", exc)
            self._fit_threshold(features)
            self._backend = "threshold"

        self._fit_features = features
        return self

    def predict(
        self,
        rv_series: pd.Series,
        iv_series: Optional[pd.Series] = None,
        vix_slope: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Predict regime states for the given series.

        Returns DataFrame with columns [state, prob_low, prob_high,
                                        vol_percentile, label].
        """
        features = self._build_features(rv_series, iv_series, vix_slope)
        if features.empty:
            return pd.DataFrame(columns=["state", "prob_low", "prob_high", "vol_percentile", "label"])

        X = features.values.astype(float)
        if self._feature_mean is not None and self._feature_std is not None:
            X = (X - self._feature_mean) / self._feature_std

        if self._backend == "hmm":
            states, probs = self._predict_hmm(X)
        else:
            states, probs = self._predict_threshold(features, rv_series)

        vol_pct = rv_series.rank(pct=True).reindex(features.index)

        rows = []
        for i, date in enumerate(features.index):
            state = int(states[i])
            # prob_high = P(stressed) = P(mid_vol or high_vol) = 1 - P(low_vol)
            # This drives the smooth regime_scale in strategy_engine correctly for both
            # slow-burn (mid_vol) and spike (high_vol) stress regimes.
            if self.n_states == 3 and self._mid_vol_state is not None:
                pl = float(probs[i, self._low_vol_state])
                ph = 1.0 - pl  # P(stressed) = P(mid) + P(high)
                if state == self._high_vol_state:
                    label = "high_vol"
                elif state == self._mid_vol_state:
                    label = "mid_vol"
                else:
                    label = "low_vol"
            else:
                ph = float(probs[i, self._high_vol_state])
                pl = 1.0 - ph
                label = "high_vol" if state == self._high_vol_state else "low_vol"
            rows.append({
                "state": state,
                "prob_low": pl,
                "prob_high": ph,
                "vol_percentile": float(vol_pct.iloc[i] if i < len(vol_pct) else 0.5),
                "label": label,
            })

        return pd.DataFrame(rows, index=features.index)

    def regime_multiplier(self, state_df: pd.DataFrame) -> pd.Series:
        """
        Scale-factor for position sizing based on regime.
        0.5 in high-vol (reduce risk), 1.0 in low-vol.
        """
        mult = np.where(state_df["label"] == "high_vol", 0.5, 1.0)
        return pd.Series(mult, index=state_df.index, name="regime_mult")

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _build_features(
        self,
        rv: pd.Series,
        iv: Optional[pd.Series],
        slope: Optional[pd.Series],
    ) -> pd.DataFrame:
        df = pd.DataFrame({"rv": rv})
        df["rv22"] = rv.rolling(22).mean()
        df["rv5"] = rv.rolling(5).mean()
        df["vol_mom"] = (df["rv5"] - df["rv22"]) / (df["rv22"] + 1e-10)
        df["log_rv"] = np.log(rv + 1e-10)

        if iv is not None:
            df["iv"] = iv.reindex(rv.index).ffill()
        else:
            df["iv"] = df["rv22"]

        if slope is not None:
            df["slope"] = slope.reindex(rv.index).ffill().fillna(0.0)
        else:
            df["slope"] = 0.0

        return df[["rv22", "vol_mom", "log_rv", "iv", "slope"]].dropna()

    # ------------------------------------------------------------------
    # HMM backend
    # ------------------------------------------------------------------

    def _fit_hmm(self, index: pd.Index, X: np.ndarray) -> None:
        from hmmlearn.hmm import GaussianHMM

        best_model = None
        best_score = -np.inf
        seeds = (7, 13, 29)
        for seed in seeds:
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=self.n_iter,
                random_state=seed,
                min_covar=1e-4,
            )
            model.fit(X)
            score = float(model.score(X))
            if score > best_score:
                best_score = score
                best_model = model

        if best_model is None:
            raise RuntimeError("HMM fit failed for all initializations")

        # Identify states by rv22 mean (ascending): low < mid < high.
        rv22_means = best_model.means_[:, 0]
        sorted_states = np.argsort(rv22_means)  # indices sorted low→high
        self._high_vol_state = int(sorted_states[-1])
        self._low_vol_state = int(sorted_states[0])
        if self.n_states == 3:
            self._mid_vol_state = int(sorted_states[1])
        else:
            self._mid_vol_state = None
        self._model = best_model
        logger.info(
            "HMM fitted: n_states=%d, low=%d, mid=%s, high=%d, ll=%.2f, rv22_means=%s",
            self.n_states,
            self._low_vol_state,
            str(self._mid_vol_state),
            self._high_vol_state,
            best_score,
            rv22_means,
        )

    def _predict_hmm(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        states = self._model.predict(X)
        probs = self._model.predict_proba(X)
        return states, probs

    # ------------------------------------------------------------------
    # Threshold fallback
    # ------------------------------------------------------------------

    def _fit_threshold(self, features: pd.DataFrame) -> None:
        """Simple percentile-based threshold fallback."""
        self._threshold = float(features["rv22"].quantile(0.60))
        self._high_vol_state = 1
        self._low_vol_state = 0
        self._mid_vol_state = None

    def _predict_threshold(
        self, features: pd.DataFrame, rv_series: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        rv22 = features["rv22"].values
        states = (rv22 > self._threshold).astype(int)
        # Soft probabilities using sigmoid distance from threshold
        scaled = (rv22 - self._threshold) / (rv22.std() + 1e-10)
        p_high = 1 / (1 + np.exp(-scaled))
        probs = np.column_stack([1 - p_high, p_high])
        return states, probs
