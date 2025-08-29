"""Transformer explanation mapper (TXM) for RF→Transformer consistency."""

import logging
from typing import Optional, Sequence, Dict, Any
import numpy as np
import pandas as pd
import torch
from ..models.encoders import dict_to_transformer_tensors
from .cross_model_attribution_fidelity_metrics import (
    sign_fidelity, rank_fidelity, prob_monotonicity
)
from ..models.encoders import dict_to_transformer_tensors

def create_feature_vector_from_event(evt, feature_names, feature_lists, embed_maps):
    """Kept for compatibility; not used by the mapper anymore."""
    row = []
    for name in feature_names:
        v = evt.get(name, None)
        if name in feature_lists['CONTINUOUS_USED']:
            try: row.append(float(v) if v is not None else 0.0)
            except (ValueError, TypeError): row.append(0.0)
        elif name in feature_lists['BOOLEAN_USED']:
            row.append((v.lower() in ('true','1','yes','on')) if isinstance(v, str) else bool(v))
        elif name in (feature_lists['HIGH_CAT_USED'] + feature_lists['LOW_CAT_USED']):
            row.append(str(v) if v is not None else "unknown")
        else:
            try: row.append(float(v) if v is not None else 0.0)
            except (ValueError, TypeError): row.append(0.0)
    return np.array(row, dtype=object)


class TransformerExplanationMapper:
    def __init__(
        self,
        rf_explainer,                    # shap.TreeExplainer on the RF ESTIMATOR (post-preprocess space)
        transformer_model,               # your pytorch model
        feature_lists: Dict[str, Sequence[str]],
        features_order: Sequence[str],   # training-time raw feature order
        embed_maps: Dict[str, Any],
        preprocessor=None,               # ColumnTransformer used in training
        rf_pipeline=None,                # optional: full imblearn/sklearn pipeline
        rf_estimator=None,               # REQUIRED: RandomForestClassifier trained on preprocessed matrix
        calibrator: Optional['TXMCalibrator']=None,
    ):
        self.rf_explainer = rf_explainer
        self.transformer_model = transformer_model
        self.feature_lists = feature_lists
        self.features_order = list(features_order)
        self.embed_maps = embed_maps
        self.preprocessor = preprocessor
        self.rf_pipeline = rf_pipeline
        self.rf_estimator = rf_estimator if rf_estimator is not None else getattr(rf_explainer, "model", None)
        self.calibrator = calibrator

        if self.rf_estimator is None:
            raise ValueError("rf_estimator is required for TXM (use the trained RandomForestClassifier).")
        if self.preprocessor is None:
            logging.warning("TXM initialized without a preprocessor; ensure rf_pipeline contains preprocessing.")

        # no-op hook; kept for compatibility
        self._compute_feature_correlation()

    def _compute_feature_correlation(self):
        self.feature_weights = {f: 1.0 for f in self.features_order}

    # ---------- dataframe & transformed matrix ----------
    def _to_dataframe(self, X: np.ndarray) -> pd.DataFrame:
        # X is (B, len(features_order)) raw – build a DF with correct columns
        if X.ndim != 2 or X.shape[1] != len(self.features_order):
            raise ValueError(f"Expected X shape (B,{len(self.features_order)}), got {X.shape}")
        df = pd.DataFrame(X, columns=self.features_order)
        # Make sure object dtypes stay object (strings) for cats; numerics as float
        for c in (self.feature_lists["CONTINUOUS_USED"] + self.feature_lists["BOOLEAN_USED"]):
            if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        for c in (self.feature_lists["HIGH_CAT_USED"] + self.feature_lists["LOW_CAT_USED"]):
            if c in df: df[c] = df[c].astype(str).fillna("unknown")
        return df


    def _to_estimator_matrix(self, X_df: pd.DataFrame):
        """
        Transform raw feature DataFrame (B x raw_features) into the matrix expected by
        the RF estimator (B x F_preprocessed). Prefer the explicit preprocessor; if
        missing, try to extract it from rf_pipeline.
        """
        # 1) Prefer explicit preprocessor
        if self.preprocessor is not None:
            X_trans = self.preprocessor.transform(X_df)
            # If sparse, SHAP/estimator can handle CSR; if needed:
            # X_trans = X_trans.toarray() if hasattr(X_trans, "toarray") else X_trans
            return X_trans

        # 2) Try to pull 'pre' step from pipeline
        if self.rf_pipeline is not None:
            pre = None
            if hasattr(self.rf_pipeline, "named_steps") and "pre" in self.rf_pipeline.named_steps:
                pre = self.rf_pipeline.named_steps["pre"]
            else:
                # imblearn/sklearn Pipeline supports slicing to drop the final estimator
                try:
                    pre = self.rf_pipeline[:-1]
                except Exception:
                    pre = None

            if pre is not None and hasattr(pre, "transform"):
                X_trans = pre.transform(X_df)
                return X_trans

        # 3) No way to transform -> error
        raise ValueError("No preprocessor available to transform raw features for the RF estimator.")


    # ---------- RF predictions in the correct space ----------
    def _rf_predict_proba(self, X_df: pd.DataFrame, X_est: Optional[np.ndarray]=None) -> np.ndarray:
        if X_est is None:
            X_est = self._to_estimator_matrix(X_df)
        return self.rf_estimator.predict_proba(X_est)

    def _predicted_classes(self, X_df: pd.DataFrame, X_est: Optional[np.ndarray]=None) -> np.ndarray:
        proba = self._rf_predict_proba(X_df, X_est=X_est)
        return np.argmax(proba, axis=1)

    def _expected_value_for_classes(self, y_pred: np.ndarray) -> np.ndarray:
        ev = self.rf_explainer.expected_value
        if np.isscalar(ev):
            return np.full(y_pred.shape[0], float(ev), dtype=float)
        ev = np.asarray(ev).reshape(-1)
        return ev[y_pred].astype(float)

    def _select_classwise_shap(self, shap_vals, y_pred: np.ndarray) -> np.ndarray:
        # normalize to (B, F) by picking predicted class per row
        if isinstance(shap_vals, list):               # list length C of (B,F)
            parts = [shap_vals[c][i] for i, c in enumerate(y_pred)]
            return np.stack(parts, axis=0)
        arr = np.asarray(shap_vals)
        if arr.ndim == 3:                             # (B,F,C)
            B, F, C = arr.shape
            out = np.zeros((B, F), dtype=arr.dtype)
            for i, c in enumerate(y_pred):
                out[i] = arr[i, :, c]
            return out
        if arr.ndim == 2:                             # (B,F)
            return arr
        if arr.ndim == 1:                             # (F,)
            return arr[None, :]
        raise ValueError(f"Unexpected SHAP shape {arr.shape}")

    # ---------- transformer side ----------
    def _get_transformer_predprob_and_class(self, X_df: pd.DataFrame) -> (np.ndarray, np.ndarray):
        # rebuild each row as dict and run through semantic encoder
        p_pred, y_pred = [], []
        with torch.no_grad():
            for _, row in X_df.iterrows():
                evt = {k: row[k] for k in self.features_order}
                # coerce types as trained
                for c in (self.feature_lists["CONTINUOUS_USED"] + self.feature_lists["BOOLEAN_USED"]):
                    evt[c] = float(evt[c])
                for c in (self.feature_lists["HIGH_CAT_USED"] + self.feature_lists["LOW_CAT_USED"]):
                    evt[c] = str(evt[c])

                cont, ch, cl = dict_to_transformer_tensors(evt, self.feature_lists, self.embed_maps, 'cpu')
                out = self.transformer_model(cont, ch, cl)
                logits = out["logits"] if isinstance(out, dict) else out
                prob = torch.softmax(logits, dim=1)  # (1,C)
                cls  = int(prob.argmax(1).item())
                y_pred.append(cls)
                p_pred.append(float(prob[0, cls].item()))
        return np.asarray(p_pred, dtype=float), np.asarray(y_pred, dtype=int)

    # ---------- main TXM ----------
    def explain_transformer_via_rf(self, X: np.ndarray, method: str = "waterfall"):
        """
        X: (B, len(features_order)) raw (unencoded) feature rows in training order.
        Returns dict with 'shap_values' (B,F) aligned to RF predicted class, plus fidelity.
        """
        try:
            # 1) DF and estimator matrix
            X_df  = self._to_dataframe(X)
            X_est = self._to_estimator_matrix(X_df)

            # 2) RF predicted class per row and class-wise baselines
            y_rf = self._predicted_classes(X_df, X_est=X_est)                 # (B,)
            rf_base = self._expected_value_for_classes(y_rf)                  # (B,)

            # 3) RF SHAP on estimator space → normalize to (B,F)
            rf_shap_raw = self.rf_explainer.shap_values(X_est, check_additivity=False)
            rf_shap = self._select_classwise_shap(rf_shap_raw, y_rf)          # (B, F)

            # 4) optional per-feature calibration
            if self.calibrator is not None and hasattr(self.calibrator, "apply"):
                rf_shap = np.stack([self.calibrator.apply(rf_shap[i]) for i in range(rf_shap.shape[0])], axis=0)

            # 5) reconstruct RF prob for comparison
            rf_margin = rf_shap.sum(axis=1) + rf_base                          # (B,)
            p_rf_hat  = rf_margin.copy()
            if (p_rf_hat.max() > 1.0) or (p_rf_hat.min() < 0.0):
                p_rf_hat = 1.0 / (1.0 + np.exp(-rf_margin))                   # sigmoid

            # 6) transformer prob of its predicted class
            p_trans, y_trans = self._get_transformer_predprob_and_class(X_df)  # both (B,)

            # 7) probability-ratio scaling
            eps = 1e-8
            alpha = np.clip(p_trans / (p_rf_hat + eps), 0.0, 10.0)             # (B,)
            mapped = rf_shap * alpha[:, None]                                  # (B, F)

            # 8) optional fidelity metrics
            fidelity = {}
            try:
                per_inst = []
                for i in range(mapped.shape[0]):
                    per_inst.append({
                        "sign": float(sign_fidelity(rf_shap[i], mapped[i])),
                        "rank_k10": float(rank_fidelity(rf_shap[i], mapped[i], k=10)),
                        "prob_monotonicity": float(prob_monotonicity(rf_shap[i], mapped[i], float(p_rf_hat[i]), float(p_trans[i]))),
                    })
                fidelity = {
                    "sign_mean": float(np.mean([m["sign"] for m in per_inst])),
                    "rank_k10_mean": float(np.mean([m["rank_k10"] for m in per_inst])),
                    "prob_monotonicity_mean": float(np.mean([m["prob_monotonicity"] for m in per_inst])),
                }
            except Exception:
                pass

            return {
                "shap_values": mapped,                      # (B, F) – 1-D vector per row
                "expected_value": rf_base,                  # (B,)  – per-row baseline
                "feature_names": self.features_order,
                "method": f"rf_mapped_to_transformer_{method}",
                "transformer_probability": p_trans,         # (B,)
                "mapping_confidence": self._mapping_confidence(p_rf_hat, p_trans),
                "txm_fidelity": fidelity,
            }
        except Exception as e:
            logging.error("Explanation mapping failed", exc_info=True)
            return None

    def _mapping_confidence(self, p_rf_hat: np.ndarray, p_trans: np.ndarray) -> float:
        try:
            if p_rf_hat.size == 1 or p_trans.size == 1:
                return float(max(0.0, 1.0 - abs(float(p_rf_hat[0]) - float(p_trans[0]))))
            corr = np.corrcoef(p_rf_hat.flatten(), p_trans.flatten())[0, 1]
            return float(max(0.0, corr)) if not np.isnan(corr) else 0.5
        except Exception:
            return 0.5
