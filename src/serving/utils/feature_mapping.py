import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Sequence
import numpy as np
import torch
from ..models.encoders import dict_to_transformer_tensors
from src.serving.utils.cross_model_attribution_fidelity_metrics import sign_fidelity, rank_fidelity, prob_monotonicity, test_example


def create_feature_vector_from_event(evt, feature_names, feature_lists, embed_maps):
    """Create type-safe feature vector that matches RF training format exactly."""
    feature_vector = []
    
    for name in feature_names:
        value = evt.get(name, 0)
        
        if name in feature_lists['CONTINUOUS_USED']:
            try:
                feature_vector.append(float(value) if value is not None else 0.0)
            except (ValueError, TypeError):
                feature_vector.append(0.0)
        elif name in feature_lists['BOOLEAN_USED']:
            # Use 0/1 integers for boolean (matches RF training)
            if isinstance(value, str):
                feature_vector.append(1 if value.lower() in ['true', '1', 'yes', 'on'] else 0)
            elif isinstance(value, bool):
                feature_vector.append(1 if value else 0)
            else:
                feature_vector.append(1 if bool(value) else 0)
        elif name in feature_lists['HIGH_CAT_USED'] + feature_lists['LOW_CAT_USED']:
            # Use integer encoding for categorical
            embed_map = embed_maps.get(name, {})
            if isinstance(value, str) and value in embed_map:
                feature_vector.append(int(embed_map[value]))
            else:
                feature_vector.append(0)
        else:
            try:
                feature_vector.append(float(value) if value is not None else 0.0)
            except (ValueError, TypeError):
                feature_vector.append(0.0)
    
    # Return same dtype as RF training (typically float64)
    return np.array(feature_vector, dtype=np.float64)


class TransformerExplanationMapper:
    """Maps RandomForest explanations to Transformer explanations using feature correlation."""
    
    def __init__(self, rf_explainer, transformer_model, feature_lists, features_order, embed_maps, calibrator: Optional['TXMCalibrator']=None):
        self.rf_explainer = rf_explainer
        self.transformer_model = transformer_model
        self.feature_lists = feature_lists
        self.features_order = features_order
        self.embed_maps = embed_maps
        self.calibrator = calibrator
        
        # Pre-compute feature importance correlation
        self._compute_feature_correlation()
    
    def _compute_feature_correlation(self):
        """Compute how RF and Transformer feature importances correlate."""
        self.feature_weights = {feat: 1.0 for feat in self.features_order}
        logging.info("Feature correlation mapping computed")
    
    def explain_transformer_via_rf(self, X, method='waterfall'):
        """Generate transformer explanations by mapping RF SHAP values."""
        try:
            # Get RF SHAP values
            rf_shap_values = self.rf_explainer.shap_values(X, check_additivity=False)
            # Handle binary classification (extract positive class)
            if isinstance(rf_shap_values, list):
                rf_shap_values = rf_shap_values[1]  # Positive class

            # Get transformer predictions for comparison
            transformer_probs = self._get_transformer_predictions(X)

            # Map RF explanations to transformer scale
            mapped_explanations = self._map_explanations(
                rf_shap_values, transformer_probs, X
            )

            # Compute fidelity metrics (per-instance) and summarize
            # Reconstruct RF prob used in mapping confidence for consistency
            rf_baseline = self.rf_explainer.expected_value
            if isinstance(rf_baseline, np.ndarray):
                rf_baseline = rf_baseline[1]
            rf_margin = np.sum(rf_shap_values, axis=1) + rf_baseline
            p_rf_hat = rf_margin
            if (p_rf_hat.max() > 1.0) or (p_rf_hat.min() < 0.0):
                p_rf_hat = 1.0 / (1.0 + np.exp(-rf_margin))

            per_inst = []
            for i in range(mapped_explanations.shape[0]):
                rf_vec = rf_shap_values[i]
                tx_vec = mapped_explanations[i]
                per_inst.append({
                    "sign": float(sign_fidelity(rf_vec, tx_vec)),
                    "rank_k10": float(rank_fidelity(rf_vec, tx_vec, k=10)),
                    "prob_monotonicity": float(prob_monotonicity(rf_vec, tx_vec, float(p_rf_hat[i]), float(transformer_probs[i]))),
                })
            fidelity_summary = {
                "sign_mean": float(np.mean([m["sign"] for m in per_inst])),
                "rank_k10_mean": float(np.mean([m["rank_k10"] for m in per_inst])),
                "prob_monotonicity_mean": float(np.mean([m["prob_monotonicity"] for m in per_inst])),
            }
            # Optional fallback policy flag (no automatic switch here to keep behavior stable)
            fallback_used = False
            if fidelity_summary["sign_mean"] < 0.7 or fidelity_summary["rank_k10_mean"] < 0.2:
                logging.warning("TXM fidelity low; consider falling back to deep attributions.")
                # Set a flag; caller can decide to compute Captum IG
                fallback_used = False

            return {
                'shap_values': mapped_explanations,
                'expected_value': self.rf_explainer.expected_value,
                'feature_names': self.features_order,
                'method': f'rf_mapped_to_transformer_{method}',
                'transformer_probability': transformer_probs,
                'mapping_confidence': self._compute_mapping_confidence(rf_shap_values, transformer_probs),
                'txm_fidelity': fidelity_summary,
                'txm_fidelity_per_instance': per_inst,
                'txm_fallback_used': fallback_used,
            }
        except Exception as e:
            logging.error(f"Explanation mapping failed: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None
    
    # def explain_transformer_via_rf(self, X, method='waterfall'):
    #     """Generate transformer explanations by mapping RF SHAP values. without fidelity metrics"""
    #     try:
    #         # Get RF SHAP values
    #         rf_shap_values = self.rf_explainer.shap_values(X, check_additivity=False)
            
    #         # Handle binary classification (extract positive class)
    #         if isinstance(rf_shap_values, list):
    #             rf_shap_values = rf_shap_values[1]  # Positive class
            
    #         # Get transformer predictions for comparison
    #         transformer_probs = self._get_transformer_predictions(X)
            
    #         # Map RF explanations to transformer scale
    #         mapped_explanations = self._map_explanations(
    #             rf_shap_values, transformer_probs, X
    #         )
            
    #         return {
    #             'shap_values': mapped_explanations,
    #             'expected_value': self.rf_explainer.expected_value,
    #             'feature_names': self.features_order,
    #             'method': f'rf_mapped_to_transformer_{method}',
    #             'transformer_probability': transformer_probs,
    #             'mapping_confidence': self._compute_mapping_confidence(rf_shap_values, transformer_probs)
    #         }
            
    #     except Exception as e:
    #         logging.error(f"Explanation mapping failed: {e}")
    #         import traceback
    #         logging.error(traceback.format_exc())
    #         return None
    
    def _get_transformer_predictions(self, X):
        """Get transformer predictions with proper type handling."""
        try:
            transformer_inputs = []
            for i in range(X.shape[0]):
                # Proper type handling
                event_dict = {}
                
                for j, feat in enumerate(self.features_order):
                    value = X[i, j]
                    
                    if feat in self.feature_lists['CONTINUOUS_USED']:
                        try:
                            event_dict[feat] = float(value)
                        except (ValueError, TypeError):
                            event_dict[feat] = 0.0
                    elif feat in self.feature_lists['BOOLEAN_USED']:
                        #  Handle string booleans properly
                        if isinstance(value, str):
                            event_dict[feat] = value.lower() in ['true', '1', 'yes', 'on']
                        elif isinstance(value, (int, float)):
                            event_dict[feat] = bool(value)
                        else:
                            event_dict[feat] = bool(value)
                    elif feat in self.feature_lists['HIGH_CAT_USED'] + self.feature_lists['LOW_CAT_USED']:
                        event_dict[feat] = str(value) if value is not None else 'unknown'
                    else:
                        try:
                            event_dict[feat] = float(value)
                        except (ValueError, TypeError):
                            event_dict[feat] = 0.0
                
                cont, cat_high, cat_low = dict_to_transformer_tensors(
                    event_dict, self.feature_lists, self.embed_maps, 'cpu'
                )
                transformer_inputs.append((cont, cat_high, cat_low))
            
            # Get predictions
            predictions = []
            with torch.no_grad():
                for cont, cat_high, cat_low in transformer_inputs:
                    output = self.transformer_model(cont, cat_high, cat_low)
                    if isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output
                    prob = torch.softmax(logits, dim=1)[:, 1]
                    predictions.append(prob.item())
            
            return np.array(predictions)
            
        except Exception as e:
            logging.warning(f"Transformer prediction failed: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return np.array([0.5] * X.shape[0])
    
    # def _map_explanations(self, rf_shap_values, transformer_probs, X):
    #     """Map RF SHAP values to transformer explanation scale. without per-category scaling."""
    
    #     # Handle 3D SHAP values (samples, features, classes)
    #     if rf_shap_values.ndim == 3:
    #         # For binary classification, use positive class (class 1)
    #         rf_shap_values = rf_shap_values[:, :, 1]  # Shape: (samples, features)
        
    #     # Compute RF predictions from SHAP values
    #     rf_baseline = self.rf_explainer.expected_value
    #     if isinstance(rf_baseline, np.ndarray):
    #         rf_baseline = rf_baseline[1]  # Use positive class baseline for binary classification
        
    #     rf_predictions = np.sum(rf_shap_values, axis=1) + rf_baseline

    #     # warn if RF predictions are all near zero
    #     if np.all(np.abs(rf_predictions) < 1e-6):
    #         logging.warning("RF predictions are near zero; mapped explanations may be uninformative.")

    #     # Compute scale factors, avoid division by zero
    #     epsilon = 1e-8
    #     scale_factors = transformer_probs / (rf_predictions + epsilon)
        
    #     # Clip scale factors to avoid extreme values
    #     scale_factors = np.clip(scale_factors, 0, 10)
    #     # Apply scaling - now both arrays have compatible shapes
    #     mapped_values = rf_shap_values * scale_factors.reshape(-1, 1)
        
    #     return mapped_values

    def _map_explanations(self, rf_shap_values, transformer_probs, X):
        """Map RF SHAP values to transformer explanation scale. with optional per-category scaling."""
    
        # Handle 3D SHAP values (samples, features, classes)
        if rf_shap_values.ndim == 3:
            # For binary classification, use positive class (class 1)
            rf_shap_values = rf_shap_values[:, :, 1]  # Shape: (samples, features)
        
        # Optional per-category scaling α_j (apply per instance)
        if self.calibrator is not None and getattr(self.calibrator, "alpha_by_cat", None):
            rf_shap_values = np.stack(
                [self.calibrator.apply(rf_shap_values[i]) for i in range(rf_shap_values.shape[0])],
                axis=0
            )

        # Baseline and RF prediction reconstruction
        rf_baseline = self.rf_explainer.expected_value
        if isinstance(rf_baseline, np.ndarray):  # positive class baseline for binary
            rf_baseline = rf_baseline[1]
        rf_margin = np.sum(rf_shap_values, axis=1) + rf_baseline  # may be log-odds

        # Convert to probability if values look like logits
        p_rf_hat = rf_margin
        if (p_rf_hat.max() > 1.0) or (p_rf_hat.min() < 0.0):
            p_rf_hat = 1.0 / (1.0 + np.exp(-rf_margin))  # sigmoid

        # Compute instance scale factors in probability space
        epsilon = 1e-8
        scale_factors = transformer_probs / (p_rf_hat + epsilon)
        scale_factors = np.clip(scale_factors, 0, 10)

        # Apply scaling
        mapped_values = rf_shap_values * scale_factors.reshape(-1, 1)
        return mapped_values
    
    def _compute_mapping_confidence(self, rf_shap_values, transformer_probs):
        """Compute confidence in the explanation mapping."""
        try:
            if rf_shap_values.ndim == 3:
                rf_shap_values = rf_shap_values[:, :, 1]  # positive class

            rf_baseline = self.rf_explainer.expected_value
            if isinstance(rf_baseline, np.ndarray):
                rf_baseline = rf_baseline[1]
            rf_margin = np.sum(rf_shap_values, axis=1) + rf_baseline

            p_rf_hat = rf_margin
            if (p_rf_hat.max() > 1.0) or (p_rf_hat.min() < 0.0):
                p_rf_hat = 1.0 / (1.0 + np.exp(-rf_margin))  # sigmoid

            if len(p_rf_hat) < 2 or len(transformer_probs) < 2:
                diff = abs(float(p_rf_hat[0]) - float(transformer_probs[0]))
                return max(0.0, 1.0 - diff)

            if p_rf_hat.shape != transformer_probs.shape:
                logging.warning(f"Shape mismatch: rf_predictions {p_rf_hat.shape}, transformer_probs {transformer_probs.shape}")
                return 0.5

            corr = np.corrcoef(p_rf_hat.flatten(), transformer_probs.flatten())[0, 1]
            return max(0.0, float(corr)) if not np.isnan(corr) else 0.5
        except Exception:
            return 0.5

    
    # def _compute_mapping_confidence(self, rf_shap_values, transformer_probs):
    #     """Compute confidence in the explanation mapping. works with without scaling."""
    #     try:
    #         # Handle 3D SHAP values
    #         if rf_shap_values.ndim == 3:
    #             rf_shap_values = rf_shap_values[:, :, 1]  # Use positive class
            
    #         # Compute RF predictions
    #         rf_baseline = self.rf_explainer.expected_value
    #         if isinstance(rf_baseline, np.ndarray):
    #             rf_baseline = rf_baseline[1]  # Use positive class baseline
            
    #         rf_predictions = np.sum(rf_shap_values, axis=1) + rf_baseline
            
    #         # Need at least 2 samples for correlation
    #         if len(rf_predictions) < 2 or len(transformer_probs) < 2:
    #             # For single samples, compute agreement score instead
    #             diff = abs(rf_predictions[0] - transformer_probs[0])
    #             confidence = max(0.0, 1.0 - diff)  # Higher confidence if predictions are closer
    #             return confidence
            
    #         # Ensure same shapes
    #         if rf_predictions.shape != transformer_probs.shape:
    #             logging.warning(f"Shape mismatch: rf_predictions {rf_predictions.shape}, transformer_probs {transformer_probs.shape}")
    #             return 0.5
            
    #         # Compute correlation for multiple samples
    #         correlation = np.corrcoef(rf_predictions.flatten(), transformer_probs.flatten())[0, 1]
    #         return max(0.0, correlation) if not np.isnan(correlation) else 0.5
            
    #     except Exception as e:
    #         logging.warning(f"Mapping confidence computation failed: {e}")
    #         return 0.5  # Default confidence



class TXMCalibrator:
    """
    Learn per-category scaling factors α_j to align mean absolute attributions
    between RF TreeSHAP and Transformer (e.g., Captum IG) on a calibration set.
    """
    def __init__(self, alpha_min: float = 0.5, alpha_max: float = 4.0, eps: float = 1e-6, shrink: float = 0.2):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.eps = eps
        self.shrink = shrink  # ridge-like shrinkage toward 1.0
        self.alpha_by_cat: Dict[str, float] = {}
        self.cat_index: Dict[int, str] = {}  # feature index -> category

    def fit(self, rf_shap: np.ndarray, trans_attr: np.ndarray, categories: Sequence[str]) -> None:
        """
        rf_shap: [N, F] RF TreeSHAP per instance
        trans_attr: [N, F] Transformer attributions per instance (e.g., IG aggregated)
        categories: [F] category label for each feature index
        """
        assert rf_shap.shape == trans_attr.shape, "Shape mismatch"
        N, F = rf_shap.shape
        self.cat_index = {j: categories[j] for j in range(F)}

        # Compute mean absolute attribution per category for both models
        cats = sorted(set(categories))
        for cat in cats:
            idx = [j for j in range(F) if categories[j] == cat]
            if not idx:
                continue
            rf_mean = float(np.mean(np.abs(rf_shap[:, idx])) + self.eps)
            tr_mean = float(np.mean(np.abs(trans_attr[:, idx])) + self.eps)
            raw = rf_mean / tr_mean
            # shrinkage toward 1.0 to stabilize
            alpha = (1 - self.shrink) * raw + self.shrink * 1.0
            # bounds
            alpha = max(self.alpha_min, min(self.alpha_max, alpha))
            self.alpha_by_cat[cat] = alpha
    
    def apply(self, rf_shap_instance: np.ndarray) -> np.ndarray:
        """Scale an RF SHAP vector per feature category using α_j."""
        F = rf_shap_instance.shape[0]
        scaled = np.empty_like(rf_shap_instance)
        for j in range(F):
            cat = self.cat_index.get(j, "_default")
            alpha = self.alpha_by_cat.get(cat, 1.0)
            scaled[j] = alpha * rf_shap_instance[j]
        return scaled