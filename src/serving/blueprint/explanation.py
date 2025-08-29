"""Explanation Blueprint for ML Model Serving - Detailed and Comparative Explanations
"""

from flask import Blueprint, request, jsonify, current_app
import logging
import numpy as np
# from ..utils.validation import handle_missing_features
from ..utils.shap_utils import _safe_get_shap_values
from ..utils.explanation_utils import (
    get_rf_explanation_data,
    get_transformer_explanation_data,
    _get_comprehensive_rf_analysis,
    _get_comprehensive_transformer_analysis,
    _analyze_feature_alignment,
    _get_safe_baseline
)
from ..models.encoders import encode_tabular  #, encode_sequence_semantic, encode_sequence_flat_for_shap
from ..models.cascade import rf_predict_proba, transformer_predict_proba
from ..utils.tau_serving_decision_helper import load_cascade_config, cascade_decision
from pathlib import Path


explanation_bp = Blueprint('explanation', __name__)


@explanation_bp.route('/explain', methods=['POST'])
def detailed_explanation():
    """Comprehensive explanation endpoint."""
    evt = request.get_json()
    if not evt:
        return jsonify({"error": "No JSON payload"}), 400
    
    # evt = handle_missing_features(evt)
    method = request.args.get('method', 'shap')  # 'shap', 'captum', 'attention', 'auto'
    
    try:
        models = current_app.ml_models
        # Ensure τ and τ2 are available (load from config if needed)
        if 'tau2' not in models:
            cfg_path = current_app.config.get('CASCADE_CONFIG_PATH', None)
            cfg_path = Path(cfg_path) if cfg_path else Path('config/cascade_config.json')
            try:
                tau_loaded, tau2_loaded, _cfg = load_cascade_config(cfg_path)
                models.setdefault('tau', float(tau_loaded))
                models['tau2'] = float(tau2_loaded)
            except Exception:
                # Fallback defaults
                models.setdefault('tau', 0.2)
                models.setdefault('tau2', 0.5)

        tau = float(models.get('tau', 0.2))
        tau2 = float(models.get('tau2', 0.5))

        # Determine which model would be used
        X_tab = encode_tabular(evt)
        rf_proba = rf_predict_proba(models['rf'], X_tab)
        # if rf_proba.max() >= models['tau']:
        # else:
        rf_max = float(rf_proba.max())

        # Apply cascade with τ and τ2
        if rf_max < tau:
            # Early exit: RF-only (benign)
            rf_payload = get_rf_explanation_data(evt, models)
            rf_payload['cascade'] = {
                "tau": tau,
                "tau2": tau2,
                "rf_max": rf_max,
                "escalated": False,
                "decision": 0  # Benign
            }
            return jsonify(rf_payload)
        else:
            # Escalate to Transformer, decide with τ2
            trans_payload = get_transformer_explanation_data(evt, models, method=method)
            probs = trans_payload.get("probabilities", [])
            p_trans = float(probs[1] if isinstance(probs, list) and len(probs) > 1 else (max(probs) if probs else 0.0))
            decision = cascade_decision(rf_max, p_trans, tau, tau2)


            # Optional fidelity gate (advisory): if low, request IG as corroboration
            txm_fid = trans_payload.get("txm_fidelity", {}) or {}
            if txm_fid.get("sign_mean", 1.0) < 0.7 or txm_fid.get("rank_k10_mean", 1.0) < 0.2:
                try:
                    ig_payload = get_transformer_explanation_data(evt, models, method="captum")
                    trans_payload["ig_backup"] = {
                        "attributions": ig_payload.get("explanation", {}).get("captum_analysis", {}),
                        "used": True
                    }
                except Exception:
                    trans_payload["ig_backup"] = {"used": False}

            trans_payload['cascade'] = {
                "tau": tau,
                "tau2": tau2,
                "rf_max": rf_max,
                "p_trans": p_trans,
                "escalated": True,
                "decision": int(decision)  # 1=Malicious, 0=Benign
            }
            return jsonify(trans_payload)
            
    except Exception as e:
        logging.error(f"Explanation error: {e}")
        return jsonify({"error": str(e)}), 500



@explanation_bp.route('/explain/compare', methods=['POST'])
def compare_explanations():
    """Compare explanations from both models."""
    evt = request.get_json()
    if not evt:
        return jsonify({"error": "No JSON payload"}), 400
    
    # evt = handle_missing_features(evt)
    
    try:
        models = current_app.ml_models
        
        # Get predictions from both models
        rf_analysis = get_rf_explanation_data(evt, models)
        transformer_analysis = get_transformer_explanation_data(evt, models)
        # rf_analysis = _get_rf_analysis(evt, models)
        # transformer_analysis = _get_transformer_analysis(evt, models)
        
        # Cascade decision logic
        cascade_decision = {
            "would_use_rf": rf_analysis["confidence"] >= models['tau'],
            "threshold": models['tau'],
            "rf_confidence": rf_analysis["confidence"],
            "transformer_confidence": transformer_analysis["confidence"],
            "agreement": rf_analysis["prediction"] == transformer_analysis["prediction"]
            # "confidence_gap": abs(rf_analysis["confidence"] - transformer_analysis["confidence"])
        }
        
        return jsonify({
            "input_event": evt,
            "rf_analysis": rf_analysis,
            "transformer_analysis": transformer_analysis,
            "cascade_decision": cascade_decision,
            # "recommendation": _get_decision_recommendation(cascade_decision)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@explanation_bp.route('/explain/comprehensive', methods=['POST'])
def comprehensive_explanation():
    """Get all explanations for both models with feature mapping."""
    evt = request.get_json()
    if not evt:
        return jsonify({"error": "No JSON payload"}), 400
    
    # evt = handle_missing_features(evt)
    
    try:
        models = current_app.ml_models
        
        # Get RF explanation
        X_tab = encode_tabular(evt)
        rf_proba = rf_predict_proba(models['rf'], X_tab)
        rf_analysis = _get_comprehensive_rf_analysis(evt, models, rf_proba)

        rf_confidence = rf_analysis.get("confidence", 0.0)
        # Get Transformer explanation with all methods
        transformer_analysis = _get_comprehensive_transformer_analysis(evt, models)
        transformer_confidence = transformer_analysis.get("confidence", 0.0)
        
        # Feature importance alignment
        alignment_analysis = _analyze_feature_alignment(rf_analysis, transformer_analysis)
        
        # Cascade decision
        cascade_decision = {
            "would_use_rf": rf_analysis["confidence"] >= models['tau'],
            "threshold": models['tau'],
            "rf_confidence": rf_confidence, #rf_analysis["confidence"],
            "transformer_confidence": transformer_analysis["confidence"],
            "prediction_agreement": rf_analysis["prediction"] == transformer_analysis["prediction"],
            "feature_alignment_score": alignment_analysis["alignment_score"]
        }
        
        return jsonify({
            "input_event": evt,
            "rf_analysis": rf_analysis,
            "transformer_analysis": transformer_analysis,
            "feature_alignment": alignment_analysis,
            "cascade_decision": cascade_decision,
            "interpretation": _generate_interpretation(cascade_decision, alignment_analysis)
        })
        
    except Exception as e:
        logging.error(f"Comprehensive explanation error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


def _generate_interpretation(cascade_decision, alignment_analysis):
    """Generate human-readable interpretation of the analysis."""
    interpretation = []
    
    # Prediction agreement
    if cascade_decision["prediction_agreement"]:
        interpretation.append("Both models agree on the prediction - high confidence result")
    else:
        interpretation.append(" Models disagree on prediction - requires careful review")
    
    # Feature alignment
    alignment_score = alignment_analysis["alignment_score"]
    if alignment_score > 0.7:
        interpretation.append(f"Strong feature importance alignment ({alignment_score:.2f}) - models focus on similar patterns")
    elif alignment_score > 0.3:
        interpretation.append(f"Moderate feature alignment ({alignment_score:.2f}) - models partially agree on important features")
    else:
        interpretation.append(f"Low feature alignment ({alignment_score:.2f}) - models focus on different aspects")
    
    # Cascade decision
    rf_confidence = cascade_decision.get("rf_confidence", 0.0)
    threshold = cascade_decision.get("threshold", 0.8)
    
    if cascade_decision["would_use_rf"]:
        interpretation.append(f"🔄 Cascade would use RF (confidence {cascade_decision['rf_confidence']:.3f} ≥ threshold {cascade_decision['threshold']:.3f})")
    else:
        interpretation.append(f"🔄 Cascade would use Transformer (RF confidence {cascade_decision['rf_confidence']:.3f} < threshold {cascade_decision['threshold']:.3f})")
    
    return interpretation


@explanation_bp.route('/explain/waterfall', methods=['POST'])
def shap_waterfall():
    """Generate SHAP waterfall plot data."""
    evt = request.get_json()
    if not evt:
        return jsonify({"error": "No JSON payload"}), 400
    
    # evt = handle_missing_features(evt)
    
    try:
        models = current_app.ml_models
        X_tab = encode_tabular(evt)
        rf_proba = rf_predict_proba(models['rf'], X_tab)
        prediction_class = int(rf_proba.argmax())
        
        # Get SHAP values
        shap_values = _safe_get_shap_values(models['rf_explainer'], X_tab.reshape(1, -1), prediction_class)
        logging.info(f"shap_values type: {type(shap_values)}, shape: {getattr(shap_values, 'shape', None)}")
        
        # Prepare waterfall data
        waterfall_data = []
        # cumulative_sum = shap_values
        # Get baseline from explainer
        expected_value = models['rf_explainer'].expected_value
        # baseline = float(expected_value[prediction_class]) if isinstance(expected_value, np.ndarray) else float(expected_value)
        baseline = _get_safe_baseline(models['rf_explainer'], prediction_class)
        cumulative_sum = baseline
        
        # Sort features by absolute SHAP value
        feature_indices = np.abs(shap_values).argsort()[::-1][:10]
        
        for i, feature_idx in enumerate(feature_indices):
            feature_name = models['feature_names'][feature_idx]
            shap_value = float(shap_values[feature_idx])
            feature_value = evt.get(feature_name, 0)
            
            waterfall_data.append({
                "feature": feature_name,
                "feature_value": feature_value,
                "shap_value": shap_value,
                "cumulative": cumulative_sum + shap_value,
                "rank": i + 1
            })
            cumulative_sum += shap_value
        
        return jsonify({
            # "shap_values": float(shap_values),
            "shap_values": shap_values.tolist(),
            "final_prediction": float(cumulative_sum),
            "prediction_class": prediction_class,
            "waterfall_data": waterfall_data,
            "model_output": float(rf_proba[prediction_class])
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500