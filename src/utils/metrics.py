"""
Model evaluation and business metrics utilities.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, 
    precision_recall_curve, auc
)
from typing import Dict, Any, List, Tuple
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_proba: np.ndarray = None) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics."""
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average='macro'),
        "recall_macro": recall_score(y_true, y_pred, average='macro'),
        "f1_macro": f1_score(y_true, y_pred, average='macro'),
        "precision_weighted": precision_score(y_true, y_pred, average='weighted'),
        "recall_weighted": recall_score(y_true, y_pred, average='weighted'),
        "f1_weighted": f1_score(y_true, y_pred, average='weighted'),
    }
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
        metrics[f"precision_class_{i}"] = p
        metrics[f"recall_class_{i}"] = r
        metrics[f"f1_class_{i}"] = f
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    
    # ROC-AUC for multiclass (if probabilities provided)
    if y_proba is not None:
        try:
            if y_proba.shape[1] > 2:  # Multiclass
                metrics["roc_auc_macro"] = roc_auc_score(y_true, y_proba, average='macro', multi_class='ovr')
                metrics["roc_auc_weighted"] = roc_auc_score(y_true, y_proba, average='weighted', multi_class='ovr')
            else:  # Binary
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")
    
    # Classification report
    metrics["classification_report"] = classification_report(y_true, y_pred, output_dict=True)
    
    return metrics


def calculate_business_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                             y_proba: np.ndarray, confidence_threshold: float = 0.7) -> Dict[str, Any]:
    """Calculate business-relevant metrics for cybersecurity use case."""
    
    # Basic counts
    total_events = len(y_true)
    true_positives = np.sum((y_true > 0) & (y_pred > 0))
    false_positives = np.sum((y_true == 0) & (y_pred > 0))
    false_negatives = np.sum((y_true > 0) & (y_pred == 0))
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    
    # Business metrics
    metrics = {
        "total_events": total_events,
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "true_negatives": int(true_negatives),
        
        # Alert-related metrics
        "total_alerts": int(np.sum(y_pred > 0)),
        "alert_rate": float(np.sum(y_pred > 0) / total_events),
        "false_positive_rate": float(false_positives / (false_positives + true_negatives)) if (false_positives + true_negatives) > 0 else 0,
        "false_negative_rate": float(false_negatives / (false_negatives + true_positives)) if (false_negatives + true_positives) > 0 else 0,
        
        # Detection metrics
        "detection_rate": float(true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 0,
        "precision_alerts": float(true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else 0,
    }
    
    # Risk level breakdown
    for risk_level in [0, 1, 2]:  # Low, High, Critical
        actual_count = np.sum(y_true == risk_level)
        predicted_count = np.sum(y_pred == risk_level)
        correct_count = np.sum((y_true == risk_level) & (y_pred == risk_level))
        
        metrics[f"risk_{risk_level}_actual"] = int(actual_count)
        metrics[f"risk_{risk_level}_predicted"] = int(predicted_count)
        metrics[f"risk_{risk_level}_correct"] = int(correct_count)
        metrics[f"risk_{risk_level}_accuracy"] = float(correct_count / actual_count) if actual_count > 0 else 0
    
    # Confidence-based metrics
    high_confidence_mask = y_proba.max(axis=1) >= confidence_threshold
    high_confidence_count = np.sum(high_confidence_mask)
    
    if high_confidence_count > 0:
        hc_accuracy = accuracy_score(y_true[high_confidence_mask], y_pred[high_confidence_mask])
        metrics["high_confidence_count"] = int(high_confidence_count)
        metrics["high_confidence_rate"] = float(high_confidence_count / total_events)
        metrics["high_confidence_accuracy"] = float(hc_accuracy)
    
    return metrics


def calculate_cascade_metrics(rf_predictions: Dict, lstm_predictions: Dict, 
                            confidence_threshold: float = 0.7) -> Dict[str, Any]:
    """Calculate metrics specific to cascade model architecture."""
    
    total_predictions = len(rf_predictions.get("y_pred", []))
    
    # RF usage (high confidence predictions)
    rf_confident_mask = rf_predictions.get("y_proba", np.array([])).max(axis=1) >= confidence_threshold
    rf_usage_count = np.sum(rf_confident_mask)
    
    # LSTM usage (low confidence RF predictions)
    lstm_usage_count = total_predictions - rf_usage_count
    
    metrics = {
        "total_predictions": total_predictions,
        "rf_usage_count": int(rf_usage_count),
        "lstm_usage_count": int(lstm_usage_count),
        "rf_usage_rate": float(rf_usage_count / total_predictions) if total_predictions > 0 else 0,
        "lstm_usage_rate": float(lstm_usage_count / total_predictions) if total_predictions > 0 else 0,
        "confidence_threshold": confidence_threshold
    }
    
    # Performance by model
    if rf_usage_count > 0 and "y_true" in rf_predictions:
        rf_mask_indices = np.where(rf_confident_mask)[0]
        rf_y_true = rf_predictions["y_true"][rf_mask_indices]
        rf_y_pred = rf_predictions["y_pred"][rf_mask_indices]
        
        metrics["rf_accuracy"] = accuracy_score(rf_y_true, rf_y_pred)
        metrics["rf_f1"] = f1_score(rf_y_true, rf_y_pred, average='weighted')
    
    if lstm_usage_count > 0 and "y_true" in lstm_predictions:
        lstm_mask_indices = np.where(~rf_confident_mask)[0]
        lstm_y_true = lstm_predictions["y_true"][lstm_mask_indices]
        lstm_y_pred = lstm_predictions["y_pred"][lstm_mask_indices]
        
        metrics["lstm_accuracy"] = accuracy_score(lstm_y_true, lstm_y_pred)
        metrics["lstm_f1"] = f1_score(lstm_y_true, lstm_y_pred, average='weighted')
    
    return metrics


def create_metrics_report(y_true: np.ndarray, y_pred: np.ndarray, 
                         y_proba: np.ndarray, model_name: str = "Model") -> Dict[str, Any]:
    """Create comprehensive metrics report."""
    
    logger.info(f"Creating metrics report for {model_name}")
    
    report = {
        "model_name": model_name,
        "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        "dataset_info": {
            "total_samples": len(y_true),
            "class_distribution": pd.Series(y_true).value_counts().to_dict()
        }
    }
    
    # Classification metrics
    report["classification_metrics"] = calculate_classification_metrics(y_true, y_pred, y_proba)
    
    # Business metrics
    report["business_metrics"] = calculate_business_metrics(y_true, y_pred, y_proba)
    
    # Summary
    cm = confusion_matrix(y_true, y_pred)
    report["summary"] = {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average='weighted')),
        "macro_f1": float(f1_score(y_true, y_pred, average='macro')),
        "total_alerts": int(np.sum(y_pred > 0)),
        "alert_precision": float(precision_score(y_true, y_pred > 0, y_pred > 0)),
        "confusion_matrix_shape": cm.shape
    }
    
    logger.info(f"Metrics report created for {model_name}")
    return report