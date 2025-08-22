"""
Enhanced business rules that work with ground truth labeling.
"""
from src.utils.business_rules import BusinessRulesConfig
from src.utils.ground_truth_labeling import GroundTruthLabeler
import pandas as pd
import numpy as np
from typing import Dict, Any

class EnhancedBusinessRulesConfig(BusinessRulesConfig):
    """Enhanced business rules that incorporate ML-discovered patterns."""
    
    def __init__(self, config_path: str = None, decoy_file_path: str = None):
        super().__init__(config_path)
        self.ground_truth_labeler = GroundTruthLabeler(decoy_file_path)
        
        # Add ML-discovered rules
        self.rules.update({
            # Anomaly detection thresholds
            "user_anomaly_threshold": 0.5,
            "cluster_anomaly_threshold": 0.5,
            "temporal_anomaly_threshold": 0.4,
            "volume_anomaly_threshold": 0.6,
            "network_anomaly_threshold": 0.7,
            
            # Composite risk thresholds
            "low_risk_threshold": 0.1,
            "high_risk_threshold": 0.3,
            "critical_risk_threshold": 0.5,
            
            # Pattern-based rules (to be learned from data)
            "suspicious_hour_pattern": [1, 2, 3, 4, 5],  # Deep night hours
            "suspicious_size_pattern": "large_burst",      # Large files in bursts
            "suspicious_domain_pattern": "high_entropy",   # High entropy domains
        })
    
    def assign_enhanced_risk_label(self, row: Dict[str, Any], is_training: bool = True) -> int:
        """Assign risk labels using both rules and ML signals."""
        
        # Get traditional rule-based score
        traditional_score = self.calculate_traditional_risk_score(row)
        
        # Get ML-based anomaly scores
        ml_score = self.calculate_ml_risk_score(row)
        
        # Combine scores
        combined_score = 0.2 * traditional_score + 0.8 * ml_score
        
        # Apply thresholds
        if combined_score >= self.rules["critical_risk_threshold"]:
            return 2  # Critical
        elif combined_score >= self.rules["high_risk_threshold"]:
            return 1
        else:
            return 0  # Low
    
    def calculate_traditional_risk_score(self, row: Dict[str, Any]) -> float:
        """Calculate risk score using traditional business rules."""
        score = 0.0
        
        # Size-based rules
        if row.get("megabytes_sent", 0) > self.rules["large_upload_threshold"]:
            score += 0.3
        
        # Temporal rules
        if self.is_after_hours(row.get("hour", 12)):
            score += 0.2
        
        # Destination rules
        if row.get("first_time_destination", False):
            score += 0.2
        
        # Burst rules
        if row.get("post_burst", 0) > self.rules.get("high_burst_threshold", 5):
            score += 0.3
        
        return min(score, 1.0)
    
    def calculate_ml_risk_score(self, row: Dict[str, Any]) -> float:
        """Calculate risk score using ML-discovered patterns."""
        score = 0.0
        
        # Anomaly-based scoring
        if row.get("user_anomaly_score", 0) > self.rules["user_anomaly_threshold"]:
            score += 0.4
        
        if row.get("cluster_anomaly_score", 0) > self.rules["cluster_anomaly_threshold"]:
            score += 0.3
        
        # Pattern-based scoring
        if row.get("hour", 12) in self.rules["suspicious_hour_pattern"]:
            score += 0.2
        
        if row.get("destination_entropy", 0) > self.rules.get("high_entropy_threshold", 3.0):
            score += 0.1
        
        return min(score, 1.0)