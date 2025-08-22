"""
Shared business rules configuration for consistent risk assessment.
"""
from typing import Dict, List, Any
import json
from pathlib import Path
import logging
from src.ingest.utils import shannon_entropy
logger = logging.getLogger(__name__)

class BusinessRulesConfig:
    """Centralized business rules for risk classification."""
    
    def __init__(self, config_path: str = None):
        # Default rules - should be set by security experts
    
        self.rules = {
            # Size thresholds (MB)
            "large_upload_threshold": 10,
            "medium_upload_threshold": 5,
            "small_upload_threshold": 2,
            
            # Temporal rules
            "after_hours_start": 19,  # 7 PM
            "after_hours_end": 7,     # 7 AM
            
            # Activity thresholds
            "high_daily_uploads": 20,
            "burst_window_seconds": 5,
            
            # Statistical thresholds (percentiles)
            "high_burst_percentile": 0.99,
            "high_entropy_percentile": 0.99,
            
            # USB rules
            "usb_weekend_large_threshold": 2,
            
            # Risk weights (for future composite scoring)
            "risk_weights": {
                # "decoy_interaction": 10.0,
                "after_hours_large_new": 10.0,
                "suspicious_domain": 6.0,
                "usb_weekend_large": 6.0,
                "high_daily_activity": 2.0,
                "high_burst": 2.0,
                "high_entropy": 2.0
            }
        }
        
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load rules from JSON file."""
        with open(config_path, 'r') as f:
            loaded_rules = json.load(f)
            self.rules.update(loaded_rules)
    
    def save_config(self, config_path: str):
        """Save rules to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(self.rules, f, indent=2)
    
    def is_after_hours(self, hour: int) -> bool:
        """Check if time is after business hours."""
        return hour < self.rules["after_hours_end"] or hour >= self.rules["after_hours_start"]
    
    def is_suspicious_domain(self, domain: str) -> bool:
        """Check if domain is suspicious."""
        return domain in self.rules["suspicious_domains"]
    
    def is_large_upload(self, megabytes: float) -> bool:
        """Check if upload is large."""
        return megabytes > self.rules["large_upload_threshold"]
    def is_medium_upload(self, megabytes: float) -> bool:
        """Check if upload is medium."""
        return megabytes > self.rules["medium_upload_threshold"]
    def is_small_upload(self, megabytes: float) -> bool:
        """Check if upload is small."""
        return megabytes > self.rules["small_upload_threshold"]
    
    # def destination_entropy(self, shanon):
    #     return shannon_entropy(shanon)
    
    def get_upload_size_category(self, megabytes: float) -> str:
        """Categorize upload size."""
        if megabytes > self.rules["large_upload_threshold"]:
            return "large"
        elif megabytes > self.rules["medium_upload_threshold"]:
            return "medium"
        elif megabytes > self.rules["small_upload_threshold"]:
            return "small"
        else:
            return "tiny"

def assign_risk_label_robust(row: Dict[str, Any], 
                           business_rules: BusinessRulesConfig,
                        #    decoy_users: set,
                        #    high_burst_threshold: float,
                            # high_entropy_threshold: float
                           ) -> int:
    """
    Robust risk label assignment using centralized business rules.
    
    Args:
        row: Event data as dictionary
        business_rules: Centralized business rules configuration
        decoy_users: Set of known decoy users
        high_burst_threshold: Data-driven burst threshold
        high_entropy_threshold: Data-driven entropy threshold
    
    Returns:
        Risk level: 0 (minor), 1 (major), 2 (critical)
    """
    
    megabytes_sent = row.get("megabytes_sent", 0) or 0
    hour = row.get("hour", 12)
    
    # Critical: decoy user interaction
    # if user in decoy_users:
    #     return 2
    
    # Major risk conditions
    major_conditions = [
        # Large upload after hours to new destination
        (business_rules.is_after_hours(hour) and 
         business_rules.is_large_upload(megabytes_sent) and 
         row.get("first_time_destination", False)),

         # After hours + medium upload + first time destination (SPECIFIC)
        ((hour < 7 or hour >= 19) and 
         megabytes_sent > business_rules.rules["medium_upload_threshold"] and 
         row.get('first_time_destination', False)),

         # Large uploads to suspicious domains (SPECIFIC)
        (megabytes_sent > business_rules.rules["large_upload_threshold"] and 
         row.get('destination_domain', '')),
         # USB + weekend + large upload
        (row.get('channel', '') == 'USB' and 
         row.get('is_weekend', False) and 
         megabytes_sent > business_rules.rules['usb_weekend_large_threshold']),
        
        # Upload to suspicious domain
        # business_rules.is_suspicious_domain(row.get("destination_domain", "")),
        
        # Large USB transfer on weekend
        (row.get("is_usb", False) and 
         business_rules.is_large_upload(megabytes_sent) and 
         row.get("is_weekend", False)),

         # Large USB transfer on weekend
        # (
        (row.get('channel', '') == 'USB'and 
        ((hour < 7 or hour >= 19) and megabytes_sent > business_rules.rules['small_upload_threshold']))
        or (row.get("megabytes_sent", 0) > business_rules.rules['small_upload_threshold'] and 
        row.get('is_weekend', False)),
        
        # High daily upload activity (fixed threshold)
        # row.get('user_upload_count', 0) > 100, #and
        #  row.get("uploads_last_24h", 20) > business_rules.rules["high_daily_uploads"]),

        #  row.get("uploads_last_24h", 20) > business_rules.rules["high_daily_uploads"]
         ((hour < 7 or hour >= 19) and megabytes_sent > business_rules.rules['small_upload_threshold'])
        or (row.get("megabytes_sent", 2) > business_rules.rules['small_upload_threshold'] and 
        row.get('is_weekend', False)),


         (row.get('hour', 12) in [1,2, 3, 4] and  # Deep night hours only
         row.get('megabytes_sent', 0) > 0.1),   # + medium size

        # ANY after hours upload > 10MB
        ((hour < 7 or hour >= 19) and megabytes_sent > business_rules.rules['medium_upload_threshold']),

         # ANY first-time destination > 5MB
        (row.get('first_time_destination', False) and megabytes_sent > 5),

        # HTTP uploads (only 543 HTTP vs 99,991 EMAIL)
       (row.get('channel', '') == 'HTTP'and 
        # row.get('user_upload_count', 0) > 10 and
        #  (row.get("uploads_last_24h", 0) > business_rules.rules["high_daily_uploads"]
        ((hour < 7 or hour >= 19) and megabytes_sent > business_rules.rules['small_upload_threshold']))
        and (row.get("megabytes_sent", 2) > business_rules.rules['small_upload_threshold'])
        and row.get('is_weekend', False),
        
        # ANY weekend activity
        (row.get('is_weekend', False) and row.get('megabytes_sent', 0) > 0.1),

        (row.get('channel', '') == 'HTTP' and row.get('is_weekend', False)
        and (row.get("megabytes_sent", 0) > 0.1)),  # Any HTTP upload on weekend

        (row.get('channel', '') == 'HTTP' and
          (row.get('hour', 12) < 7 or row.get('hour', 12) >= 19)),  # Any HTTP upload after hours

        (
        row.get('channel', '') == 'HTTP' and
        row.get('is_weekend', False) and
        row.get('megabytes_sent', 0) > 0.1 and  # Significant data
        row.get('first_time_destination', False)  # New destination
        ),

        (
        row.get('channel', '') == 'HTTP' and
        row.get('is_weekend', False) and
        row.get('after_hours', False)  # Weekend + Night = Very suspicious
        ),

        (
        row.get('channel', '') == 'EMAIL' and
        row.get('is_weekend', False) and
        row.get('destination_domain', '').endswith(('lockheedmartin.com', 'northropgrumman.com', 'boeing.com', '.org')) or
        # row.get('destination_domain', '') not in ['dtaa.com', 'company.com'] and
        (row.get('hour', 12) < 7 or row.get('hour', 12) >= 19) and
        row.get('megabytes_sent', 0) > 0.1  # Any file to competitor
        ),

        (
        row.get('channel', '') == 'HTTP' and
        row.get('is_weekend', False) and
        row.get('destination_domain', '').endswith(('lockheedmartin.com', 'northropgrumman.com', 'boeing.com', '.org')) or
        row.get('destination_domain', '') not in ['dtaa.com', 'company.com'] and
        (row.get('hour', 12) < 7 or row.get('hour', 12) >= 19) and
        row.get('megabytes_sent', 0) > 0.1  # Any file to competitor
        ),

        (
        row.get('destination_domain', '') not in ['dtaa.com', 'company.com'] and
        row.get('destination_domain', '') != '' and  # Not empty
        (row.get('hour', 12) < 7 or row.get('hour', 12) >= 19)
        ),

        # HTTP weekend to competitor domains
        (
            row.get('channel', '') == 'HTTP' and
            row.get('is_weekend', False) and
            row.get('destination_domain', '').endswith(('lockheedmartin.com', 'northropgrumman.com', 'boeing.com', '.org'))
        ),

        # Cross-organizational communication risk potential industrial espionage!
        (row.get('destination_domain', '') in ['lockheedmartin.com', 'northropgrumman.com', 'boeing.com'] and
        row.get('megabytes_sent', 0) > 0.1),  # Any file to competitor

        # Industry peer communication after hours potential industrial espionage!
        (row.get('destination_domain', '').endswith(('.com', '.org')) and 
        row.get('destination_domain', '') not in ['dtaa.com', 'gmail.com', 'yahoo.com'] and
        (row.get('hour', 12) < 7 or row.get('hour', 12) >= 19)),

        # # Rare domain category + large upload
        row.get('channel', '') == 'HTTP' and (row.get('rare_domain_category', False)),
        # row.get('megabytes_sent', 0) > 1

        # # High daily upload activity
        row.get('channel', '') == 'HTTP' and row.get('user_upload_count', 0) > 1
        and ('is_weekend', False) or (row.get('hour', 12) < 7 or row.get('hour', 12) >= 19),

        #row.get('channel', '') == 'EMAIL' and  row.get('user_upload_count', 0) > 95
        # and ('is_weekend', False) #or (row.get('hour', 12) < 7 or row.get('hour', 12) >= 19)
        # Statistical outliers
        row.get("post_burst", 0) > business_rules.rules["high_burst_threshold"],
        (row.get("destination_entropy", 0) > business_rules.rules["high_entropy_threshold"]),
        row.get('max_session_duration_anomaly', 0) > 2.0,
        row.get('max_device_usage_anomaly', 0) > 0.8,
        row.get('had_multiple_device_sessions', False),
        row.get('logon_risk_score', 0) > 0.7
    ]
    
    if any(major_conditions):
        return 1
    
    # Minor: everything else
    return 0

# Add this simple function to your existing business_rules.py:

def assign_risk_label_simple(row: Dict[str, Any], business_rules: BusinessRulesConfig) -> int:
    """Simple business rule classification."""
    score = 0.0
    
    # Large file
    if row.get("megabytes_sent", 0) > 10:
        score += 0.4
    
    # After hours
    if row.get("after_hours", False):
        score += 0.3
    
    # First time destination
    if row.get("first_time_destination", False):
        score += 0.2
    
    # Weekend
    if row.get("is_weekend", False):
        score += 0.1
    
    return 1 if score >= 0.5 else 0  # Binary classification


# def rule_baseline_robust(row_features: Dict[str, Any], 
#                         business_rules: BusinessRulesConfig,
#                         decoy_users: set = None,
#                         use_statistical_thresholds: bool = False, # Default to False
#                         high_burst_threshold: float = None,
#                         high_entropy_threshold: float = None) -> int:
#     """
#     Robust rule-based baseline that mirrors assign_risk_label logic.
#     the purpose of your rule baseline:
#     Tests if expert rules alone can compete with ML
#     Fair comparison - doesn't use training data statistics
    
#     Args:
#         row_features: Feature Dictionary
#         business_rules: business rules configuration
#         decoy_users: Set of decoy users (if available)
#         use_statistcal_thresholds: Whether to use data-driven thresholds
#         high_burst_threshold: Data-driven burst threshold (if use_statistical_thresholds = True)
#         high_entropy_threshold: Data-driven entropy threshold (if use_statistical_thresholds = True)
    
#     Returns:
#         Risk level: 0 (minor), 1 (major)
#     """

#     if use_statistical_thresholds and high_burst_threshold is not None and high_entropy_threshold is not None:
#         # Use exact same logic as training labels
#         return assign_risk_label_robust(
#             row_features, business_rules, decoy_users or set(),
#             high_burst_threshold, high_entropy_threshold
#         )
#     else:
#         # Pure rule-based logic without data-driven thresholds
#         user = row_features.get("user", "")
#         megabytes_sent = row_features.get("megabytes_sent", 0)
#         hour = row_features.get("hour", 12)
#         first_time_destination = row_features.get("first_time_destination", False)
        
#         # Critical: decoy user detection
#         # if decoy_users and user in decoy_users:
#         #     return 2
        
#         # Major risk conditions (subset of assign_risk_label logic)
#         major_conditions = [
#             # Large upload after hours to new destination
#             (business_rules.is_after_hours(hour) and 
#             business_rules.is_large_upload(megabytes_sent) and 
#             first_time_destination),
            
#             # Upload to suspicious domain
#             business_rules.is_suspicious_domain(row_features.get("destination_domain", "")),
            
#             # Large USB transfer on weekend
#             (row_features.get("is_usb", False) and 
#             business_rules.is_large_upload(megabytes_sent) and 
#             row_features.get("is_weekend", False)),
            
#             # High daily upload activity
#             row_features.get("uploads_last_24h", 0) > business_rules.rules["high_daily_uploads"],
            
#             # Simple size-based rules (without statistical thresholds)
#             megabytes_sent > business_rules.rules["medium_upload_threshold"],

#             # # Add statistical thresholds if available
#             # (high_burst_threshold is not None and 
#             # row_features.get("post_burst", 0) > high_burst_threshold),
            
#             # (high_entropy_threshold is not None and 
#             # row_features.get("destination_entropy", 0) > high_entropy_threshold),
#         ]
        
#         if any(major_conditions):
#             return 1
        
#         return 0