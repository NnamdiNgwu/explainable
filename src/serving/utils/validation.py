from flask import current_app
from typing import Dict, List
from ..config.settings import Config

def handle_missing_features(event: Dict) -> Dict:
    """Add missing features with defaults from training."""
    defaults = current_app.ml_models['feature_defaults']
    
    complete_event = defaults.copy()
    complete_event.update(event)
    return complete_event

def validate_event(event: Dict) -> List[str]:
    """Check for missing required fields."""
    required_features = set(current_app.ml_models['feature_names'])
    provided_features = set(event.keys())

    missing  = list(required_features - provided_features)
    return missing

    

def get_sample_event() -> Dict:
    """Return a valid sample event using training features."""
    # # Use one of the background events, but ensure it has all required features
    
    if Config.BACKGROUND_EVENTS:
        base_event = Config.BACKGROUND_EVENTS[0].copy()
    else:
        base_event = {}
    
    # Fill with defaults for any missing features
    return handle_missing_features(base_event)


