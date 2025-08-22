from flask import Flask
from flask_cors import CORS
import logging
import os
from src.models.safe_smote import SafeSMOTE
import sys
sys.modules[__name__].SafeSMOTE = SafeSMOTE
# Ensure SafeSMOTE is available in the app context
from .config.settings import Config
from .models.model_loader import ModelLoader
from .blueprints import (
    health_bp,
    prediction_bp,
    explanation_bp,
    dashboard_bp,
    monitoring_bp,
    admin_bp
)
def _validate_feature_consistency(app):
    """Validate that features are consistent across components."""
    with app.app_context():
        from .utils.validation import get_sample_event
        from .models.encoders import encode_tabular, encode_sequence_semantic
        
        # Test with sample data
        sample = get_sample_event()
        
        try:
            # Test both encoding paths
            X_tab = encode_tabular(sample)
            cont, cat_high, cat_low = encode_sequence_semantic(
                sample,
                app.ml_models['feature_lists'],
                app.ml_models['embed_maps']
              )
            
            logging.info("✅ Feature consistency validation passed")
            logging.info(f"   - Features: {len(app.ml_models['feature_names'])}")
            logging.info(f"   - RF input shape: {X_tab.shape}")
            logging.info(f"   - Transformer input shapes: {cont.shape}, {cat_high.shape}, {cat_low.shape}")
            
        except Exception as e:
            raise ValueError(f"Feature consistency validation failed: {e}")

def create_app(config_name='production'):
    """Application factory pattern."""
    app = Flask(__name__)
    
    # Load configuration
    if config_name == 'production':
        app.config.from_object(Config)
    
    # Initialize extensions
    CORS(app)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load models once at startup
    try:
        with app.app_context():
            app.ml_models = ModelLoader.load_all_models(app.config.get('MODEL_DIR', 'data_processed'))
            logging.info("All models loaded successfully")
        
        # Validate feature consistency
        _validate_feature_consistency(app)
    except Exception as e:
        logging.error(f"Failed to load models: {e}")
        raise
    
    # Register blueprints
    app.register_blueprint(health_bp, url_prefix='/api/v1')
    app.register_blueprint(prediction_bp, url_prefix='/api/v1')
    app.register_blueprint(explanation_bp, url_prefix='/api/v1')
    app.register_blueprint(dashboard_bp, url_prefix='/api/v1/dashboard')
    app.register_blueprint(monitoring_bp, url_prefix='/api/v1/monitoring')
    app.register_blueprint(admin_bp, url_prefix='/api/v1/admin')
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {"error": "Endpoint not found"}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return {"error": "Internal server error"}, 500
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)