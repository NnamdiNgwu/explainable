from flask import Flask
from flask_cors import CORS
import logging
import os
from models.safe_smote import SafeSMOTE
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
        # _validate_feature_consistency(app)
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