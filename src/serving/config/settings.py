import os
import pathlib

class Config:
    """Base configuration."""
    
    # Model configuration
    MODEL_DIR = os.environ.get(
        "MODEL_DIR", 
        pathlib.Path(__file__).resolve().parents[3] / "data_processed"
    )
    
    DEVICE = os.environ.get(
        "DEVICE", 
        "cuda:0" if __import__('torch').cuda.is_available() else "cpu"
    )
    
    DEBUG_MODE = bool(int(os.environ.get("DEBUG", "0")))
    
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')
    
    # CORS configuration
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')

    # Monitoring
    ENABLE_METRICS = bool(int(os.environ.get("ENABLE_METRICS", "1")))
    
    # Logging configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # Background events for explanations. Could stream 
    BACKGROUND_EVENTS = [
        {
            "post_burst": 5, "destination_entropy": 6.2, "hour": 23, "megabytes_sent": 25.8,
            "uploads_last_24h": 12, "user_upload_count": 8, "user_mean_upload_size": 15.3,
            "user_std_upload_size": 4.2, "user_unique_destinations": 6, "user_destination_count": 10,
            "attachment_count": 2, "bcc_count": 1, "cc_count": 0, "size": 25.8,
            "first_time_destination": True, "after_hours": True, "is_large_upload": True,
            "to_suspicious_domain": True, "is_usb": False, "is_weekend": False,
            "has_attachments": True, "is_from_user": True,
            "destination_domain": "suspicious.net", "user": "anomalous_user", 
            "channel": "HTTP", "from_domain": "company.com"
        },
        {
            "post_burst": 1, "destination_entropy": 2.1, "hour": 14, "megabytes_sent": 2.4,
            "uploads_last_24h": 2, "user_upload_count": 3, "user_mean_upload_size": 2.8,
            "user_std_upload_size": 0.5, "user_unique_destinations": 1, "user_destination_count": 2,
            "attachment_count": 0, "bcc_count": 0, "cc_count": 1, "size": 2.4,
            "first_time_destination": False, "after_hours": False, "is_large_upload": False,
            "to_suspicious_domain": False, "is_usb": False, "is_weekend": False,
            "has_attachments": False, "is_from_user": True,
            "destination_domain": "internal.com", "user": "normal_user", 
            "channel": "HTTP", "from_domain": "company.com"
        }
    ]

    
class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    LOG_LEVEL = 'INFO'