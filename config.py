"""
Configuration settings for the Tone Converter application
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Server settings
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    
    # Model settings
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')
    DATASET_PATH = os.path.join(os.path.dirname(__file__), 'datasets')
    
    # API settings
    MAX_TEXT_LENGTH = 5000
    MIN_TEXT_LENGTH = 3
    DEFAULT_MODE = 'polite'
    
    # CORS settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*')
    
    # Available conversion modes
    AVAILABLE_MODES = [
        'polite',
        'formal',
        'informal',
        'professional',
        'friendly',
        'neutral'
    ]

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    # Add production-specific settings

class TestConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestConfig,
    'default': DevelopmentConfig
}

def get_config(env=None):
    """Get configuration based on environment"""
    if env is None:
        env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])
