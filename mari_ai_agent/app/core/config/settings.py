# mari_ai_agent/app/core/config/settings.py
import os
from pydantic_settings import BaseSettings
from app.core.config.database import DatabaseSettings

class Settings(BaseSettings):
    """Configuración global de la aplicación"""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Mari AI - Academic Intelligence Agent"
    PROJECT_VERSION: str = "1.0.0"
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = ENVIRONMENT == "development"
    
    # CORS
    BACKEND_CORS_ORIGINS: list = ["*"]  # In production, specify allowed origins
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("API_KEY", "")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Database
    database: DatabaseSettings = DatabaseSettings()
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignorar variables extra del .env

# Global settings instance
settings = Settings()