# mari_ai_agent/app/core/config/database.py
import os
from pydantic_settings import BaseSettings
from typing import Optional

class DatabaseSettings(BaseSettings):
    """Configuración de base de datos académica"""
    
    # Mari AI Academic Database - LOCAL SETUP
    ACADEMIC_DB_HOST: str = "localhost"
    ACADEMIC_DB_PORT: int = 5432
    ACADEMIC_DB_NAME: str = "aca_2"
    ACADEMIC_DB_USER: str = "postgres"
    ACADEMIC_DB_PASSWORD: str = os.getenv("ACADEMIC_DB_PASSWORD", "samuel1902")
    
    # Connection Pool Settings
    ACADEMIC_DB_POOL_SIZE: int = 10
    ACADEMIC_DB_MAX_OVERFLOW: int = 20
    ACADEMIC_DB_POOL_TIMEOUT: int = 30
    
    # Redis for Caching
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CACHE_TTL: int = 3600  # 1 hour
    
    @property
    def academic_database_url(self) -> str:
        """URL completa de conexión a BD académica"""
        return f"postgresql://{self.ACADEMIC_DB_USER}:{self.ACADEMIC_DB_PASSWORD}@{self.ACADEMIC_DB_HOST}:{self.ACADEMIC_DB_PORT}/{self.ACADEMIC_DB_NAME}"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignorar variables extra del .env

# Instancia global
db_settings = DatabaseSettings()