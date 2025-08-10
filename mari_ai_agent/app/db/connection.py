# mari_ai_agent/app/db/connection.py
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator
import logging
from app.core.config.database import db_settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Gestor centralizado de conexiones a BD académica"""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Inicializa el engine de SQLAlchemy"""
        try:
            self.engine = create_engine(
                db_settings.academic_database_url,
                poolclass=QueuePool,
                pool_size=db_settings.ACADEMIC_DB_POOL_SIZE,
                max_overflow=db_settings.ACADEMIC_DB_MAX_OVERFLOW,
                pool_timeout=db_settings.ACADEMIC_DB_POOL_TIMEOUT,
                pool_pre_ping=True,
                echo=False  # Set to True for SQL debugging
            )
            
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info("✅ Database engine initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Context manager para sesiones de BD"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Prueba la conexión a la BD"""
        try:
            with self.get_session() as session:
                result = session.execute(text("SELECT 1"))
                result.fetchone()
            logger.info("✅ Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False
    
    def get_table_count(self, table_name: str) -> int:
        """Obtiene el conteo de registros de una tabla"""
        try:
            with self.get_session() as session:
                result = session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                count = result.scalar()
            return count
        except Exception as e:
            logger.error(f"Error counting {table_name}: {e}")
            return 0

# Base para modelos SQLAlchemy
Base = declarative_base()

# Instancia global del gestor
db_manager = DatabaseManager()

# Dependency para FastAPI
def get_db_session() -> Generator[Session, None, None]:
    """Dependency para obtener sesión de BD en endpoints"""
    with db_manager.get_session() as session:
        yield session