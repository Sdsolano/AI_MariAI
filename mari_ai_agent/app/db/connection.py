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
        # <<< CAMBIO: Ya no inicializamos un solo engine.
        # Ahora guardaremos un engine y un sessionmaker para cada DB que nos pidan.
        self.engines = {}
        self.SessionMakers = {}
    
    # <<< CAMBIO: Este método reemplaza al _initialize_engine.
    # Es el "cerebro" que crea y guarda (cachea) los pools de conexiones.
    def _get_or_create_engine_for_url(self, db_url: str):
        """
        Crea un engine y un SessionMaker para una URL de base de datos específica
        solo si no existen previamente.
        """
        # Si ya tenemos un engine para esta URL, no hacemos nada.
        if db_url in self.engines:
            return

        # Si es la primera vez que vemos esta URL, creamos un nuevo engine y su pool.
        logger.info(f"Creando nuevo pool de conexiones para: {db_url.split('@')[-1]}")
        engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=10,       # Puedes tomar estos valores de un config
            max_overflow=20,
            pool_timeout=30,
            pool_pre_ping=True,
            echo=False          # Recomendable ponerlo en False para producción
        )
        
        # Guardamos el engine y el sessionmaker en nuestros diccionarios.
        self.engines[db_url] = engine
        self.SessionMakers[db_url] = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # <<< CAMBIO: Este es el nuevo método público.
    # Ahora requiere la db_url para saber a qué base de datos conectarse.
    @contextmanager
    def get_session_for_url(self, db_url: str) -> Generator[Session, None, None]:
        """Context manager para obtener una sesión de BD usando una URL dinámica."""
        if not db_url:
            raise ValueError("La URL de la base de datos no puede estar vacía.")

        # 1. Asegurarse de que el engine para esta URL esté listo.
        self._get_or_create_engine_for_url(db_url)
        
        # 2. Obtener el SessionMaker específico para esta URL.
        SessionLocal = self.SessionMakers[db_url]
        session = SessionLocal()
        
        # 3. El resto de la lógica es la misma (transacciones seguras).
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error en la sesión para {db_url.split('@')[-1]}: {e}")
            raise
        finally:
            session.close()
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
                echo=True  # Set to True for SQL debugging
            )
            
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info("✅ Database engine initialized successfully")
            logger.info(f"🔗 Database URL: {db_settings.academic_database_url}")
            
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