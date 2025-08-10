# mari_ai_agent/app/db/session.py
"""
Database session management
"""

from sqlalchemy.orm import Session
from app.db.connection import db_manager
from typing import Generator

def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get database session
    """
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()

# Alias for compatibility
get_db_session = get_db
