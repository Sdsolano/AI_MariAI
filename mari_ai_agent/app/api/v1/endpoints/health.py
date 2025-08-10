# mari_ai_agent/app/api/v1/endpoints/health.py
"""
Health check endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.connection import db_manager
from typing import Dict, Any

router = APIRouter()

@router.get("/", response_model=Dict[str, Any])
async def health_check():
    """
    Basic health check endpoint
    """
    return {
        "status": "healthy",
        "service": "Mari AI Agent",
        "version": "1.0.0"
    }

@router.get("/database", response_model=Dict[str, Any])
async def database_health_check(db: Session = Depends(get_db)):
    """
    Database connectivity health check
    """
    try:
        # Test database connection
        connection_ok = db_manager.test_connection()
        
        if not connection_ok:
            raise HTTPException(status_code=503, detail="Database connection failed")
        
        # Get table counts
        students_count = db_manager.get_table_count("estu_estudiantes")
        grades_count = db_manager.get_table_count("acad_actividades_notas")
        attendance_count = db_manager.get_table_count("student_attendance")
        
        return {
            "status": "healthy",
            "database": "connected",
            "tables": {
                "students": students_count,
                "grades": grades_count,
                "attendance": attendance_count
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database health check failed: {str(e)}")

@router.get("/services", response_model=Dict[str, Any])
async def services_health_check():
    """
    Services health check
    """
    try:
        # Check various services
        services_status = {
            "database": "healthy",
            "openai": "healthy",  # Could add actual OpenAI API check
            "redis": "unknown",   # Could add Redis connectivity check
        }
        
        return {
            "status": "healthy",
            "services": services_status
        }
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Services health check failed: {str(e)}")
