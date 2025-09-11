# mari_ai_agent/app/api/v1/endpoints/academic.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from app.db.session import get_db
from app.services.academic_service import AcademicDataService, AcademicStatsService
from app.schemas.academic_schemas import (
    StudentProfile, StudentRequest, GradeRequest, 
    BatchStudentsRequest, InstitutionalOverview
)
from app.utils.exceptions import StudentNotFoundError, DataValidationError
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# ========== ENDPOINTS DE ESTUDIANTES ==========

@router.get("/student/{student_id}", response_model=StudentProfile)
async def get_student_profile(
    student_id: int,
    db: Session = Depends(get_db)
):
    """
    Obtiene el perfil académico completo de un estudiante.
    
    Incluye:
    - Información básica del estudiante
    - Métricas académicas (promedio, asignaturas, etc.)
    - Métricas de uso de plataforma
    - Indicadores de riesgo académico
    - Recomendaciones personalizadas
    """
    try:
        service = AcademicDataService(db)
        profile = await service.get_student_profile(student_id)
        return profile
        
    except StudentNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting student profile: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/students/grade/{grade}")
async def get_students_by_grade(
    grade: str,
    limit: int = Query(50, ge=1, le=200, description="Límite de resultados"),
    db: Session = Depends(get_db)
):
    """
    Obtiene estudiantes de un grado específico con métricas básicas.
    
    Parámetros:
    - grade: Nombre del grado (ej: "5", "Quinto", "Grado 5")
    - limit: Número máximo de estudiantes a retornar
    """
    try:
        service = AcademicDataService(db)
        students = await service.get_students_by_grade(grade, limit)
        
        return {
            "grade": grade,
            "total_students": len(students),
            "students": students
        }
        
    except Exception as e:
        logger.error(f"Error getting students by grade: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/students/batch")
async def get_batch_student_data(
    request: BatchStudentsRequest,
    db: Session = Depends(get_db)
):
    """
    Obtiene datos de múltiples estudiantes en una sola petición.
    
    Útil para análisis masivos y entrenamiento de modelos ML.
    """
    try:
        service = AcademicDataService(db)
        results = await service.get_batch_student_data(request.student_ids)
        
        return {
            "requested_students": len(request.student_ids),
            "processed_students": len(results),
            "students_data": results
        }
        
    except Exception as e:
        logger.error(f"Error in batch student data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ========== ENDPOINTS DE ESTADÍSTICAS ==========

@router.get("/stats/institutional", response_model=InstitutionalOverview)
async def get_institutional_overview(
    db: Session = Depends(get_db)
):
    """
    Obtiene overview general de métricas institucionales.
    
    Incluye:
    - Número total de estudiantes activos
    - Promedios institucionales
    - Calidad de datos disponibles
    """
    try:
        service = AcademicStatsService(db)
        overview = await service.get_institutional_overview()
        return overview
        
    except Exception as e:
        logger.error(f"Error getting institutional overview: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/health/database")
async def check_database_health(
    db: Session = Depends(get_db)
):
    """
    Verifica el estado de la conexión a la base de datos académica.
    """
    try:
        from app.db.connection import db_manager
        
        # Test basic connection
        connection_ok = db_manager.test_connection()
        
        if not connection_ok:
            raise HTTPException(status_code=503, detail="Database connection failed")
        
        # Get basic stats
        students_count = db_manager.get_table_count("estu_estudiantes")
        grades_count = db_manager.get_table_count("acad_actividades_notas")
        attendance_count = db_manager.get_table_count("student_attendance")
        
        return {
            "status": "healthy",
            "database": "academic_db",
            "connection": "active",
            "tables": {
                "students": students_count,
                "grades": grades_count,
                "attendance": attendance_count
            },
            "timestamp": str(datetime.now())
        }
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Database health check failed: {str(e)}")

# ========== ENDPOINTS DE TESTING ==========

@router.get("/test/sample-students")
async def get_sample_students(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """
    Obtiene una muestra de estudiantes para testing.
    Solo para desarrollo y debugging.
    """
    try:
        from app.repositories.academic_repository import AcademicRepository
        
        repo = AcademicRepository(db)
        students = repo.get_students_with_activity(min_grades=3, limit=limit)
        
        # Simplificar output para testing
        sample = []
        for student in students:
            sample.append({
                "student_id": student["student_id"],
                "nombre": student["nombre_completo"],
                "total_notas": student["total_notas"],
                "promedio": round(float(student["promedio_notas"] or 0), 2)
            })
        
        return {
            "sample_size": len(sample),
            "students": sample,
            "note": "Sample data for testing purposes only"
        }
        
    except Exception as e:
        logger.error(f"Error getting sample students: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

