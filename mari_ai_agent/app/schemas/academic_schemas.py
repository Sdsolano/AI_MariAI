# mari_ai_agent/app/schemas/academic_schemas.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from enum import Enum

class RiskLevel(str, Enum):
    MINIMO = "MINIMO"
    BAJO = "BAJO"
    MEDIO = "MEDIO"
    ALTO = "ALTO"
    UNKNOWN = "UNKNOWN"

class PerformanceLevel(str, Enum):
    EXCELENTE = "Excelente"
    BUENO = "Bueno"
    ACEPTABLE = "Aceptable"
    BAJO = "Bajo"
    SIN_EVALUAR = "Sin evaluar"

class StudentBasicInfo(BaseModel):
    student_id: int
    nombre_completo: str
    edad: Optional[int] = None
    genero: Optional[str] = None
    estrato: Optional[int] = None

class AcademicMetrics(BaseModel):
    total_grades: int = 0
    average_grade: float = 0.0
    subjects_count: int = 0
    attendance_rate: float = 0.0
    performance_distribution: Dict[str, int] = {}

class PlatformMetrics(BaseModel):
    total_sessions: int = 0
    total_hours: float = 0.0
    avg_session_hours: float = 0.0
    days_analyzed: int = 30

class RiskIndicators(BaseModel):
    risk_score: int = Field(..., ge=0, le=100, description="Risk score from 0-100")
    risk_level: RiskLevel
    factors: Dict[str, float] = {}
    recommendations_needed: bool = False

class StudentProfile(BaseModel):
    student_id: int
    student_info: StudentBasicInfo
    academic_metrics: AcademicMetrics
    platform_metrics: PlatformMetrics
    risk_indicators: RiskIndicators
    recommendations: List[str] = []
    extraction_date: str

    class Config:
        use_enum_values = True

class StudentRequest(BaseModel):
    student_id: int = Field(..., gt=0, description="ID del estudiante")

class GradeRequest(BaseModel):
    grade: str = Field(..., min_length=1, description="Nombre del grado")
    limit: int = Field(50, gt=0, le=200, description="Límite de resultados")

class BatchStudentsRequest(BaseModel):
    student_ids: List[int] = Field(..., min_items=1, max_items=100)

class InstitutionalOverview(BaseModel):
    institution_metrics: Dict[str, float]
    data_quality: Dict[str, int]
    analysis_date: str