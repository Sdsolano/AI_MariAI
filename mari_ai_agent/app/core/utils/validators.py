# mari_ai_agent/app/core/utils/validators.py
from typing import Any, Optional
import re

def validate_student_id(student_id: Any) -> bool:
    """Valida que el ID del estudiante sea válido"""
    if not isinstance(student_id, int):
        return False
    return student_id > 0

def validate_grade_name(grade: str) -> bool:
    """Valida nombre de grado"""
    if not isinstance(grade, str):
        return False
    return len(grade.strip()) > 0

def validate_email(email: str) -> bool:
    """Valida formato de email"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_limit(limit: Any, max_limit: int = 1000) -> int:
    """Valida y sanitiza parámetro de límite"""
    if not isinstance(limit, int):
        return 50  # Default
    return min(max(1, limit), max_limit)
