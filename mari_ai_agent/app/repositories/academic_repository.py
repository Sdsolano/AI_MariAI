# mari_ai_agent/app/repositories/academic_repository.py
from sqlalchemy.orm import Session
from sqlalchemy import text, func, and_, desc
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, date, timedelta
import pandas as pd

from app.models.academic import (
    Student, StudentEnrollment, AcademicEnrollment, 
    ActivityGrade, StudentAttendance, AcademicPeriod, Subject, UserSession
)

class AcademicRepository:
    """Repository para datos académicos"""
    
    def __init__(self, session: Session):
        self.session = session
    
    # ========== ESTUDIANTES ==========
    
    def get_student_by_id(self, student_id: int) -> Optional[Student]:
        """Obtiene estudiante por ID"""
        return self.session.query(Student).filter(
            Student.id == student_id,
            Student.estado == 'on'
        ).first()
    
    def get_students_by_grade(self, grade_name: str, limit: int = 100) -> List[Student]:
        """Obtiene estudiantes por grado"""
        query = text("""
            SELECT DISTINCT e.*
            FROM estu_estudiantes e
            JOIN acad_estumatricula em ON e.id = em.idestudiante
            JOIN acad_gradosgrupos gg ON em.idgrados_grupos = gg.id
            JOIN acad_grados g ON gg.idgrado = g.id
            WHERE e.estado = 'on' 
              AND g.nombre ILIKE :grade_name
            ORDER BY e.id DESC
            LIMIT :limit
        """)
        
        result = self.session.execute(query, {
            'grade_name': f'%{grade_name}%',
            'limit': limit
        })
        
        # Convert to Student objects
        students = []
        for row in result:
            student = Student()
            for i, column in enumerate(result.keys()):
                setattr(student, column, row[i])
            students.append(student)
        
        return students
    
    def get_students_with_activity(self, min_grades: int = 5, limit: int = 200) -> List[Dict[str, Any]]:
        """Obtiene estudiantes con actividad académica mínima"""
        query = text("""
            SELECT 
                e.id as student_id,
                e.primer_nombre || ' ' || e.primer_apellido as nombre_completo,
                e.genero,
                EXTRACT(YEAR FROM AGE(e.fecha_nacimiento)) as edad,
                COUNT(DISTINCT an.id) as total_notas,
                COUNT(DISTINCT ma.idgrados_asignatura) as materias_count,
                AVG(an.nota) as promedio_notas,
                MIN(an.fecha_creacion::date) as primera_actividad,
                MAX(an.fecha_creacion::date) as ultima_actividad
            FROM estu_estudiantes e
            JOIN acad_estumatricula em ON e.id = em.idestudiante
            JOIN acad_matricula_academica ma ON em.id = ma.idestumatricula
            LEFT JOIN acad_actividades_notas an ON ma.id = an.idmatricula AND an.estado = 'on'
            WHERE e.estado = 'on'
            GROUP BY e.id, e.primer_nombre, e.primer_apellido, e.genero, e.fecha_nacimiento
            HAVING COUNT(DISTINCT an.id) >= :min_grades
            ORDER BY COUNT(DISTINCT an.id) DESC
            LIMIT :limit
        """)
        
        result = self.session.execute(query, {
            'min_grades': min_grades,
            'limit': limit
        })
        
        return [dict(row._mapping) for row in result]
    
    # ========== CALIFICACIONES ==========
    
    def get_student_grades(self, student_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtiene calificaciones completas de un estudiante"""
        query = text("""
            SELECT 
                an.id as nota_id,
                an.nota,
                an.fecha_creacion::date as fecha_nota,
                EXTRACT(YEAR FROM an.fecha_creacion) as ano_nota,
                EXTRACT(MONTH FROM an.fecha_creacion) as mes_nota,
                CASE 
                    WHEN an.nota >= 4.5 THEN 'Excelente'
                    WHEN an.nota >= 4.0 THEN 'Bueno'
                    WHEN an.nota >= 3.0 THEN 'Aceptable'
                    WHEN an.nota < 3.0 THEN 'Bajo'
                    ELSE 'Sin clasificar'
                END as rendimiento,
                COALESCE(asig.nombre, 'Asignatura no especificada') as asignatura,
                COALESCE(p.nombre, 'Período no especificado') as periodo,
                an.idactividad as actividad_id,
                asig.id as asignatura_id,
                p.id as periodo_id
            FROM estu_estudiantes e
            JOIN acad_estumatricula em ON e.id = em.idestudiante
            JOIN acad_matricula_academica ma ON em.id = ma.idestumatricula
            JOIN acad_actividades_notas an ON ma.id = an.idmatricula
            LEFT JOIN acad_actividades a ON an.idactividad = a.id
            LEFT JOIN acad_periodos p ON a.idperiodo = p.id
            LEFT JOIN acad_gradosasignaturas ga ON ma.idgrados_asignatura = ga.id
            LEFT JOIN acad_asignaturas asig ON ga.idasignatura = asig.id
            WHERE e.id = :student_id
              AND an.estado = 'on'
              AND an.nota IS NOT NULL
              AND an.nota BETWEEN 0 AND 5
            ORDER BY an.fecha_creacion DESC
            LIMIT :limit
        """)
        
        result = self.session.execute(query, {
            'student_id': student_id,
            'limit': limit
        })
        
        return [dict(row._mapping) for row in result]
    
    def get_grades_summary_by_subject(self, student_id: int) -> List[Dict[str, Any]]:
        """Resumen de calificaciones por asignatura"""
        query = text("""
            SELECT 
                asig.id as asignatura_id,
                asig.nombre as asignatura,
                COUNT(an.id) as total_notas,
                AVG(an.nota) as promedio,
                STDDEV(an.nota) as desviacion,
                MIN(an.nota) as nota_minima,
                MAX(an.nota) as nota_maxima,
                MIN(an.fecha_creacion::date) as primera_nota,
                MAX(an.fecha_creacion::date) as ultima_nota
            FROM estu_estudiantes e
            JOIN acad_estumatricula em ON e.id = em.idestudiante
            JOIN acad_matricula_academica ma ON em.id = ma.idestumatricula
            JOIN acad_actividades_notas an ON ma.id = an.idmatricula
            JOIN acad_gradosasignaturas ga ON ma.idgrados_asignatura = ga.id
            JOIN acad_asignaturas asig ON ga.idasignatura = asig.id
            WHERE e.id = :student_id
              AND an.estado = 'on'
              AND an.nota IS NOT NULL
            GROUP BY asig.id, asig.nombre
            ORDER BY promedio DESC
        """)
        
        result = self.session.execute(query, {'student_id': student_id})
        return [dict(row._mapping) for row in result]
    
    # ========== ASISTENCIA ==========
    
    def get_student_attendance(self, student_id: int, days: int = 90) -> List[Dict[str, Any]]:
        """Obtiene registros de asistencia recientes"""
        since_date = date.today() - timedelta(days=days)
        
        return self.session.query(StudentAttendance).filter(
            StudentAttendance.student_id == student_id,
            StudentAttendance.date >= since_date
        ).order_by(desc(StudentAttendance.date)).limit(100).all()
    
    def get_attendance_rate(self, student_id: int, days: int = 30) -> float:
        """Calcula tasa de asistencia"""
        since_date = date.today() - timedelta(days=days)
        
        total_records = self.session.query(StudentAttendance).filter(
            StudentAttendance.student_id == student_id,
            StudentAttendance.date >= since_date
        ).count()
        
        attended_records = self.session.query(StudentAttendance).filter(
            StudentAttendance.student_id == student_id,
            StudentAttendance.date >= since_date,
            StudentAttendance.attended == True
        ).count()
        
        return attended_records / total_records if total_records > 0 else 0.0
    
    # ========== USO DE PLATAFORMA ==========
    
    def get_platform_usage(self, student_id: int, days: int = 30) -> Dict[str, Any]:
        """Obtiene estadísticas de uso de plataforma"""
        since_date = datetime.now() - timedelta(days=days)
        
        sessions = self.session.query(UserSession).filter(
            UserSession.user_id == student_id,
            UserSession.login_timestamp >= since_date
        ).all()
        
        total_sessions = len(sessions)
        total_hours = 0.0
        
        for session in sessions:
            if session.logout_timestamp:
                duration = session.logout_timestamp - session.login_timestamp
                total_hours += duration.total_seconds() / 3600
        
        return {
            'total_sessions': total_sessions,
            'total_hours': round(total_hours, 2),
            'avg_session_hours': round(total_hours / total_sessions, 2) if total_sessions > 0 else 0.0,
            'days_analyzed': days
        }
    
    # ========== ANÁLISIS AGREGADO ==========
    
    def get_student_comprehensive_data(self, student_id: int) -> Dict[str, Any]:
        """Obtiene datos comprehensivos de un estudiante para ML"""
        student = self.get_student_by_id(student_id)
        if not student:
            return {}
        
        grades = self.get_student_grades(student_id)
        grades_summary = self.get_grades_summary_by_subject(student_id)
        attendance_rate = self.get_attendance_rate(student_id)
        platform_usage = self.get_platform_usage(student_id)
        
        # Features calculados
        total_grades = len(grades)
        avg_grade = sum(g['nota'] for g in grades) / total_grades if total_grades > 0 else 0.0
        subjects_count = len(grades_summary)
        
        # Rendimiento por categoría
        performance_dist = {}
        for grade in grades:
            perf = grade['rendimiento']
            performance_dist[perf] = performance_dist.get(perf, 0) + 1
        
        return {
            'student_id': student_id,
            'student_info': {
                'nombre_completo': student.nombre_completo,
                'edad': student.edad,
                'genero': student.genero,
                'estrato': student.estrato
            },
            'academic_metrics': {
                'total_grades': total_grades,
                'average_grade': round(avg_grade, 2),
                'subjects_count': subjects_count,
                'attendance_rate': round(attendance_rate, 2),
                'performance_distribution': performance_dist
            },
            'platform_metrics': platform_usage,
            'recent_grades': grades[:10],  # Últimas 10 notas
            'subjects_summary': grades_summary,
            'extraction_date': datetime.now().isoformat()
        }
    
    def get_batch_student_data(self, student_ids: List[int]) -> List[Dict[str, Any]]:
        """Obtiene datos de múltiples estudiantes en batch"""
        results = []
        for student_id in student_ids:
            try:
                data = self.get_student_comprehensive_data(student_id)
                if data:
                    results.append(data)
            except Exception as e:
                print(f"Error processing student {student_id}: {e}")
                continue
        
        return results