# mari_ai_agent/app/models/academic.py
from sqlalchemy import Column, Integer, String, Date, DateTime, Numeric, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.connection import Base
from datetime import datetime, date
from typing import Optional

class Student(Base):
    """Modelo para estudiantes (estu_estudiantes)"""
    __tablename__ = "estu_estudiantes"
    
    id = Column(Integer, primary_key=True, index=True)
    identificacion = Column(String, nullable=False, index=True)
    primer_nombre = Column(String, nullable=False)
    segundo_nombre = Column(String)
    primer_apellido = Column(String, nullable=False)
    segundo_apellido = Column(String)
    genero = Column(String)
    fecha_nacimiento = Column(Date)
    direccion = Column(String)
    telefono = Column(String)
    celular = Column(String)
    email = Column(String)
    estrato = Column(Integer)
    estado = Column(String, default='on')
    fecha_registro = Column(DateTime, default=func.now())
    
    @property
    def nombre_completo(self) -> str:
        """Nombre completo del estudiante"""
        nombres = [self.primer_nombre]
        if self.segundo_nombre:
            nombres.append(self.segundo_nombre)
        apellidos = [self.primer_apellido]
        if self.segundo_apellido:
            apellidos.append(self.segundo_apellido)
        return f"{' '.join(nombres)} {' '.join(apellidos)}"
    
    @property
    def edad(self) -> Optional[int]:
        """Edad calculada del estudiante"""
        if self.fecha_nacimiento:
            today = date.today()
            return today.year - self.fecha_nacimiento.year - ((today.month, today.day) < (self.fecha_nacimiento.month, self.fecha_nacimiento.day))
        return None

class StudentEnrollment(Base):
    """Modelo para matrículas de estudiantes (acad_estumatricula)"""
    __tablename__ = "acad_estumatricula"
    
    id = Column(Integer, primary_key=True, index=True)
    idestudiante = Column(Integer, ForeignKey("estu_estudiantes.id"), nullable=False)
    idgrados_grupos = Column(Integer, nullable=False)
    fecha_matricula = Column(DateTime, default=func.now())
    estado = Column(String, default='on')
    
    # Relationship
    student = relationship("Student", backref="enrollments")

class AcademicEnrollment(Base):
    """Modelo para matrícula académica (acad_matricula_academica)"""
    __tablename__ = "acad_matricula_academica"
    
    id = Column(Integer, primary_key=True, index=True)
    idestumatricula = Column(Integer, ForeignKey("acad_estumatricula.id"), nullable=False)
    idgrados_asignatura = Column(Integer, nullable=False)
    idperiodo = Column(Integer)
    observacion = Column(Text)
    fecha_creacion = Column(DateTime, default=func.now())
    
    # Relationship
    enrollment = relationship("StudentEnrollment", backref="academic_enrollments")

class ActivityGrade(Base):
    """Modelo para calificaciones de actividades (acad_actividades_notas)"""
    __tablename__ = "acad_actividades_notas"
    
    id = Column(Integer, primary_key=True, index=True)
    idactividad = Column(Integer)
    idmatricula = Column(Integer, ForeignKey("acad_matricula_academica.id"), nullable=False)
    nota = Column(Numeric(3, 2))  # Nota con 2 decimales
    fecha_creacion = Column(DateTime, default=func.now())
    estado = Column(String, default='on')
    idcompetencia_niveleducativo = Column(Integer)
    
    # Relationship
    academic_enrollment = relationship("AcademicEnrollment", backref="grades")
    
    @property
    def performance_level(self) -> str:
        """Nivel de rendimiento basado en la nota"""
        if self.nota is None:
            return 'Sin evaluar'
        elif self.nota >= 4.5:
            return 'Excelente'
        elif self.nota >= 4.0:
            return 'Bueno'
        elif self.nota >= 3.0:
            return 'Aceptable'
        else:
            return 'Bajo'

class StudentAttendance(Base):
    """Modelo para asistencia de estudiantes (student_attendance)"""
    __tablename__ = "student_attendance"
    
    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, nullable=False)
    student_id = Column(Integer, ForeignKey("estu_estudiantes.id"), nullable=False)
    student_name = Column(String, nullable=False)
    date = Column(Date, nullable=False)
    subject_id = Column(Integer, nullable=False)
    period_id = Column(Integer, nullable=False)
    idanio_lectivo = Column(Integer, nullable=False)
    attended = Column(Boolean)
    not_attended = Column(Boolean)
    late = Column(Boolean)
    excuse = Column(Boolean)
    observation = Column(Text)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationship
    student = relationship("Student", backref="attendance_records")

class AcademicPeriod(Base):
    """Modelo para períodos académicos (acad_periodos)"""
    __tablename__ = "acad_periodos"
    
    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String)
    porcentaje = Column(Numeric(5, 2))
    fecha_inicio = Column(Date)
    fecha_fin = Column(Date)
    idanio_lectivo = Column(Integer, nullable=False)
    fecha_inicio_recuperacion = Column(Date)
    fecha_fin_recuperacion = Column(Date)
    estado = Column(String, default='on')

class Subject(Base):
    """Modelo para asignaturas (acad_asignaturas)"""
    __tablename__ = "acad_asignaturas"
    
    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String)
    abr = Column(String, nullable=False)  # Abreviatura
    idarea = Column(Integer)
    estado = Column(String, default='on')
    iconid = Column(Integer)

class UserSession(Base):
    """Modelo para sesiones de usuario (user_sessions)"""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("estu_estudiantes.id"), nullable=False)
    login_timestamp = Column(DateTime, nullable=False)
    logout_timestamp = Column(DateTime)
    session_duration = Column(String)  # Interval type
    status = Column(String, default='active')
    
    # Relationship
    user = relationship("Student", backref="sessions")

# Modelo para resumen de estudiante (para ML)
class StudentSummary:
    """Clase para resumen de datos de estudiante (no tabla DB)"""
    
    def __init__(self, student_id: int):
        self.student_id = student_id
        self.total_grades = 0
        self.average_grade = 0.0
        self.attendance_rate = 0.0
        self.platform_usage_hours = 0.0
        self.subjects_count = 0
        self.risk_level = 'unknown'
        self.last_activity_date = None