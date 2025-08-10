# mari_ai_agent/app/services/academic_service.py
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from app.repositories.academic_repository import AcademicRepository
from app.models.academic import StudentSummary
from app.core.utils.validators import validate_student_id
from app.utils.exceptions import StudentNotFoundError, DataValidationError
import logging

logger = logging.getLogger(__name__)

class AcademicDataService:
    """Servicio principal para datos académicos"""
    
    def __init__(self, session: Session):
        self.session = session
        self.repository = AcademicRepository(session)
    
    async def get_student_profile(self, student_id: int) -> Dict[str, Any]:
        """Obtiene perfil completo del estudiante"""
        try:
            # Validar ID
            if not validate_student_id(student_id):
                raise DataValidationError(f"Invalid student ID: {student_id}")
            
            # Obtener datos comprehensivos
            profile = self.repository.get_student_comprehensive_data(student_id)
            
            if not profile:
                raise StudentNotFoundError(f"Student {student_id} not found")
            
            # Enriquecer con análisis adicional
            profile['risk_indicators'] = self._calculate_risk_indicators(profile)
            profile['recommendations'] = self._generate_basic_recommendations(profile)
            
            logger.info(f"✅ Student profile generated for ID: {student_id}")
            return profile
            
        except Exception as e:
            logger.error(f"❌ Error getting student profile {student_id}: {e}")
            raise
    
    async def get_students_by_grade(self, grade: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Obtiene estudiantes por grado con métricas básicas"""
        try:
            students = self.repository.get_students_by_grade(grade, limit)
            
            # Enriquecer con métricas básicas
            enriched_students = []
            for student in students:
                try:
                    profile = await self.get_student_profile(student.id)
                    enriched_students.append({
                        'student_id': student.id,
                        'nombre_completo': student.nombre_completo,
                        'basic_metrics': profile.get('academic_metrics', {}),
                        'risk_level': profile.get('risk_indicators', {}).get('risk_level', 'unknown')
                    })
                except Exception as e:
                    logger.warning(f"Error enriching student {student.id}: {e}")
                    continue
            
            logger.info(f"✅ Retrieved {len(enriched_students)} students for grade {grade}")
            return enriched_students
            
        except Exception as e:
            logger.error(f"❌ Error getting students by grade {grade}: {e}")
            raise
    
    async def get_batch_student_data(self, student_ids: List[int]) -> List[Dict[str, Any]]:
        """Obtiene datos en batch para múltiples estudiantes"""
        try:
            results = self.repository.get_batch_student_data(student_ids)
            
            # Agregar indicadores de riesgo a cada estudiante
            for result in results:
                result['risk_indicators'] = self._calculate_risk_indicators(result)
            
            logger.info(f"✅ Batch data retrieved for {len(results)}/{len(student_ids)} students")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in batch student data retrieval: {e}")
            raise
    
    def _calculate_risk_indicators(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula indicadores básicos de riesgo académico"""
        try:
            metrics = student_data.get('academic_metrics', {})
            platform_metrics = student_data.get('platform_metrics', {})
            
            # Factores de riesgo
            avg_grade = metrics.get('average_grade', 0.0)
            attendance_rate = metrics.get('attendance_rate', 0.0)
            platform_hours = platform_metrics.get('total_hours', 0.0)
            total_grades = metrics.get('total_grades', 0)
            
            # Calcular score de riesgo (0-100, donde 100 es mayor riesgo)
            risk_score = 0
            
            # Factor nota promedio (40% del score)
            if avg_grade < 3.0:
                risk_score += 40
            elif avg_grade < 3.5:
                risk_score += 25
            elif avg_grade < 4.0:
                risk_score += 10
            
            # Factor asistencia (30% del score)
            if attendance_rate < 0.7:
                risk_score += 30
            elif attendance_rate < 0.8:
                risk_score += 20
            elif attendance_rate < 0.9:
                risk_score += 10
            
            # Factor actividad en plataforma (20% del score)
            if platform_hours < 5:
                risk_score += 20
            elif platform_hours < 10:
                risk_score += 10
            
            # Factor cantidad de evaluaciones (10% del score)
            if total_grades < 5:
                risk_score += 10
            elif total_grades < 10:
                risk_score += 5
            
            # Determinar nivel de riesgo
            if risk_score >= 70:
                risk_level = 'ALTO'
            elif risk_score >= 40:
                risk_level = 'MEDIO'
            elif risk_score >= 20:
                risk_level = 'BAJO'
            else:
                risk_level = 'MINIMO'
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'factors': {
                    'academic_performance': avg_grade,
                    'attendance_rate': attendance_rate,
                    'platform_engagement': platform_hours,
                    'evaluation_frequency': total_grades
                },
                'recommendations_needed': risk_score >= 40
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk indicators: {e}")
            return {
                'risk_score': 50,
                'risk_level': 'UNKNOWN',
                'factors': {},
                'recommendations_needed': True
            }
    
    def _generate_basic_recommendations(self, student_data: Dict[str, Any]) -> List[str]:
        """Genera recomendaciones básicas basadas en el perfil"""
        recommendations = []
        
        try:
            risk_indicators = student_data.get('risk_indicators', {})
            risk_level = risk_indicators.get('risk_level', 'UNKNOWN')
            factors = risk_indicators.get('factors', {})
            
            if risk_level in ['ALTO', 'MEDIO']:
                # Recomendaciones por factor específico
                if factors.get('academic_performance', 0) < 3.5:
                    recommendations.append("Programar sesiones de tutoría académica")
                    recommendations.append("Revisar material de refuerzo en asignaturas con bajo rendimiento")
                
                if factors.get('attendance_rate', 1.0) < 0.8:
                    recommendations.append("Contactar al estudiante para mejorar asistencia")
                    recommendations.append("Identificar barreras para la asistencia regular")
                
                if factors.get('platform_engagement', 0) < 10:
                    recommendations.append("Motivar mayor participación en actividades virtuales")
                    recommendations.append("Proporcionar capacitación en uso de plataforma")
                
                if factors.get('evaluation_frequency', 0) < 8:
                    recommendations.append("Incentivar participación en más actividades evaluativas")
            
            elif risk_level == 'BAJO':
                recommendations.append("Mantener el buen rendimiento actual")
                recommendations.append("Considerar actividades de liderazgo o tutoría entre pares")
            
            elif risk_level == 'MINIMO':
                recommendations.append("Excelente rendimiento - considerar programas de enriquecimiento")
                recommendations.append("Oportunidades de mentoría para otros estudiantes")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations = ["Revisar perfil del estudiante con el equipo académico"]
        
        return recommendations

class AcademicStatsService:
    """Servicio para estadísticas académicas institucionales"""
    
    def __init__(self, session: Session):
        self.session = session
        self.repository = AcademicRepository(session)
    
    async def get_institutional_overview(self) -> Dict[str, Any]:
        """Obtiene overview general de la institución"""
        try:
            # Estudiantes activos con actividad
            active_students = self.repository.get_students_with_activity(min_grades=1, limit=1000)
            
            # Estadísticas generales
            total_students = len(active_students)
            avg_grades_per_student = sum(s['total_notas'] for s in active_students) / total_students if total_students > 0 else 0
            avg_subjects_per_student = sum(s['materias_count'] for s in active_students) / total_students if total_students > 0 else 0
            overall_average = sum(float(s['promedio_notas'] or 0) for s in active_students) / total_students if total_students > 0 else 0
            
            return {
                'institution_metrics': {
                    'total_active_students': total_students,
                    'avg_grades_per_student': round(avg_grades_per_student, 1),
                    'avg_subjects_per_student': round(avg_subjects_per_student, 1),
                    'overall_grade_average': round(overall_average, 2)
                },
                'data_quality': {
                    'students_with_sufficient_data': len([s for s in active_students if s['total_notas'] >= 10]),
                    'students_with_moderate_data': len([s for s in active_students if 5 <= s['total_notas'] < 10]),
                    'students_with_limited_data': len([s for s in active_students if s['total_notas'] < 5])
                },
                'analysis_date': str(datetime.now().date())
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting institutional overview: {e}")
            raise