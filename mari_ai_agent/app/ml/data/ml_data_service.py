# app/ml/data/ml_data_service.py
"""
🤖 SERVICIO DE EXTRACCIÓN DE DATOS PARA ML
==========================================
Extrae datos de la BD académica y los prepara para entrenamiento de modelos ML
"""

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from app.repositories.academic_repository import AcademicRepository
from app.services.academic_service import AcademicDataService

logger = logging.getLogger(__name__)

class MLDataExtractionService:
    """Servicio para extraer y preparar datos para Machine Learning"""
    
    def __init__(self, session: Session):
        self.session = session
        self.repository = AcademicRepository(session)
        self.academic_service = AcademicDataService(session)
    
    async def extract_students_dataframe(
        self, 
        min_grades: int = 5,
        limit: int = 1000,
        include_risk_labels: bool = True
    ) -> pd.DataFrame:
        """
        Extrae datos de estudiantes y retorna DataFrame listo para ML
        
        Args:
            min_grades: Mínimo número de notas por estudiante
            limit: Máximo número de estudiantes a extraer
            include_risk_labels: Si incluir etiquetas de riesgo pre-calculadas
            
        Returns:
            DataFrame con features ML-ready
        """
        try:
            logger.info(f"🔍 Extracting student data for ML (min_grades={min_grades}, limit={limit})")
            
            # 1. Obtener estudiantes con actividad suficiente
            active_students = self.repository.get_students_with_activity(
                min_grades=min_grades, 
                limit=limit
            )
            
            if not active_students:
                raise ValueError("No students found with sufficient activity")
            
            logger.info(f"✅ Found {len(active_students)} active students")
            
            # 2. Extraer datos detallados para cada estudiante
            students_data = []
            
            for i, student_basic in enumerate(active_students):
                student_id = student_basic['student_id']
                
                try:
                    # Obtener perfil completo
                    profile = await self.academic_service.get_student_profile(student_id)
                    
                    if profile:
                        # Extraer features estructuradas
                        features = self._extract_student_features(profile)
                        students_data.append(features)
                        
                        if (i + 1) % 50 == 0:
                            logger.info(f"   📊 Processed {i + 1}/{len(active_students)} students")
                            
                except Exception as e:
                    logger.warning(f"⚠️ Error processing student {student_id}: {e}")
                    continue
            
            # 3. Crear DataFrame
            df = pd.DataFrame(students_data)
            
            if df.empty:
                raise ValueError("No valid student data extracted")
            
            # 4. Limpiar y validar datos
            df = self._clean_dataframe(df)
            
            # 5. Agregar features derivadas
            df = self._add_derived_features(df)
            
            # 6. Agregar target variable (riesgo académico)
            if include_risk_labels:
                df = self._add_risk_labels(df)
            
            logger.info(f"🎉 DataFrame created successfully:")
            logger.info(f"   📊 Shape: {df.shape}")
            logger.info(f"   📋 Features: {list(df.columns)}")
            logger.info(f"   📈 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error extracting students dataframe: {e}")
            raise
    
    def _extract_student_features(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae features de ML del perfil de estudiante"""
        
        # Información básica
        basic_info = profile.get('basic_info', {})
        academic_metrics = profile.get('academic_metrics', {})
        platform_metrics = profile.get('platform_metrics', {})
        risk_indicators = profile.get('risk_indicators', {})
        
        features = {
            # === IDENTIFICACIÓN ===
            'student_id': basic_info.get('id'),
            'edad': basic_info.get('edad'),
            'genero': basic_info.get('genero'),
            'estrato': basic_info.get('estrato'),
            'grado': basic_info.get('grado'),
            
            # === MÉTRICAS ACADÉMICAS ===
            'total_grades': academic_metrics.get('total_grades', 0),
            'average_grade': academic_metrics.get('average_grade', 0.0),
            'subjects_count': academic_metrics.get('subjects_count', 0),
            'min_grade': academic_metrics.get('min_grade', 0.0),
            'max_grade': academic_metrics.get('max_grade', 0.0),
            'grade_std': academic_metrics.get('grade_std', 0.0),
            
            # Distribución de rendimiento
            'grades_excellent': academic_metrics.get('performance_distribution', {}).get('Excelente', 0),
            'grades_good': academic_metrics.get('performance_distribution', {}).get('Bueno', 0),
            'grades_satisfactory': academic_metrics.get('performance_distribution', {}).get('Satisfactorio', 0),
            'grades_low': academic_metrics.get('performance_distribution', {}).get('Bajo', 0),
            'grades_very_low': academic_metrics.get('performance_distribution', {}).get('Muy Bajo', 0),
            
            # === MÉTRICAS DE PLATAFORMA ===
            'total_sessions': platform_metrics.get('total_sessions', 0),
            'total_hours': platform_metrics.get('total_hours', 0.0),
            'avg_session_duration': platform_metrics.get('avg_session_duration', 0.0),
            'days_since_last_session': platform_metrics.get('days_since_last_session', 999),
            
            # === MÉTRICAS DE ASISTENCIA ===
            'attendance_rate': platform_metrics.get('attendance_rate', 0.0),
            
            # === INDICADORES DE RIESGO PRE-CALCULADOS ===
            'risk_score': risk_indicators.get('risk_score', 0),
            'risk_level': risk_indicators.get('risk_level', 'UNKNOWN'),
            
            # === TIMESTAMP ===
            'extraction_date': datetime.now().isoformat()
        }
        
        return features
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia y valida el DataFrame"""
        
        # Eliminar filas sin ID válido
        df = df.dropna(subset=['student_id'])
        df = df[df['student_id'] > 0]
        
        # Convertir tipos de datos
        numeric_columns = [
            'edad', 'total_grades', 'average_grade', 'subjects_count',
            'min_grade', 'max_grade', 'grade_std',
            'grades_excellent', 'grades_good', 'grades_satisfactory', 'grades_low', 'grades_very_low',
            'total_sessions', 'total_hours', 'avg_session_duration', 'days_since_last_session',
            'attendance_rate', 'risk_score'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Limpiar valores extremos
        df['days_since_last_session'] = np.clip(df['days_since_last_session'], 0, 365)
        df['attendance_rate'] = np.clip(df['attendance_rate'], 0, 1)
        df['average_grade'] = np.clip(df['average_grade'], 0, 5)
        
        # Eliminar duplicados por student_id
        df = df.drop_duplicates(subset=['student_id'], keep='first')
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrega features derivadas/calculadas"""
        
        # === FEATURES ACADÉMICAS DERIVADAS ===
        
        # Proporción de notas por categoría
        df['pct_excellent'] = df['grades_excellent'] / (df['total_grades'] + 1e-6)
        df['pct_good'] = df['grades_good'] / (df['total_grades'] + 1e-6)
        df['pct_low_combined'] = (df['grades_low'] + df['grades_very_low']) / (df['total_grades'] + 1e-6)
        
        # Rango de notas
        df['grade_range'] = df['max_grade'] - df['min_grade']
        
        # Consistencia académica (inverso del std)
        df['grade_consistency'] = 1 / (df['grade_std'] + 0.1)
        
        # === FEATURES DE ACTIVIDAD DE PLATAFORMA ===
        
        # Intensidad de uso
        df['hours_per_session'] = df['total_hours'] / (df['total_sessions'] + 1e-6)
        df['sessions_per_week'] = df['total_sessions'] / 12  # Asumiendo ~12 semanas de periodo
        
        # Engagement reciente
        df['is_recent_user'] = (df['days_since_last_session'] <= 7).astype(int)
        df['is_active_user'] = (df['total_sessions'] >= 10).astype(int)
        
        # === FEATURES DE RIESGO COMBINADAS ===
        
        # Score compuesto simple
        df['academic_performance_score'] = (
            df['average_grade'] * 0.4 +
            df['pct_excellent'] * 2.0 +
            df['pct_good'] * 1.0 -
            df['pct_low_combined'] * 1.5
        )
        
        # Engagement score
        df['platform_engagement_score'] = (
            np.log1p(df['total_sessions']) * 0.3 +
            np.log1p(df['total_hours']) * 0.3 +
            df['is_recent_user'] * 0.4
        )
        
        return df
    
    def _add_risk_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrega etiquetas de riesgo como target variables"""
        
        # Mapear risk_level a valores numéricos
        risk_mapping = {
            'MINIMO': 0,
            'BAJO': 1,
            'MEDIO': 2,
            'ALTO': 3,
            'UNKNOWN': -1
        }
        
        df['risk_level_numeric'] = df['risk_level'].map(risk_mapping).fillna(-1)
        
        # Target binario: Alto riesgo vs No alto riesgo
        df['is_high_risk'] = (df['risk_level_numeric'] >= 2).astype(int)
        
        # Target multi-clase
        df['risk_category'] = df['risk_level_numeric']
        
        return df
    
    async def get_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Genera resumen estadístico de las features"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        summary = {
            'dataset_info': {
                'total_students': len(df),
                'total_features': len(df.columns),
                'numeric_features': len(numeric_cols),
                'categorical_features': len(categorical_cols),
                'missing_values': df.isnull().sum().sum()
            },
            'academic_metrics': {
                'avg_total_grades': df['total_grades'].mean(),
                'avg_grade_average': df['average_grade'].mean(),
                'avg_subjects_count': df['subjects_count'].mean()
            },
            'risk_distribution': df['risk_level'].value_counts().to_dict() if 'risk_level' in df else {},
            'feature_correlations': df[numeric_cols].corr()['risk_score'].abs().sort_values(ascending=False).head(10).to_dict() if 'risk_score' in df else {}
        }
        
        return summary

# ================================
# FUNCIÓN DE UTILIDAD PRINCIPAL
# ================================

async def create_ml_dataframe(
    session: Session,
    min_grades: int = 5,
    limit: int = 1000
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Función principal para crear DataFrame ML-ready
    
    Returns:
        Tuple[DataFrame, Summary]: DataFrame listo para ML y resumen estadístico
    """
    
    service = MLDataExtractionService(session)
    
    # Extraer datos
    df = await service.extract_students_dataframe(
        min_grades=min_grades,
        limit=limit,
        include_risk_labels=True
    )
    
    # Generar resumen
    summary = await service.get_feature_summary(df)
    
    return df, summary