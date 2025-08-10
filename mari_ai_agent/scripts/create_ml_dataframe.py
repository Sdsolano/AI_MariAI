# scripts/create_direct_ml_dataframe.py
"""
🚀 EXTRACCIÓN DIRECTA DE DATAFRAME ML
====================================
Versión simplificada que extrae datos directamente del repository
"""

import asyncio
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sqlalchemy import text
from typing import Dict, List, Any

# Agregar el directorio raíz al path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()

from app.db.connection import db_manager
from app.repositories.academic_repository import AcademicRepository

async def extract_ml_dataframe_direct(min_grades: int = 1, limit: int = 200) -> pd.DataFrame:
    """
    Extrae DataFrame ML directamente desde la base de datos
    Sin pasar por servicios complejos
    """
    
    print(f"🚀 DIRECT ML DATAFRAME EXTRACTION")
    print(f"   min_grades: {min_grades}")
    print(f"   limit: {limit}")
    print("-" * 50)
    
    with db_manager.get_session() as session:
        
        # 1. Query directa para obtener datos completos
        print("📊 Executing comprehensive student query...")
        
        comprehensive_query = text("""
            SELECT 
                e.id as student_id,
                e.primer_nombre || ' ' || COALESCE(e.segundo_nombre, '') || ' ' || 
                e.primer_apellido || ' ' || COALESCE(e.segundo_apellido, '') as nombre_completo,
                e.genero,
                EXTRACT(YEAR FROM AGE(CURRENT_DATE, e.fecha_nacimiento)) as edad,
                e.estrato,
                
                -- Métricas académicas básicas
                COUNT(DISTINCT an.id) as total_grades,
                AVG(an.nota) as average_grade,
                STDDEV(an.nota) as grade_std,
                MIN(an.nota) as min_grade,
                MAX(an.nota) as max_grade,
                COUNT(DISTINCT ma.idgrados_asignatura) as subjects_count,
                
                -- Distribución de rendimiento
                COUNT(CASE WHEN an.nota >= 4.5 THEN 1 END) as grades_excellent,
                COUNT(CASE WHEN an.nota >= 4.0 AND an.nota < 4.5 THEN 1 END) as grades_good,
                COUNT(CASE WHEN an.nota >= 3.0 AND an.nota < 4.0 THEN 1 END) as grades_satisfactory,
                COUNT(CASE WHEN an.nota >= 2.0 AND an.nota < 3.0 THEN 1 END) as grades_low,
                COUNT(CASE WHEN an.nota < 2.0 THEN 1 END) as grades_very_low,
                
                -- Fechas para análisis temporal
                MIN(an.fecha_creacion::date) as primera_actividad,
                MAX(an.fecha_creacion::date) as ultima_actividad,
                
                -- Cálculo de períodos activos
                COUNT(DISTINCT DATE_TRUNC('month', an.fecha_creacion)) as meses_activos
                
            FROM estu_estudiantes e
            JOIN acad_estumatricula em ON e.id = em.idestudiante
            JOIN acad_matricula_academica ma ON em.id = ma.idestumatricula
            LEFT JOIN acad_actividades_notas an ON ma.id = an.idmatricula AND an.estado = 'on'
            WHERE e.estado = 'on'
            GROUP BY e.id, e.primer_nombre, e.segundo_nombre, e.primer_apellido, 
                     e.segundo_apellido, e.genero, e.fecha_nacimiento, e.estrato
            HAVING COUNT(DISTINCT an.id) >= :min_grades
            ORDER BY COUNT(DISTINCT an.id) DESC
            LIMIT :limit
        """)
        
        result = session.execute(comprehensive_query, {
            'min_grades': min_grades,
            'limit': limit
        })
        
        # 2. Convertir a DataFrame
        rows = result.fetchall()
        columns = result.keys()
        
        if not rows:
            print("❌ No data returned from query!")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows, columns=columns)
        print(f"✅ Query returned {len(df)} students")
        
        # 3. Limpiar y preparar datos
        print("🧹 Cleaning and preparing data...")
        
        # Convertir tipos de datos
        numeric_columns = [
            'edad', 'total_grades', 'average_grade', 'grade_std', 
            'min_grade', 'max_grade', 'subjects_count',
            'grades_excellent', 'grades_good', 'grades_satisfactory', 
            'grades_low', 'grades_very_low', 'meses_activos'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Limpiar valores extremos
        df['edad'] = np.clip(df['edad'], 5, 25)  # Rango razonable para estudiantes
        df['average_grade'] = np.clip(df['average_grade'], 0, 5)
        df['grade_std'] = df['grade_std'].fillna(0)
        
        # 4. Agregar features derivadas
        print("🔧 Creating derived features...")
        
        # Proporciones de rendimiento
        df['pct_excellent'] = df['grades_excellent'] / (df['total_grades'] + 1e-6)
        df['pct_good'] = df['grades_good'] / (df['total_grades'] + 1e-6)
        df['pct_satisfactory'] = df['grades_satisfactory'] / (df['total_grades'] + 1e-6)
        df['pct_low_combined'] = (df['grades_low'] + df['grades_very_low']) / (df['total_grades'] + 1e-6)
        
        # Métricas de rendimiento
        df['grade_range'] = df['max_grade'] - df['min_grade']
        df['grade_consistency'] = 1 / (df['grade_std'] + 0.1)
        
        # Score de rendimiento académico
        df['academic_performance_score'] = (
            df['average_grade'] * 0.4 +
            df['pct_excellent'] * 2.0 +
            df['pct_good'] * 1.0 -
            df['pct_low_combined'] * 1.5
        )
        
        # Actividad temporal
        df['grades_per_month'] = df['total_grades'] / (df['meses_activos'] + 1)
        
        # 5. Calcular métricas de riesgo académico
        print("⚠️ Calculating risk metrics...")
        
        # Score de riesgo basado en múltiples factores
        df['risk_score'] = 0
        
        # Factor 1: Promedio bajo (0-40 puntos)
        df.loc[df['average_grade'] < 3.0, 'risk_score'] += 40
        df.loc[(df['average_grade'] >= 3.0) & (df['average_grade'] < 3.5), 'risk_score'] += 25
        df.loc[(df['average_grade'] >= 3.5) & (df['average_grade'] < 4.0), 'risk_score'] += 10
        
        # Factor 2: Alta proporción de notas bajas (0-30 puntos)
        df.loc[df['pct_low_combined'] > 0.3, 'risk_score'] += 30
        df.loc[(df['pct_low_combined'] > 0.2) & (df['pct_low_combined'] <= 0.3), 'risk_score'] += 20
        df.loc[(df['pct_low_combined'] > 0.1) & (df['pct_low_combined'] <= 0.2), 'risk_score'] += 10
        
        # Factor 3: Inconsistencia en notas (0-20 puntos)
        df.loc[df['grade_std'] > 1.0, 'risk_score'] += 20
        df.loc[(df['grade_std'] > 0.7) & (df['grade_std'] <= 1.0), 'risk_score'] += 10
        
        # Factor 4: Baja actividad (0-10 puntos)
        df.loc[df['grades_per_month'] < 5, 'risk_score'] += 10
        df.loc[(df['grades_per_month'] >= 5) & (df['grades_per_month'] < 10), 'risk_score'] += 5
        
        # Asegurar que risk_score esté en rango 0-100
        df['risk_score'] = np.clip(df['risk_score'], 0, 100)
        
        # Categorías de riesgo
        df['risk_level'] = 'BAJO'
        df.loc[df['risk_score'] >= 25, 'risk_level'] = 'MEDIO'
        df.loc[df['risk_score'] >= 50, 'risk_level'] = 'ALTO'
        df.loc[df['risk_score'] >= 75, 'risk_level'] = 'CRITICO'
        
        # Target binario
        df['is_high_risk'] = (df['risk_score'] >= 50).astype(int)
        
        # 6. Agregar metadata
        df['extraction_date'] = pd.Timestamp.now().isoformat()
        df['extraction_method'] = 'direct_query'
        
        # 7. Procesar fechas
        df['primera_actividad'] = pd.to_datetime(df['primera_actividad'])
        df['ultima_actividad'] = pd.to_datetime(df['ultima_actividad'])
        df['days_since_last_activity'] = (pd.Timestamp.now() - df['ultima_actividad']).dt.days
        
        print(f"🎉 DataFrame processing completed!")
        print(f"   📊 Final shape: {df.shape}")
        print(f"   📋 Features: {df.shape[1]}")
        
        return df

async def main():
    """Función principal"""
    
    print("🤖 MARI AI - DIRECT ML DATAFRAME CREATION")
    print("=" * 55)
    
    try:
        # 1. Verificar conexión
        print("🔍 Testing database connection...")
        if not db_manager.test_connection():
            print("❌ Database connection failed!")
            return
        print("✅ Database connection successful")
        
        # 2. Extraer DataFrame con diferentes configuraciones
        configurations = [
            {'min_grades': 1, 'limit': 50, 'name': 'Small Sample'},
            {'min_grades': 10, 'limit': 100, 'name': 'Medium Quality'},
            {'min_grades': 5, 'limit': 200, 'name': 'Large Dataset'}
        ]
        
        for config in configurations:
            print(f"\n🚀 EXTRACTING: {config['name']}")
            print("-" * 40)
            
            df = await extract_ml_dataframe_direct(
                min_grades=config['min_grades'],
                limit=config['limit']
            )
            
            if df.empty:
                print(f"❌ No data extracted for {config['name']}")
                continue
                
            # Mostrar resumen
            print(f"\n📈 DATASET SUMMARY - {config['name']}:")
            print(f"   👥 Students: {len(df):,}")
            print(f"   📊 Features: {df.shape[1]:,}")
            print(f"   📚 Avg grades per student: {df['total_grades'].mean():.1f}")
            print(f"   🎯 Avg grade average: {df['average_grade'].mean():.2f}")
            print(f"   📖 Avg subjects: {df['subjects_count'].mean():.1f}")
            
            # Distribución de riesgo
            risk_dist = df['risk_level'].value_counts()
            print(f"   ⚠️ Risk distribution:")
            for risk, count in risk_dist.items():
                pct = (count / len(df)) * 100
                print(f"      {risk}: {count} ({pct:.1f}%)")
            
            # Top correlaciones con risk_score
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if 'risk_score' in numeric_cols:
                correlations = df[numeric_cols].corr()['risk_score'].abs().sort_values(ascending=False)
                print(f"   🔗 Top correlations with risk_score:")
                for feature, corr in correlations.head(5).items():
                    if feature != 'risk_score':
                        print(f"      {feature}: {corr:.3f}")
            
            # Muestra de datos
            print(f"\n📋 SAMPLE DATA:")
            important_cols = ['student_id', 'edad', 'total_grades', 'average_grade', 'risk_score', 'risk_level']
            available_cols = [col for col in important_cols if col in df.columns]
            print(df[available_cols].head(3).to_string(index=False))
            
            # Guardar automáticamente
            output_dir = root_dir / "data" / "ml"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mari_ai_direct_{config['name'].lower().replace(' ', '_')}_{timestamp}.csv"
            filepath = output_dir / filename
            
            df.to_csv(filepath, index=False)
            print(f"   💾 Saved to: {filename}")
            
            # Si es el dataset grande y funciona, usarlo como principal
            if config['name'] == 'Large Dataset' and len(df) > 50:
                main_file = output_dir / "mari_ai_ml_dataset_main.csv"
                df.to_csv(main_file, index=False)
                print(f"   🎯 Main dataset saved to: mari_ai_ml_dataset_main.csv")
                
                # Retornar el dataset principal para uso posterior
                return df
        
        print(f"\n🎉 All extractions completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(main())
    
    if result is not None:
        print(f"\n🚀 SUCCESS! ML DataFrame ready with {len(result)} students!")
        print(f"📊 Next step: Implement ML prediction model")
    else:
        print(f"\n💥 FAILED! Check error messages above.")