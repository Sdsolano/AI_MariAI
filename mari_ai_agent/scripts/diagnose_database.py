# scripts/diagnose_database.py
"""
🔍 SCRIPT DE DIAGNÓSTICO - BASE DE DATOS ACADÉMICA
================================================
Analiza paso a paso la BD para entender por qué no hay estudiantes
"""

import asyncio
import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()

from app.db.connection import db_manager
from app.repositories.academic_repository import AcademicRepository
from sqlalchemy import text

async def diagnose_database():
    """Diagnóstico completo de la base de datos académica"""
    
    print("🔍 MARI AI - DATABASE DIAGNOSTIC")
    print("=" * 50)
    
    try:
        # 1. Test conexión básica
        print("🔌 Testing database connection...")
        if not db_manager.test_connection():
            print("❌ Database connection failed!")
            return
        print("✅ Database connection successful")
        
        with db_manager.get_session() as session:
            repo = AcademicRepository(session)
            
            # 2. Analizar tablas principales
            print("\n📊 ANALYZING MAIN TABLES:")
            print("-" * 30)
            
            # Estudiantes totales
            query = text("SELECT COUNT(*) as total FROM estu_estudiantes WHERE estado = 'on'")
            result = session.execute(query).fetchone()
            total_students = result[0]
            print(f"👥 Total active students: {total_students:,}")
            
            # Matriculas
            query = text("SELECT COUNT(*) as total FROM acad_estumatricula")
            result = session.execute(query).fetchone()
            total_enrollments = result[0]
            print(f"📋 Total enrollments: {total_enrollments:,}")
            
            # Matriculas académicas
            query = text("SELECT COUNT(*) as total FROM acad_matricula_academica")
            result = session.execute(query).fetchone()
            total_academic_enrollments = result[0]
            print(f"🎓 Total academic enrollments: {total_academic_enrollments:,}")
            
            # Notas totales
            query = text("SELECT COUNT(*) as total FROM acad_actividades_notas WHERE estado = 'on'")
            result = session.execute(query).fetchone()
            total_grades = result[0]
            print(f"📚 Total active grades: {total_grades:,}")
            
            # 3. Analizar distribución de notas por estudiante
            print("\n📈 ANALYZING GRADE DISTRIBUTION:")
            print("-" * 35)
            
            query = text("""
                SELECT 
                    COUNT(DISTINCT an.id) as notas_count,
                    COUNT(DISTINCT e.id) as estudiantes_count
                FROM estu_estudiantes e
                JOIN acad_estumatricula em ON e.id = em.idestudiante
                JOIN acad_matricula_academica ma ON em.id = ma.idestumatricula
                LEFT JOIN acad_actividades_notas an ON ma.id = an.idmatricula AND an.estado = 'on'
                WHERE e.estado = 'on'
                GROUP BY e.id
                ORDER BY notas_count DESC
                LIMIT 20
            """)
            
            results = session.execute(query).fetchall()
            
            if results:
                print("📊 Top students by grade count:")
                for i, row in enumerate(results[:10]):
                    print(f"   Student {i+1}: {row[0]} grades")
                
                # Estadísticas de distribución
                grades_counts = [row[0] for row in results]
                avg_grades = sum(grades_counts) / len(grades_counts)
                max_grades = max(grades_counts)
                min_grades = min(grades_counts)
                
                print(f"\n📈 Grade distribution stats:")
                print(f"   Average grades per student: {avg_grades:.1f}")
                print(f"   Maximum grades: {max_grades}")
                print(f"   Minimum grades: {min_grades}")
                
                # Contar estudiantes por rangos
                students_5_plus = len([g for g in grades_counts if g >= 5])
                students_3_plus = len([g for g in grades_counts if g >= 3])
                students_1_plus = len([g for g in grades_counts if g >= 1])
                
                print(f"\n👥 Students by grade count:")
                print(f"   With 5+ grades: {students_5_plus}")
                print(f"   With 3+ grades: {students_3_plus}")
                print(f"   With 1+ grades: {students_1_plus}")
            
            # 4. Test query específica get_students_with_activity
            print("\n🔧 TESTING get_students_with_activity QUERY:")
            print("-" * 45)
            
            for min_grades in [1, 3, 5, 10]:
                try:
                    students = repo.get_students_with_activity(min_grades=min_grades, limit=10)
                    print(f"   min_grades={min_grades}: {len(students)} students found")
                    
                    if students and min_grades == 1:
                        # Mostrar detalle del primer estudiante
                        first_student = students[0]
                        print(f"   Sample student: ID={first_student['student_id']}, "
                              f"grades={first_student['total_notas']}, "
                              f"avg={first_student['promedio_notas']:.2f}")
                
                except Exception as e:
                    print(f"   min_grades={min_grades}: ERROR - {e}")
            
            # 5. Verificar query manual directa
            print("\n🛠️ MANUAL QUERY TEST:")
            print("-" * 25)
            
            manual_query = text("""
                SELECT 
                    e.id as student_id,
                    e.primer_nombre || ' ' || e.primer_apellido as nombre_completo,
                    COUNT(DISTINCT an.id) as total_notas,
                    AVG(an.nota) as promedio_notas
                FROM estu_estudiantes e
                JOIN acad_estumatricula em ON e.id = em.idestudiante
                JOIN acad_matricula_academica ma ON em.id = ma.idestumatricula
                LEFT JOIN acad_actividades_notas an ON ma.id = an.idmatricula AND an.estado = 'on'
                WHERE e.estado = 'on'
                GROUP BY e.id, e.primer_nombre, e.primer_apellido
                HAVING COUNT(DISTINCT an.id) >= 1
                ORDER BY COUNT(DISTINCT an.id) DESC
                LIMIT 10
            """)
            
            manual_results = session.execute(manual_query).fetchall()
            
            if manual_results:
                print(f"✅ Manual query returned {len(manual_results)} students:")
                for row in manual_results:
                    print(f"   ID: {row[0]}, Name: {row[1]}, Grades: {row[2]}, Avg: {row[3]:.2f}")
            else:
                print("❌ Manual query returned 0 students")
                
                # Debug: verificar cada tabla por separado
                print("\n🔍 DEBUGGING INDIVIDUAL TABLES:")
                
                # Solo estudiantes
                query = text("SELECT COUNT(*) FROM estu_estudiantes WHERE estado = 'on'")
                count = session.execute(query).scalar()
                print(f"   Active students: {count}")
                
                # Join con matriculas
                query = text("""
                    SELECT COUNT(DISTINCT e.id) 
                    FROM estu_estudiantes e
                    JOIN acad_estumatricula em ON e.id = em.idestudiante
                    WHERE e.estado = 'on'
                """)
                count = session.execute(query).scalar()
                print(f"   Students with enrollments: {count}")
                
                # Join con matriculas académicas
                query = text("""
                    SELECT COUNT(DISTINCT e.id) 
                    FROM estu_estudiantes e
                    JOIN acad_estumatricula em ON e.id = em.idestudiante
                    JOIN acad_matricula_academica ma ON em.id = ma.idestumatricula
                    WHERE e.estado = 'on'
                """)
                count = session.execute(query).scalar()
                print(f"   Students with academic enrollments: {count}")
                
                # Join con notas
                query = text("""
                    SELECT COUNT(DISTINCT e.id) 
                    FROM estu_estudiantes e
                    JOIN acad_estumatricula em ON e.id = em.idestudiante
                    JOIN acad_matricula_academica ma ON em.id = ma.idestumatricula
                    JOIN acad_actividades_notas an ON ma.id = an.idmatricula
                    WHERE e.estado = 'on' AND an.estado = 'on'
                """)
                count = session.execute(query).scalar()
                print(f"   Students with grades: {count}")
            
            # 6. Verificar estructura de algunas tablas clave
            print("\n📋 TABLE STRUCTURE VERIFICATION:")
            print("-" * 35)
            
            # Verificar si existen las columnas esperadas
            tables_to_check = [
                "estu_estudiantes",
                "acad_estumatricula", 
                "acad_matricula_academica",
                "acad_actividades_notas"
            ]
            
            for table in tables_to_check:
                try:
                    query = text(f"""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = '{table}'
                        ORDER BY ordinal_position
                        LIMIT 10
                    """)
                    columns = session.execute(query).fetchall()
                    print(f"   {table}: {len(columns)} columns")
                    # Mostrar primeras 5 columnas
                    for col in columns[:5]:
                        print(f"      - {col[0]} ({col[1]})")
                
                except Exception as e:
                    print(f"   {table}: ERROR - {e}")
            
            # 7. Recomendaciones
            print("\n💡 RECOMMENDATIONS:")
            print("-" * 20)
            
            if total_students == 0:
                print("❌ No active students found. Check 'estado' field values.")
            elif total_grades == 0:
                print("❌ No active grades found. Check grades table and 'estado' field.")
            elif manual_results:
                print("✅ Data exists! Issue is in the repository method.")
                print("   🔧 Fix: Adjust get_students_with_activity() method")
                print("   🔧 Or: Use lower min_grades threshold")
            else:
                print("❌ Data relationship issue. Check table JOINs.")
                print("   🔧 Verify foreign key relationships")
                print("   🔧 Check if column names match schema")
    
    except Exception as e:
        print(f"❌ Diagnostic error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(diagnose_database())