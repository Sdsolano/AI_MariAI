#!/usr/bin/env python3
"""
Buscar estudiantes de 6to grado con datos académicos reales
"""

import sys
sys.path.append('.')

from app.db.connection import db_manager
from sqlalchemy import text

def find_6th_grade_students():
    """Buscar estudiantes de 6to grado con notas"""
    
    try:
        with db_manager.get_session() as session:
            print("🔍 SEARCHING FOR 6TH GRADE STUDENTS")
            print("="*50)
            
            # Query para estudiantes de 6° con notas académicas
            query = text("""
                SELECT 
                    e.id as student_id,
                    em.id as matricula_id,
                    e.primer_nombre,
                    e.primer_apellido,
                    g.nombre as grado,
                    COUNT(an.id) as total_notas,
                    AVG(an.nota) as promedio,
                    MIN(an.nota) as nota_min,
                    MAX(an.nota) as nota_max
                FROM acad_actividades_notas an
                JOIN acad_matricula_academica ma ON an.idmatricula = ma.id
                JOIN acad_estumatricula em ON ma.idestumatricula = em.id
                JOIN estu_estudiantes e ON em.idestudiante = e.id
                JOIN acad_gradosgrupos gg ON em.idgrados_grupos = gg.id
                JOIN acad_grados g ON gg.idgrado = g.id
                WHERE an.estado = 'on'
                  AND e.estado = 'on'
                  AND em.estado = 'on'
                  AND an.nota IS NOT NULL
                  AND an.nota > 0
                  AND g.nombre ILIKE '%6%'
                GROUP BY e.id, em.id, e.primer_nombre, e.primer_apellido, g.nombre
                HAVING COUNT(an.id) >= 20  -- Al menos 20 notas
                ORDER BY COUNT(an.id) DESC
                LIMIT 10
            """)
            
            result = session.execute(query)
            students = result.fetchall()
            
            print(f"📊 Found {len(students)} 6th grade students with academic data:")
            print()
            
            for i, student in enumerate(students, 1):
                promedio = f"{student.promedio:.2f}" if student.promedio else "N/A"
                nota_range = f"{student.nota_min:.1f}-{student.nota_max:.1f}"
                
                print(f"{i}. 👨‍🎓 {student.primer_nombre} {student.primer_apellido}")
                print(f"   Grade: {student.grado}")
                print(f"   Student ID: {student.student_id} | Matricula ID: {student.matricula_id}")
                print(f"   Notes: {student.total_notas} | Average: {promedio} | Range: {nota_range}")
                print()
            
            if students:
                recommended = students[0]
                print(f"✅ RECOMMENDED FOR 6TH GRADE TESTING:")
                print(f"   Name: {recommended.primer_nombre} {recommended.primer_apellido}")
                print(f"   Grade: {recommended.grado}")
                print(f"   Matricula ID: {recommended.matricula_id}")
                print(f"   Notes: {recommended.total_notas}")
                print(f"   Average: {recommended.promedio:.2f}")
                
                return recommended.matricula_id
            else:
                print("❌ No 6th grade students found with sufficient academic data")
                return None
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    find_6th_grade_students()