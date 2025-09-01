"""
Script to check user status in the database
"""
import asyncio
import sys
from sqlalchemy import text
sys.path.append('.')
from app.db.connection import db_manager

async def check_user_status(matricula_id: int):
    try:
        with db_manager.get_session() as session:
            # First find the student and user from matricula_id
            matricula_query = text("""
                SELECT 
                    u.id as user_id,
                    u.estado as user_estado,
                    e.id as student_id,
                    e.estado as student_estado,
                    e.primer_nombre,
                    e.primer_apellido,
                    em.id as matricula_id,
                    em.estado as matricula_estado
                FROM acad_estumatricula em
                JOIN estu_estudiantes e ON em.idestudiante = e.id
                JOIN usua_usuarios u ON e.idusuario = u.id
                WHERE em.id = :matricula_id
            """)
            result = session.execute(matricula_query, {"matricula_id": matricula_id})
            row = result.fetchone()
            
            if not row:
                print(f"❌ Matrícula {matricula_id} no encontrada")
                return
                
            print(f"📚 Información encontrada para matrícula {matricula_id}:")
            print(f"\n👤 Usuario (usua_usuarios):")
            print(f"   ID: {row.user_id}")
            print(f"   Estado: {row.user_estado}")
            
            print(f"\n👨‍🎓 Estudiante (estu_estudiantes):")
            print(f"   ID: {row.student_id}")
            print(f"   Nombre: {row.primer_nombre} {row.primer_apellido}")
            print(f"   Estado: {row.student_estado}")
            
            print(f"\n📝 Matrícula (acad_estumatricula):")
            print(f"   ID: {row.matricula_id}")
            print(f"   Estado: {row.matricula_estado}")
            
            # Check enrollment details
            enrollment_query = text("""
                SELECT g.grado, g.grupo
                FROM acad_estumatricula em
                JOIN acad_grados_grupos g ON em.idgrados_grupos = g.id
                WHERE em.id = :matricula_id
            """)
            enrollment_result = session.execute(enrollment_query, {"matricula_id": matricula_id})
            enrollment_row = enrollment_result.fetchone()
            
            if enrollment_row:
                print(f"\n� Detalles académicos:")
                print(f"   Grado: {enrollment_row.grado}")
                print(f"   Grupo: {enrollment_row.grupo}")
                
            print("\n💡 Para usar el chat, debes usar el ID de usuario:")
            print(f"   user_id: {row.user_id}")
            print("\nEjemplo de request correcto:")
            print("""{
    "user_id": "%d",
    "message": "¿Cómo puedo mejorar en matemáticas?",
    "conversation_history": [],
    "context": {}
}""" % row.user_id)
            
    except Exception as e:
        print(f"❌ Error checking user status: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_user_status.py <user_id>")
        sys.exit(1)
        
    user_id = int(sys.argv[1])
    asyncio.run(check_user_status(user_id))
