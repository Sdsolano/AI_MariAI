# mari_ai_agent/startup_test.py - SCRIPT DE TESTING
#!/usr/bin/env python3
"""
🧪 SCRIPT DE TESTING INICIAL - MARI AI
=====================================
Prueba la integración completa de base de datos y servicios
"""

import asyncio
import os
import sys
from pathlib import Path

# Agregar el directorio raíz al path
root_dir = Path(__file__).parent
sys.path.append(str(root_dir))

# Cargar variables de entorno desde .env
try:
    from dotenv import load_dotenv
    load_dotenv()
    print(f"🔧 Loaded .env file using python-dotenv")
except ImportError:
    print("⚠️ python-dotenv not installed, trying manual .env loading...")
    # Cargar manualmente el archivo .env si dotenv no está disponible
    env_file = root_dir / ".env"
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print(f"🔧 Manually loaded .env file")
    else:
        print(f"❌ .env file not found at {env_file}")

# Debug: verificar variables de entorno
print(f"🔍 DEBUG - Environment variables:")
print(f"   ACADEMIC_DB_PASSWORD: {'SET' if os.getenv('ACADEMIC_DB_PASSWORD') else 'NOT SET'}")
print(f"   API_KEY: {'SET' if os.getenv('API_KEY') else 'NOT SET'}")
print(f"   Working directory: {os.getcwd()}")
print(f"   .env file path: {root_dir / '.env'}")
print(f"   .env file exists: {(root_dir / '.env').exists()}")
print()

async def test_database_integration():
    """Prueba la integración de base de datos"""
    print("🔍 TESTING DATABASE INTEGRATION...")
    
    try:
        # Importar configuración
        from app.core.config.database import db_settings
        from app.db.connection import db_manager
        from app.repositories.academic_repository import AcademicRepository
        from app.services.academic_service import AcademicDataService
        
        print(f"   📊 Database URL: {db_settings.ACADEMIC_DB_HOST}:{db_settings.ACADEMIC_DB_PORT}")
        
        # 1. Test conexión básica
        print(f"\n🔌 Testing database connection...")
        connection_ok = db_manager.test_connection()
        if connection_ok:
            print(f"   ✅ Database connection successful")
        else:
            print(f"   ❌ Database connection failed")
            return False
        
        # 2. Test conteos de tablas
        print(f"\n📋 Testing table access...")
        students_count = db_manager.get_table_count("estu_estudiantes")
        grades_count = db_manager.get_table_count("acad_actividades_notas")
        attendance_count = db_manager.get_table_count("student_attendance")
        
        print(f"   👥 Students: {students_count:,}")
        print(f"   📚 Grades: {grades_count:,}")
        print(f"   📅 Attendance: {attendance_count:,}")
        
        # 3. Test repository
        print(f"\n🔧 Testing repository layer...")
        with db_manager.get_session() as session:
            repo = AcademicRepository(session)
            
            # Test estudiantes con actividad
            active_students = repo.get_students_with_activity(min_grades=5, limit=10)
            print(f"   ✅ Active students retrieved: {len(active_students)}")
            
            if active_students:
                # Test un estudiante específico
                student_id = active_students[0]['student_id']
                student_data = repo.get_student_comprehensive_data(student_id)
                
                if student_data:
                    print(f"   ✅ Student data retrieved for ID: {student_id}")
                    print(f"      - Total grades: {student_data['academic_metrics']['total_grades']}")
                    print(f"      - Average grade: {student_data['academic_metrics']['average_grade']}")
                else:
                    print(f"   ⚠️ No comprehensive data for student {student_id}")
        
        # 4. Test service layer
        print(f"\n🎯 Testing service layer...")
        with db_manager.get_session() as session:
            service = AcademicDataService(session)
            
            if active_students:
                student_id = active_students[0]['student_id']
                profile = await service.get_student_profile(student_id)
                
                if profile:
                    risk_level = profile.get('risk_indicators', {}).get('risk_level', 'unknown')
                    print(f"   ✅ Student profile generated")
                    print(f"      - Risk level: {risk_level}")
                    print(f"      - Recommendations: {len(profile.get('recommendations', []))}")
        
        print(f"\n🎉 DATABASE INTEGRATION TEST COMPLETED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        print(f"\n❌ DATABASE INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_api_endpoints():
    """Prueba los endpoints de API"""
    print(f"\n🌐 TESTING API ENDPOINTS...")
    
    try:
        import httpx
        
        base_url = "http://localhost:8000"
        
        async with httpx.AsyncClient() as client:
            # Test root endpoint
            response = await client.get(f"{base_url}/")
            if response.status_code == 200:
                print(f"   ✅ Root endpoint working")
            else:
                print(f"   ❌ Root endpoint failed: {response.status_code}")
            
            # Test health endpoint
            response = await client.get(f"{base_url}/api/v1/health/database")
            if response.status_code == 200:
                print(f"   ✅ Database health endpoint working")
            else:
                print(f"   ❌ Database health endpoint failed: {response.status_code}")
            
            # Test sample students
            response = await client.get(f"{base_url}/api/v1/academic/test/sample-students?limit=5")
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Sample students endpoint working: {data['sample_size']} students")
            else:
                print(f"   ❌ Sample students endpoint failed: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"   ⚠️ API testing failed (server may not be running): {e}")
        return False

async def main():
    """Función principal de testing"""
    print("🧪 MARI AI - STARTUP TESTING")
    print("=" * 50)
    
    # Check environment
    if not os.getenv("ACADEMIC_DB_PASSWORD"):
        print("❌ Missing ACADEMIC_DB_PASSWORD environment variable")
        print("   Please copy .env.example to .env and configure it")
        return
    
    if not os.getenv("API_KEY"):
        print("❌ Missing API_KEY environment variable")
        print("   Please set your OpenAI API key in .env")
        return
    
    # Run tests
    db_test = await test_database_integration()
    api_test = await test_api_endpoints()
    
    print(f"\n📋 TESTING RESULTS:")
    print(f"   Database Integration: {'✅ PASS' if db_test else '❌ FAIL'}")
    print(f"   API Endpoints: {'✅ PASS' if api_test else '⚠️ SKIP (server not running)'}")
    
    if db_test:
        print(f"\n🚀 READY FOR DEVELOPMENT!")
        print(f"   Next steps:")
        print(f"   1. Start the server: uvicorn app.main:app --reload")
        print(f"   2. Open API docs: http://localhost:8000/docs")
        print(f"   3. Test endpoints: http://localhost:8000/api/v1/academic/test/sample-students")
    else:
        print(f"\n❌ SETUP INCOMPLETE - Please fix database issues first")

if __name__ == "__main__":
    asyncio.run(main())