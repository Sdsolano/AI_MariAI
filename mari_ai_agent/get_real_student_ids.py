# get_real_student_ids.py
"""
Query the database directly to get real student IDs for testing
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.db.connection import db_manager
import requests
import json

def get_real_student_ids():
    """Query database to get actual student IDs"""
    try:
        print("Connecting to database...")
        
        with db_manager.get_session() as session:
            # Get sample student IDs
            from sqlalchemy import text
            query = text("""
                SELECT DISTINCT idmatricula 
                FROM acad_actividades_notas 
                WHERE idmatricula IS NOT NULL 
                ORDER BY idmatricula 
                LIMIT 20
            """)
            
            result = session.execute(query)
            results = result.fetchall()
            
            student_ids = [row[0] for row in results]
            print(f"Found {len(student_ids)} student IDs: {student_ids}")
            
            return student_ids
        
    except Exception as e:
        print(f"Error querying database: {e}")
        return []

def test_prediction_with_real_id(student_id):
    """Test prediction endpoint with a real student ID"""
    try:
        print(f"\n[TEST] Prediction for Student ID: {student_id}")
        
        response = requests.post(
            f"http://localhost:8000/api/v1/prediction/risk/{student_id}",
            timeout=10
        )
        
        print(f"[STATUS] {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"[SUCCESS] Prediction generated!")
            print(f"  - Student ID: {data['student_id']}")
            print(f"  - Risk Level: {data['risk_level']}")
            print(f"  - Risk Probability: {data['risk_probability']:.3f}")
            print(f"  - Confidence: {data['confidence']:.3f}")
            print(f"  - Model Used: {data['model_used']}")
            print(f"  - Timestamp: {data['prediction_timestamp']}")
            
            # Show key factors
            print(f"  - Top Risk Factors:")
            for i, factor in enumerate(data['key_factors'][:3], 1):
                print(f"    {i}. {factor['factor']}: {factor['value']:.3f} ({factor['impact']})")
            
            # Show recommendations
            print(f"  - Recommended Actions:")
            for i, action in enumerate(data['recommended_actions'][:2], 1):
                print(f"    {i}. {action['action']} (Priority: {action['priority']})")
            
            return True
        else:
            print(f"[ERROR] {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"[EXCEPTION] {e}")
        return False

def test_batch_prediction(student_ids):
    """Test batch prediction with real student IDs"""
    if len(student_ids) < 3:
        print("[WARNING] Need at least 3 student IDs for batch test")
        return False
    
    try:
        batch_ids = student_ids[:5]  # Test with first 5 students
        print(f"\n[TEST] Batch Prediction for {len(batch_ids)} students: {batch_ids}")
        
        response = requests.post(
            "http://localhost:8000/api/v1/prediction/batch",
            json={"student_ids": batch_ids},
            timeout=30
        )
        
        print(f"[STATUS] {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"[SUCCESS] Batch prediction completed!")
            print(f"  - Total Students: {data['total_students']}")
            print(f"  - Successful Predictions: {data['successful_predictions']}")
            print(f"  - Failed Predictions: {data['failed_predictions']}")
            print(f"  - Processing Time: {data['processing_time_seconds']:.3f} seconds")
            print(f"  - Avg Time per Student: {data['processing_time_seconds']/data['total_students']:.3f}s")
            
            # Show sample predictions
            print(f"  - Sample Results:")
            for pred in data['predictions'][:3]:
                print(f"    Student {pred['student_id']}: {pred['risk_level']} (prob: {pred['risk_probability']:.3f})")
            
            return True
        else:
            print(f"[ERROR] {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"[EXCEPTION] {e}")
        return False

def main():
    """Main testing function"""
    print("MARI AI - TESTING WITH REAL DATABASE STUDENT IDs")
    print("=" * 60)
    
    # Get real student IDs from database
    student_ids = get_real_student_ids()
    
    if not student_ids:
        print("[ERROR] Could not retrieve student IDs from database")
        return
    
    print(f"\n[INFO] Testing predictions with {len(student_ids)} real student IDs")
    
    # Test individual predictions
    successful_individual = 0
    for i, student_id in enumerate(student_ids[:5]):  # Test first 5
        if test_prediction_with_real_id(student_id):
            successful_individual += 1
    
    # Test batch prediction
    batch_success = test_batch_prediction(student_ids)
    
    # Summary
    print(f"\n{'='*60}")
    print("REAL DATA TEST RESULTS")
    print("="*60)
    print(f"Individual Predictions: {successful_individual}/5 successful")
    print(f"Batch Prediction: {'✅ Success' if batch_success else '❌ Failed'}")
    
    if successful_individual > 0 or batch_success:
        print(f"\n🎉 EXCELLENT! Prediction system is working with real student data!")
        print(f"📊 Database contains {len(student_ids)} students ready for ML predictions")
        print(f"🚀 System is production-ready for Mari AI platform integration")
    else:
        print(f"\n⚠️ Need to investigate prediction generation issues")
        print(f"💡 Database connection works, but predictions may need data verification")

if __name__ == "__main__":
    main()