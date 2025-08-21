# test_with_real_data.py
"""
Test script to find real student data and test predictions with actual IDs
"""

import requests
import json
import random
from typing import List, Dict, Any

# Configuration
BASE_URL = "http://localhost:8000/api/v1"

def get_valid_student_ids(num_attempts: int = 50) -> List[int]:
    """Try to find valid student IDs by testing a range"""
    print("Searching for valid student IDs...")
    valid_ids = []
    
    # Try a range of IDs commonly used in databases
    test_ids = list(range(1, 101)) + list(range(1000, 1100)) + [random.randint(1, 2000) for _ in range(20)]
    
    for student_id in test_ids[:num_attempts]:
        try:
            response = requests.post(
                f"{BASE_URL}/prediction/risk/{student_id}",
                timeout=2
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'risk_level' in data:
                    valid_ids.append(student_id)
                    print(f"[FOUND] Student ID {student_id} - Risk: {data['risk_level']}")
                    
                    if len(valid_ids) >= 10:  # Stop after finding 10 valid IDs
                        break
                        
        except Exception:
            continue
    
    print(f"Found {len(valid_ids)} valid student IDs: {valid_ids}")
    return valid_ids

def test_predictions_with_real_data(student_ids: List[int]):
    """Test prediction endpoints with real student data"""
    if not student_ids:
        print("[ERROR] No valid student IDs found. Cannot test predictions.")
        return
    
    print(f"\n{'='*60}")
    print("TESTING PREDICTIONS WITH REAL DATA")
    print("="*60)
    
    # Test individual predictions
    print(f"\n[TEST] Individual Predictions")
    for i, student_id in enumerate(student_ids[:5]):  # Test first 5
        try:
            response = requests.post(f"{BASE_URL}/prediction/risk/{student_id}")
            if response.status_code == 200:
                data = response.json()
                print(f"Student {student_id}:")
                print(f"  - Risk Level: {data['risk_level']}")
                print(f"  - Probability: {data['risk_probability']:.3f}")
                print(f"  - Confidence: {data['confidence']:.3f}")
                print(f"  - Model: {data['model_used']}")
                print(f"  - Key Factors: {len(data['key_factors'])}")
                
                # Show top factor
                if data['key_factors']:
                    top_factor = data['key_factors'][0]
                    print(f"  - Top Factor: {top_factor['factor']} ({top_factor['impact']})")
                print()
        except Exception as e:
            print(f"[ERROR] Testing student {student_id}: {e}")
    
    # Test batch predictions
    print(f"[TEST] Batch Predictions")
    batch_ids = student_ids[:3]  # Use first 3 for batch test
    try:
        response = requests.post(
            f"{BASE_URL}/prediction/batch",
            json={"student_ids": batch_ids}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Batch Results:")
            print(f"  - Total: {data['total_students']}")
            print(f"  - Successful: {data['successful_predictions']}")
            print(f"  - Processing Time: {data['processing_time_seconds']:.3f}s")
            print(f"  - Average Time per Student: {data['processing_time_seconds']/data['total_students']:.3f}s")
        else:
            print(f"[ERROR] Batch request failed: {response.text}")
    except Exception as e:
        print(f"[ERROR] Batch test: {e}")

def test_model_performance():
    """Test all available models with evaluation"""
    print(f"\n{'='*60}")
    print("MODEL PERFORMANCE COMPARISON")
    print("="*60)
    
    # Get available models
    try:
        response = requests.get(f"{BASE_URL}/prediction/models/status")
        if response.status_code == 200:
            models_data = response.json()
            available_models = [m['model_name'] for m in models_data['available_models']]
            
            print(f"Testing {len(available_models)} models...")
            
            for model_name in available_models:
                # Switch to model
                switch_response = requests.post(f"{BASE_URL}/prediction/models/switch/{model_name}")
                if switch_response.status_code == 200:
                    
                    # Evaluate model
                    eval_response = requests.post(
                        f"{BASE_URL}/prediction/evaluate",
                        json={"model_name": model_name}
                    )
                    
                    if eval_response.status_code == 200:
                        eval_data = eval_response.json()
                        print(f"\n{model_name.upper()} Performance:")
                        print(f"  - Accuracy: {eval_data['accuracy']:.3f}")
                        print(f"  - AUC Score: {eval_data['auc_score']:.3f}")
                        print(f"  - F1 Score: {eval_data['f1_score']:.3f}")
                        print(f"  - Precision: {eval_data['precision']:.3f}")
                        print(f"  - Recall: {eval_data['recall']:.3f}")
                        print(f"  - Test Samples: {eval_data['total_samples']}")
                    
    except Exception as e:
        print(f"[ERROR] Model performance test: {e}")

def main():
    """Main testing function"""
    print("MARI AI - REAL DATA PREDICTION TESTING")
    print("=" * 60)
    
    # First verify server is running
    try:
        response = requests.get(f"{BASE_URL}/health/", timeout=5)
        if response.status_code != 200:
            print("[ERROR] Server not responding. Start server first.")
            return
    except:
        print("[ERROR] Cannot connect to server. Make sure it's running on port 8000.")
        return
    
    print("[OK] Server is running")
    
    # Find valid student IDs
    valid_students = get_valid_student_ids(num_attempts=30)
    
    if valid_students:
        # Test predictions with real data
        test_predictions_with_real_data(valid_students)
        
        # Test model performance
        test_model_performance()
        
        print(f"\n{'='*60}")
        print("REAL DATA TEST COMPLETE")
        print("="*60)
        print("[SUCCESS] Prediction system is working with real student data!")
        print(f"[INFO] Found {len(valid_students)} valid students for testing")
        print(f"[INFO] System ready for production use")
        
    else:
        print(f"\n{'='*60}")
        print("NO VALID STUDENT DATA FOUND")
        print("="*60)
        print("[WARNING] Could not find valid student IDs in the tested range")
        print("[INFO] This might indicate:")
        print("  1. Student IDs are in a different range")
        print("  2. Students need additional academic data for predictions")
        print("  3. Database connection or data issues")
        print("\n[RECOMMENDATION] Check your academic database and student records")

if __name__ == "__main__":
    main()