# test_prediction_endpoints.py
"""
Test script for ML prediction endpoints
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000/api/v1"
TEST_STUDENT_IDS = [1, 2, 3, 4, 5]  # Adjust based on your actual student IDs

def test_endpoint(url: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
    """Test an endpoint and return results"""
    try:
        print(f"\n🧪 Testing {method} {url}")
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        else:
            print(f"❌ Unsupported method: {method}")
            return {"error": "Unsupported method"}
        
        if response.status_code == 200:
            print(f"✅ Success: {response.status_code}")
            return response.json()
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return {"error": response.text, "status_code": response.status_code}
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return {"error": str(e)}

def main():
    """Run comprehensive endpoint tests"""
    print("🚀 MARI AI - ML PREDICTION ENDPOINTS TEST")
    print("=" * 60)
    
    # Wait for server to be ready
    print("\n⏳ Waiting for server to be ready...")
    for i in range(5):
        try:
            response = requests.get(f"{BASE_URL}/health/")
            if response.status_code == 200:
                print("✅ Server is ready!")
                break
        except:
            print(f"   Attempt {i+1}/5...")
            time.sleep(2)
    else:
        print("❌ Server not responding. Make sure to run: uvicorn app.main:app --reload")
        return
    
    # Test 1: Health Check
    print(f"\n{'='*60}")
    print("📊 HEALTH CHECKS")
    print("="*60)
    
    health_result = test_endpoint(f"{BASE_URL}/health/")
    database_result = test_endpoint(f"{BASE_URL}/health/database")
    
    # Test 2: Models Status
    print(f"\n{'='*60}")
    print("🤖 MODELS STATUS")
    print("="*60)
    
    models_status = test_endpoint(f"{BASE_URL}/prediction/models/status")
    if "available_models" in models_status:
        print(f"   Available models: {len(models_status['available_models'])}")
        print(f"   Active model: {models_status['active_model']}")
        for model in models_status['available_models']:
            print(f"   - {model['model_name']}: {'✅' if model['loaded'] else '❌'}")
    
    # Test 3: Individual Prediction
    print(f"\n{'='*60}")
    print("🎯 INDIVIDUAL PREDICTIONS")
    print("="*60)
    
    # Test with first student ID
    test_student = TEST_STUDENT_IDS[0]
    prediction_result = test_endpoint(
        f"{BASE_URL}/prediction/risk/{test_student}",
        method="POST"
    )
    
    if "risk_level" in prediction_result:
        print(f"   Student {test_student}:")
        print(f"   - Risk Level: {prediction_result['risk_level']}")
        print(f"   - Probability: {prediction_result['risk_probability']}")
        print(f"   - Confidence: {prediction_result['confidence']}")
        print(f"   - Model Used: {prediction_result['model_used']}")
        print(f"   - Key Factors: {len(prediction_result['key_factors'])}")
        print(f"   - Recommendations: {len(prediction_result['recommended_actions'])}")
    
    # Test 4: Batch Prediction
    print(f"\n{'='*60}")
    print("📦 BATCH PREDICTIONS")
    print("="*60)
    
    batch_data = {"student_ids": TEST_STUDENT_IDS[:3]}  # Test with first 3 students
    batch_result = test_endpoint(
        f"{BASE_URL}/prediction/batch",
        method="POST",
        data=batch_data
    )
    
    if "total_students" in batch_result:
        print(f"   Total students: {batch_result['total_students']}")
        print(f"   Successful: {batch_result['successful_predictions']}")
        print(f"   Failed: {batch_result['failed_predictions']}")
        print(f"   Processing time: {batch_result['processing_time_seconds']}s")
        
        if batch_result['predictions']:
            print(f"   Sample prediction:")
            sample = batch_result['predictions'][0]
            print(f"   - Student {sample['student_id']}: {sample['risk_level']}")
    
    # Test 5: Model Switching
    print(f"\n{'='*60}")
    print("🔄 MODEL SWITCHING")
    print("="*60)
    
    # Try switching to gradient boosting
    switch_result = test_endpoint(
        f"{BASE_URL}/prediction/models/switch/gradient_boosting",
        method="POST"
    )
    
    if "new_model" in switch_result:
        print(f"   Switched to: {switch_result['new_model']}")
        
        # Test prediction with new model
        new_prediction = test_endpoint(
            f"{BASE_URL}/prediction/risk/{test_student}",
            method="POST"
        )
        if "model_used" in new_prediction:
            print(f"   Prediction with new model: {new_prediction['model_used']}")
    
    # Test 6: Risk Summary
    print(f"\n{'='*60}")
    print("📈 RISK SUMMARY")
    print("="*60)
    
    summary_result = test_endpoint(f"{BASE_URL}/prediction/risk/summary")
    if "total_students_analyzed" in summary_result:
        print(f"   Total students analyzed: {summary_result['total_students_analyzed']}")
        print(f"   Risk distribution:")
        for level, pct in summary_result['risk_distribution'].items():
            print(f"   - {level}: {pct}%")
    
    # Test 7: Model Evaluation
    print(f"\n{'='*60}")
    print("📊 MODEL EVALUATION")
    print("="*60)
    
    eval_data = {"model_name": "random_forest"}
    eval_result = test_endpoint(
        f"{BASE_URL}/prediction/evaluate",
        method="POST",
        data=eval_data
    )
    
    if "accuracy" in eval_result:
        print(f"   Model: {eval_result['model_name']}")
        print(f"   Accuracy: {eval_result['accuracy']}")
        print(f"   AUC Score: {eval_result['auc_score']}")
        print(f"   F1 Score: {eval_result['f1_score']}")
    
    # Final Summary
    print(f"\n{'='*60}")
    print("📋 TEST SUMMARY")
    print("="*60)
    print("✅ All core endpoints tested!")
    print("🎯 Ready for integration with Mari AI platform")
    print("📖 API documentation: http://localhost:8000/docs")
    print("🔍 Alternative docs: http://localhost:8000/redoc")

if __name__ == "__main__":
    main()