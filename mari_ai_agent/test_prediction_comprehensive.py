# test_prediction_comprehensive.py
"""
Comprehensive test suite for Mari AI ML prediction endpoints
Includes performance testing, error scenarios, and detailed validation
"""

import requests
import json
import time
import statistics
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import sys

# Configuration
BASE_URL = "http://localhost:8000/api/v1"
PERFORMANCE_URL = "http://localhost:8000"
TEST_STUDENT_IDS = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]  # Mix of IDs for testing

class PredictionTester:
    def __init__(self):
        self.results = {
            'passed': 0,
            'failed': 0,
            'performance_metrics': {},
            'errors': []
        }
    
    def log_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
        
        if success:
            self.results['passed'] += 1
        else:
            self.results['failed'] += 1
            self.results['errors'].append(f"{test_name}: {details}")
    
    def test_endpoint(self, url: str, method: str = "GET", data: Dict = None, 
                     expected_status: int = 200) -> Dict[str, Any]:
        """Test an endpoint and return results"""
        try:
            start_time = time.time()
            
            if method == "GET":
                response = requests.get(url)
            elif method == "POST":
                response = requests.post(url, json=data)
            else:
                return {"error": f"Unsupported method: {method}"}
            
            response_time = time.time() - start_time
            
            result = {
                "status_code": response.status_code,
                "response_time": response_time,
                "success": response.status_code == expected_status
            }
            
            if response.status_code == expected_status:
                try:
                    result["data"] = response.json()
                except:
                    result["data"] = response.text
            else:
                result["error"] = response.text
            
            return result
            
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def wait_for_server(self, max_attempts: int = 10, delay: int = 2) -> bool:
        """Wait for server to be ready"""
        print("Waiting for server to be ready...")
        
        for i in range(max_attempts):
            try:
                response = requests.get(f"{BASE_URL}/health/", timeout=5)
                if response.status_code == 200:
                    print("[OK] Server is ready!")
                    return True
            except:
                print(f"   Attempt {i+1}/{max_attempts}...")
                time.sleep(delay)
        
        print("[ERROR] Server not responding after maximum attempts")
        return False
    
    def test_health_endpoints(self):
        """Test all health and status endpoints"""
        print(f"\n{'='*60}")
        print("HEALTH & STATUS TESTS")
        print("="*60)
        
        # Basic health check
        result = self.test_endpoint(f"{BASE_URL}/health/")
        self.log_result("Health Check", result['success'])
        
        # Database health
        result = self.test_endpoint(f"{BASE_URL}/health/database")
        self.log_result("Database Health", result['success'])
        
        # System status
        result = self.test_endpoint(f"{PERFORMANCE_URL}/status")
        self.log_result("System Status", result['success'])
        
        # Root endpoint
        result = self.test_endpoint(f"{PERFORMANCE_URL}/")
        self.log_result("Root Endpoint", result['success'])
    
    def test_models_management(self):
        """Test model management endpoints"""
        print(f"\n{'='*60}")
        print("MODEL MANAGEMENT TESTS")
        print("="*60)
        
        # Get models status
        result = self.test_endpoint(f"{BASE_URL}/prediction/models/status")
        success = result['success'] and 'available_models' in result.get('data', {})
        
        if success:
            models_data = result['data']
            available_models = [m['model_name'] for m in models_data['available_models']]
            active_model = models_data['active_model']
            
            details = f"Found {len(available_models)} models: {available_models}. Active: {active_model}"
            self.log_result("Models Status", True, details)
            
            # Test model switching for each available model
            for model_name in available_models:
                if model_name != active_model:  # Don't switch to the same model
                    switch_result = self.test_endpoint(
                        f"{BASE_URL}/prediction/models/switch/{model_name}",
                        method="POST"
                    )
                    switch_success = switch_result['success'] and 'new_model' in switch_result.get('data', {})
                    self.log_result(f"Switch to {model_name}", switch_success)
                    
                    if switch_success:
                        time.sleep(0.5)  # Brief delay between switches
        else:
            self.log_result("Models Status", False, "Could not retrieve models status")
    
    def test_individual_predictions(self):
        """Test individual prediction endpoints"""
        print(f"\n{'='*60}")
        print("INDIVIDUAL PREDICTION TESTS")
        print("="*60)
        
        response_times = []
        successful_predictions = 0
        
        for student_id in TEST_STUDENT_IDS[:5]:  # Test first 5 students
            result = self.test_endpoint(
                f"{BASE_URL}/prediction/risk/{student_id}",
                method="POST"
            )
            
            if result['success']:
                data = result['data']
                required_fields = ['student_id', 'risk_level', 'risk_probability', 
                                 'confidence', 'key_factors', 'recommended_actions']
                
                has_all_fields = all(field in data for field in required_fields)
                valid_probability = 0 <= data.get('risk_probability', -1) <= 1
                valid_confidence = 0 <= data.get('confidence', -1) <= 1
                
                if has_all_fields and valid_probability and valid_confidence:
                    successful_predictions += 1
                    response_times.append(result['response_time'])
                    details = f"Student {student_id}: {data['risk_level']} (prob: {data['risk_probability']:.3f})"
                    self.log_result(f"Prediction Student {student_id}", True, details)
                else:
                    self.log_result(f"Prediction Student {student_id}", False, "Invalid response format")
            else:
                self.log_result(f"Prediction Student {student_id}", False, result.get('error', 'Unknown error'))
        
        # Performance summary
        if response_times:
            avg_time = statistics.mean(response_times)
            max_time = max(response_times)
            self.results['performance_metrics']['individual_prediction_avg'] = avg_time
            self.results['performance_metrics']['individual_prediction_max'] = max_time
            print(f"    Performance: Avg {avg_time:.3f}s, Max {max_time:.3f}s")
    
    def test_batch_predictions(self):
        """Test batch prediction endpoints"""
        print(f"\n{'='*60}")
        print("BATCH PREDICTION TESTS")
        print("="*60)
        
        # Test small batch (3 students)
        small_batch = {"student_ids": TEST_STUDENT_IDS[:3]}
        result = self.test_endpoint(
            f"{BASE_URL}/prediction/batch",
            method="POST",
            data=small_batch
        )
        
        if result['success']:
            data = result['data']
            expected_total = len(small_batch['student_ids'])
            actual_total = data.get('total_students', 0)
            
            batch_success = (expected_total == actual_total and 
                           data.get('successful_predictions', 0) > 0)
            
            details = f"Processed {actual_total}/{expected_total} students in {data.get('processing_time_seconds', 0):.3f}s"
            self.log_result("Small Batch (3 students)", batch_success, details)
            
            if batch_success:
                self.results['performance_metrics']['batch_small_time'] = data.get('processing_time_seconds', 0)
        else:
            self.log_result("Small Batch (3 students)", False, result.get('error', 'Unknown error'))
        
        # Test larger batch (7 students)
        large_batch = {"student_ids": TEST_STUDENT_IDS[:7]}
        result = self.test_endpoint(
            f"{BASE_URL}/prediction/batch",
            method="POST",
            data=large_batch
        )
        
        if result['success']:
            data = result['data']
            expected_total = len(large_batch['student_ids'])
            actual_total = data.get('total_students', 0)
            
            batch_success = (expected_total == actual_total)
            details = f"Processed {actual_total}/{expected_total} students in {data.get('processing_time_seconds', 0):.3f}s"
            self.log_result("Large Batch (7 students)", batch_success, details)
            
            if batch_success:
                self.results['performance_metrics']['batch_large_time'] = data.get('processing_time_seconds', 0)
        else:
            self.log_result("Large Batch (7 students)", False, result.get('error', 'Unknown error'))
    
    def test_risk_summary(self):
        """Test risk summary endpoint"""
        print(f"\n{'='*60}")
        print("RISK SUMMARY TESTS")
        print("="*60)
        
        result = self.test_endpoint(f"{BASE_URL}/prediction/risk/summary")
        
        if result['success']:
            data = result['data']
            required_fields = ['total_students_analyzed', 'risk_distribution', 'summary_timestamp']
            
            has_all_fields = all(field in data for field in required_fields)
            has_risk_levels = all(level in data.get('risk_distribution', {}) 
                                for level in ['BAJO', 'MEDIO', 'ALTO', 'CRITICO'])
            
            if has_all_fields and has_risk_levels:
                total_students = data['total_students_analyzed']
                distribution = data['risk_distribution']
                details = f"Analyzed {total_students} students. Distribution: " + \
                         ", ".join([f"{k}:{v}%" for k, v in distribution.items()])
                self.log_result("Risk Summary", True, details)
            else:
                self.log_result("Risk Summary", False, "Invalid response format")
        else:
            self.log_result("Risk Summary", False, result.get('error', 'Unknown error'))
    
    def test_model_evaluation(self):
        """Test model evaluation endpoint"""
        print(f"\n{'='*60}")
        print("MODEL EVALUATION TESTS")
        print("="*60)
        
        # Test evaluation of default model
        eval_data = {"model_name": "random_forest"}
        result = self.test_endpoint(
            f"{BASE_URL}/prediction/evaluate",
            method="POST",
            data=eval_data
        )
        
        if result['success']:
            data = result['data']
            required_metrics = ['accuracy', 'auc_score', 'precision', 'recall', 'f1_score']
            
            has_all_metrics = all(metric in data for metric in required_metrics)
            valid_metrics = all(0 <= data.get(metric, -1) <= 1 for metric in required_metrics)
            
            if has_all_metrics and valid_metrics:
                details = f"Accuracy: {data['accuracy']:.3f}, AUC: {data['auc_score']:.3f}, F1: {data['f1_score']:.3f}"
                self.log_result("Model Evaluation", True, details)
                self.results['performance_metrics']['model_accuracy'] = data['accuracy']
                self.results['performance_metrics']['model_auc'] = data['auc_score']
            else:
                self.log_result("Model Evaluation", False, "Invalid metrics format")
        else:
            self.log_result("Model Evaluation", False, result.get('error', 'Unknown error'))
    
    def test_error_scenarios(self):
        """Test error handling"""
        print(f"\n{'='*60}")
        print("ERROR SCENARIO TESTS")
        print("="*60)
        
        # Test invalid student ID
        result = self.test_endpoint(
            f"{BASE_URL}/prediction/risk/999999",
            method="POST",
            expected_status=404
        )
        self.log_result("Invalid Student ID", result['success'])
        
        # Test invalid model switch
        result = self.test_endpoint(
            f"{BASE_URL}/prediction/models/switch/invalid_model",
            method="POST",
            expected_status=400
        )
        self.log_result("Invalid Model Switch", result['success'])
        
        # Test empty batch request
        result = self.test_endpoint(
            f"{BASE_URL}/prediction/batch",
            method="POST",
            data={"student_ids": []},
            expected_status=400
        )
        self.log_result("Empty Batch Request", result['success'])
        
        # Test malformed batch request
        result = self.test_endpoint(
            f"{BASE_URL}/prediction/batch",
            method="POST",
            data={"invalid_field": [1, 2, 3]},
            expected_status=422
        )
        self.log_result("Malformed Batch Request", result['success'])
    
    def test_concurrent_requests(self, num_concurrent: int = 5):
        """Test concurrent request handling"""
        print(f"\n{'='*60}")
        print(f"CONCURRENT REQUESTS TEST ({num_concurrent} concurrent)")
        print("="*60)
        
        def make_prediction_request(student_id: int):
            return self.test_endpoint(
                f"{BASE_URL}/prediction/risk/{student_id}",
                method="POST"
            )
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [
                executor.submit(make_prediction_request, student_id)
                for student_id in TEST_STUDENT_IDS[:num_concurrent]
            ]
            
            results = []
            for future in as_completed(futures):
                results.append(future.result())
        
        total_time = time.time() - start_time
        successful_requests = sum(1 for r in results if r['success'])
        
        concurrent_success = successful_requests == num_concurrent
        details = f"Completed {successful_requests}/{num_concurrent} requests in {total_time:.3f}s"
        self.log_result("Concurrent Requests", concurrent_success, details)
        
        if concurrent_success:
            avg_response_time = statistics.mean([r['response_time'] for r in results if r['success']])
            self.results['performance_metrics']['concurrent_avg_time'] = avg_response_time
            self.results['performance_metrics']['concurrent_total_time'] = total_time
    
    def print_final_summary(self):
        """Print comprehensive test summary"""
        print(f"\n{'='*60}")
        print("COMPREHENSIVE TEST SUMMARY")
        print("="*60)
        
        total_tests = self.results['passed'] + self.results['failed']
        success_rate = (self.results['passed'] / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Test Results:")
        print(f"   Passed: {self.results['passed']}")
        print(f"   Failed: {self.results['failed']}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        if self.results['performance_metrics']:
            print(f"\nPerformance Metrics:")
            for metric, value in self.results['performance_metrics'].items():
                if 'time' in metric:
                    print(f"   {metric}: {value:.3f}s")
                else:
                    print(f"   {metric}: {value:.3f}")
        
        if self.results['errors']:
            print(f"\nFailed Tests:")
            for error in self.results['errors']:
                print(f"   - {error}")
        
        # Overall assessment
        print(f"\nOverall Assessment:")
        if success_rate >= 90:
            print("   [EXCELLENT] System is production ready")
        elif success_rate >= 80:
            print("   [GOOD] Minor issues to address")
        elif success_rate >= 70:
            print("   [NEEDS WORK] Several issues found")
        else:
            print("   [CRITICAL] Major issues require immediate attention")
        
        print(f"\nUseful Links:")
        print(f"   API Documentation: http://localhost:8000/docs")
        print(f"   Alternative docs: http://localhost:8000/redoc")
        print(f"   Health check: http://localhost:8000/api/v1/health/")
        print(f"   System status: http://localhost:8000/status")
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("MARI AI - COMPREHENSIVE ML PREDICTION ENDPOINTS TEST")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check if server is ready
        if not self.wait_for_server():
            print("[ERROR] Cannot proceed - server is not responding")
            return False
        
        # Run all test categories
        self.test_health_endpoints()
        self.test_models_management()
        self.test_individual_predictions()
        self.test_batch_predictions()
        self.test_risk_summary()
        self.test_model_evaluation()
        self.test_error_scenarios()
        self.test_concurrent_requests()
        
        # Print final summary
        self.print_final_summary()
        
        return self.results['failed'] == 0

def main():
    """Main test runner"""
    tester = PredictionTester()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()