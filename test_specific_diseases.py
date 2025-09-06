# test_specific_diseases.py
import sys
import os
import json
import numpy as np
import re
import time

# Add the current directory to Python path (where manage.py is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medical_chatbot.settings')

import django
django.setup()

# Now import your chatbot
from chatbot_app.medical_ai.chat_bot import MedicalChatBot

class DiseaseTester:
    def __init__(self, debug_mode=True):
        self.debug_mode = debug_mode
        self.chatbot = None
        self.initialize_chatbot()
        
    def initialize_chatbot(self):
        """Initialize a fresh chatbot instance"""
        if self.debug_mode:
            print("🔄 Initializing new chatbot instance...")
        
        self.chatbot = MedicalChatBot()
        self.initialized = self.chatbot.initialize_bot()
        
        if self.debug_mode:
            print(f"✅ Chatbot initialized: {self.initialized}")
        return self.initialized
        
    def load_test_cases(self, specific_diseases=None):
        """Load test cases for specific diseases only"""
        test_cases_path = os.path.join(os.path.dirname(__file__), 'test_cases.json')
        try:
            with open(test_cases_path, 'r') as f:
                all_test_cases = json.load(f)
                
            if specific_diseases:
                # Filter for specific diseases
                filtered_cases = []
                for test_case in all_test_cases:
                    expected_condition = test_case['expected_condition']
                    if any(disease.lower() in expected_condition.lower() for disease in specific_diseases):
                        filtered_cases.append(test_case)
                
                if self.debug_mode:
                    print(f"✅ Loaded {len(filtered_cases)} test cases for diseases: {specific_diseases}")
                return filtered_cases
            else:
                if self.debug_mode:
                    print(f"✅ Loaded {len(all_test_cases)} test cases")
                return all_test_cases
                
        except Exception as e:
            print(f"❌ Error loading test cases: {e}")
            return []
    
    def force_reset_chatbot(self):
        """COMPLETE RESET - Create a brand new chatbot instance"""
        if self.debug_mode:
            print("🔄 FORCE RESET: Creating completely new chatbot instance")
        
        # Create a brand new instance to ensure complete isolation
        self.initialize_chatbot()
            
    def extract_condition_from_response(self, response, expected_condition):
        """Improved condition detection"""
        if not response:
            return False, 0
            
        response_lower = response.lower()
        expected_lower = expected_condition.lower()
        
        # Direct match
        if expected_lower in response_lower:
            return True, 100
        
        # Partial matches
        partial_matches = [
            expected_lower,
            expected_lower.replace("fever", "").strip(),
            expected_lower.replace("disease", "").strip(),
            expected_lower.replace("infection", "").strip(),
        ]
        
        for partial in partial_matches:
            if partial and partial in response_lower:
                return True, 90
        
        # Disease-specific keywords
        keyword_patterns = {
            "appendicitis": ["appendicitis", "appendix"],
            "heart attack": ["heart attack", "myocardial", "cardiac"],
            "common cold": ["common cold", "cold", "rhinovirus"],
            "chikungunya": ["chikungunya", "chik"],
            "dengue": ["dengue", "df"]
        }
        
        for condition_key, keywords in keyword_patterns.items():
            if condition_key in expected_lower:
                for keyword in keywords:
                    if keyword in response_lower:
                        return True, 85
        
        return False, 0
    
    def extract_confidence_from_response(self, response):
        """Extract confidence score from response"""
        if not response:
            return 0
            
        confidence_patterns = [
            r"Confidence Level:[\s]*([\d.]+)%",
            r"Confidence:[\s]*([\d.]+)%",
            r"([\d.]+)% confidence",
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return 0
    
    def is_final_diagnosis(self, response):
        """Check if this response contains a final diagnosis"""
        if not response:
            return False
            
        diagnosis_indicators = [
            "SUSPECTED CONDITION:",
            "MEDICAL ASSESSMENT",
            "RECOMMENDATIONS",
            "final diagnosis",
            "assessment complete"
        ]
        
        return any(indicator in response for indicator in diagnosis_indicators)
    
    def run_test_case(self, test_case):
        """Run a single test case and return results"""
        if not self.initialized:
            return {
                'success': False,
                'expected_condition': test_case['expected_condition'],
                'detected_condition': "Chatbot not initialized",
                'confidence': 0,
                'condition_detected': False,
                'error': 'Chatbot not initialized'
            }
        
        # COMPLETE RESET for each test
        self.force_reset_chatbot()
        
        conversation = test_case['conversation']
        expected_condition = test_case['expected_condition']
        expected_confidence = test_case.get('expected_confidence', 70)
        
        responses = []
        final_diagnosis = None
        confidence = 0
        condition_detected = False
        match_confidence = 0
        got_final_diagnosis = False
        
        if self.debug_mode:
            print(f"\n🔍 Starting test: {test_case['name']}")
            print(f"   Expected: {expected_condition}")
        
        # Simulate the conversation
        for i, user_input in enumerate(conversation):
            if self.debug_mode:
                print(f"\n   User {i+1}: {user_input}")
            
            query_type = self.chatbot.get_query_type(user_input)
            
            if self.debug_mode:
                print(f"   Query type: {query_type}")
            
            try:
                if query_type == 'medical':
                    response, should_continue = self.chatbot.process_medical_query(user_input)
                    responses.append(response)
                    
                    if self.debug_mode:
                        print(f"   Response: {response[:150]}..." if response else "   Response: None")
                        print(f"   Should continue: {should_continue}")
                    
                    # Check if we got a final diagnosis
                    is_final = self.is_final_diagnosis(response)
                    if self.debug_mode and is_final:
                        print("   ✅ FINAL DIAGNOSIS DETECTED")
                    
                    if is_final and not got_final_diagnosis:
                        final_diagnosis = response
                        got_final_diagnosis = True
                        
                        confidence = self.extract_confidence_from_response(response)
                        condition_detected, match_confidence = self.extract_condition_from_response(
                            response, expected_condition
                        )
                        
                        if self.debug_mode:
                            print(f"   Confidence: {confidence}%")
                            print(f"   Condition detected: {condition_detected}")
                
                elif query_type == 'informational':
                    response = self.chatbot.get_informational_response(user_input)
                    responses.append(response)
                    
                else:
                    response = self.chatbot.get_casual_response(user_input)
                    responses.append(response)
                        
            except Exception as e:
                if self.debug_mode:
                    print(f"   ❌ Error: {e}")
                return {
                    'success': False,
                    'expected_condition': expected_condition,
                    'detected_condition': f"Error: {str(e)}",
                    'confidence': 0,
                    'condition_detected': False,
                    'error': str(e)
                }
        
        success = condition_detected and confidence >= expected_confidence
        
        if self.debug_mode:
            print(f"   Test result: {'✅ PASS' if success else '❌ FAIL'}")
            print(f"   Condition found: {condition_detected}")
            print(f"   Confidence: {confidence}% (required: {expected_confidence}%)")
            print(f"   Final diagnosis reached: {got_final_diagnosis}")
        
        return {
            'success': success,
            'expected_condition': expected_condition,
            'detected_condition': final_diagnosis,
            'confidence': confidence,
            'condition_detected': condition_detected,
            'match_confidence': match_confidence,
            'final_diagnosis_reached': got_final_diagnosis
        }
    
    def run_specific_disease_tests(self, diseases):
        """Run tests only for specific diseases"""
        test_cases = self.load_test_cases(diseases)
        results = []
        
        if not test_cases:
            print("❌ No test cases found for the specified diseases")
            return results
        
        print(f"\n🧪 Testing {len(test_cases)} cases for diseases: {diseases}")
        print("=" * 60)
        
        for i, test_case in enumerate(test_cases):
            print(f"\n{i+1}/{len(test_cases)}: {test_case['name']}")
            result = self.run_test_case(test_case)
            results.append(result)
            
            # Small delay between tests
            time.sleep(0.1)
        
        return results
    
    def calculate_accuracy(self, results):
        """Calculate accuracy metrics"""
        total = len(results)
        successful = sum(1 for r in results if r.get('success', False))
        accuracy = (successful / total) * 100 if total > 0 else 0
        
        confidences = [r['confidence'] for r in results if r.get('confidence', 0) > 0]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        final_diagnosis_count = sum(1 for r in results if r.get('final_diagnosis_reached', False))
        
        return {
            'total_tests': total,
            'successful_tests': successful,
            'accuracy': accuracy,
            'average_confidence': avg_confidence,
            'final_diagnosis_reached': final_diagnosis_count
        }
    
    def print_results(self, results, metrics):
        """Print comprehensive results"""
        print(f"\n📈 Test Results:")
        print("=" * 50)
        print(f"Accuracy: {metrics['accuracy']:.2f}%")
        print(f"Average Confidence: {metrics['average_confidence']:.2f}%")
        print(f"Final Diagnosis Reached: {metrics['final_diagnosis_reached']}/{metrics['total_tests']}")
        print(f"Successful Tests: {metrics['successful_tests']}/{metrics['total_tests']}")
        
        print(f"\n📋 Detailed Results:")
        print("=" * 50)
        for i, result in enumerate(results):
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            final_dx = "✓" if result.get('final_diagnosis_reached', False) else "✗"
            print(f"{i+1:2d}. {status} {final_dx} - {result['expected_condition']}")
            print(f"    Confidence: {result['confidence']}% | Condition Found: {result['condition_detected']}")

if __name__ == "__main__":
    # Test ONLY these specific diseases
    TARGET_DISEASES = [
        "Appendicitis",
        "Heart Attack", 
        "Common Cold or Flu",
        "Chikungunya Fever",
        "Dengue Fever"
    ]
    
    print("Starting specific disease tests...")
    
    try:
        tester = DiseaseTester(debug_mode=True)
        results = tester.run_specific_disease_tests(TARGET_DISEASES)
        
        if results:
            metrics = tester.calculate_accuracy(results)
            tester.print_results(results, metrics)
        else:
            print("❌ No results to display")
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        import traceback
        traceback.print_exc()