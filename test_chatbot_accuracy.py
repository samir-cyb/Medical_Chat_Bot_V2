# test_chatbot_accuracy.py
import sys
import os
import json
import numpy as np
import re

# Add the current directory to Python path (where manage.py is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medical_chatbot.settings')

import django
django.setup()

# Now import your chatbot - use the correct path
from chatbot_app.medical_ai.chat_bot import MedicalChatBot

class ChatBotTester:
    def __init__(self, debug_mode=False):
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
        
    def load_test_cases(self):
        """Load test cases from the JSON file in the same directory"""
        test_cases_path = os.path.join(os.path.dirname(__file__), 'test_cases.json')
        try:
            with open(test_cases_path, 'r') as f:
                test_cases = json.load(f)
                
            if self.debug_mode:
                print(f"✅ Loaded {len(test_cases)} test cases")
                
            return test_cases
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
        """Improved condition detection with multiple strategies"""
        if not response:
            return False, 0
            
        response_lower = response.lower()
        expected_lower = expected_condition.lower()
        
        # Strategy 1: Direct match
        if expected_lower in response_lower:
            return True, 100  # 100% confidence in match
        
        # Strategy 2: Partial matches
        partial_matches = [
            expected_lower,
            expected_lower.replace("fever", "").strip(),
            expected_lower.replace("disease", "").strip(),
            expected_lower.replace("infection", "").strip(),
            expected_lower.replace(" or ", " ").replace("/", " ").strip(),
        ]
        
        for partial in partial_matches:
            if partial and partial in response_lower:
                return True, 90  # 90% confidence
        
        # Strategy 3: Keyword-based matching
        keyword_patterns = {
            "dengue": ["dengue", "df"],
            "chikungunya": ["chikungunya", "chik"],
            "malaria": ["malaria", "plasmodium"],
            "influenza": ["influenza", "flu"],
            "appendicitis": ["appendicitis", "appendix"],
            "heart attack": ["heart attack", "myocardial", "cardiac arrest"],
            "acid reflux": ["acid reflux", "gerd", "gastroesophageal"],
            "common cold": ["common cold", "cold", "rhinovirus"],
            "headache": ["headache", "migraine", "cephalgia"],
            "anxiety": ["anxiety", "panic attack", "panic disorder"]
        }
        
        for condition_key, keywords in keyword_patterns.items():
            if condition_key in expected_lower:
                for keyword in keywords:
                    if keyword in response_lower:
                        return True, 85  # 85% confidence
        
        # Strategy 4: Try to extract condition from the response
        condition_patterns = [
            r"SUSPECTED CONDITION:([^\n]+)",
            r"Condition:([^\n]+)",
            r"diagnosis:([^\n]+)",
            r"likely([^\n]+)condition",
        ]
        
        for pattern in condition_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted_condition = match.group(1).strip()
                if expected_lower in extracted_condition.lower():
                    return True, 95
                # Check if any word matches
                expected_words = expected_lower.split()
                extracted_words = extracted_condition.lower().split()
                if any(word in extracted_words for word in expected_words):
                    return True, 80
        
        return False, 0
    
    def extract_confidence_from_response(self, response):
        """Extract confidence score from response"""
        if not response:
            return 0
            
        # Look for confidence patterns
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
            
        # Look for patterns that indicate a final diagnosis
        diagnosis_indicators = [
            "SUSPECTED CONDITION:",
            "MEDICAL ASSESSMENT",
            "RECOMMENDATIONS",
            "final diagnosis",
            "assessment complete",
            "recommended action",
            "seek immediate medical",
            "consult a healthcare professional"
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
            
            # Get query type
            query_type = self.chatbot.get_query_type(user_input)
            
            if self.debug_mode:
                print(f"   Query type: {query_type}")
            
            try:
                if query_type == 'medical':
                    response, should_continue = self.chatbot.process_medical_query(user_input)
                    responses.append(response)
                    
                    if self.debug_mode:
                        print(f"   Response: {response[:200]}..." if response else "   Response: None")
                        print(f"   Should continue: {should_continue}")
                    
                    # Check if we got a final diagnosis
                    is_final = self.is_final_diagnosis(response)
                    if self.debug_mode and is_final:
                        print("   ✅ FINAL DIAGNOSIS DETECTED")
                    
                    if is_final and not got_final_diagnosis:
                        final_diagnosis = response
                        got_final_diagnosis = True
                        
                        # Extract confidence
                        confidence = self.extract_confidence_from_response(response)
                        
                        # Extract condition match
                        condition_detected, match_confidence = self.extract_condition_from_response(
                            response, expected_condition
                        )
                        
                        if self.debug_mode:
                            print(f"   Confidence extracted: {confidence}%")
                            print(f"   Condition detected: {condition_detected} (match confidence: {match_confidence})")
                
                elif query_type == 'informational':
                    response = self.chatbot.get_informational_response(user_input)
                    responses.append(response)
                    
                    if self.debug_mode:
                        print(f"   Info response: {response[:100]}...")
                    
                    # Check for condition in informational responses too
                    if not got_final_diagnosis:
                        condition_detected, match_confidence = self.extract_condition_from_response(
                            response, expected_condition
                        )
                
                else:
                    response = self.chatbot.get_casual_response(user_input)
                    responses.append(response)
                    
                    if self.debug_mode:
                        print(f"   Casual response: {response[:100]}...")
                        
            except Exception as e:
                if self.debug_mode:
                    print(f"   ❌ Error processing query: {e}")
                return {
                    'success': False,
                    'expected_condition': expected_condition,
                    'detected_condition': f"Error: {str(e)}",
                    'confidence': 0,
                    'condition_detected': False,
                    'error': str(e)
                }
        
        # For tests that didn't reach final diagnosis, check all responses
        if not got_final_diagnosis and self.debug_mode:
            print("   ⚠️  No final diagnosis detected in any response")
            print("   🔍 Searching all responses for condition...")
            
            # Check all responses for the condition
            for i, response in enumerate(responses):
                if response:
                    detected, conf = self.extract_condition_from_response(response, expected_condition)
                    if detected:
                        condition_detected = True
                        match_confidence = max(match_confidence, conf)
                        if self.debug_mode:
                            print(f"   Found condition in response {i+1} with confidence {conf}")
        
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
            'final_diagnosis_reached': got_final_diagnosis,
            'conversation': conversation,
            'responses': responses
        }
    
    def run_all_tests(self):
        """Run all test cases and return results"""
        test_cases = self.load_test_cases()
        results = []
        
        if not test_cases:
            print("❌ No test cases loaded")
            return results
        
        for i, test_case in enumerate(test_cases):
            print(f"Running test case {i+1}/{len(test_cases)}: {test_case['name']}")
            result = self.run_test_case(test_case)
            results.append(result)
            
            # Add small delay between tests to avoid any resource conflicts
            import time
            time.sleep(0.1)
        
        return results
    
    def calculate_accuracy(self, results):
        """Calculate accuracy metrics"""
        total = len(results)
        successful = sum(1 for r in results if r.get('success', False))
        accuracy = (successful / total) * 100 if total > 0 else 0
        
        confidences = [r['confidence'] for r in results if r.get('confidence', 0) > 0]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        match_confidences = [r['match_confidence'] for r in results if r.get('match_confidence', 0) > 0]
        avg_match_confidence = np.mean(match_confidences) if match_confidences else 0
        
        final_diagnosis_count = sum(1 for r in results if r.get('final_diagnosis_reached', False))
        
        return {
            'total_tests': total,
            'successful_tests': successful,
            'accuracy': accuracy,
            'average_confidence': avg_confidence,
            'average_match_confidence': avg_match_confidence,
            'final_diagnosis_reached': final_diagnosis_count
        }
    
    def generate_report(self, results, metrics):
        """Generate a detailed test report"""
        report = {
            'metrics': metrics,
            'detailed_results': [
                {
                    'name': f"Test {i+1}",
                    'expected': r['expected_condition'],
                    'success': r['success'],
                    'confidence': r['confidence'],
                    'match_confidence': r.get('match_confidence', 0),
                    'condition_detected': r.get('condition_detected', False),
                    'final_diagnosis': r.get('final_diagnosis_reached', False),
                    'error': r.get('error', None)
                }
                for i, r in enumerate(results)
            ]
        }
        
        # Save report to file
        report_path = os.path.join(os.path.dirname(__file__), 'test_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def debug_single_test(self, test_case_name):
        """Debug a specific test case in detail"""
        test_cases = self.load_test_cases()
        for test_case in test_cases:
            if test_case['name'] == test_case_name:
                print(f"\n🔍 DEBUGGING: {test_case['name']}")
                print("=" * 60)
                
                result = self.run_test_case(test_case)
                
                print(f"\n📊 RESULTS:")
                print(f"Expected: {result['expected_condition']}")
                print(f"Success: {result['success']}")
                print(f"Confidence: {result['confidence']}%")
                print(f"Condition detected: {result['condition_detected']}")
                print(f"Match confidence: {result.get('match_confidence', 0)}")
                print(f"Final diagnosis reached: {result.get('final_diagnosis_reached', False)}")
                
                if result['detected_condition']:
                    print(f"\n📝 FULL RESPONSE:")
                    print(result['detected_condition'])
                
                # Show all responses
                print(f"\n📋 ALL RESPONSES:")
                for i, response in enumerate(result['responses']):
                    if response:
                        print(f"{i+1}. {response[:150]}...")
                
                return result
        
        print(f"❌ Test case '{test_case_name}' not found")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test chatbot accuracy')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--single', type=str, help='Debug a single test case by name')
    args = parser.parse_args()
    
    print("Starting chatbot accuracy tests...")
    
    try:
        tester = ChatBotTester(debug_mode=args.debug)
        
        if args.single:
            # Debug a single test case
            result = tester.debug_single_test(args.single)
        else:
            # Run all tests
            results = tester.run_all_tests()
            metrics = tester.calculate_accuracy(results)
            report = tester.generate_report(results, metrics)
            
            print(f"\n📈 Test Results:")
            print("=" * 50)
            print(f"Accuracy: {metrics['accuracy']:.2f}%")
            print(f"Average Confidence: {metrics['average_confidence']:.2f}%")
            print(f"Average Match Confidence: {metrics['average_match_confidence']:.2f}%")
            print(f"Final Diagnosis Reached: {metrics['final_diagnosis_reached']}/{metrics['total_tests']}")
            print(f"Successful Tests: {metrics['successful_tests']}/{metrics['total_tests']}")
            
            # Print detailed results
            print(f"\n📋 Detailed Results:")
            print("=" * 50)
            for i, result in enumerate(results):
                status = "✅ PASS" if result['success'] else "❌ FAIL"
                final_dx = "✓" if result.get('final_diagnosis_reached', False) else "✗"
                print(f"{i+1:2d}. {status} {final_dx} - {result['expected_condition']}")
                print(f"    Confidence: {result['confidence']}% | Match: {result.get('match_confidence', 0)}")
                if 'error' in result:
                    print(f"    Error: {result['error']}")
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        import traceback
        traceback.print_exc()