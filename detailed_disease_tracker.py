# detailed_disease_tracker.py
import sys
import os
import json
import numpy as np
import re
import time
from datetime import datetime

# Add the current directory to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medical_chatbot.settings')

import django
django.setup()

from chatbot_app.medical_ai.chat_bot import MedicalChatBot

class DetailedDiseaseTracker:
    def __init__(self, debug_mode=True):
        self.debug_mode = debug_mode
        self.chatbot = None
        self.initialize_chatbot()
        self.interaction_log = []
        
    def initialize_chatbot(self):
        """Initialize a fresh chatbot instance"""
        self.chatbot = MedicalChatBot()
        self.initialized = self.chatbot.initialize_bot()
        return self.initialized
        
    def load_test_cases(self, specific_diseases=None):
        """Load test cases for specific diseases only"""
        test_cases_path = os.path.join(os.path.dirname(__file__), 'test_cases.json')
        try:
            with open(test_cases_path, 'r') as f:
                all_test_cases = json.load(f)
                
            if specific_diseases:
                filtered_cases = []
                for test_case in all_test_cases:
                    expected_condition = test_case['expected_condition']
                    if any(disease.lower() in expected_condition.lower() for disease in specific_diseases):
                        filtered_cases.append(test_case)
                return filtered_cases
            return all_test_cases
                
        except Exception as e:
            print(f"❌ Error loading test cases: {e}")
            return []
    
    def force_reset_chatbot(self):
        """COMPLETE RESET - Create a brand new chatbot instance"""
        self.initialize_chatbot()
    
    def log_interaction(self, test_name, step, user_input, response, should_continue, symptoms, asked_questions):
        """Log every interaction for detailed analysis"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'test_name': test_name,
            'step': step,
            'user_input': user_input,
            'response': response[:500] if response else None,
            'should_continue': should_continue,
            'symptoms': symptoms.copy(),
            'asked_questions': list(asked_questions),
            'is_medical_chat': self.chatbot.is_medical_chat
        }
        self.interaction_log.append(interaction)
        
        if self.debug_mode:
            print(f"   📝 LOG: Step {step}, Continue: {should_continue}")
            print(f"   📝 Symptoms: {symptoms}")
            print(f"   📝 Asked: {list(asked_questions)}")
    
    def save_interaction_log(self, filename):
        """Save the detailed interaction log to file"""
        log_path = os.path.join(os.path.dirname(__file__), filename)
        with open(log_path, 'w') as f:
            json.dump(self.interaction_log, f, indent=2)
        print(f"✅ Interaction log saved to {filename}")
    
    def run_test_case_with_tracking(self, test_case):
        """Run a test case with detailed tracking of every step"""
        self.force_reset_chatbot()
        
        conversation = test_case['conversation']
        expected_condition = test_case['expected_condition']
        expected_confidence = test_case.get('expected_confidence', 70)
        
        final_diagnosis = None
        confidence = 0
        condition_detected = False
        got_final_diagnosis = False
        
        print(f"\n🔍 TRACKING: {test_case['name']}")
        print(f"   Expected: {expected_condition}")
        print("=" * 60)
        
        # Track each conversation step
        for i, user_input in enumerate(conversation):
            print(f"\n   Step {i+1}: User: '{user_input}'")
            
            query_type = self.chatbot.get_query_type(user_input)
            print(f"   Query type: {query_type}")
            
            if query_type == 'medical':
                response, should_continue = self.chatbot.process_medical_query(user_input)
                
                # Log this interaction
                self.log_interaction(
                    test_case['name'], i+1, user_input, response, should_continue,
                    self.chatbot.user_symptoms, self.chatbot.asked_questions
                )
                
                print(f"   Response: {response[:150]}..." if response else "   Response: None")
                print(f"   Should continue: {should_continue}")
                print(f"   Current symptoms: {self.chatbot.user_symptoms}")
                print(f"   Asked questions: {list(self.chatbot.asked_questions)}")
                
                # Check for final diagnosis
                if not should_continue and response and any(x in response for x in ["SUSPECTED CONDITION:", "MEDICAL ASSESSMENT"]):
                    got_final_diagnosis = True
                    final_diagnosis = response
                    
                    # Extract confidence
                    confidence_match = re.search(r"Confidence Level:[\s]*([\d.]+)%", response)
                    if confidence_match:
                        confidence = float(confidence_match.group(1))
                    
                    # Check if expected condition is mentioned
                    condition_detected = expected_condition.lower() in response.lower()
                    
                    print(f"   🎯 FINAL DIAGNOSIS REACHED!")
                    print(f"   Confidence: {confidence}%")
                    print(f"   Condition detected: {condition_detected}")
                    break
            
            else:
                print(f"   ❌ WRONG QUERY TYPE: {query_type} (expected: medical)")
                break
        
        # If no final diagnosis, analyze why
        if not got_final_diagnosis:
            print(f"   ❌ NO FINAL DIAGNOSIS REACHED")
            print(f"   Final symptoms: {self.chatbot.user_symptoms}")
            print(f"   Final asked questions: {list(self.chatbot.asked_questions)}")
            print(f"   Is medical chat: {self.chatbot.is_medical_chat}")
            
            # Check if we have enough symptoms for a diagnosis
            expected_symptoms = self.get_expected_symptoms(expected_condition)
            matched_symptoms = [s for s in self.chatbot.user_symptoms if s in expected_symptoms]
            print(f"   Expected symptoms: {expected_symptoms}")
            print(f"   Matched symptoms: {matched_symptoms} ({len(matched_symptoms)}/{len(expected_symptoms)})")
        
        success = condition_detected and confidence >= expected_confidence
        
        print(f"   🎯 TEST RESULT: {'✅ PASS' if success else '❌ FAIL'}")
        
        return {
            'success': success,
            'expected_condition': expected_condition,
            'detected_condition': final_diagnosis,
            'confidence': confidence,
            'condition_detected': condition_detected,
            'final_diagnosis_reached': got_final_diagnosis,
            'symptoms_count': len(self.chatbot.user_symptoms),
            'questions_asked': len(self.chatbot.asked_questions)
        }
    
    def get_expected_symptoms(self, condition_name):
        """Get expected symptoms for a condition from knowledge base"""
        knowledge_path = os.path.join(BASE_DIR, 'chatbot_app', 'medical_ai', 'medical_knowledge.json')
        try:
            with open(knowledge_path, 'r') as f:
                knowledge = json.load(f)
            
            for condition in knowledge['medical_data']:
                if condition_name.lower() in condition.get('condition', '').lower():
                    return condition.get('symptoms', [])
        except:
            pass
        return []
    
    def run_all_tests_with_detailed_tracking(self, diseases):
        """Run all tests with detailed tracking"""
        test_cases = self.load_test_cases(diseases)
        results = []
        
        if not test_cases:
            print("❌ No test cases found")
            return results
        
        print(f"\n🧪 DETAILED TRACKING for {len(test_cases)} test cases")
        print("=" * 70)
        
        for i, test_case in enumerate(test_cases):
            print(f"\n📋 TEST {i+1}/{len(test_cases)}: {test_case['name']}")
            result = self.run_test_case_with_tracking(test_case)
            results.append(result)
            time.sleep(0.5)  # Small delay between tests
        
        return results
    
    def generate_detailed_report(self, results):
        """Generate a comprehensive report"""
        total = len(results)
        successful = sum(1 for r in results if r.get('success', False))
        accuracy = (successful / total) * 100 if total > 0 else 0
        
        final_diagnosis_count = sum(1 for r in results if r.get('final_diagnosis_reached', False))
        
        print(f"\n📊 DETAILED ANALYSIS REPORT:")
        print("=" * 60)
        print(f"Total Tests: {total}")
        print(f"Successful: {successful} ({accuracy:.1f}%)")
        print(f"Final Diagnosis Reached: {final_diagnosis_count}/{total}")
        
        # Analyze failures
        failures = [r for r in results if not r['success']]
        if failures:
            print(f"\n🔍 FAILURE ANALYSIS ({len(failures)} cases):")
            for failure in failures:
                print(f"   ❌ {failure['expected_condition']}:")
                if not failure['final_diagnosis_reached']:
                    print(f"      No final diagnosis reached")
                elif not failure['condition_detected']:
                    print(f"      Wrong condition diagnosed")
                else:
                    print(f"      Low confidence: {failure['confidence']}%")
        
        # Save interaction log
        self.save_interaction_log('detailed_interaction_log.json')
        
        return {
            'total_tests': total,
            'successful_tests': successful,
            'accuracy': accuracy,
            'final_diagnosis_reached': final_diagnosis_count
        }

# Test specific diseases
TARGET_DISEASES = ["Appendicitis", "Heart Attack", "Common Cold or Flu", "Chikungunya Fever", "Dengue Fever"]

if __name__ == "__main__":
    print("Starting Detailed Disease Tracking...")
    print("This will track EVERY step of each conversation")
    print("=" * 70)
    
    try:
        tracker = DetailedDiseaseTracker(debug_mode=True)
        results = tracker.run_all_tests_with_detailed_tracking(TARGET_DISEASES)
        
        if results:
            report = tracker.generate_detailed_report(results)
            
            print(f"\n🎯 KEY INSIGHTS:")
            print("=" * 40)
            for i, result in enumerate(results):
                status = "✅" if result['success'] else "❌"
                print(f"{i+1}. {status} {result['expected_condition']}")
                print(f"   Symptoms: {result['symptoms_count']} | Questions: {result['questions_asked']}")
                print(f"   Final DX: {result['final_diagnosis_reached']} | Confidence: {result['confidence']}%")
                
        else:
            print("❌ No results to display")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()