# chatbot_app/tests/test_accuracy.py
import json
import os
from django.test import TestCase
from django.conf import settings

# Import your chatbot
from chatbot_app.medical_ai.chat_bot import MedicalChatBot

class ChatBotAccuracyTest(TestCase):
    
    def setUp(self):
        self.chatbot = MedicalChatBot()
        self.chatbot.initialize_bot()
        
        # Load test cases
        test_cases_path = os.path.join(settings.BASE_DIR, 'test_cases.json')
        with open(test_cases_path, 'r') as f:
            self.test_cases = json.load(f)
    
    def reset_chatbot(self):
        """Reset chatbot state between tests"""
        self.chatbot.reset_conversation()
    
    def run_test_case(self, test_case):
        """Run a single test case and return results"""
        self.reset_chatbot()
        
        conversation = test_case['conversation']
        expected_condition = test_case['expected_condition']
        expected_confidence = test_case.get('expected_confidence', 70)
        
        final_diagnosis = None
        confidence = 0
        
        # Simulate the conversation
        for user_input in conversation:
            query_type = self.chatbot.get_query_type(user_input)
            
            if query_type == 'medical':
                response, should_continue = self.chatbot.process_medical_query(user_input)
                
                # Check if we got a final diagnosis
                if not should_continue and "SUSPECTED CONDITION:" in response:
                    final_diagnosis = response
                    # Extract confidence if available
                    if "Confidence Level:" in response:
                        try:
                            confidence_str = response.split("Confidence Level:")[1].split("%")[0].strip()
                            confidence = float(confidence_str)
                        except:
                            confidence = 0
            
            # For other query types, we don't need the response for accuracy testing
        
        # Check if expected condition was detected
        condition_detected = expected_condition.lower() in final_diagnosis.lower() if final_diagnosis else False
        
        return condition_detected and confidence >= expected_confidence
    
    def test_chatbot_accuracy(self):
        """Test all test cases and calculate accuracy"""
        results = []
        for test_case in self.test_cases:
            success = self.run_test_case(test_case)
            results.append(success)
            print(f"Test '{test_case['name']}': {'PASS' if success else 'FAIL'}")
        
        # Calculate accuracy
        total = len(results)
        successful = sum(results)
        accuracy = (successful / total) * 100 if total > 0 else 0
        
        print(f"\nOverall Accuracy: {accuracy:.2f}%")
        print(f"Successful Tests: {successful}/{total}")
        
        # You can add assertions here
        self.assertGreaterEqual(accuracy, 60, "Accuracy should be at least 60%")