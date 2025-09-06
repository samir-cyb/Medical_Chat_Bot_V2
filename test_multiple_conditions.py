# test_multiple_conditions.py
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medical_chatbot.settings')

import django
django.setup()

from chatbot_app.medical_ai.chat_bot import MedicalChatBot

def test_different_conditions():
    chatbot = MedicalChatBot()
    chatbot.initialize_bot()
    
    test_cases = [
        {
            'name': 'Dengue Symptoms',
            'conversation': [
                "I have high fever for 4 days",
                "My temperature is 103.5",
                "Yes, I have a severe headache",
                "I was in Dhaka last week",
                "Yes, I have pain behind my eyes"
            ],
            'expected': 'Dengue'
        },
        {
            'name': 'Common Cold Symptoms', 
            'conversation': [
                "I have a runny nose and cough",
                "Yes, I have a sore throat",
                "No fever",
                "It started 2 days ago"
            ],
            'expected': 'Cold'
        },
        {
            'name': 'Appendicitis Symptoms',
            'conversation': [
                "I have severe pain in my lower right abdomen",
                "The pain is about 8 out of 10",
                "Yes, I feel nauseous",
                "It started yesterday and has been getting worse"
            ],
            'expected': 'Appendicitis'
        },
        {
            'name': 'Heart Attack Symptoms',
            'conversation': [
                "I have chest pain",
                "The pain is severe, about 9 out of 10", 
                "Yes, it's radiating to my left arm",
                "Yes, I'm sweating and feeling nauseous"
            ],
            'expected': 'Heart'
        }
    ]
    
    print("🧪 Testing Multiple Conditions")
    print("=" * 40)
    
    for test_case in test_cases:
        print(f"\n🔍 Testing: {test_case['name']}")
        print("-" * 30)
        
        chatbot.reset_conversation()
        
        for i, user_input in enumerate(test_case['conversation']):
            print(f"{i+1}. User: {user_input}")
            
            response, should_continue = chatbot.process_medical_query(user_input)
            print(f"   → Continue: {should_continue}")
            
            # Check if expected condition is mentioned
            if test_case['expected'].lower() in response.lower():
                print(f"   ✅ '{test_case['expected']}' mentioned!")
            
            if not should_continue:
                print(f"   🎯 FINAL DIAGNOSIS REACHED!")
                print(f"   Response: {response[:150]}...")
                break
                
        print(f"   Final symptoms: {chatbot.user_symptoms}")

if __name__ == "__main__":
    test_different_conditions()