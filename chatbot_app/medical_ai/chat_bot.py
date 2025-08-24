import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "medical_chatbot.settings")  # replace with your project settings path
django.setup()


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import re
import json
import random
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import numpy as np
from .config import KNOWLEDGE_BASE_PATH, PERSIST_DIRECTORY

class MedicalChatBot:
    
    
    
    def is_informational_question(self, user_input: str) -> bool:
        """Check if the user is asking for general information rather than describing symptoms"""
        input_lower = user_input.lower()
        
        informational_phrases = [
            # Symptoms
            'symptoms of', 'signs of', 'what are symptoms', 'what is symptom',
            'tell me symptoms', 'explain symptoms', 'describe symptoms',
            'what does', 'what do', 'what is', 'what are',
            
            # Causes
            'what causes', 'cause of', 'why do people get', 'why does happen',
            'risk factors', 'dangers of', 'is it contagious', 'how does spread',
            
            # Treatment
            'how to treat', 'treatment for', 'ways to treat', 'cure for',
            'how do you cure', 'medicine for', 'remedy for', 'how is treated',
            
            # Prevention
            'how to prevent', 'prevention of', 'avoid getting', 'protect yourself',
            'can you get vaccinated', 'vaccine for',
            
            # General info
            'information about', 'can you tell me about', 'explain about',
            'give me details on', 'overview of', 'definition of', 'what is a ',
            'what is an ', 'what is the ', 'what does mean', 'how common is',
            'who is at risk', 'how long does last', 'how serious is',
            'can you die from', 'is it dangerous', 'when to see a doctor'
        ]
        
        # Check if it starts with informational phrases
        for phrase in informational_phrases:
            if input_lower.startswith(phrase) or f" {phrase}" in input_lower:
                return True
        
        # Check for pattern: "disease_name" + "symptoms"/"treatment" etc.
        disease_info_patterns = [
            r'(\w+)\s+symptoms',
            r'(\w+)\s+treatment',
            r'(\w+)\s+causes',
            r'(\w+)\s+prevention',
            r'symptoms of (\w+)',
            r'treatment for (\w+)',
            r'causes of (\w+)'
        ]
        
        for pattern in disease_info_patterns:
            if re.search(pattern, input_lower):
                return True
        
        return False
        
    def reset_conversation(self):
        """Reset the conversation state"""
        self.chat_history = []
        self.asked_questions.clear()
        self.current_follow_ups = []
        self.user_symptoms = []
        self.user_info = {
            "fever_duration": None,
            "fever_temperature": None,
            "travel_history": None,
            "symptom_onset": None,
            "age": None,
            "preexisting_conditions": None,
            "pain_location": None,
            "pain_severity": None,
            "symptom_duration": None,
            "current_medications": None
        }
        self.diagnostic_stage = "initial"
        self.suspected_conditions = []
        self.current_focus = None
    
    def __init__(self):
        self.qa_chain = None
        self.vectordb = None
        self.initialized = False
        self.chat_history = []
        self.asked_questions = set()
        self.is_medical_chat = False
        self.current_follow_ups = []
        self.user_symptoms = []
        self.symptom_weights = {}
        self.symptom_mapping = {}
        self.medical_data = []
        self.dengue_prone_areas = []
        self.chikungunya_prone_areas = []
        self.malaria_prone_areas = []
        self.user_info = {
            "fever_duration": None,
            "fever_temperature": None,
            "travel_history": None,
            "symptom_onset": None,
            "age": None,
            "preexisting_conditions": None,
            "pain_location": None,
            "pain_severity": None,
            "symptom_duration": None,
            "current_medications": None
        }
        self.diagnostic_stage = "initial"
        self.suspected_conditions = []
        self.current_focus = None
        self.last_question = None
        
    def load_knowledge_base(self):
        """Load the knowledge base with symptom weights and mappings"""
        try:
            with open(KNOWLEDGE_BASE_PATH, 'r') as f:
                full_data = json.load(f)
            
            self.medical_data = full_data["medical_data"]
            self.symptom_weights = full_data["symptom_weights"]
            self.symptom_mapping = full_data["symptom_mapping"]
            self.dengue_prone_areas = full_data["dengue_prone_areas"]
            self.chikungunya_prone_areas = full_data["chikungunya_prone_areas"]
            if "malaria_prone_areas" in full_data:
                self.malaria_prone_areas = full_data["malaria_prone_areas"]
            return True
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            return False
            
    def initialize_bot(self) -> bool:
        """Initializes and returns the QA chain for the chatbot."""
        try:
            # Load knowledge base first
            if not self.load_knowledge_base():
                return False
                
            # 1. Create embedding function
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False}
            )

            # 2. Load the existing vector database
            self.vectordb = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings,
                collection_name="medical_knowledge"
            )

            # 3. Initialize the local LLM with fallback options
            model_options = ["gemma:2b", "llama2:7b", "llama2", "mistral"]
            self.llm = None
            
            for model_name in model_options:
                try:
                    self.llm = Ollama(model=model_name)
                    print(f"✅ Using model: {model_name}")
                    break
                except Exception as e:
                    print(f"❌ Model {model_name} not available: {e}")
                    continue
            
            if self.llm is None:
                print("❌ No suitable Ollama model found. Please install at least one model.")
                return False

            # 4. Build a diagnostic prompt template
            prompt_template = """
            You are a compassionate and thorough medical diagnostic assistant. Your goal is to:
            1. Understand the patient's symptoms through careful questioning
            2. Identify potential medical conditions based on symptoms
            3. Provide appropriate medical guidance and recommendations
            
            Medical Knowledge Context:
            {context}
            
            Current Conversation History:
            {chat_history}
            
            Patient's Current Statement:
            {question}
            
            Patient's Known Information:
            - Symptoms: {symptoms}
            - Fever duration: {fever_duration}
            - Fever temperature: {fever_temperature}
            - Travel history: {travel_history}
            - Pain location: {pain_location}
            - Symptom duration: {symptom_duration}
            - Symptom onset: {symptom_onset}
            - Preexisting conditions: {preexisting_conditions}
            - Current medications: {current_medications}
            
            Diagnostic Instructions:
            1. Analyze the patient's statement in context of the conversation history
            2. If more information is needed, ask SPECIFIC follow-up questions
            3. If enough information is available, provide:
               - Potential diagnosis (most likely conditions)
               - Recommended next steps
               - First aid advice if appropriate
               - Doctor specialty recommendation
            4. Always be empathetic, professional, and clear
            5. For urgent symptoms, emphasize immediate medical attention
            
            Your Response:
            """
            
            PROMPT = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "chat_history", "question", "symptoms", 
                               "fever_duration", "fever_temperature", "travel_history",
                               "pain_location", "symptom_duration"]
            )

            # 5. Create a retrieval chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectordb.as_retriever(search_kwargs={"k": 5}),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            self.initialized = True
            return True

        except Exception as e:
            print(f"❌ Failed to initialize chatbot: {e}")
            return False

    def extract_symptoms(self, user_input: str) -> List[str]:
        """Extract and normalize symptoms from user input using the symptom mapping"""
        user_input_lower = user_input.lower()
        found_symptoms = set()
        
        # Create a reverse mapping for easy lookup
        reverse_mapping = {}
        for normalized, variations in self.symptom_mapping.items():
            for variation in variations:
                reverse_mapping[variation] = normalized
        
        # Check for each possible symptom variation
        for symptom_variation, normalized_symptom in reverse_mapping.items():
            if re.search(r'\b' + re.escape(symptom_variation) + r'\b', user_input_lower):
                found_symptoms.add(normalized_symptom)
        
        # Also check for direct matches with known symptoms
        for symptom in self.symptom_weights.keys():
            if re.search(r'\b' + re.escape(symptom) + r'\b', user_input_lower):
                found_symptoms.add(symptom)
                
        # Extract fever temperature if mentioned
        fever_match = re.search(r'(\d{2,3}(?:\.\d{1,2})?)\s*(degrees|°|fahrenheit|f|celcius|c|degree|deg|temp|temperature)?', user_input_lower)
        if fever_match:
            try:
                temp = float(fever_match.group(1))
                self.user_info["fever_temperature"] = temp
                found_symptoms.add("fever")
                if temp >= 102:
                    found_symptoms.add("high fever")
                elif temp <= 101:
                    found_symptoms.add("mild fever")
            except ValueError:
                pass  # Handle invalid temperature format
        
        # Extract fever duration if mentioned
        duration_match = re.search(r'(\d+)\s*(day|days|hour|hours|week|weeks)', user_input_lower)
        if duration_match:
            self.user_info["fever_duration"] = f"{duration_match.group(1)} {duration_match.group(2)}"
            self.user_info["symptom_duration"] = f"{duration_match.group(1)} {duration_match.group(2)}"
            
        # Extract age if mentioned
        age_match = re.search(r'(\d+)\s*(year|years|yr|yrs|yo|year old)', user_input_lower)
        if age_match:
            self.user_info["age"] = age_match.group(1)
            
        # Extract travel history if mentioned
        travel_areas = self.dengue_prone_areas + self.chikungunya_prone_areas + self.malaria_prone_areas
        for area in travel_areas:
            if re.search(r'\b' + re.escape(area.lower()) + r'\b', user_input_lower):
                self.user_info["travel_history"] = area
                break
                
        # Extract pain location if mentioned
        pain_locations = {
            "eyes": ["eyes", "eye", "behind eyes", "ocular", "retro-orbital"],
            "chest": ["chest", "breast", "sternum"],
            "abdomen": ["abdomen", "stomach", "belly", "tummy"],
            "head": ["head", "headache", "migraine", "cranial"],
            "joint": ["joint", "knee", "elbow", "wrist", "ankle", "arthralgia"],
            "back": ["back", "spine", "spinal"],
            "neck": ["neck", "cervical"]
        }
        
        for location, keywords in pain_locations.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', user_input_lower):
                    self.user_info["pain_location"] = location
                    break
        
        # Extract pain severity if mentioned
        if any(term in user_input_lower for term in ["severe pain", "extreme pain", "worst pain", "unbearable pain", "10/10", "ten out of ten"]):
            self.user_info["pain_severity"] = "severe"
        elif any(term in user_input_lower for term in ["moderate pain", "medium pain", "manageable pain", "5/10", "6/10", "7/10"]):
            self.user_info["pain_severity"] = "moderate"
        elif any(term in user_input_lower for term in ["mild pain", "slight pain", "minor pain", "1/10", "2/10", "3/10", "4/10"]):
            self.user_info["pain_severity"] = "mild"
                
        return list(found_symptoms)

    def calculate_symptom_score(self, user_symptoms: List[str], condition_symptoms: List[str]) -> float:
        """Calculate a weighted match score with synonym matching"""
        score = 0
        max_possible = 0
        
        # Create symptom mapping for fuzzy matching
        symptom_equivalents = {
            "high fever": ["high fever with sudden onset", "very high fever", "spiking fever"],
            "fever": ["temperature", "hot", "burning up"],
            "severe headache": ["bad headache", "worst headache", "excruciating headache"],
            "pain behind eyes": ["eye socket pain", "retro-orbital pain", "deep eye pain"],
            # Add more mappings as needed
        }
        
        # Check each user symptom against condition symptoms
        for user_symptom in user_symptoms:
            # Get all equivalent ways to say this symptom
            equivalents = symptom_equivalents.get(user_symptom, []) + [user_symptom]
            
            # Check if any equivalent matches any condition symptom
            for equivalent in equivalents:
                if equivalent in condition_symptoms:
                    score += self.symptom_weights.get(user_symptom, 5)
                    break
        
        # Calculate max possible based on condition symptoms
        for symptom in condition_symptoms:
            max_possible += self.symptom_weights.get(symptom, 5)
                
        return (score / max_possible * 100) if max_possible > 0 else 0

    def find_best_matches(self, user_symptoms: List[str], top_n: int = 3) -> List[Dict]:
        """Find the top conditions that match the user's symptoms"""
        matches = []
        
        for condition in self.medical_data:
            condition_symptoms = condition["symptoms"]
            score = self.calculate_symptom_score(user_symptoms, condition_symptoms)
            
            if score > 30:  # Only consider matches with at least 30% score
                matches.append({
                    "condition": condition,
                    "score": score,
                    "matched_symptoms": [s for s in user_symptoms if s in condition_symptoms]
                })
        
        # Sort by score descending
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:top_n]

    def get_diagnostic_questions(self, suspected_conditions: List[Dict]) -> List[str]:
        """Generate specific diagnostic questions based on suspected conditions"""
        questions = []
        
        # If we have suspected conditions, ask differentiating questions
        if suspected_conditions:
            top_condition = suspected_conditions[0]["condition"]
            condition_name = top_condition.get("condition", "").lower()
            
            # Ask condition-specific questions
            if "dengue" in condition_name:
                questions.extend([
                    "Have you noticed any bleeding from your gums or nose?",
                    "Do you have any red spots or rash on your skin?",
                    "Have you traveled to any dengue-prone areas recently?"
                ])
            elif "chikungunya" in condition_name:
                questions.extend([
                    "Are your joints swollen or painful to move?",
                    "Is the joint pain affecting your ability to perform daily activities?",
                    "Have you been in areas with known chikungunya outbreaks?"
                ])
            elif "appendicitis" in condition_name:
                questions.extend([
                    "Is the pain specifically in your lower right abdomen?",
                    "Does the pain get worse when you move or cough?",
                    "Have you experienced loss of appetite or nausea?"
                ])
            elif any(term in condition_name for term in ["heart", "cardiac"]):
                questions.extend([
                    "Does the pain radiate to your arm, jaw, or back?",
                    "Are you experiencing shortness of breath or sweating?",
                    "Do you have a history of heart problems?"
                ])
        
        # General diagnostic questions based on symptoms
        if any(s in self.user_symptoms for s in ["fever", "high fever"]):
            if not self.user_info["fever_duration"]:
                questions.append("How long have you had fever?")
            if not self.user_info["fever_temperature"]:
                questions.append("What is your current temperature?")
                
        if any(s in self.user_symptoms for s in ["pain", "abdominal pain", "chest pain"]):
            if not self.user_info["pain_location"]:
                questions.append("Can you tell me exactly where the pain is located?")
            if not self.user_info["pain_severity"]:
                questions.append("On a scale of 1-10, how severe is the pain?")
                
        if any(s in self.user_symptoms for s in ["rash", "bleeding gums", "nosebleeds"]):
            questions.append("When did you first notice these symptoms?")
            
        if not self.user_info["travel_history"] and any(s in self.user_symptoms for s in ["fever", "rash", "joint pain"]):
            questions.append("Have you traveled anywhere recently?")
            
        if (not self.user_info.get("travel_history") and 
        self.user_info.get("travel_history") != "No travel" and
        any(s in self.user_symptoms for s in ["fever", "rash", "joint pain"])):
            questions.append("Have you traveled anywhere recently?")
                
        # Remove duplicates and limit to 3 questions
        questions = list(set(questions))[:3]
        
        return questions

    def generate_empathetic_response(self) -> str:
        """Generate an empathetic introduction to questions"""
        empathetic_intros = [
            "I understand this must be concerning. To help you better, ",
            "I want to make sure I understand your situation completely. ",
            "I know health concerns can be worrying. Let me ask a few questions to clarify. ",
            "Thank you for sharing that. To provide the best advice, ",
            "I appreciate you telling me about this. To help me understand better, "
        ]
        
        return random.choice(empathetic_intros)
    
    def get_informational_response(self, user_input: str) -> str:
        """Handle informational questions about diseases and symptoms"""
        input_lower = user_input.lower()
        
        # Extract disease name using simpler method
        disease_name = None
        
        # Try to find disease name after common informational phrases
        patterns = [
            r'symptoms of (\w+)',
            r'signs of (\w+)', 
            r'what causes (\w+)',
            r'treatment for (\w+)',
            r'how to treat (\w+)',
            r'information about (\w+)',
            r'what is (\w+)',
            r'(\w+) symptoms',
            r'(\w+) treatment',
            r'(\w+) causes'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, input_lower)
            if match:
                disease_name = match.group(1).strip()
                # Remove question words that might have been captured
                disease_name = re.sub(r'\b(what|is|are|of|for|about|the|a|an)\b', '', disease_name)
                disease_name = disease_name.strip()
                if disease_name:  # Only use if we have a real disease name
                    break
        
        if disease_name:
            return self.get_disease_information(disease_name)
        
        # If no disease detected, ask for clarification
        return "I'd be happy to provide information about medical conditions. Could you specify which disease or condition you're asking about?"
        
    def debug_query_type(self, user_input: str):
        """Debug method to see how the query is being classified"""
        input_lower = user_input.lower().strip()
        print(f"DEBUG: Input: '{user_input}'")
        print(f"DEBUG: Lower: '{input_lower}'")
        
        # Check informational first
        is_info = self.is_informational_question(user_input)
        print(f"DEBUG: Is informational: {is_info}")
        
        if is_info:
            print("DEBUG: Should return 'informational'")
            return
        
        # Check other types
        print(f"DEBUG: is_medical_chat: {self.is_medical_chat}")
        print(f"DEBUG: chat_history length: {len(self.chat_history)}")
        
        # Check medical keywords
        medical_keywords = ['fever', 'pain', 'symptom']  # Add your actual keywords
        has_medical = any(keyword in input_lower for keyword in medical_keywords)
        print(f"DEBUG: Has medical keywords: {has_medical}")

    # Add this temporarily to your get_query_type method for debugging
    
    def get_query_type(self, user_input: str) -> str:
        """Determine if the user input is a greeting, medical query, or casual conversation"""
        input_lower = user_input.lower().strip()
        
        if not input_lower:
            return 'casual'
        
        # DEBUG: Print what we're analyzing
        print(f"ANALYZING: '{user_input}'")
        print(f"is_medical_chat: {self.is_medical_chat}")
        print(f"chat_history: {self.chat_history}")
        
        # 1. Check for exit commands FIRST (highest priority)
        if any(exit_cmd in input_lower for exit_cmd in ['quit', 'exit', 'bye', 'goodbye']):
            print("RETURNING: exit")
            return 'exit'
        
        # 2. Check for informational questions SECOND
        is_info = self.is_informational_question(user_input)
        print(f"is_informational_question: {is_info}")
        
        if is_info:
            print("RETURNING: informational")
            return 'informational'
        
        # 3. If we're already in a medical conversation, continue it
        if self.is_medical_chat and len(self.chat_history) > 0:
            print("RETURNING: medical (continuation)")
            return 'medical'
        
        # 4. Check for medical keywords (but exclude informational patterns)
        medical_keywords = [
            r'\bpain\b', r'\bhurt\b', r'\bache\b', r'\bsore\b', r'\binjury\b', r'\bwound\b', 
            r'\bbleed\b', r'\bblood\b', r'\bfever\b', r'\btemperature\b', r'\bcough\b', 
            r'\bsneeze\b', r'\bcold\b', r'\bflu\b', r'\bheadache\b', r'\bstomach\b', 
            r'\bchest\b', r'\barm\b', r'\bleg\b', r'\bback\b', r'\bneck\b', r'\bnausea\b', 
            r'\bvomit\b', r'\bdizzy\b', r'\bdizziness\b', r'\brush\b', r'\bitch\b', 
            r'\bbreath\b', r'\bbreathe\b', r'\bheart\b', r'\bpalpitation\b', r'\bdoctor\b', 
            r'\bhospital\b', r'\bemergency\b', r'\burgent\b', r'\bclinic\b', 
            r'\bappointment\b', r'\bmedicine\b', r'\bpill\b', r'\bsymptom\b', r'\bill\b',
            r'\bsick\b', r'\bunwell\b', r'\bhealth\b', r'\bmedical\b', r'\bchikungunya\b',
            r'\bdengue\b', r'\bjoint pain\b', r'\brash\b', r'\bvomiting\b', r'\bdehydration\b',
            r'\bbleeding\b', r'\bnosebleed\b', r'\bgum bleeding\b', r'\babdominal pain\b',
            r'\bmosquito\b', r'\bbite\b', r'\btravel\b', r'\bvisited\b', r'\bfeeling unwell\b',
            r'\bnot feeling well\b', r'\bnot well\b', r'\bunwell\b', r'\bi have\b', r'\bi feel\b',
            r'\bmy\b.*\bhurts\b', r'\bexperiencing\b', r'\bhaving\b'
        ]
        
        has_medical_keyword = any(re.search(keyword, input_lower) for keyword in medical_keywords)
        print(f"has_medical_keyword: {has_medical_keyword}")
        
        if has_medical_keyword:
            print("RETURNING: medical (new conversation)")
            return 'medical'
        
        # 5. Casual greetings and small talk
        casual_patterns = [
            r'^hello$', r'^hi$', r'^hey$', r'how are you', r'good morning', r'good afternoon',
            r'good evening', r'what\'s up', r'howdy', r'greetings', r'thank you', r'thanks',
            r'who are you', r'what can you do', r'help me', r'your name', r'about you'
        ]
        
        if any(re.search(pattern, input_lower) for pattern in casual_patterns):
            print("RETURNING: greeting")
            return 'greeting'
        
        print("RETURNING: casual (default)")
        return 'casual'
    
    def is_informational_question(self, user_input: str) -> bool:
        """Check if the user is asking for general information rather than describing symptoms"""
        input_lower = user_input.lower()
        
        # Phrases that indicate informational questions
        informational_starters = [
            'what are', 'what is', 'tell me about', 'explain', 'describe',
            'how to', 'can you tell me', 'information about', 'details about',
            'what does', 'what do', 'how does', 'why does'
        ]
        
        # Check if it starts with informational phrases
        for starter in informational_starters:
            if input_lower.startswith(starter):
                return True
        
        # Specific informational patterns about diseases/symptoms
        informational_patterns = [
            r'symptoms of', r'signs of', r'treatment for', r'causes of',
            r'prevention of', r'medicine for', r'cure for', r'vaccine for',
            r'risk factors for', r'dangers of', r'information about',
            r'what are.*symptoms', r'what is.*treatment', r'how to treat',
            r'how to prevent', r'can you die from', r'is it contagious'
        ]
        
        for pattern in informational_patterns:
            if re.search(pattern, input_lower):
                return True
        
        # Check if it's asking ABOUT a disease rather than describing symptoms
        disease_words = ['dengue', 'malaria', 'chikungunya', 'flu', 'fever', 'headache']
        question_words = ['what', 'how', 'why', 'when', 'where', 'can', 'does', 'is', 'are']
        
        has_disease_word = any(word in input_lower for word in disease_words)
        has_question_word = any(word in input_lower for word in question_words)
        has_first_person = any(phrase in input_lower for phrase in ['i have', 'i feel', 'my', 'i am', 'i\'m'])
        
        # If it has disease words and question words but NOT first-person experience
        if has_disease_word and has_question_word and not has_first_person:
            return True
        
        return False
    
    
    def get_disease_information(self, disease_name: str) -> str:
        """Get information about a specific disease from the knowledge base"""
        disease_name_lower = disease_name.lower()
        
        # Search for matching conditions
        matching_conditions = []
        for condition in self.medical_data:
            cond_name = condition.get('condition', '').lower()
            if disease_name_lower in cond_name or any(disease_name_lower in symptom.lower() for symptom in condition.get('symptoms', [])):
                matching_conditions.append(condition)
        
        if not matching_conditions:
            return f"I don't have specific information about {disease_name}. Please describe your symptoms and I can help with possible conditions."
        
        # Build informative response using only available fields
        response = f"📋 Information about **{disease_name.title()}**:\n\n"
        
        for condition in matching_conditions:
            response += f"🔹 Condition: {condition.get('condition', 'Unknown')}\n"
            
            if condition.get('symptoms'):
                response += f"🤒 Symptoms: {', '.join(condition['symptoms'])}\n"
            
            response += f"⏱️ Urgency: {condition.get('urgency', 'Not specified')}\n"
            response += f"✅ Recommended action: {condition.get('suggested_action', 'Not specified')}\n"
            
            if condition.get('first_aid'):
                response += f"⛑️ First aid: {condition['first_aid']}\n"
            
            response += f"🏥 Specialty: {condition.get('specialty', 'General Medicine')}\n"
            
            # Add some general information based on condition name
            if "dengue" in disease_name_lower:
                response += "🧬 Causes: Mosquito-borne viral infection\n"
                response += "🛡️ Prevention: Use mosquito repellent, eliminate standing water\n"
                response += "💊 Treatment: Supportive care, fluid replacement, avoid aspirin\n"
            elif "malaria" in disease_name_lower:
                response += "🧬 Causes: Parasite transmitted through mosquito bites\n"
                response += "🛡️ Prevention: Mosquito nets, antimalarial medication\n"
                response += "💊 Treatment: Antimalarial drugs prescribed by doctor\n"
            
            response += "\n" + "-"*50 + "\n\n"
        
        response += "⚠️ Remember: This is **general information**. Please consult a doctor for personal medical advice."
        return response

    def generate_final_recommendation(self, condition: Dict, confidence: float) -> str:
        """Generate a structured final recommendation"""
        response = "\n" + "="*60
        response += "\nMEDICAL ASSESSMENT & RECOMMENDATIONS"
        response += "\n" + "="*60
        
        # Condition and confidence
        response += f"\n\nSUSPECTED CONDITION: {condition.get('condition', 'Medical Condition')}"
        response += f"\nConfidence Level: {confidence:.1f}%"
        
        if 'stage' in condition:
            response += f"\nStage: {condition['stage']}"
        
        # Urgency assessment
        urgency = condition.get('urgency', '').lower()
        if urgency in ['emergency', 'dangerous', 'urgent']:
            response += "\n\n🚨 URGENCY: IMMEDIATE MEDICAL ATTENTION REQUIRED"
            response += "\nPlease seek emergency care right away."
        else:
            response += f"\n\nUrgency: {condition.get('urgency', 'Consultation recommended')}"
        
        # Recommended actions
        response += "\n\nRECOMMENDED ACTIONS:"
        response += f"\n• {condition.get('suggested_action', 'Consult a healthcare professional')}"
        
        # First aid advice
        if 'first_aid' in condition:
            response += "\n\nFIRST AID / SELF-CARE:"
            response += f"\n• {condition['first_aid']}"
        
        # Test recommendations
        response += "\n\nRECOMMENDED TESTS:"
        if "dengue" in condition.get('condition', '').lower():
            response += f"\n• {self.get_dengue_test_recommendation()}"
        elif "chikungunya" in condition.get('condition', '').lower():
            response += f"\n• {self.get_chikungunya_test_recommendation()}"
        else:
            response += "\n• Diagnostic tests as recommended by your doctor based on examination"
        
        # Doctor specialty
        response += "\n\nSPECIALIST RECOMMENDATION:"
        response += f"\n• Please consult a: {condition.get('specialty', 'Primary Care Physician')}"
        
        # Travel history note
        if self.user_info["travel_history"]:
            travel_risk = self.check_travel_history_risk(self.user_info["travel_history"])
            if travel_risk != "none":
                response += f"\n\nNOTE: Your recent travel to {self.user_info['travel_history']} may be relevant to your symptoms."
        
        # Final reassurance
        comforting_closing = [
            "\n\nRemember, early medical consultation leads to better outcomes.",
            "\n\nPlease don't hesitate to seek professional medical advice.",
            "\n\nYour health is important - please follow up with a healthcare provider.",
            "\n\nMany similar conditions are treatable with proper medical care."
        ]
        
        response += random.choice(comforting_closing)
        
        # Disclaimer
        response += "\n\n" + "="*60
        response += "\nREMINDER: I am an AI assistant for informational purposes only."
        response += "\nPlease consult a qualified healthcare professional for any medical concerns."
        response += "\n" + "="*60
        
        return response

    def get_dengue_test_recommendation(self):
        """Recommend appropriate dengue test based on fever duration"""
        if not self.user_info["fever_duration"]:
            return "NS1 antigen test or IgM antibody test"
        
        try:
            fever_days = int(self.user_info["fever_duration"].split()[0])
            if fever_days <= 3:
                return "NS1 antigen test (most accurate in early infection)"
            else:
                return "IgM antibody test (more accurate after 3-4 days of fever)"
        except:
            return "NS1 antigen test or IgM antibody test"

    def get_chikungunya_test_recommendation(self):
        """Recommend appropriate chikungunya test based on fever duration"""
        if not self.user_info["fever_duration"]:
            return "RT-PCR test or IgM antibody test"
        
        try:
            fever_days = int(self.user_info["fever_duration"].split()[0])
            if fever_days <= 5:
                return "RT-PCR test (detects virus directly in early infection)"
            else:
                return "IgM antibody test (detects immune response after 5-7 days)"
        except:
            return "RT-PCR test or IgM antibody test"

    def check_travel_history_risk(self, location: str) -> str:
        """Check if travel history suggests risk for dengue or chikungunya"""
        if not location:
            return "none"
            
        location_lower = location.lower()
        
        dengue_risk = any(area.lower() in location_lower for area in self.dengue_prone_areas)
        chikungunya_risk = any(area.lower() in location_lower for area in self.chikungunya_prone_areas)
        malaria_risk = any(area.lower() in location_lower for area in self.malaria_prone_areas)
        
        if dengue_risk and chikungunya_risk:
            return "both"
        elif dengue_risk:
            return "dengue"
        elif chikungunya_risk:
            return "chikungunya"
        elif malaria_risk:
            return "malaria"
        else:
            return "none"

    def get_query_type(self, user_input: str) -> str:
        """Determine if the user input is a greeting, medical query, or casual conversation"""
        input_lower = user_input.lower().strip()
        
        if not input_lower:
            return 'casual'
        
        if self.is_informational_question(user_input):
            return 'informational'  # NEW type
        
        # Exit commands
        if any(exit_cmd in input_lower for exit_cmd in ['quit', 'exit', 'bye', 'goodbye']):
            return 'exit'
        
        # If we're in a medical conversation, treat most inputs as medical
        if self.is_medical_chat and len(self.chat_history) > 0:
            return 'medical'
        
        # Medical keywords
        medical_keywords = [
            r'\bpain\b', r'\bhurt\b', r'\bache\b', r'\bsore\b', r'\binjury\b', r'\bwound\b', 
            r'\bbleed\b', r'\bblood\b', r'\bfever\b', r'\btemperature\b', r'\bcough\b', 
            r'\bsneeze\b', r'\bcold\b', r'\bflu\b', r'\bheadache\b', r'\bstomach\b', 
            r'\bchest\b', r'\barm\b', r'\bleg\b', r'\bback\b', r'\bneck\b', r'\bnausea\b', 
            r'\bvomit\b', r'\bdizzy\b', r'\bdizziness\b', r'\brush\b', r'\bitch\b', 
            r'\bbreath\b', r'\bbreathe\b', r'\bheart\b', r'\bpalpitation\b', r'\bdoctor\b', 
            r'\bhospital\b', r'\bemergency\b', r'\burgent\b', r'\bclinic\b', 
            r'\bappointment\b', r'\bmedicine\b', r'\bpill\b', r'\bsymptom\b', r'\bill\b',
            r'\bsick\b', r'\bunwell\b', r'\bhealth\b', r'\bmedical\b', r'\bchikungunya\b',
            r'\bdengue\b', r'\bjoint pain\b', r'\brash\b', r'\bvomiting\b', r'\bdehydration\b',
            r'\bbleeding\b', r'\bnosebleed\b', r'\bgum bleeding\b', r'\babdominal pain\b',
            r'\bmosquito\b', r'\bbite\b', r'\btravel\b', r'\bvisited\b', r'\bfeeling unwell\b',
            r'\bnot feeling well\b', r'\bnot well\b', r'\bunwell\b'
        ]
        
        # Check for medical keywords with word boundaries
        has_medical_keywords = any(re.search(keyword, input_lower) for keyword in medical_keywords)
        if has_medical_keywords and self.is_informational_question(user_input):
            return 'informational'
        elif has_medical_keywords:
            return 'medical'
        # Casual greetings and small talk
        casual_patterns = [
            r'^hello$', r'^hi$', r'^hey$', r'how are you', r'good morning', r'good afternoon',
            r'good evening', r'what\'s up', r'howdy', r'greetings', r'thank you', r'thanks',
            r'who are you', r'what can you do', r'help me', r'your name', r'about you'
        ]
        
        # Check for exact matches for simple casual queries
        if any(re.search(pattern, input_lower) for pattern in casual_patterns):
            return 'greeting'
        
        return 'casual'
    
    def detect_affirmative_negative(self, user_input: str) -> str:
        """Detect if the user response is affirmative, negative, or unclear"""
        input_lower = user_input.lower().strip()
        
        negative_words = ['no', 'not', 'never', 'none', "don't", "didn't", "haven't", "hasn't", 
                        "wasn't", "isn't", "aren't", "won't", "can't", "couldn't", "wouldn't", 
                        "shouldn't", "nope", "nah", "negative"]
        affirmative_words = ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'alright', 'certainly', 
                            'definitely', 'absolutely', 'of course', 'have', 'did', 'was', 'is', 
                            'are', 'will', 'can', 'could', 'would', 'should', 'i have', 'i did',
                            'i was', 'i am']
        
        # Check for negative words
        negative_count = sum(1 for word in negative_words if re.search(r'\b' + re.escape(word) + r'\b', input_lower))
        
        # Check for affirmative words
        affirmative_count = sum(1 for word in affirmative_words if re.search(r'\b' + re.escape(word) + r'\b', input_lower))
        
        if negative_count > affirmative_count:
            return "negative"
        elif affirmative_count > negative_count:
            return "affirmative"
        else:
            return "unclear"

    def get_casual_response(self, user_input: str) -> str:
        """Handles non-medical conversation."""
        input_lower = user_input.lower().strip()
        
        greetings = [
            "Hello! I'm here to help with any health concerns you might have.",
            "Hi there! How can I assist you with your health today?",
            "Greetings! I'm your medical assistant. What health concerns would you like to discuss?",
            "Hi! I'm here and ready to help. Please describe any symptoms you're experiencing."
        ]

        how_are_you_responses = [
            "I'm here to help you with your health questions! How are you feeling today?",
            "I'm functioning well and ready to assist you. What can I help you with?",
            "As an AI, I don't have feelings, but I'm here to help you feel better. What's on your mind?"
        ]

        who_are_you_responses = [
            "I am a medical diagnostic assistant. My purpose is to help you understand your symptoms and guide you to appropriate care.",
            "I'm a helpful bot designed to provide information about medical symptoms and recommendations.",
            "You can call me your friendly health guide! I'm here to help you understand your symptoms and suggest appropriate care."
        ]

        what_can_you_do_responses = [
            "I can help you understand what might be causing your symptoms and suggest appropriate medical care. Just describe how you're feeling!",
            "My purpose is to offer guidance on symptoms, potential conditions, and recommend appropriate medical professionals.",
            "I can provide information about symptoms, first aid, and recommend specialists. What's bothering you?"
        ]

        thank_you_responses = [
            "You're welcome! Is there anything else I can help you with?",
            "Glad I could help! Feel better soon!",
            "You're very welcome! Remember to take care of yourself.",
            "My pleasure! Let me know if you have other questions."
        ]
        
        if 'how are you' in input_lower:
            return random.choice(how_are_you_responses)
        elif 'who are you' in input_lower or 'your name' in input_lower:
            return random.choice(who_are_you_responses)
        elif 'what can you do' in input_lower or 'help' in input_lower:
            return random.choice(what_can_you_do_responses)
        elif any(thanks in input_lower for thanks in ['thank', 'thanks', 'appreciate']):
            return random.choice(thank_you_responses)
        elif any(greeting in input_lower for greeting in ['hi', 'hello', 'hey', 'greetings', 'howdy']):
            return random.choice(greetings)
        else:
            # Fallback response for other casual input
            return "I'm here to help with medical concerns. Please describe any symptoms or health issues you're experiencing."

    def extract_answered_questions(self, user_input: str) -> Dict:
        """Extract information from user responses to update user_info"""
        user_input_lower = user_input.lower()
        extracted_info = {}
        
        
        response_type = self.detect_affirmative_negative(user_input)
    
    # Check if this is a response to the last asked question
        if self.last_question:
            last_question_lower = self.last_question.lower()
            
            # Handle bleeding-related questions
            if any(term in last_question_lower for term in ['bleeding', 'gums', 'nose', 'blood']):
                if response_type == "affirmative":
                    extracted_info["bleeding_symptoms"] = True
                    extracted_info["gum_bleeding"] = 'gums' in last_question_lower
                    extracted_info["nose_bleeding"] = 'nose' in last_question_lower
                elif response_type == "negative":
                    extracted_info["bleeding_symptoms"] = False
                    extracted_info["gum_bleeding"] = False
                    extracted_info["nose_bleeding"] = False
            
            # Handle rash-related questions
            elif any(term in last_question_lower for term in ['rash', 'skin', 'red spots']):
                if response_type == "affirmative":
                    extracted_info["rash"] = True
                elif response_type == "negative":
                    extracted_info["rash"] = False
            
            # Handle joint-related questions
            elif any(term in last_question_lower for term in ['joint', 'swollen', 'painful to move']):
                if response_type == "affirmative":
                    extracted_info["joint_swelling"] = True
                elif response_type == "negative":
                    extracted_info["joint_swelling"] = False
            
            # Handle travel-related questions
            elif 'travel' in last_question_lower:
                if response_type == "affirmative":
                    extracted_info["travel_mentioned"] = True
                elif response_type == "negative":
                    extracted_info["travel_history"] = "No travel"
                    extracted_info["travel_question_answered"] = True
        
        # Extract duration information
        duration_match = re.search(r'(\d+)\s*(day|days|hour|hours|week|weeks|month|months)', user_input_lower)
        if duration_match:
            extracted_info["fever_duration"] = f"{duration_match.group(1)} {duration_match.group(2)}"
            extracted_info["symptom_duration"] = f"{duration_match.group(1)} {duration_match.group(2)}"
        
        # Extract temperature
        temp_match = re.search(r'(\d{2,3}(?:\.\d{1,2})?)\s*(degrees|°|fahrenheit|f|celcius|c|degree|deg|temp|temperature)?', user_input_lower)
        if temp_match:
            try:
                extracted_info["fever_temperature"] = float(temp_match.group(1))
            except ValueError:
                pass  # Handle invalid temperature format
        
        # Extract pain severity (only if pain is mentioned)
        pain_severity_patterns = {
            "severe": r"\b(10\/10|ten out of ten|severe|extreme|worst|unbearable|excruciating)\b",
            "moderate": r"\b(5\/10|6\/10|7\/10|moderate|medium|manageable)\b", 
            "mild": r"\b(1\/10|2\/10|3\/10|4\/10|mild|slight|minor)\b"
        }
        if any(word in user_input_lower for word in ["pain", "hurt", "ache", "sore"]):
            for severity, pattern in pain_severity_patterns.items():
                if re.search(pattern, user_input_lower):
                    extracted_info["pain_severity"] = severity
                    break
        
        # NEW: Handle travel history responses (both positive and negative)
        travel_areas = self.dengue_prone_areas + self.chikungunya_prone_areas + self.malaria_prone_areas
        
        # Check for positive travel responses
        travel_mentioned = False
        travel_location = None
        
        for area in travel_areas:
            if re.search(r'\b' + re.escape(area.lower()) + r'\b', user_input_lower):
                travel_mentioned = True
                travel_location = area
                break
        
        # Check for explicit travel mentions without specific location
        travel_keywords = ['travel', 'visited', 'been to', 'went to', 'trip to']
        if any(keyword in user_input_lower for keyword in travel_keywords):
            travel_mentioned = True
        
        # Check for negative responses
        negative_keywords = ['no', 'not', "haven't", "hasn't", "didn't", "never", "none", "nope", "nah"]
        positive_keywords = ['yes', 'yeah', 'yep', 'sure', 'have', 'did']
        
        has_negative = any(re.search(r'\b' + re.escape(word) + r'\b', user_input_lower) for word in negative_keywords)
        has_positive = any(re.search(r'\b' + re.escape(word) + r'\b', user_input_lower) for word in positive_keywords)
        
        # Handle travel question responses
        if "travel" in user_input_lower or travel_mentioned:
            if has_negative:
                # User said they haven't traveled
                extracted_info["travel_history"] = "No travel"
            elif has_positive and travel_location:
                # User said they traveled to a specific location
                extracted_info["travel_history"] = travel_location
            elif has_positive:
                # User said they traveled but didn't specify where
                extracted_info["travel_mentioned"] = True
            elif travel_location:
                # User mentioned a location without yes/no
                extracted_info["travel_history"] = travel_location
        
        # Extract pain location
        pain_locations = {
            "eyes": ["eyes", "eye", "behind eyes", "ocular", "retro-orbital"],
            "chest": ["chest", "breast", "sternum"],
            "abdomen": ["abdomen", "stomach", "belly", "tummy"],
            "head": ["head", "headache", "migraine", "cranial"],
            "joint": ["joint", "knee", "elbow", "wrist", "ankle", "arthralgia"],
            "back": ["back", "spine", "spinal"],
            "neck": ["neck", "cervical"]
        }
        for location, keywords in pain_locations.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', user_input_lower):
                    extracted_info["pain_location"] = location
                    break
        
        # Extract age
        age_match = re.search(r'(\d+)\s*(year|years|yr|yrs|yo|year old)', user_input_lower)
        if age_match:
            extracted_info["age"] = age_match.group(1)
        
        # Extract symptom onset
        onset_patterns = [
            r"(started|began|onset)\s+(?:about|approximately|around)?\s*(\d+)\s*(day|days|hour|hours|week|weeks)\s+ago",
            r"(\d+)\s*(day|days|hour|hours|week|weeks)\s+ago"
        ]
        for pattern in onset_patterns:
            onset_match = re.search(pattern, user_input_lower)
            if onset_match:
                # Use the last two groups for the time and unit
                if len(onset_match.groups()) >= 2:
                    extracted_info["symptom_onset"] = f"{onset_match.group(len(onset_match.groups())-1)} {onset_match.group(len(onset_match.groups()))} ago"
                break
        
        # Extract preexisting conditions
        condition_keywords = ["diabetes", "hypertension", "high blood pressure", "heart disease", "asthma", 
                            "allergies", "arthritis", "kidney disease", "liver disease"]
        for condition in condition_keywords:
            if re.search(r'\b' + re.escape(condition) + r'\b', user_input_lower):
                if "preexisting_conditions" not in extracted_info:
                    extracted_info["preexisting_conditions"] = []
                extracted_info["preexisting_conditions"].append(condition)
        
        # Extract current medications
        med_keywords = ["taking", "on", "using", "prescribed", "medication", "medicine", "pill", "tablet"]
        if any(keyword in user_input_lower for keyword in med_keywords):
            med_match = re.search(r"(aspirin|ibuprofen|paracetamol|acetaminophen|antibiotic|antihistamine|insulin|steroid)", user_input_lower)
            if med_match:
                if "current_medications" not in extracted_info:
                    extracted_info["current_medications"] = []
                extracted_info["current_medications"].append(med_match.group(1))
        
        return extracted_info

    def process_medical_query(self, user_input: str) -> Tuple[str, bool]:
        if not self.initialized:
            return "I'm sorry, the medical assistant is not properly initialized. Please try again later.", False
        
        try:
            # Add user input to history
            self.chat_history.append(f"User: {user_input}")
            
             # Store the last question for context tracking
            current_last_question = self.last_question
            self.last_question = None  # Reset for new processing
            
            # Extract symptoms from user input
            new_symptoms = self.extract_symptoms(user_input)
            self.user_symptoms.extend(new_symptoms)
            self.user_symptoms = list(set(self.user_symptoms))  # Remove duplicates
            
            # Extract information from user response
            extracted_info = self.extract_answered_questions(user_input)
            for key, value in extracted_info.items():
                if value:  # Only update if we got a value
                    self.user_info[key] = value
            
            # Check if we have enough information for diagnosis
            has_fever = any(s in self.user_symptoms for s in ["fever", "high fever", "mild fever"])
            has_headache = "headache" in self.user_symptoms or "severe headache" in self.user_symptoms
            has_travel_history = self.user_info["travel_history"] is not None
            has_fever_info = self.user_info["fever_temperature"] is not None and self.user_info["fever_duration"] is not None
            
            # If we have fever + headache + travel to dengue area, we should provide diagnosis
            if (has_fever and has_headache and has_travel_history and 
                self.user_info["travel_history"] in self.dengue_prone_areas and
                has_fever_info):
                
                # Find dengue-related conditions
                dengue_conditions = [cond for cond in self.medical_data 
                                   if "dengue" in cond.get("condition", "").lower()]
                
                if dengue_conditions:
                    # Select the most appropriate dengue condition based on symptoms
                    best_dengue_match = None
                    best_score = 0
                    
                    for condition in dengue_conditions:
                        score = self.calculate_symptom_score(self.user_symptoms, condition["symptoms"])
                        if score > best_score:
                            best_score = score
                            best_dengue_match = condition
                    
                    if best_dengue_match:
                        response = self.generate_final_recommendation(best_dengue_match, best_score)
                        self.chat_history.append(f"Bot: {response}")
                        
                        # Reset for next conversation
                        self.is_medical_chat = False
                        self.asked_questions.clear()
                        self.current_follow_ups = []
                        self.user_symptoms = []
                        self.user_info = {key: None for key in self.user_info}
                        self.diagnostic_stage = "initial"
                        
                        return response, False
            
            # Find best matching conditions
            best_matches = self.find_best_matches(self.user_symptoms)
            
            # Update diagnostic stage
            if len(self.user_symptoms) < 2:
                self.diagnostic_stage = "initial"
            elif best_matches and best_matches[0]["score"] >= 70:
                self.diagnostic_stage = "confirmation"
            else:
                self.diagnostic_stage = "symptom_clarification"
            
            # If we have a high-confidence match, provide final recommendation
            if self.diagnostic_stage == "confirmation" and best_matches:
                best_match = best_matches[0]
                response = self.generate_final_recommendation(best_match["condition"], best_match["score"])
                
                # Add bot response to history
                self.chat_history.append(f"Bot: {response}")
                
                # Reset for next conversation
                self.is_medical_chat = False
                self.asked_questions.clear()
                self.current_follow_ups = []
                self.user_symptoms = []
                self.user_info = {key: None for key in self.user_info}
                self.diagnostic_stage = "initial"
                
                return response, False
            
            # Otherwise, ask diagnostic questions
            diagnostic_questions = self.get_diagnostic_questions(best_matches)
            
            diagnostic_questions = self.get_diagnostic_questions(best_matches)
        
            if diagnostic_questions:
                # Store the question we're about to ask
                self.last_question = diagnostic_questions[0]
                
                # Use empathetic introduction
                empathetic_intro = self.generate_empathetic_response()
                question = empathetic_intro + diagnostic_questions[0]
            
            # If we detected travel but no location, ask for specific location
            if (self.user_info.get("travel_mentioned") and 
                not self.user_info.get("travel_history") and
                self.user_info.get("travel_history") != "No travel"):
                diagnostic_questions.insert(0, "Which specific area did you travel to?")
            
            # If we have fever but no temperature details, ask
            if any(s in self.user_symptoms for s in ["fever", "high fever", "mild fever"]):
                if self.user_info["fever_temperature"] is None:
                    diagnostic_questions.insert(0, "What is your current temperature?")
                if self.user_info["fever_duration"] is None:
                    diagnostic_questions.insert(0, "How long have you had fever?")
            
            if diagnostic_questions:
                # Use empathetic introduction
                empathetic_intro = self.generate_empathetic_response()
                question = empathetic_intro + diagnostic_questions[0]
                
                # Store the question to track what we've asked
                self.asked_questions.add(diagnostic_questions[0])
                self.current_follow_ups = diagnostic_questions[1:]  # Store remaining questions
                
                # Add bot response to history
                self.chat_history.append(f"Bot: {question}")
                
                return question, True
            
            # If we don't have questions but still need more info
            if not diagnostic_questions and len(self.user_symptoms) < 3:
                empathetic_prompts = [
                    "I want to make sure I understand your situation correctly. Could you describe your symptoms in more detail?",
                    "Let me help you get the right care. Please tell me more about what you're experiencing.",
                    "I'm here to help you figure this out. Can you provide more details about your symptoms?",
                    "I understand this might be worrying. Let's work through it together. What other symptoms are you experiencing?"
                ]
                
                response = random.choice(empathetic_prompts)
                self.chat_history.append(f"Bot: {response}")
                return response, True
            
            # If we reach here, we don't have enough information but can't ask more questions
            # Provide gentle guidance to see a doctor
            empathetic_responses = [
                "I understand you're concerned about your symptoms. Based on the information provided, I recommend consulting a healthcare professional for a proper evaluation.",
                "I've done my best to understand your symptoms, but I think it would be best for you to see a doctor for a definitive diagnosis.",
                "Your symptoms deserve proper medical attention. I encourage you to consult with a healthcare provider who can examine you properly."
            ]
            
            response = random.choice(empathetic_responses)
            self.chat_history.append(f"Bot: {response}")
            return response, False
            
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error processing your medical query. Please try again or consult a healthcare professional."
            self.chat_history.append(f"Bot: {error_msg}")
            return error_msg, False

    def run_chat(self):
        """Runs an interactive chat session with the user."""
        print("Medical Diagnostic Assistant Bot")
        print("=" * 60)
        print("Describe your symptoms and I will help identify potential causes.")
        print("I'll ask questions to understand your situation better.")
        print("Type 'quit', 'exit', or 'bye' to end the chat.")
        print("REMINDER: This is for informational purposes only. PLEASE CONSULT A DOCTOR for proper medical advice.")
        print()

        if not self.initialize_bot():
            print("Failed to initialize the chatbot. Please check if the vector database exists and Ollama is running.")
            return

        print("Chatbot initialized successfully. How can I help you today?\n")

        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # TEMPORARY DEBUG
                print(f"=== DEBUG ===")
                print(f"Input: '{user_input}'")
                print(f"Current is_medical_chat: {self.is_medical_chat}")
                print(f"Chat history length: {len(self.chat_history)}")
                
                # Determine query type
                query_type = self.get_query_type(user_input)
                print(f"Query type determined: {query_type}")
                print(f"=== END DEBUG ===\n")
                
                if query_type == 'exit':
                    print("\nBot: Thank you for chatting. Please take care of your health!")
                    break
                
                # Handle different query types
                if query_type == 'medical':
                    self.is_medical_chat = True
                    # Only reset if this is a new medical conversation
                    if not self.chat_history or "Bot:" not in self.chat_history[-1]:
                        self.chat_history = []  # Start fresh medical conversation
                    
                    response, should_continue = self.process_medical_query(user_input)
                    print(f"\nBot: {response}\n")
                    
                elif query_type in ['greeting', 'casual']:
                    response = self.get_casual_response(user_input)
                    print(f"\nBot: {response}\n")
                    
                    # If the casual response invites medical discussion, set flag
                    if any(keyword in response.lower() for keyword in ['symptom', 'health', 'medical']):
                        self.is_medical_chat = True
                        self.chat_history = []
                        
                elif query_type == 'informational':
                    # RESET medical chat state for informational questions
                    self.is_medical_chat = False
                    self.chat_history = []
                    
                    response = self.get_informational_response(user_input)
                    print(f"\nBot: {response}\n")
                
                print("-" * 60 + "\n")

            except KeyboardInterrupt:
                print("\n\nBot: Session ended. Take care!")
                break
            except Exception as e:
                print(f"\nBot: I encountered an unexpected error. Please try again.")
                print(f"Error: {e}")
                # Reset state but keep the bot running
                self.chat_history = []
                self.asked_questions.clear()
                self.current_follow_ups = []
                self.is_medical_chat = False
                self.user_symptoms = []
                self.user_info = {key: None for key in self.user_info}
                self.diagnostic_stage = "initial"
                continue

def main():
    """Main function to run the chatbot."""
    chatbot = MedicalChatBot()
    chatbot.run_chat()

if __name__ == "__main__":
    main()