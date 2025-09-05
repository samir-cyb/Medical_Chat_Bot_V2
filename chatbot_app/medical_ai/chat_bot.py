import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "medical_chatbot.settings")  # replace with your project settings path
django.setup()"""


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
    
    
    
    """def is_informational_question(self, user_input: str) -> bool:
        
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
        
        return False"""
        
    def reset_conversation(self):
        """Reset the conversation state"""
        self.chat_history = []
        self.asked_questions.clear()
        self.current_follow_ups = []
        self.user_symptoms = []
        self.is_medical_chat = False
        self.last_question = None
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
            
            print(f"DEBUG: Loading vector database from: {PERSIST_DIRECTORY}")
            print(f"DEBUG: Database exists: {os.path.exists(PERSIST_DIRECTORY)}")
            
            if os.path.exists(PERSIST_DIRECTORY):
                print("DEBUG: Vector database files found:")
                for file in os.listdir(PERSIST_DIRECTORY):
                    print(f"  - {file}")
            # 2. Load the existing vector database
            self.vectordb = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings,
                collection_name="medical_knowledge"
            )
            
            test_results = self.vectordb.similarity_search("fever", k=1)
            print(f"DEBUG: Database test query returned {len(test_results)} results")

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
        print(f"🔍 extract_symptoms analyzing: '{user_input}'")
        user_input_lower = user_input.lower()
        print(f"DEBUG: Lowercase: '{user_input_lower}'")
        found_symptoms = set()
        user_input = ' '.join(user_input.split())
        
        # First, check for pain severity scale responses (like "6 out of 10")
        pain_scale_match = re.search(r'(\d+)\s*(?:\/|out of|out-of)\s*10', user_input_lower)
        if pain_scale_match:
            pain_level = int(pain_scale_match.group(1))
            #print(f"🔍 [extract_symptoms] Detected pain scale: {pain_level}/10")
            
            # Map pain level to severity description
            if pain_level >= 7:
                found_symptoms.add("severe pain")
                #print(f"✅ [extract_symptoms] Added severe pain (scale: {pain_level}/10)")
            elif pain_level >= 4:
                found_symptoms.add("moderate pain")
                #print(f"✅ [extract_symptoms] Added moderate pain (scale: {pain_level}/10)")
            else:
                found_symptoms.add("mild pain")
                #print(f"✅ [extract_symptoms] Added mild pain (scale: {pain_level}/10)")
    
        
        # Comprehensive variation mapping for all diseases
        variation_mapping = {
            # Dengue-related variations
            "high fever": "high fever",
            "very high fever": "high fever", 
            "spiking fever": "high fever",
            "sudden fever": "high fever",
            "pain behind my eyes": "pain behind eyes",
            "eye pain": "pain behind eyes",
            "eyes hurting": "pain behind eyes",
            "behind eyes": "pain behind eyes",
            "retro orbital pain": "pain behind eyes",
            "really bad headache": "severe headache",
            "worst headache": "severe headache",
            "excruciating headache": "severe headache",
            "debilitating headache": "severe headache",
            "muscle aches": "severe muscle pain",
            "body hurting": "severe muscle pain",
            "whole body pain": "severe muscle pain",
            "throwing up": "vomiting",
            "puking": "vomiting",
            "feeling sick": "nausea",
            "queasy": "nausea",
            
            # Chikungunya-related variations
            "joint aches": "debilitating joint pain",
            "joints hurting": "debilitating joint pain",
            "can't move joints": "debilitating joint pain",
            "swollen joints": "joint swelling",
            "puffy joints": "joint swelling",
            "inflamed joints": "joint swelling",
            "red eyes": "conjunctival injection",
            "bloodshot eyes": "conjunctival injection",
            "eye redness": "conjunctival injection",
            "bumpy rash": "maculopapular rash",
            "raised rash": "maculopapular rash",
            
            # Appendicitis-related variations
            "stomach pain": "abdominal pain",
            "belly pain": "abdominal pain",
            "tummy ache": "abdominal pain",
            "right side pain": "sharp pain lower right abdomen",
            "lower right pain": "sharp pain lower right abdomen",
            "rlq pain": "right lower quadrant pain",
            "rebound pain": "rebound tenderness",
            "tender abdomen": "rebound tenderness",
            
            # Heart-related variations
            "chest discomfort": "chest pain",
            "chest tightness": "chest pain",
            "arm pain": "pain radiating to arm or jaw",
            "jaw pain": "pain radiating to arm or jaw",
            "back pain": "pain radiating to arm or jaw",
            "short of breath": "shortness of breath",
            "breathing problems": "shortness of breath",
            "can't breathe": "shortness of breath",
            "heart racing": "heart palpitations",
            "pounding heart": "heart palpitations",
            
            # General symptom variations
            "fever": "high fever",  # Map general fever to high fever for better matching
            "temperature": "fever",
            "hot": "fever",
            "burning up": "fever",
            "head hurting": "headache",
            "head ache": "headache",
            "coughing": "cough",
            "hacking": "cough",
            "runny nose": "runny nose",
            "nose running": "runny nose",
            "sore throat": "sore throat",
            "throat pain": "sore throat",
            "body pains": "body aches",
            "aching body": "body aches",
            "skin rash": "rash",
            "red spots": "rash",
            "skin irritation": "rash",
            "feeling sick": "nausea",
            "upset stomach": "nausea",
            "vomiting": "vomiting",
            "throwing up": "vomiting",
            "stomach pain": "abdominal pain",
            "belly ache": "abdominal pain",
            "bleeding gums": "bleeding gums",
            "gums bleeding": "bleeding gums",
            "nose bleed": "nosebleeds",
            "nose bleeding": "nosebleeds",
            "tired": "fatigue",
            "exhausted": "fatigue",
            "no energy": "lethargy",
            "confused": "confusion",
            "disoriented": "confusion",
            "stiff neck": "stiff neck",
            "neck stiffness": "stiff neck",
            "light hurts eyes": "light sensitivity",
            "sensitive to light": "light sensitivity",
            
            # Severity variations
            "mild fever": "mild fever",
            "low grade fever": "mild fever",
            "slight fever": "mild fever",
            "bad headache": "severe headache",
            "terrible headache": "severe headache",
            "unbearable headache": "severe headache",
            "constant cough": "persistent cough",
            "nonstop coughing": "persistent cough",
            "severe stomach pain": "severe abdominal pain",
            "extreme abdominal pain": "severe abdominal pain",
            "can't breathe properly": "difficulty breathing",
            "struggling to breathe": "difficulty breathing",
            
            "dizziness": ["dizzy", "light.headed", "vertigo", "spinning"],
            "weakness": ["weak", "fatigue", "exhaust", "no energy", "can't move"],
            "body pain": ["body pain", "whole body hurt", "muscle ache", "aching"],
            "sore throat": ["sore throat", "throat hurt", "throat pain", "scratchy throat"],
            "fever": ["fever", "temperature", "hot", "burning", "warm"],
            "runny nose": ["runny nose", "nose run", "nasal", "stuffy nose"]
        }
        
        # Check for variations first (most specific to least specific)
        for variation, normalized in variation_mapping.items():
            if isinstance(normalized, list):
                # If it's a list, use the first item as the normalized symptom
                # and check all variations in the list
                for var in normalized:
                    if re.search(r'\b' + re.escape(var) + r'\b', user_input_lower):
                        found_symptoms.add(normalized[0])  # Use first item as normalized
                        break
            else:
                # If it's a string, process normally
                if re.search(r'\b' + re.escape(variation) + r'\b', user_input_lower):
                    found_symptoms.add(normalized)
    
        
        # Create a reverse mapping for easy lookup
        reverse_mapping = {}
        for normalized, variations in self.symptom_mapping.items():
            for variation in variations:
                reverse_mapping[variation] = normalized
        
        # Check for each possible symptom variation
        for symptom_variation, normalized_symptom in reverse_mapping.items():
            if re.search(r'\b' + re.escape(symptom_variation) + r'\b', user_input_lower):
                found_symptoms.add(normalized_symptom)
                #print(f"✅ [extract_symptoms] Mapped '{symptom_variation}' -> '{normalized_symptom}'")
        
        # Also check for direct matches with known symptoms
        for symptom in self.symptom_weights.keys():
            if re.search(r'\b' + re.escape(symptom) + r'\b', user_input_lower):
                found_symptoms.add(symptom)
                #print(f"✅ [extract_symptoms] Direct match: '{symptom}'")

        # Extract fever temperature if mentioned
        temp_match = re.search(r'(\d{2,3}(?:\.\d{1,2})?)\s*(degrees|°|fahrenheit|f|celcius|c|degree|deg|temp|temperature)?', user_input_lower)
        if temp_match:
            try:
                temp = float(temp_match.group(1))
                self.user_info["fever_temperature"] = temp
                
                # Remove any fever symptoms from current extraction (to prevent duplicates)
                for fever_symptom in ["fever", "high fever", "mild fever"]:
                    if fever_symptom in found_symptoms:
                        found_symptoms.remove(fever_symptom)
                        
                # Add appropriate fever symptom based on temperature
                if temp >= 102:
                    found_symptoms.add("high fever")
                elif temp >= 100.4 and temp < 102:
                    found_symptoms.add("fever")
                else:
                    found_symptoms.add("mild fever")
                    
                print(f"✅ [extract_symptoms] Temperature-based fever symptom: {list(found_symptoms)}")
                
                # NOW remove the old fever symptoms from user_symptoms since we have temperature
                # This prevents duplicates but only AFTER we have the temperature info
                for fever_symptom in ["fever", "high fever", "mild fever"]:
                    if fever_symptom in self.user_symptoms:
                        self.user_symptoms.remove(fever_symptom)
                        
            except ValueError:
                pass
        
        # comment out for finding better contitions
        """if fever_match:
            try:
                temp = float(fever_match.group(1))
                self.user_info["fever_temperature"] = temp
                print(f"✅ [extract_symptoms] Extracted temperature: {temp}°F")
                if temp >= 102:
                    found_symptoms.add("high fever")
                elif temp >= 100.4:  # Standard fever threshold
                    found_symptoms.add("fever")
                else:
                    found_symptoms.add("mild fever")
            except ValueError:
                pass  # Handle invalid temperature format"""
        
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
            "eyes": ["eyes", "eye", "behind eyes", "ocular", "retro-orbital","behind my eyes"],
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
            
            
            # comment out for dublicate function
        
        if "high fever" in found_symptoms and "fever" in found_symptoms:
            found_symptoms.remove("fever")
            
        print(f"✅ Found symptoms: {list(found_symptoms)}")
        return list(found_symptoms)

    def calculate_symptom_score(self, user_symptoms: List[str], condition_symptoms: List[str]) -> float:
        """Calculate a weighted match score between user symptoms and condition symptoms"""
        
        # 🔥 NEW CODE: Remove duplicate fever symptoms
        user_symptoms = user_symptoms.copy()  # Don't modify original list
        if "high fever" in user_symptoms and "fever" in user_symptoms:
            user_symptoms.remove("fever")
            print("⚠️ [calculate_symptom_score] Removed 'fever' because 'high fever' is present")
        if "mild fever" in user_symptoms and "fever" in user_symptoms:
            user_symptoms.remove("fever") 
            print("⚠️ [calculate_symptom_score] Removed 'fever' because 'mild fever' is present")
    
        
        print(f"🔍 [calculate_symptom_score] User symptoms: {user_symptoms}")
        print(f"🔍 [calculate_symptom_score] Condition symptoms: {condition_symptoms}\n\n")
        
        # Normalize all symptoms by stripping whitespace
        user_symptoms = [s.strip().lower() for s in user_symptoms]
        condition_symptoms = [s.strip().lower() for s in condition_symptoms]
        
        if not user_symptoms or not condition_symptoms:
            print("⚠️ [calculate_symptom_score] Empty symptoms list, returning 0")
            return 0
        
        
        
        # 1. Count how many of USER's symptoms match the condition
        matches_count = 0
        matched_symptoms = []
        
        for user_symptom in user_symptoms:
            if user_symptom in condition_symptoms:
                matches_count += 1
                matched_symptoms.append(user_symptom)
                print(f"✅ [calculate_symptom_score] Matched: {user_symptom}")
        
        # 2. Base score: Percentage of user symptoms matched (max 50%)
        base_score = (matches_count / len(user_symptoms)) * 50 if user_symptoms else 0
        
        # 3. Specificity bonus: Extra points for rare symptoms (max 50%)
        specificity_bonus = 0
        for symptom in matched_symptoms:
            weight = self.symptom_weights.get(symptom, 5)
            if weight >= 6:  # Rare/specific symptom
                specificity_bonus += 10
                print(f"⭐ [calculate_symptom_score] Rare symptom bonus: {symptom} (+10)")
        
        # Cap specificity bonus at 50%
        specificity_bonus = min(specificity_bonus, 50)
        # 4. Calculate initial score (before any adjustments)
        initial_score = base_score + specificity_bonus
        
        
        # Add this after specificity_bonus calculation:

        # 4. Uniqueness bonus: Extra points for symptoms that only match this condition (max 20%)
        uniqueness_bonus = 0
        for symptom in matched_symptoms:
            # Check how many other conditions have this symptom
            symptom_count = 0
            for condition in self.medical_data:
                if symptom in condition["symptoms"]:
                    symptom_count += 1
            
            # If this symptom only appears in 1-2 conditions, give bonus
            if symptom_count <= 2:
                uniqueness_bonus += 2
                print(f"🎯 [calculate_symptom_score] Unique symptom bonus: {symptom} (+2)")

        # Cap uniqueness bonus at 10%
        uniqueness_bonus = min(uniqueness_bonus, 10)

        # Add to final score
        initial_score = base_score + specificity_bonus + uniqueness_bonus
                
        
        
        # 5. Apply condition-specific adjustments FIRST
        condition_name = str(condition_symptoms).lower()
        # For cardiac conditions
        if any(term in condition_name for term in ["heart", "cardiac", "attack", "myocardial"]):
            cardiac_specific = ["pain radiating to arm or jaw", "heart palpitations", 
                            "sweating", "nausea with chest pain", "shortness of breath with chest pain"]
            has_cardiac_specific = any(s in user_symptoms for s in cardiac_specific)
            if not has_cardiac_specific:
                initial_score *= 0.8
                print(f"❤️ [calculate_symptom_score] Cardiac condition penalty: No specific cardiac symptoms. Score reduced to {initial_score:.1f}")
        
        # For gastric conditions
        if any(term in condition_name for term in ["gastric", "acid", "reflux", "gerd", "indigestion"]):
            gastric_symptoms = ["heartburn", "acid reflux", "bloating", "indigestion", "belching"]
            has_gastric_symptoms = any(s in user_symptoms for s in gastric_symptoms)
            if has_gastric_symptoms:
                initial_score *= 1.1
                print(f"🍽️ [calculate_symptom_score] Gastric condition bonus: Gastric symptoms present. Score increased to {initial_score:.1f}")
        
        
        
        
        # 6. Add travel history bonus AFTER condition adjustments
        travel_bonus = 0
        if self.user_info.get("travel_history") and self.user_info["travel_history"] != "No travel":
            condition_str = str(condition_symptoms).lower()
            
            # Check if this looks like DENGUE (specific symptom combination)
            if self.user_info["travel_history"] in self.dengue_prone_areas:
                # DENGUE requires fever + at least ONE specific dengue symptom
                dengue_specific = ["bleeding gums", "nosebleeds", "petechiae", "severe abdominal pain", "pain behind eyes"]
                has_dengue_specific = any(symptom in condition_str for symptom in dengue_specific)
                has_fever = any(symptom in condition_str for symptom in ["high fever", "fever", "persistent high fever"])
                
                if has_fever and has_dengue_specific:  # BOTH required for dengue
                    travel_bonus = 20
                    print(f"✈️ Travel bonus for DENGUE (fever + specific symptom): +20")
            
            # Check if this looks like CHIKUNGUNYA
            if self.user_info["travel_history"] in self.chikungunya_prone_areas:
                # CHIKUNGUNYA requires fever + joint symptoms
                has_joint_symptoms = any(symptom in condition_str for symptom in ["joint pain", "joint swelling", "severe joint pain"])
                has_fever = any(symptom in condition_str for symptom in ["high fever", "fever"])
                
                if has_fever and has_joint_symptoms:  # BOTH required for chikungunya
                    travel_bonus = 20
                    print(f"✈️ Travel bonus for CHIKUNGUNYA (fever + joint pain): +20")
            
            # Check if this looks like MALARIA
            if self.user_info["travel_history"] in self.malaria_prone_areas:
                # MALARIA requires fever + chills/cyclic pattern
                has_malaria_pattern = any(symptom in condition_str for symptom in ["chills", "sweating", "cyclic fever", "periodic fever"])
                has_fever = any(symptom in condition_str for symptom in ["high fever", "fever"])
                
                if has_fever and has_malaria_pattern:  # BOTH required for malaria
                    travel_bonus = 20
                    print(f"✈️ Travel bonus for MALARIA (fever + chills): +20")
        
        # 6. Calculate final score
        final_score = initial_score + travel_bonus
        final_score = min(final_score, 100)
        
        print(f"📊 [calculate_symptom_score] Final score: {final_score:.1f}% (Base: {base_score:.1f}, Specificity: +{specificity_bonus}, Travel: +{travel_bonus})")
        
        return final_score
    
    
    def check_clear_diagnosis(self) -> Optional[Dict]:
        """Check if there's a clear diagnosis based on key symptoms and travel history"""
        # Only consider clear diagnosis if we have multiple specific symptoms
        if len(self.user_symptoms) < 3:
            return None
        
        has_high_fever = any(s in self.user_symptoms for s in ["high fever", "fever"])
        has_travel_to_dengue_area = self.user_info.get("travel_history") in self.dengue_prone_areas
        has_travel_to_chikungunya_area = self.user_info.get("travel_history") in self.chikungunya_prone_areas
        
        # For dengue: require fever + travel + at least ONE dengue-specific symptom
        if has_high_fever and has_travel_to_dengue_area:
            dengue_specific_symptoms = ["rash", "bleeding gums", "nosebleeds", "severe headache", "pain behind eyes"]
            has_dengue_specific = any(symptom in self.user_symptoms for symptom in dengue_specific_symptoms)
            
            if has_dengue_specific:
                for condition in self.medical_data:
                    if "dengue" in condition.get("condition", "").lower():
                        # Calculate ACTUAL score, don't hardcode!
                        actual_score = self.calculate_symptom_score(self.user_symptoms, condition["symptoms"])
                        return {
                            "condition": condition,
                            "confidence": actual_score,  # ← USE REAL SCORE
                            "reason": "Fever, travel to dengue area, and specific dengue symptoms"
                        }
        
        # For chikungunya: require fever + travel + joint symptoms
        if has_high_fever and has_travel_to_chikungunya_area:
            chikungunya_specific_symptoms = ["joint pain", "joint swelling", "severe joint pain"]
            has_chikungunya_specific = any(symptom in self.user_symptoms for symptom in chikungunya_specific_symptoms)
            
            if has_chikungunya_specific:
                for condition in self.medical_data:
                    if "chikungunya" in condition.get("condition", "").lower():
                        actual_score = self.calculate_symptom_score(self.user_symptoms, condition["symptoms"])
                        return {
                            "condition": condition,
                            "confidence": actual_score,  # ← USE REAL SCORE
                            "reason": "Fever, travel to chikungunya area, and joint symptoms"
                        }
        
        return None


    def get_matched_symptoms_for_display(self, user_symptoms: List[str], condition_symptoms: List[str]) -> List[str]:
        """Get matched symptoms for display using the same logic as scoring"""
        
        user_symptoms = user_symptoms.copy()
        if "high fever" in user_symptoms and "fever" in user_symptoms:
            user_symptoms.remove("fever")
        if "mild fever" in user_symptoms and "fever" in user_symptoms:
            user_symptoms.remove("fever")
        # ▲▲▲ ADD THIS DUPLICATE REMOVAL ▲▲▲
        
        matched = []
        user_symptoms_normalized = [s.strip().lower() for s in user_symptoms]
        condition_symptoms_normalized = [s.strip().lower() for s in condition_symptoms]
        
        # Enhanced symptom mapping for better matching (same as in calculate_symptom_score)
        symptom_equivalents = {
            "abdominal pain": ["severe abdominal pain", "stomach pain", "belly pain", "tummy ache"],
            "chest pain": ["chest discomfort", "chest tightness", "pain in chest"],
            "headache": ["severe headache", "mild headache", "head pain", "throbbing head"],
            "joint pain": ["severe joint pain", "debilitating joint pain", "joints hurting"],
            "fever": ["high fever", "mild fever", "temperature", "hot", "burning up"],
            "high fever": ["very high fever", "spiking fever", "high fever with sudden onset"],
            "mild fever": ["slight fever", "low grade fever"],
            "shortness of breath": ["difficulty breathing", "breathing difficulty", "can't catch breath"],
            "nausea": ["feeling sick", "queasy", "upset stomach"],
            "vomiting": ["throwing up", "puking"],
            "rash": ["skin rash", "red spots", "widespread rash", "maculopapular rash"],
            "fatigue": ["extreme fatigue", "tiredness", "lethargy"],
            "pain behind eyes": ["eye pain", "eye socket pain", "retro-orbital pain"]
        }
        
        for user_symptom in user_symptoms_normalized:
            # Direct match
            if user_symptom in condition_symptoms_normalized:
                matched.append(user_symptom)
            else:
                # Check equivalent symptoms
                equivalents = symptom_equivalents.get(user_symptom, [])
                for equivalent in equivalents:
                    if equivalent in condition_symptoms_normalized:
                        matched.append(user_symptom)
                        break
        
        return matched



    def find_best_matches(self, user_symptoms: List[str], top_n: int = 3) -> List[Dict]:
        """Find the top conditions that match the user's symptoms"""
        print(f"🔍 [find_best_matches] Finding matches for symptoms: {user_symptoms}")
        #print(f"🔍 [find_best_matches] User info: {self.user_info}")
        matches = []
        for i, condition in enumerate(self.medical_data):
            condition_symptoms = condition["symptoms"]
            condition_name = condition.get("condition", "Unknown")
            score = self.calculate_symptom_score(user_symptoms, condition_symptoms)
            condition_name_lower = condition_name.lower()

           # For emergency conditions, require higher threshold but don't penalize a strong single symptom
            urgency = condition.get("urgency", "").lower()
            if urgency in ["emergency", "dangerous", "urgent"] and score > 0:
                specific_symptoms = [s for s in condition_symptoms if self.symptom_weights.get(s, 5) >= 7]
                matched_specific = [s for s in user_symptoms if s in specific_symptoms]
                # Add these debug prints in your penalty logic:
                print(f"DEBUG: Condition symptoms: {condition_symptoms}")
                print(f"DEBUG: Specific symptoms (weight≥7): {specific_symptoms}")
                print(f"DEBUG: User symptoms: {user_symptoms}")
                print(f"DEBUG: Matched specific symptoms: {matched_specific}")
                print(f"DEBUG: Expected 2 specific symptoms, found: {len(matched_specific)}")
                
                # Only apply a penalty if the user has multiple symptoms but is missing the key ones for this emergency
                # If the user only has one symptom (like 'high fever'), it might be the start of the condition. Don't penalize it heavily.
                """if len(matched_specific) < 2 and len(user_symptoms) > 1:
                    penalty_factor = 0.95  # Reduced penalty: only 5% reduction
                    original_score = score
                    score *= penalty_factor
                    print(f"⚠️ [find_best_matches] Emergency condition '{condition_name}' penalty...")"""
                # If the user has only one symptom and it's a strong match, don't penalize it.
                # The bot will just ask more questions to confirm.
    
            if score > 20:  # Only consider matches with at least 20% score
                match_data = {
                    "condition": condition,
                    "score": score,
                    "matched_symptoms": self.get_matched_symptoms_for_display(user_symptoms, condition_symptoms),
                    "condition_name": condition_name
                }
                matches.append(match_data)
                print(f"✅ [find_best_matches] Added match: {condition_name} ({score:.1f}%)")
            #else:
                #print(f"❌ [find_best_matches] Skipped: {condition_name} (score: {score:.1f}% < 20%)")
        
        # Sort by score descending
        matches.sort(key=lambda x: x["score"], reverse=True)
        
        print(f"\n📊 [find_best_matches] Found {len(matches)} potential matches:")
        for i, match in enumerate(matches[:top_n]):
            print(f"   {i+1}. {match['condition_name']} - {match['score']:.1f}%")
            print(f"      Matched symptoms: {match['matched_symptoms']}")
        
        return matches[:top_n]

    
    def get_diagnostic_questions(self, suspected_conditions: List[Dict]) -> List[str]:
        """Generate specific diagnostic questions based on suspected conditions"""
        print(f"=== DEBUG: get_diagnostic_questions() called ===")
        print(f"User symptoms: {self.user_symptoms}")
        print(f"Suspected conditions: {[cond['condition_name'] for cond in suspected_conditions]}")
        
        questions = []
        asked_questions_set = set(self.asked_questions)
        
        
        if suspected_conditions and suspected_conditions[0]["condition_name"].lower() == "appendicitis":
            # Always ask these critical questions for appendicitis
            appendicitis_questions = [
                "Do you have nausea or vomiting?",
                "Do you have a fever?",
                "Does the pain get worse when you move or cough?",
                "Have you lost your appetite?"
            ]
            
            # Filter out already asked questions
            new_questions = [q for q in appendicitis_questions if q not in asked_questions_set]
            
            if new_questions:
                return new_questions[:2]  # Return first 2 unasked questions
        
        # STAGE 1: Basic symptom questions FIRST (always prioritize these)
        if any(s in self.user_symptoms for s in ["fever", "high fever"]):
            if not self.user_info["fever_duration"] and "How long have you had fever?" not in asked_questions_set:
                questions.append("How long have you had fever?")
            
            if not self.user_info["fever_temperature"] and "What is your current temperature?" not in asked_questions_set:
                questions.append("What is your current temperature?")
        
        # Check if we have basic fever info to proceed to specific questions
        has_basic_fever_info = (self.user_info.get("fever_duration") is not None and 
                            self.user_info.get("fever_temperature") is not None)
        
        # STAGE 2: Disease differentiation questions (ONLY after basic info)
        if has_basic_fever_info and suspected_conditions:
            top_conditions = suspected_conditions[:3]
            condition_names = [cond["condition_name"].lower() for cond in top_conditions]
            print(f"DEBUG: Top conditions: {condition_names}")
            
            # Check if we have fever-related diseases
            fever_diseases = ["dengue", "chikungunya", "influenza", "flu", "common cold", "cold"]
            has_fever_diseases = any(any(disease in name for disease in fever_diseases) for name in condition_names)
            
            if has_fever_diseases and len(top_conditions) >= 2:
                # Ask about RASH first (key differentiator for all fever diseases)
                if "Do you have any skin rash or red spots?" not in asked_questions_set:
                    questions.append("Do you have any skin rash or red spots?")
                
                # If we have rash info or multiple conditions, ask more specific
                if "rash" in self.user_symptoms or len(top_conditions) >= 2:
                    differentiating_questions = []
                    
                    # DENGUE-specific questions (ask these first if suspected)
                    if any("dengue" in name for name in condition_names):
                        dengue_questions = [
                            "Have you noticed bleeding from gums?",
                            "Have you experienced nosebleeds?",
                            "Do you have severe headache or pain behind your eyes?",
                            "Do you have severe abdominal pain?"
                        ]
                        differentiating_questions.extend(dengue_questions)
                    
                    # CHIKUNGUNYA-specific questions
                    if any("chikungunya" in name for name in condition_names):
                        chikungunya_questions = [
                            "Do you have severe joint pain or swelling?",
                            "Are your joints painful to move?",
                            "Do you have muscle pain?"
                        ]
                        differentiating_questions.extend(chikungunya_questions)
                    
                    # INFLUENZA-specific questions
                    if any("influenza" in name for name in condition_names) or any("flu" in name for name in condition_names):
                        influenza_questions = [
                            "Do you have cough or respiratory symptoms?",
                            "Do you have runny nose or sore throat?",
                            "Do you have body aches or fatigue?",
                            "Is your cough persistent or severe?",
                            "Are you experiencing extreme fatigue or weakness?",
                            "Do you have difficulty breathing or chest pain?",
                            "Are your body aches severe or debilitating?"
                            
                        ]
                        differentiating_questions.extend(influenza_questions)
                        
                    # Add this section with the other disease-specific questions:
                    if any("common cold" in name for name in condition_names) or any("cold" in name for name in condition_names):
                        common_cold_questions = [
                            "Do you have runny nose or nasal congestion?",
                            "Do you have sore throat?",
                            "Do you have sneezing or watery eyes?",
                            "Is your cough productive (with phlegm)?"
                        ]
                        differentiating_questions.extend(common_cold_questions)
                    
                    # Add unique questions only (limit to 2 per turn)
                    for question in differentiating_questions:
                        if (question not in asked_questions_set and 
                            question not in questions and 
                            len(questions) < 3):  # Increased limit to 3
                            questions.append(question)
        
        # STAGE 3: Travel history questions (ONLY after we have specific symptoms)
        has_fever = any(s in self.user_symptoms for s in ["fever", "high fever"])
        travel_unknown = (not self.user_info.get("travel_history") and 
                        self.user_info.get("travel_history") != "No travel")
        
        # Only ask travel if we have specific symptoms beyond just fever
        has_specific_symptoms = len(self.user_symptoms) > 1  # More than just fever
        
        if (has_fever and travel_unknown and has_basic_fever_info and has_specific_symptoms and
            "Have you traveled anywhere recently?" not in asked_questions_set and
            "Have you traveled anywhere recently?" not in questions):
            
            # Check if we have travel-relevant diseases
            travel_relevant = False
            if suspected_conditions:
                for condition in suspected_conditions[:2]:
                    condition_name = condition["condition_name"].lower()
                    if any(disease in condition_name for disease in ["dengue", "chikungunya", "malaria"]):
                        travel_relevant = True
                        break
            
            if travel_relevant:
                questions.append("Have you traveled anywhere recently?")
        
        # STAGE 4: Follow-up questions based on travel response
        if (self.user_info.get("travel_mentioned") and 
            not self.user_info.get("travel_history") and
            self.user_info.get("travel_history") != "No travel" and
            "Which specific area did you travel to?" not in asked_questions_set and
            len(questions) < 3):
            questions.append("Which specific area did you travel to?")
        
        # STAGE 5: Current location question if no travel
        if (self.user_info.get("travel_history") == "No travel" and
            "What is your current location?" not in asked_questions_set and
            "What is your current location?" not in questions and
            len(questions) < 3):
            questions.append("What is your current location?")
        
        # STAGE 6: General symptom questions (only if we have space)
        if len(questions) < 2:  # Only add if we have room
            if any(s in self.user_symptoms for s in ["pain", "abdominal pain", "chest pain"]):
                if not self.user_info["pain_location"] and "Can you tell me exactly where the pain is located?" not in asked_questions_set:
                    questions.append("Can you tell me exactly where the pain is located?")
                
                if not self.user_info["pain_severity"] and "On a scale of 1-10, how severe is the pain?" not in asked_questions_set and len(questions) < 3:
                    questions.append("On a scale of 1-10, how severe is the pain?")
                    
            if any(s in self.user_symptoms for s in ["rash", "bleeding gums", "nosebleeds"]):
                if "When did you first notice these symptoms?" not in asked_questions_set and len(questions) < 3:
                    questions.append("When did you first notice these symptoms?")
        
        # Remove duplicates and prioritize emergency questions
        questions = list(set(questions))
        priority_questions = []
        secondary_questions = []
        
        for q in questions:
            if any(keyword in q.lower() for keyword in ["bleeding", "emergency", "shortness of breath", "severe", "radiate"]):
                priority_questions.append(q)
            else:
                secondary_questions.append(q)
        
        # Combine with priority questions first
        final_questions = priority_questions + secondary_questions
        final_questions = final_questions[:3]  # Limit to 3 questions
        
        print(f"DEBUG: Final questions: {final_questions}")
        return final_questions
    
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
        #print(f"DEBUG: Input: '{user_input}'")
        #print(f"DEBUG: Lower: '{input_lower}'")
        
        # Check informational first
        is_info = self.is_informational_question(user_input)
        #print(f"DEBUG: Is informational: {is_info}")
        
        if is_info:
            #print("DEBUG: Should return 'informational'")
            return
        
        # Check other types
        #print(f"DEBUG: is_medical_chat: {self.is_medical_chat}")
        #print(f"DEBUG: chat_history length: {len(self.chat_history)}")
        
        # Check medical keywords
        medical_keywords = ['fever', 'pain', 'symptom']  # Add your actual keywords
        has_medical = any(keyword in input_lower for keyword in medical_keywords)
        #print(f"DEBUG: Has medical keywords: {has_medical}")

    
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

        # 1. Check for exit commands FIRST (highest priority)
        exit_commands = ['quit', 'exit', 'bye', 'goodbye', 'stop', 'end']
        if any(exit_cmd in input_lower for exit_cmd in exit_commands):
            return 'exit'
        
        # 2. Check if this is likely an answer to a medical question (CONTEXT-AWARE)
        if self.is_medical_chat and self.last_question:
            # Check for numeric answers to scale questions
            last_question_lower = self.last_question.lower()
            if any(term in last_question_lower for term in ['scale', '1-10', 'rating', 'how severe', 'scale of']):
                if (input_lower.isdigit() or 
                    any(word in input_lower for word in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'])):
                    return 'medical'
            
            # Check for duration/time answers
            if any(term in last_question_lower for term in ['how long', 'duration', 'when did', 'how many days', 'how many hours']):
                if (any(unit in input_lower for unit in ['day', 'days', 'hour', 'hours', 'week', 'weeks', 'month', 'months']) or
                    input_lower.isdigit()):
                    return 'medical'
            
            # Check for yes/no answers
            if any(term in last_question_lower for term in ['are you', 'do you', 'have you', 'is there', 'does it', 'can you']):
                if any(word in input_lower for word in ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'no', 'nope', 'not', 'never', 'nah']):
                    return 'medical'
            
            # Check for temperature answers
            if any(term in last_question_lower for term in ['temperature', 'temp', 'fever', 'degree', 'how high']):
                if (any(char in input_lower for char in ['°', 'deg']) or 
                    input_lower.isdigit() or 
                    any(word in input_lower for word in ['high', 'low', 'normal'])):
                    return 'medical'
        
        # 3. Check for informational questions
        is_info = self.is_informational_question(user_input)
        
        if is_info:
            return 'informational'
        
        # 4. If we're already in a medical conversation, continue it
        if self.is_medical_chat and len(self.chat_history) > 0:
            return 'medical'
        
        # 5. Check for medical keywords (but exclude informational patterns)
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
        
        if has_medical_keyword and not is_info:
            return 'medical'
        
        # 6. Casual greetings and small talk
        casual_patterns = [
            r'^hello$', r'^hi$', r'^hey$', r'how are you', r'good morning', r'good afternoon',
            r'good evening', r'what\'s up', r'howdy', r'greetings', r'thank you', r'thanks',
            r'who are you', r'what can you do', r'help me', r'your name', r'about you'
        ]
        
        if any(re.search(pattern, input_lower) for pattern in casual_patterns):
            return 'greeting'
        
        return 'casual'
    
    def detect_affirmative_negative(self, user_input: str) -> str:
        """Detect if the user response is affirmative, negative, or unclear"""
        input_lower = user_input.lower().strip()
        #print(f"DEBUG detect_affirmative_negative: Input='{input_lower}'")
        
        # Handle simple yes/no responses first
        if input_lower == 'yes':
            print("DEBUG: Simple 'yes' detected")
            return "affirmative"
        elif input_lower == 'no':
            print("DEBUG: Simple 'no' detected")
            return "negative"
        
        # Then handle more complex cases with word counting
        negative_words = ['no', 'not', 'never', 'none', "don't", "didn't", "haven't", "hasn't", 
                        "wasn't", "isn't", "aren't", "won't", "can't", "couldn't", "wouldn't", 
                        "shouldn't", "nope", "nah", "negative"]
        
        affirmative_words = ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'alright', 'certainly', 
                            'definitely', 'absolutely', 'of course', 'have', 'did', 'was', 'is', 
                            'are', 'will', 'can', 'could', 'would', 'should', 'i have', 'i did',
                            'i was', 'i am']
        
        negative_count = sum(1 for word in negative_words if re.search(r'\b' + re.escape(word) + r'\b', input_lower))
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
                # Friendly & approachable
                "Hello! I'm here to help with any health concerns you might have.",
                "Hi there! How can I assist you with your health today?",
                "Hi! I'm here and ready to help. Please describe any symptoms you're experiencing.",
                "Hey! I'm here to support you. What’s on your mind regarding your health?",

                # Professional & clear
                "Greetings! I'm your medical assistant. What health concerns would you like to discuss?",
                "Welcome! I’m here to help you understand your symptoms and guide you toward the right care.",
                "Good day! I can provide information about your symptoms and suggest next steps. What’s troubling you?",

                # Warm & reassuring
                "Hi! I’m glad you reached out. Tell me what symptoms you’ve been experiencing, and we’ll go from there.",
                "Hello! Your health matters. Share your concerns with me, and I’ll do my best to guide you.",
                "Hi there! Don’t worry, you’re not alone—I’m here to help you make sense of your symptoms."
            ]


        how_are_you_responses = [
            "I'm here to help you with your health questions! How are you feeling today?",
            "I'm functioning well and ready to assist you. What can I help you with?",
            "As an AI, I don't have feelings, but I'm here to help you feel better. What's on your mind?"
        ]

        who_are_you_responses = [
                    # Professional & clear
                    "I am a medical diagnostic assistant. My purpose is to help you understand your symptoms and guide you to appropriate care.",
                    "I'm a helpful bot designed to provide information about medical symptoms and recommendations.",
                    "I’m your digital health assistant, here to give you information and guide you toward the right care.",

                    # Friendly & approachable
                    "You can call me your friendly health guide! I'm here to help you understand your symptoms and suggest appropriate care.",
                    "Think of me as your health companion—I’ll listen to your concerns and point you in the right direction.",
                    "I'm your virtual health buddy, here to provide useful information about your symptoms and possible next steps.",

                    # Reassuring & supportive
                    "I’m an assistant built to support you with health information and guidance, but I’m not a doctor.",
                    "I’m here to help you make sense of your symptoms and encourage you to seek professional care when needed.",
                    "My role is to guide you with trustworthy health information so you feel more confident about your next steps."
                ]


        what_can_you_do_responses = [
                            # Friendly & inviting
                            "I can help you understand what might be causing your symptoms and suggest appropriate medical care. Just describe how you're feeling!",
                            "I can listen to your symptoms and guide you toward the type of medical help that might be useful.",
                            "If you tell me what you're experiencing, I can provide information and point you in the right direction.",

                            # Professional & clear
                            "My purpose is to offer guidance on symptoms, potential conditions, and recommend appropriate medical professionals.",
                            "I can help you better understand possible reasons for your symptoms and suggest when it may be time to see a doctor.",
                            "I can provide information about symptoms, first aid, and recommend specialists. What's bothering you?",

                            # Supportive & reassuring
                            "I'm here to support you by making sense of your symptoms and helping you decide what steps to take next.",
                            "Think of me as a guide—I can explain possible causes, give you general health information, and suggest professional care when needed.",
                            "I can’t replace a doctor, but I can help you organize your symptoms and figure out what kind of healthcare provider to consult."
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
        #print(f"DEBUG extract_answered_questions: Input='{user_input}', Last question='{self.last_question}'")
        response_type = self.detect_affirmative_negative(user_input)
        #print(f"DEBUG: Response type detected: {response_type}")
        
        # First check for pain scale responses
        pain_scale_match = re.search(r'(\d+)\s*(?:\/|out of|out-of)\s*10', user_input_lower)
        if pain_scale_match and self.last_question and any(term in self.last_question.lower() for term in ['scale', '1-10', 'rating', 'how severe']):
                pain_level = int(pain_scale_match.group(1))
                extracted_info["pain_severity"] = f"{pain_level}/10"
                #print(f"✅ [extract_answered_questions] Extracted pain severity: {pain_level}/10")
                
                # Add pain severity symptom
                if pain_level >= 7:
                    self.user_symptoms.append("severe pain")
                elif pain_level  < 7 and pain_level >=4:
                    self.user_symptoms.append("moderate pain")
                else:
                    self.user_symptoms.append("mild pain")

        # Check if this is a response to the last asked question
        if self.last_question:
            last_question_lower = self.last_question.lower()
                        
                # REPLACE the bleeding handling code with this:

            if any(term in last_question_lower for term in ['bleeding', 'hemorrhage', 'blood loss', 'oozing', 'flowing', 'gushing', 'blood', 'plasma', 'clot', 'gore']):
                if response_type == "affirmative":
                    extracted_info["bleeding_symptoms"] = True
                    
                    # Check SPECIFICALLY what type of bleeding was asked about
                    if 'gums' in last_question_lower and 'nose' not in last_question_lower:
                        # Only gums mentioned
                        extracted_info["gum_bleeding"] = True
                        extracted_info["nose_bleeding"] = False
                        self.user_symptoms.append("bleeding gums")
                        
                    elif 'nose' in last_question_lower and 'gums' not in last_question_lower:
                        # Only nose mentioned  
                        extracted_info["nose_bleeding"] = True
                        extracted_info["gum_bleeding"] = False
                        self.user_symptoms.append("nosebleeds")
                        
                    elif 'gums' in last_question_lower and 'nose' in last_question_lower:
                        # Both mentioned - need to ask follow-up to specify which one
                        extracted_info["bleeding_mentioned"] = True  # Don't assume which one
                        # DON'T add any symptoms yet - ask for clarification
                        print("⚠️ Both gums and nose mentioned, need clarification")
                        
                elif response_type == "negative":
                    extracted_info["bleeding_symptoms"] = False
                    extracted_info["gum_bleeding"] = False
                    extracted_info["nose_bleeding"] = False
                    # Remove both symptoms
                    if "bleeding gums" in self.user_symptoms:
                        self.user_symptoms.remove("bleeding gums")
                    if "nosebleeds" in self.user_symptoms:
                        self.user_symptoms.remove("nosebleeds")
            
            
            # Add this handler for the pain worsening question:
            elif any(term in last_question_lower for term in ['pain get worse', 'worse when move', 'worse when cough']):
                if response_type == "affirmative":
                    extracted_info["worsening_pain"] = True
                    self.user_symptoms.append("worsening pain")  # ← This should be in Appendicitis symptoms
                elif response_type == "negative":
                    extracted_info["worsening_pain"] = False
                    if "worsening pain" in self.user_symptoms:
                        self.user_symptoms.remove("worsening pain")
            
            # Handle common cold symptoms
            elif any(term in last_question_lower for term in ['runny nose', 'nasal congestion', 'stuffy nose', 'nasal']):
                if response_type == "affirmative":
                    extracted_info["runny_nose"] = True
                    self.user_symptoms.append("runny nose")
                elif response_type == "negative":
                    extracted_info["runny_nose"] = False
                    if "runny nose" in self.user_symptoms:
                        self.user_symptoms.remove("runny nose")

            elif any(term in last_question_lower for term in ['sore throat', 'throat pain', 'throat sore']):
                if response_type == "affirmative":
                    extracted_info["sore_throat"] = True
                    self.user_symptoms.append("sore throat")
                elif response_type == "negative":
                    extracted_info["sore_throat"] = False
                    if "sore throat" in self.user_symptoms:
                        self.user_symptoms.remove("sore throat")

            elif any(term in last_question_lower for term in ['sneezing', 'watery eyes', 'itchy eyes']):
                if response_type == "affirmative":
                    extracted_info["sneezing"] = True
                    self.user_symptoms.append("sneezing")
                elif response_type == "negative":
                    extracted_info["sneezing"] = False
                    if "sneezing" in self.user_symptoms:
                        self.user_symptoms.remove("sneezing")

            elif any(term in last_question_lower for term in ['productive cough', 'phlegm', 'mucus']):
                if response_type == "affirmative":
                    extracted_info["productive_cough"] = True
                    self.user_symptoms.append("productive cough")
                elif response_type == "negative":
                    extracted_info["productive_cough"] = False
                    if "productive cough" in self.user_symptoms:
                        self.user_symptoms.remove("productive cough")
            
            # Add this section with the other symptom handlers:
            elif any(term in last_question_lower for term in ['cough', 'respiratory symptoms', 'breathing problems', 'coughing']):
                if response_type == "affirmative":
                    extracted_info["cough"] = True
                    self.user_symptoms.append("cough")
                elif response_type == "negative":
                    extracted_info["cough"] = False
                    if "cough" in self.user_symptoms:
                        self.user_symptoms.remove("cough")
                        
            elif any(term in last_question_lower for term in ['persistent cough', 'constant cough', 'nonstop coughing']):
                if response_type == "affirmative":
                    extracted_info["persistent_cough"] = True
                    self.user_symptoms.append("persistent cough")
                elif response_type == "negative":
                    extracted_info["persistent_cough"] = False
                    if "persistent cough" in self.user_symptoms:
                        self.user_symptoms.remove("persistent cough")
                        
            # Add this section with the other symptom handlers:

            # Handler for extreme fatigue question
            elif any(term in last_question_lower for term in ['extreme fatigue', 'extreme weakness', 'severe fatigue']):
                if response_type == "affirmative":
                    extracted_info["extreme_fatigue"] = True
                    self.user_symptoms.append("extreme fatigue")  # ← This gets weight 6
                elif response_type == "negative":
                    extracted_info["extreme_fatigue"] = False
                    if "extreme fatigue" in self.user_symptoms:
                        self.user_symptoms.remove("extreme fatigue")

            # Handler for difficulty breathing question  
            elif any(term in last_question_lower for term in ['difficulty breathing', 'chest pain', 'breathing problems', 'shortness of breath']):
                if response_type == "affirmative":
                    if 'breathing' in last_question_lower or 'shortness' in last_question_lower:
                        extracted_info["shortness_of_breath"] = True
                        self.user_symptoms.append("difficulty breathing")  # ← This gets weight 8
                    if 'chest' in last_question_lower or 'pain' in last_question_lower:
                        extracted_info["chest_pain"] = True
                        self.user_symptoms.append("chest pain")  # ← This gets weight 7
                elif response_type == "negative":
                    if 'breathing' in last_question_lower:
                        extracted_info["shortness_of_breath"] = False
                        if "difficulty breathing" in self.user_symptoms:
                            self.user_symptoms.remove("difficulty breathing")
                    if 'chest' in last_question_lower:
                        extracted_info["chest_pain"] = False
                        if "chest pain" in self.user_symptoms:
                            self.user_symptoms.remove("chest pain")
            
            
            # Handle heart problems question            
            elif any(term in last_question_lower for term in [
                                            'heart problems', 'cardiac problems', 'heart condition', 'cardiac disorder', 
                                            'cardiovascular issues', 'cardiac complications', 
                                            'heart disease', 'cardiovascular disease', 'coronary artery disease', 
                                            'ischemic heart disease', 'coronary heart disease', 'cardiac illness', 
                                            'heart disorder', 'cardiopathy', 
                                            'heart history', 'cardiac history', 'cardiovascular history', 
                                            'prior heart condition', 'past cardiac illness', 'history of heart disease', 
                                            'previous heart problems'
                                            ]):
                if response_type == "affirmative":
                    if "preexisting_conditions" not in extracted_info:
                        extracted_info["preexisting_conditions"] = []
                    extracted_info["preexisting_conditions"].append("heart problems")
                elif response_type == "negative":
                    extracted_info["no_heart_problems"] = True
                    
                        # Handle shortness of breath question
            elif any(term in last_question_lower for term in [
            'shortness of breath', 'breathlessness', 'difficulty breathing', 'dyspnea', 
            'labored breathing', 'trouble breathing', 'respiratory distress', 'hard to catch breath', 'winded',
            'breathing', 'respiration', 'inhaling', 'exhaling', 'ventilation', 'gas exchange', 'breathing rate', 'pulmonary function',
            'sweating', 'perspiration', 'diaphoresis', 'clamminess', 'damp skin', 
            'excessive sweating', 'hyperhidrosis', 'moist skin', 'cold sweat']):
                if response_type == "affirmative":
                    extracted_info["shortness_of_breath"] = True
                    self.user_symptoms.append("shortness of breath")
                elif response_type == "negative":
                    extracted_info["shortness_of_breath"] = False
                    if "shortness of breath" in self.user_symptoms:
                        self.user_symptoms.remove("shortness of breath")
                    
             # Handle pain severity question
            elif any(term in last_question_lower for term in [
            'scale', 'rating scale', 'measurement scale', 'severity scale', 'numeric scale', 'pain scale', 'grading scale',
            '1-10', 'one to ten', 'scale of ten', 'out of ten', 'ten-point scale', 'pain score 0-10', 'zero to ten',
            'how severe', 'severity level', 'degree of pain', 'how bad', 'intensity', 'level of discomfort', 'extent of pain', 'how much it hurts',
            'pain level', 'pain rating', 'pain severity', 'pain intensity', 'discomfort level', 'ache level', 'distress level', 'suffering scale']):
                if user_input_lower.isdigit():
                    pain_level = int(user_input_lower)
                    extracted_info["pain_severity"] = f"{pain_level}/10"
                    if pain_level >= 7:
                        self.user_symptoms.append("severe pain")
                        
            # Handle rash-related questions
            elif any(term in last_question_lower for term in [
            'rash', 'skin rash', 'eruption', 'breakout', 'skin irritation', 'dermatitis', 
            'hives', 'welts', 'skin outbreak', 'allergic rash',
            'skin', 'epidermis', 'dermis', 'cutaneous tissue', 'outer layer', 
            'complexion', 'flesh', 'integument',
            'red spots', 'red bumps', 'blotches', 'patches', 'red marks', 
            'red dots', 'lesions', 'papules', 'eruptions', 'skin discoloration']):
                if response_type == "affirmative":
                    extracted_info["rash"] = True
                    self.user_symptoms.append("rash")  # ADD RASH TO SYMPTOMS
                elif response_type == "negative":
                    extracted_info["rash"] = False
                    # REMOVE RASH FROM SYMPTOMS
                    if "rash" in self.user_symptoms:
                        self.user_symptoms.remove("rash")
            
            # Add this section with the other symptom handlers:
            elif any(term in last_question_lower for term in ['body aches', 'muscle aches', 'fatigue', 'tiredness', 'exhaustion']):
                if response_type == "affirmative":
                    if 'body ache' in last_question_lower or 'muscle ache' in last_question_lower:
                        extracted_info["body_aches"] = True
                        self.user_symptoms.append("body aches")
                    if 'fatigue' in last_question_lower or 'tired' in last_question_lower or 'exhaustion' in last_question_lower:
                        extracted_info["fatigue"] = True
                        self.user_symptoms.append("fatigue")
                elif response_type == "negative":
                    if 'body ache' in last_question_lower or 'muscle ache' in last_question_lower:
                        extracted_info["body_aches"] = False
                        if "body aches" in self.user_symptoms:
                            self.user_symptoms.remove("body aches")
                    if 'fatigue' in last_question_lower or 'tired' in last_question_lower or 'exhaustion' in last_question_lower:
                        extracted_info["fatigue"] = False
                        if "fatigue" in self.user_symptoms:
                            self.user_symptoms.remove("fatigue")
            
            
            # Add this section with the other symptom handlers:
            elif any(term in last_question_lower for term in ['muscle pain', 'muscle ache', 'myalgia', 'sore muscles']):
                if response_type == "affirmative":
                    extracted_info["muscle_pain"] = True
                    self.user_symptoms.append("muscle pain")
                elif response_type == "negative":
                    extracted_info["muscle_pain"] = False
                    if "muscle pain" in self.user_symptoms:
                        self.user_symptoms.remove("muscle pain")
            
            
            # Handle joint-related questions
            elif any(term in last_question_lower for term in [
            'joint', 'articulation', 'hinge', 'synovial joint', 'bone connection', 'cartilage junction', 'musculoskeletal joint',
            'swollen', 'puffy', 'enlarged', 'inflamed', 'edematous', 'distended', 'bloated', 'tender swelling',
            'painful to move', 'restricted movement', 'stiff joint', 'limited mobility', 'reduced range of motion',
            'sore when moving', 'movement pain', 'joint stiffness', 'difficulty moving', 'aching with motion']):
                if response_type == "affirmative":
                    extracted_info["joint_swelling"] = True
                    self.user_symptoms.append("joint swelling")  # ADD JOINT SWELLING TO SYMPTOMS
                elif response_type == "negative":
                    extracted_info["joint_swelling"] = False
                    # REMOVE JOINT SWELLING FROM SYMPTOMS
                    if "joint swelling" in self.user_symptoms:
                        self.user_symptoms.remove("joint swelling")
            
            # Handle travel-related questions
            elif 'travel' in last_question_lower:
                if response_type == "affirmative":
                    extracted_info["travel_mentioned"] = True
                elif response_type == "negative":
                    extracted_info["travel_history"] = "No travel"
                    extracted_info["travel_question_answered"] = True
                    
            elif any(term in last_question_lower for term in [
 'how long', 'length of time', 'duration', 'period of time', 'time span', 'over what period', 'how many days', 'since when',
 'fever duration', 'duration of fever', 'length of fever', 'time fever lasted', 'period of fever', 'how long fever lasted', 'fever timeline',
 'had fever', 'experienced fever', 'history of fever', 'suffered from fever', 'been febrile', 'fever episode', 'previously had fever'
]):
                # Extract duration from answer
                duration_match = re.search(r'(\d+)\s*(day|days|hour|hours|week|weeks)', user_input_lower)
                if duration_match:
                    extracted_info["fever_duration"] = f"{duration_match.group(1)} {duration_match.group(2)}"
                    extracted_info["symptom_duration"] = f"{duration_match.group(1)} {duration_match.group(2)}"
        
        # Extract duration information
        duration_match = re.search(r'(\d+)\s*(day|days|hour|hours|week|weeks|month|months)', user_input_lower)
        if duration_match:
            extracted_info["fever_duration"] = f"{duration_match.group(1)} {duration_match.group(2)}"
            extracted_info["symptom_duration"] = f"{duration_match.group(1)} {duration_match.group(2)}"
        
        # Extract temperature
        temp_match = re.search(r'(\d{2,3}(?:\.\d{1,2})?)\s*(degrees|°|fahrenheit|f|celcius|c|degree|deg|temp|temperature)?', user_input_lower)
        if temp_match:
            try:
                temperature = float(temp_match.group(1))
                extracted_info["fever_temperature"] = temperature
                # ADD FEVER TO SYMPTOMS BASED ON TEMPERATURE
                if temperature >= 100.4:  # Fever threshold
                    self.user_symptoms.append("fever")
                    if temperature >= 102:  # High fever
                        self.user_symptoms.append("high fever")
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
        
        # Handle travel history responses (both positive and negative)
        travel_areas = self.dengue_prone_areas + self.chikungunya_prone_areas + self.malaria_prone_areas
        
        # Check for positive travel responses
        travel_mentioned = False
        travel_location = None
        
        for area in travel_areas:
            if re.search(r'\b' + re.escape(area.lower()) + r'\b', user_input_lower):
                travel_mentioned = True
                extracted_info["travel_history"] = area
                print(f"✅ [extract_answered_questions] Detected travel to: {area}")
                break
        
        # Check for explicit travel mentions without specific location
        travel_keywords = [
 'travel', 'journey', 'voyage', 'excursion', 'exploration', 'tourism', 'commute', 'expedition', 'trip',
 'visited', 'went', 'stopped by', 'saw', 'called on', 'attended', 'dropped in', 'paid a visit',
 'been to', 'traveled to', 'stayed in', 'spent time in', 'passed through', 'resided in', 'gone to',
 'went to', 'headed to', 'moved toward', 'journeyed to', 'made a trip to', 'set off for', 'traveled toward',
 'trip to', 'holiday in', 'vacation in', 'journey to', 'tour of', 'outing to', 'visit to'
]

        if any(keyword in user_input_lower for keyword in travel_keywords):
            travel_mentioned = True
        
        # Check for negative responses
        negative_keywords = [
 'no', 'not', 'never', 'none',
 "haven't", "hasn't", "didn't", "won't", "can't", "wouldn't", "shouldn't", "couldn't", "ain't",
 'nope', 'nah', 'naw', 'nuh-uh', 'negative', 'no way', 'not at all', 'uh-uh',
 'absolutely not', 'definitely not', 'under no circumstances', 'by no means', 'never ever', 'not really'
]

        positive_keywords = [
 'yes', 'yeah', 'yep', 'yup', 'sure', 'of course', 'certainly', 'absolutely', 'definitely',
 'uh-huh', 'mm-hmm', 'right', 'you bet', 'sure thing', 'ok', 'okay', 'k',
 'have', 'did', 'done', 'got it', 'already did', 'I have', 'I did', 'completed',
 'for sure', 'no doubt', 'without a doubt', 'affirmative', 'indeed', 'exactly'
]

        
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
    "eyes": [
        "eyes", "eye", "behind eyes", "ocular", "retro-orbital",
        "eye socket", "around eyes", "vision pain", "optic"
    ],
    "chest": [
        "chest", "breast", "sternum", "thorax", "rib cage",
        "pecs", "lungs area", "heart area"
    ],
    "abdomen": [
        "abdomen", "stomach", "belly", "tummy", "gut",
        "abdominal area", "navel", "lower stomach", "upper abdomen"
    ],
    "head": [
        "head", "headache", "migraine", "cranial", "skull",
        "forehead", "temple", "occipital", "scalp", "cranium"
    ],
    "joint": [
        "joint", "knee", "elbow", "wrist", "ankle", "arthralgia",
        "shoulder", "hip", "knuckles", "toe joint", "finger joint"
    ],
    "back": [
        "back", "spine", "spinal", "lower back", "upper back",
        "lumbar", "thoracic", "vertebrae", "dorsal"
    ],
    "neck": [
        "neck", "cervical", "throat", "nape", "back of neck",
        "cervix (colloquial misentry)", "neckline"
    ]
}

        for location, keywords in pain_locations.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', user_input_lower):
                    extracted_info["pain_location"] = location
                    # ADD PAIN SYMPTOM BASED ON LOCATION
                    if location == "eyes":
                        self.user_symptoms.append("pain behind eyes")
                    elif location == "head":
                        self.user_symptoms.append("headache")
                    elif location == "joint":
                        self.user_symptoms.append("joint pain")
                    elif location == "abdomen":
                        self.user_symptoms.append("abdominal pain")
                    elif location == "chest":
                        self.user_symptoms.append("chest pain")
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
        condition_keywords = [
    # Metabolic
    "diabetes", "diabetes mellitus", "type 1 diabetes", "type 2 diabetes", "high blood sugar", "prediabetes",
    
    # Cardiovascular
    "hypertension", "high blood pressure", "blood pressure problems", 
    "heart disease", "cardiac disease", "coronary artery disease", 
    "ischemic heart disease", "heart condition", "heart problems",
    
    # Respiratory
    "asthma", "bronchial asthma", "respiratory disease", "breathing problems",
    
    # Immune / Allergic
    "allergies", "allergic reaction", "hay fever", "allergic rhinitis", "food allergy", "dust allergy",
    
    # Musculoskeletal
    "arthritis", "osteoarthritis", "rheumatoid arthritis", "joint disease", "arthropathy",
    
    # Kidney
    "kidney disease", "renal disease", "renal failure", "chronic kidney disease", "CKD", "nephropathy",
    
    # Liver
    "liver disease", "hepatic disease", "cirrhosis", "hepatitis", "fatty liver", "liver condition"
]

        for condition in condition_keywords:
            if re.search(r'\b' + re.escape(condition) + r'\b', user_input_lower):
                if "preexisting_conditions" not in extracted_info:
                    extracted_info["preexisting_conditions"] = []
                extracted_info["preexisting_conditions"].append(condition)
        
        # Extract current medications
        med_keywords = [
    # General usage
    "taking", "on", "using", "prescribed", 
    "medication", "medicine", "drug", "remedy", "treatment",

    # Forms of meds
    "pill", "tablet", "capsule", "caplet", "lozenge",
    "syrup", "liquid", "drops", "ointment", "cream", "gel",
    "inhaler", "injection", "shot", "vaccine",

    # Casual mentions
    "meds", "script", "rx", "dose", "dosing",
    "painkiller", "antibiotic", "supplement", "vitamin"
]

        if any(keyword in user_input_lower for keyword in med_keywords):
            med_match = re.search(r"(aspirin|ibuprofen|paracetamol|acetaminophen|antibiotic|antihistamine|insulin|steroid)", user_input_lower)
            if med_match:
                if "current_medications" not in extracted_info:
                    extracted_info["current_medications"] = []
                extracted_info["current_medications"].append(med_match.group(1))
        
        # Remove duplicates from user_symptoms
        self.user_symptoms = list(set(self.user_symptoms))
        
        return extracted_info


    def process_medical_query(self, user_input: str) -> Tuple[str, bool]:
        print(f"🔍 process_medical_query called with: '{user_input}'")
        
        print(f"\n=== DEBUG: process_medical_query() called ===")
        print(f"Input: '{user_input}'")
        print(f"Current user_symptoms: {self.user_symptoms}")

        clear_diagnosis = self.check_clear_diagnosis()
        if clear_diagnosis:
            response = self.generate_final_recommendation(
                clear_diagnosis["condition"], 
                clear_diagnosis["confidence"]
            )
            # Reset conversation
            self.reset_conversation()
            return response, False
        
        old_value = self.is_medical_chat
        self.is_medical_chat = True
        
        if not self.initialized:
            print("❌ Chatbot not initialized")
            return "I'm sorry, the medical assistant is not properly initialized. Please try again later.", False
            
        try:
            # Add user input to history
            self.chat_history.append(f"User: {user_input}")
            
            # Store the last question for context tracking
            current_last_question = self.last_question

            new_symptoms = self.extract_symptoms(user_input)
            print(f"✅ Extracted symptoms: {new_symptoms}")
            self.user_symptoms.extend(new_symptoms)
            self.user_symptoms = list(set(self.user_symptoms))  # Remove duplicates
            
            # Extract information from user response
            extracted_info = self.extract_answered_questions(user_input)
            for key, value in extracted_info.items():
                if value:  # Only update if we got a value
                    self.user_info[key] = value
            
            print(f"🔍 Extracted info: {extracted_info}")
            print(f"🔍 User info: {self.user_info}")
            print(f"🔍 User symptoms: {self.user_symptoms}")
                
            # Check if we have enough information for diagnosis
            has_fever = any(s in self.user_symptoms for s in ["fever", "high fever", "mild fever"])
            has_headache = "headache" in self.user_symptoms or "severe headache" in self.user_symptoms
            has_travel_history = self.user_info["travel_history"] is not None
            has_fever_info = self.user_info["fever_temperature"] is not None and self.user_info["fever_duration"] is not None
            
            # If we have fever + headache + travel to dengue area, we should provide diagnosis
            if (has_fever and has_headache and has_travel_history and 
                self.user_info["travel_history"] in self.dengue_prone_areas and
                has_fever_info and len(self.user_symptoms) >= 4):
                
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
            print("🔍 Finding best matches...")
            best_matches = self.find_best_matches(self.user_symptoms)
            print(f"✅ Best matches found: {len(best_matches)}")
            
            # Update diagnostic stage
            if len(self.user_symptoms) < 3:
                self.diagnostic_stage = "initial"
            elif best_matches and best_matches[0]["score"] >= 80:
                self.diagnostic_stage = "confirmation"
            else:
                self.diagnostic_stage = "symptom_clarification"
            
            # If we have a high-confidence match, provide final recommendation
            """if self.diagnostic_stage == "confirmation" and best_matches:
                best_match = best_matches[0]
                
                # SAFETY CHECK: Don't diagnose emergency conditions without multiple specific symptoms
                condition_name = best_match["condition"].get("condition", "").lower()
                urgency = best_match["condition"].get("urgency", "").lower()
                
                # For emergency/dangerous conditions, require at least 2 specific symptoms
                if urgency in ["emergency", "dangerous", "urgent"]:
                    specific_symptoms = [s for s in best_match["condition"]["symptoms"] 
                                    if self.symptom_weights.get(s, 5) >= 7]
                    matched_specific = len([s for s in self.user_symptoms if s in specific_symptoms])
                    
                    if matched_specific < 2:
                        print(f"⚠️ SAFETY CHECK: Emergency condition '{condition_name}' needs at least 2 specific symptoms, but only has {matched_specific}")
                        # Don't provide final diagnosis, ask more questions instead
                        self.diagnostic_stage = "symptom_clarification"
                
                # Only provide final diagnosis if it passes safety checks
                if self.diagnostic_stage == "confirmation":
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
                
                return response, False"""
            
            if self.diagnostic_stage == "confirmation" and best_matches:
                best_match = best_matches[0]
                
                # SAFETY CHECK: Don't diagnose emergency conditions without multiple specific symptoms
                condition_name = best_match["condition"].get("condition", "").lower()
                urgency = best_match["condition"].get("urgency", "").lower()
                
                # For emergency/dangerous conditions, require at least 2 specific symptoms
                safety_check_passed = True
                if urgency in ["emergency", "dangerous", "urgent"]:
                    specific_symptoms = [s for s in best_match["condition"]["symptoms"] 
                                    if self.symptom_weights.get(s, 5) >= 7]
                    matched_specific = len([s for s in self.user_symptoms if s in specific_symptoms])
                    
                    if matched_specific < 2:
                        print(f"⚠️ SAFETY CHECK: Emergency condition '{condition_name}' needs at least 2 specific symptoms, but only has {matched_specific}")
                        # Don't provide final diagnosis, ask more questions instead
                        self.diagnostic_stage = "symptom_clarification"
                        safety_check_passed = False
                
                # Only provide final diagnosis if it passes safety checks
                if self.diagnostic_stage == "confirmation" and safety_check_passed:
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
                else:
                    # Safety check failed or diagnostic stage changed - ask more questions
                    diagnostic_questions = self.get_diagnostic_questions(best_matches)
                    
                    if diagnostic_questions:
                        # Use empathetic introduction
                        empathetic_intro = self.generate_empathetic_response()
                        question = empathetic_intro + diagnostic_questions[0]
                        
                        # Store the question we're about to ask
                        self.last_question = diagnostic_questions[0]
                        self.asked_questions.add(diagnostic_questions[0])
                        self.current_follow_ups = diagnostic_questions[1:]
                        
                        # Add bot response to history
                        self.chat_history.append(f"Bot: {question}")
                        
                        return question, True
                    else:
                        # No more questions available - provide gentle guidance
                        response = "I understand you're concerned about your symptoms. Based on the information provided, I recommend consulting a healthcare professional for a proper evaluation."
                        self.chat_history.append(f"Bot: {response}")
                        
                        # Reset for next conversation
                        self.is_medical_chat = False
                        self.asked_questions.clear()
                        self.current_follow_ups = []
                        self.user_symptoms = []
                        self.user_info = {key: None for key in self.user_info}
                        self.diagnostic_stage = "initial"
                        
                        return response, False
            
            
            
            # Otherwise, ask diagnostic questions - REMOVE THE MANUAL TRAVEL QUESTION INSERTION
            diagnostic_questions = self.get_diagnostic_questions(best_matches)
            
            # DEBUG: Check what questions we got
            print(f"🔍 DEBUG: Diagnostic questions from get_diagnostic_questions: {diagnostic_questions}")
            
            # REMOVED THE MANUAL TRAVEL QUESTION INSERTION LOGIC HERE
            # The travel question logic is now handled inside get_diagnostic_questions
            
            if diagnostic_questions:
                # Store the question we're about to ask
                self.last_question = diagnostic_questions[0]
                
                # Use empathetic introduction
                empathetic_intro = self.generate_empathetic_response()
                question = empathetic_intro + diagnostic_questions[0]
                self.asked_questions.add(diagnostic_questions[0])
                self.current_follow_ups = diagnostic_questions[1:]
                
                # Add bot response to history
                self.chat_history.append(f"Bot: {question}")
                
                return question, True
            
            # If we don't have questions but still need more info
            if not diagnostic_questions and len(self.user_symptoms) < 3:
                empathetic_prompts = [
                    "I want to make sure I understand your situation correctly. Could you describe your symptoms in more detail?",
                    "Let me help you get the right care. Please tell me more about what you're experiencing.",
                    "I'm here to help you figure this out. Can you provide more details about your symptoms?",
                    "I understand this might be worrying. Let's work through it together. What other symptoms are you experiencing?",
                    "That sounds uncomfortable. You're not alone—can you share a bit more about what you're going through?",
                    "I hear your concern. Could you walk me through when these symptoms started?",
                    "It's important we get a clear picture. Can you tell me if anything makes your symptoms better or worse?",
                    "You're doing the right thing by talking about this. What other changes have you noticed?",
                    "Thank you for sharing this with me. Can you describe the severity or frequency of your symptoms?",
                    "I want to make sure we cover everything. Are there any other symptoms or details you think I should know?",
                    "I can imagine this feels difficult. Let's take it step by step—what symptom bothers you the most?",
                    "Your health matters, and I want to support you. Could you explain how these symptoms affect your daily life?",
                    "That must be tough to deal with. When did you first notice these symptoms?"
                ]
                
                response = random.choice(empathetic_prompts)
                self.chat_history.append(f"Bot: {response}")
                return response, True
            
            # If we reach here, we don't have enough information but can't ask more questions
            empathetic_responses = [
                "I understand you're concerned about your symptoms. Based on the information provided, I recommend consulting a healthcare professional for a proper evaluation.",
                "I've done my best to understand your symptoms, but I think it would be best for you to see a doctor for a definitive diagnosis.",
                "Your symptoms deserve proper medical attention. I encourage you to consult with a healthcare provider who can examine you properly.",
                "I know it can be worrying to have these symptoms. You're not alone—please reach out to a healthcare professional who can help you further.",
                "Your health and peace of mind are important. A doctor will be able to give you the best advice and care.",
                "Thank you for sharing these details. To ensure you get the right treatment, I recommend contacting a healthcare provider.",
                "Since symptoms can sometimes overlap between different conditions, it's safest to have a medical professional evaluate your situation.",
                "I can't provide a diagnosis, but a doctor will be able to examine you properly and suggest the right treatment.",
                "It's important to get clarity on what's causing your symptoms. Please consult with a healthcare professional as soon as you can.",
                "You're taking the right step by discussing your symptoms. Now I encourage you to follow up with a medical professional for a thorough check.",
                "It's always better to be safe when it comes to your health. Please schedule a visit with a doctor to get a proper evaluation.",
                "Your well-being matters. A healthcare provider will be able to guide you with the most accurate advice."
            ]
            
            response = random.choice(empathetic_responses)
            self.chat_history.append(f"Bot: {response}")
            return response, False
            
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error processing your medical query. Please try again or consult a healthcare professional."
            self.chat_history.append(f"Bot: {error_msg}")
            print(f"❌ Error in process_medical_query: {e}")
            import traceback
            traceback.print_exc()
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
                    print(f"✅ Setting is_medical_chat to True")
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