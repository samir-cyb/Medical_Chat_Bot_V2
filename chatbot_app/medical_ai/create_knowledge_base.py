import json
import os
from .config import KNOWLEDGE_BASE_PATH

# Symptom weights for more accurate matching - EXPANDED
symptom_weights = {
    # High urgency symptoms
    "sharp pain lower right abdomen": 9, "right lower quadrant pain": 9, "rebound tenderness": 10,
    "chest pain": 10, "pain radiating to arm or jaw": 10, "shortness of breath": 10, "heart palpitations": 9,
    "difficulty breathing": 10, "severe dehydration": 9, "bleeding gums": 8, "nosebleeds": 8, 
    "severe abdominal pain": 9, "persistent vomiting": 9, "lethargy": 8, "restlessness": 7,
    "sudden drop in temperature": 9, "confusion": 9, "very high fever": 8, "severe bleeding": 10,
    "stiff neck": 8, "light sensitivity": 7, "severe dizziness": 8, "fainting": 9,
    
    # Dengue specific symptoms
    "high fever with sudden onset": 8, "severe headache": 7, "pain behind eyes": 8, "severe muscle pain": 7,
    "severe joint pain": 7, "skin rash": 6, "mild bleeding": 7, "low platelet count": 8,
    "easy bruising": 7, "petechiae": 8, "red spots on skin": 6, "increased thirst": 5,
    
    # Chikungunya specific symptoms
    "high fever with abrupt onset": 7, "debilitating joint pain": 9, "joint swelling": 7, "maculopapular rash": 6,
    "conjunctival injection": 6, "nausea": 5, "fatigue": 5, "severe joint stiffness": 7,
    "symmetric joint involvement": 6, "joint redness": 5, "morning joint stiffness": 6,
    
    # Medium urgency symptoms
    "high fever": 7, "severe body aches": 7, "persistent cough": 6, "extreme fatigue": 6,
    "widespread rash": 6, "eye pain": 6, "nausea": 6, "vomiting": 6, "diarrhea": 5,
    "loss of appetite": 4, "swollen lymph nodes": 6, "ear pain": 5, "sinus pressure": 4,
    
    # Low urgency symptoms
    "mild fever": 4, "slight runny nose": 2, "occasional sneezing": 2, "mild headache": 3,
    "joint pain": 5, "rash": 5, "mild muscle pain": 3, "runny nose": 3, "body aches": 4,
    "cough": 4, "sore throat": 4, "changing mole": 5, "new skin growth": 5, "severe acne": 4,
    "headache": 4, "migraine": 5, "head pain": 4, "throbbing head": 4, "mild fatigue": 3,
    "occasional cough": 3, "tiredness": 3, "mild sore throat": 3, "nasal congestion": 3
}

# Symptom mapping for normalization - EXPANDED
symptom_mapping = {
    "fever": ["fever", "temperature", "hot", "burning up", "have a temp", "running a fever", "feeling feverish"],
    "high fever": ["high fever", "very hot", "102 fever", "103 fever", "104 fever", "105 fever", "spiking fever", "very high temperature"],
    "mild fever": ["mild fever", "slight fever", "low grade fever", "99 fever", "100 fever", "101 fever", "low-grade temperature"],
    "headache": ["headache", "head pain", "head hurting", "head ache", "head throbbing"],
    "severe headache": ["severe headache", "bad headache", "worst headache", "excruciating headache", "debilitating headache", "unbearable headache"],
    "mild headache": ["mild headache", "slight headache", "minor headache", "small headache"],
    "cough": ["cough", "coughing", "hacking", "hacking cough", "coughing fit"],
    "persistent cough": ["persistent cough", "constant cough", "nonstop coughing", "continuous cough", "won't stop coughing"],
    "runny nose": ["runny nose", "nose running", "rhinorrhea", "dripping nose", "nasal discharge"],
    "sore throat": ["sore throat", "throat pain", "throat hurting", "painful throat", "scratchy throat", "irritated throat"],
    "body aches": ["body aches", "muscle aches", "aching body", "body pains", "generalized ache", "whole body hurts"],
    "muscle pain": ["muscle pain", "myalgia", "muscles hurting", "sore muscles", "muscle soreness", "aching muscles"],
    "joint pain": ["joint pain", "aching joints", "joints hurting", "sore joints", "arthralgia", "joint discomfort"],
    "severe joint pain": ["severe joint pain", "bad joint pain", "joints very painful", "excruciating joint pain", "debilitating joint pain"],
    "rash": ["rash", "skin rash", "red spots", "skin irritation", "skin eruption", "skin redness"],
    "maculopapular rash": ["maculopapular rash", "bumpy rash", "raised rash", "rash with bumps", "papular rash"],
    "widespread rash": ["widespread rash", "rash all over", "body rash", "rash covering body", "extensive rash"],
    "nausea": ["nausea", "queasy", "feeling sick", "sick to stomach", "upset stomach", "feel like vomiting"],
    "vomiting": ["vomiting", "throwing up", "puking", "emesis", "regurgitating"],
    "persistent vomiting": ["persistent vomiting", "constant vomiting", "can't stop vomiting", "repeated vomiting", "continuous vomiting"],
    "abdominal pain": ["abdominal pain", "stomach pain", "belly pain", "tummy ache", "abdominal discomfort"],
    "severe abdominal pain": ["severe abdominal pain", "bad stomach pain", "extreme abdominal pain", "excruciating stomach pain"],
    "shortness of breath": ["shortness of breath", "breathing difficulty", "can't catch breath", "labored breathing", "breathlessness"],
    "chest pain": ["chest pain", "chest hurting", "pain in chest", "chest discomfort", "tightness in chest"],
    "bleeding gums": ["bleeding gums", "gums bleeding", "blood when brushing", "blood from gums", "gingival bleeding"],
    "nosebleeds": ["nosebleeds", "nose bleeding", "blood from nose", "epistaxis", "nasal bleeding"],
    "lethargy": ["lethargy", "extremely tired", "no energy", "profound fatigue", "complete exhaustion", "can't get out of bed"],
    "confusion": ["confusion", "disorientation", "can't think clearly", "mental fog", "altered mental state"],
    "pain behind eyes": ["pain behind eyes", "eye socket pain", "retro-orbital pain", "pain behind eyeballs", "deep eye pain"],
    "conjunctival injection": ["conjunctival injection", "red eyes", "bloodshot eyes", "eye redness", "inflamed eyes"],
    "joint swelling": ["joint swelling", "swollen joints", "puffy joints", "inflamed joints", "joint inflammation"],
    "easy bruising": ["easy bruising", "bruising easily", "unexplained bruises", "bruises without injury"],
    "petechiae": ["petechiae", "pinpoint red spots", "small red dots", "red specks on skin"],
    "stiff neck": ["stiff neck", "neck stiffness", "can't move neck", "neck rigidity"],
    "light sensitivity": ["light sensitivity", "sensitive to light", "photophobia", "lights hurt eyes"],
    "loss of appetite": ["loss of appetite", "not hungry", "no desire to eat", "reduced appetite"],
    "swollen lymph nodes": ["swollen lymph nodes", "swollen glands", "enlarged lymph nodes", "lumps in neck"],
    "dehydration": ["dehydration", "dehydrated", "dry mouth", "excessive thirst", "dark urine"]
}

# Dengue-prone areas in Bangladesh (2025)
dengue_prone_areas = [
    "Dhaka", "Chattogram", "Barishal", "Khulna", "Rajshahi", "Sylhet",
    "Gazipur", "Narayanganj", "Mymensingh", "Rangpur"
]

# Chikungunya-prone areas in Bangladesh (2025)
chikungunya_prone_areas = [
    "Dhaka", "Tejgaon", "Mohakhali"
]

# Malaria-prone areas in Bangladesh (2025)
malaria_prone_areas = [
    "Bandarban", "Rangamati", "Khagrachhari"
]

# This is our medical knowledge dataset.
medical_data = [
    {
        "symptoms": ["sharp pain lower right abdomen", "right lower quadrant pain", "abdominal pain with nausea", "rebound tenderness"],
        "specialty": "Gastroenterology / Emergency Medicine",
        "suggested_action": "Seek immediate medical attention at an Urgent Care or Emergency Room. This could indicate appendicitis, which is a medical emergency.",
        "urgency": "Urgent",
        "first_aid": "Do not eat or drink anything. Avoid taking pain medication. Apply a cold compress to the area.",
        "follow_up_questions": ["Is the pain getting worse?", "Do you have a fever?", "Is there any vomiting?"],
        "condition": "Appendicitis"
    },
    {
        "symptoms": ["chest pain", "pain radiating to arm or jaw", "shortness of breath", "heart palpitations"],
        "specialty": "Cardiology / Emergency Medicine",
        "suggested_action": "This is a medical emergency. Go to the nearest Emergency Room or call emergency services immediately. Do not drive yourself.",
        "urgency": "Emergency",
        "first_aid": "If you are with the person, have them sit down and rest. If they have prescribed nitroglycerin, help them take it. If they are unresponsive, call for emergency help.",
        "follow_up_questions": ["Is the pain sharp or dull?", "Are you feeling dizzy?", "Is there shortness of breath?"],
        "condition": "Heart Attack"
    },
    {
        "symptoms": ["persistent rash", "changing mole", "new skin growth", "severe acne"],
        "specialty": "Dermatology",
        "suggested_action": "Schedule a routine appointment with a dermatologist for evaluation.",
        "urgency": "Routine",
        "first_aid": "Keep the affected area clean and dry. Avoid scratching. Use over-the-counter hydrocortisone cream if it's an itch.",
        "follow_up_questions": ["Has the rash spread?", "Is it itchy?", "Is there any pain or pus?"],
        "condition": "Skin Condition"
    },
    {
        "symptoms": ["fever", "cough", "sore throat", "runny nose", "body aches"],
        "specialty": "Primary Care / General Practice",
        "suggested_action": "Schedule an appointment with your primary care doctor or visit an Urgent Care clinic for evaluation. Rest and hydrate.",
        "urgency": "Routine",
        "first_aid": "Drink plenty of fluids, get rest, and take over-the-counter fever reducers if needed. Use a saline nasal spray for congestion.",
        "follow_up_questions": ["How long have you had these symptoms?", "Is the fever high?", "Do you have a productive cough?"],
        "condition": "Common Cold or Flu"
    },
    {
        "symptoms": ["headache", "migraine", "head pain", "throbbing head"],
        "specialty": "Neurology / Primary Care",
        "suggested_action": "If the headache is severe or sudden, seek urgent care. For chronic headaches, schedule an appointment with a neurologist or your primary care doctor.",
        "urgency": "Varies",
        "first_aid": "Rest in a dark, quiet room. Apply a cold or warm compress to your head. Stay hydrated.",
        "follow_up_questions": ["Is this a new type of headache for you?", "Are you experiencing any vision changes?", "Is the pain on one side of your head?"],
        "condition": "Headache"
    },
    {
        "symptoms": ["mild fever", "slight runny nose", "occasional sneezing", "mild headache"],
        "specialty": "Primary Care / General Practice",
        "suggested_action": "Rest at home, drink plenty of fluids, and take over-the-counter cold medicine if needed. Monitor your symptoms.",
        "urgency": "Normal",
        "first_aid": "Get plenty of rest, drink warm fluids like tea with honey, use a humidifier, and take over-the-counter cold medicine.",
        "follow_up_questions": ["How long have you had these symptoms?", "Is your fever above 100.4°F (38°C)?", "Are you experiencing any body aches?"],
        "stage": "Normal Stage",
        "condition": "Common Cold"
    },
    {
        "symptoms": ["high fever", "severe body aches", "persistent cough", "extreme fatigue"],
        "specialty": "Primary Care / General Practice",
        "suggested_action": "Schedule an appointment with your doctor today. You may need antiviral medication if diagnosed early.",
        "urgency": "Intermediate",
        "first_aid": "Rest, stay hydrated, take fever reducers like acetaminophen or ibuprofen, and use a warm compress for body aches.",
        "follow_up_questions": ["How high is your fever?", "Are you having difficulty breathing?", "Do you have any underlying health conditions?"],
        "stage": "Intermediate Stage",
        "condition": "Influenza (Flu)"
    },
    {
        "symptoms": ["very high fever", "difficulty breathing", "chest pain", "severe dehydration", "confusion"],
        "specialty": "Emergency Medicine",
        "suggested_action": "Seek emergency medical care immediately. These symptoms could indicate a serious complication.",
        "urgency": "Dangerous",
        "first_aid": "While waiting for emergency care, try to keep the person hydrated with small sips of water and use cool compresses to reduce fever.",
        "follow_up_questions": ["When did the breathing difficulties start?", "Is the person able to speak in full sentences?", "What is their current temperature?"],
        "stage": "Dangerous Stage",
        "condition": "Severe Respiratory Infection"
    },
    {
        "symptoms": ["mild fever", "joint pain", "rash", "headache", "mild muscle pain"],
        "specialty": "Infectious Disease / Primary Care",
        "suggested_action": "Schedule an appointment with your doctor for evaluation. Rest and take acetaminophen for pain and fever.",
        "urgency": "Normal",
        "first_aid": "Rest, stay hydrated, take pain relievers for joint discomfort, and use calamine lotion for rash relief.",
        "follow_up_questions": ["When did the symptoms start?", "Have you traveled to areas with known Chikungunya outbreaks?", "Is the rash spreading?"],
        "stage": "Normal Stage",
        "condition": "Chikungunya Fever"
    },
    {
        "symptoms": ["high fever", "severe joint pain", "widespread rash", "eye pain", "nausea"],
        "specialty": "Infectious Disease",
        "suggested_action": "Seek medical attention within 24 hours. You may need specific testing and management.",
        "urgency": "Intermediate",
        "first_aid": "Rest in a cool environment, stay well-hydrated, take pain medication as directed, and use cool compresses for fever.",
        "follow_up_questions": ["How severe is the joint pain?", "Are you able to move your joints comfortably?", "Have you noticed any bleeding?"],
        "stage": "Intermediate Stage",
        "condition": "Chikungunya Fever"
    },
    {
        "symptoms": ["persistent high fever", "severe dehydration", "bleeding gums", "nosebleeds", "severe abdominal pain"],
        "specialty": "Emergency Medicine / Infectious Disease",
        "suggested_action": "Go to the emergency room immediately. This could indicate severe Chikungunya complications.",
        "urgency": "Dangerous",
        "first_aid": "While waiting for emergency care, keep the person lying down with elevated legs if showing signs of shock. Do not give aspirin or NSAIDs.",
        "follow_up_questions": ["Is there any visible bleeding?", "How long has the fever persisted?", "Is the person conscious and responsive?"],
        "stage": "Dangerous Stage",
        "condition": "Chikungunya Fever"
    },
    {
        "symptoms": ["mild fever", "headache", "mild body aches", "rash"],
        "specialty": "Infectious Disease / Primary Care",
        "suggested_action": "Schedule a doctor's appointment for evaluation. Monitor for any warning signs.",
        "urgency": "Normal",
        "first_aid": "Rest, drink plenty of fluids, take acetaminophen for fever and pain, and avoid mosquito bites to prevent spreading.",
        "follow_up_questions": ["When did the symptoms start?", "Have you been in areas with dengue outbreaks?", "Is the rash itchy?"],
        "stage": "Normal Stage",
        "condition": "Dengue Fever"
    },
    {
        "symptoms": ["high fever with sudden onset", "severe headache", "pain behind eyes", "severe muscle pain", "vomiting"],
        "specialty": "Infectious Disease",
        "suggested_action": "Seek medical attention within 24 hours. You may need blood tests and close monitoring.",
        "urgency": "Intermediate",
        "first_aid": "Rest, stay hydrated with oral rehydration solutions, take acetaminophen for pain (avoid aspirin/NSAIDs), and use cool compresses.",
        "follow_up_questions": ["How many days have you had fever?", "Are you able to keep fluids down?", "Have you noticed any bleeding or bruising?"],
        "stage": "Intermediate Stage",
        "condition": "Dengue Fever"
    },
    {
        "symptoms": ["sudden drop in temperature", "severe abdominal pain", "persistent vomiting", "bleeding", "lethargy", "restlessness"],
        "specialty": "Emergency Medicine / Infectious Disease",
        "suggested_action": "This is a medical emergency. Go to the hospital immediately. This could indicate dengue hemorrhagic fever or dengue shock syndrome.",
        "urgency": "Dangerous",
        "first_aid": "While waiting for emergency care, keep the person hydrated with small sips of oral rehydration solution. Do not give aspirin or NSAIDs.",
        "follow_up_questions": ["Is there any visible bleeding?", "When was the last time they urinated?", "Are they conscious and responsive?"],
        "stage": "Dangerous Stage",
        "condition": "Dengue Fever"
    },
    {
        "symptoms": ["fever", "stiff neck", "headache", "confusion", "light sensitivity"],
        "specialty": "Emergency Medicine / Neurology",
        "suggested_action": "Seek emergency medical care immediately. These could be signs of meningitis.",
        "urgency": "Emergency",
        "first_aid": "Keep the person in a quiet, dark room. Do not give anything by mouth if they are confused or vomiting.",
        "follow_up_questions": ["When did the neck stiffness start?", "Is there a rash anywhere on the body?", "Any recent head injury?"],
        "condition": "Possible Meningitis"
    },
    {
        "symptoms": ["fever", "sore throat", "swollen lymph nodes", "fatigue", "loss of appetite"],
        "specialty": "Primary Care / Infectious Disease",
        "suggested_action": "Schedule an appointment with your doctor. This could be mononucleosis or another viral infection.",
        "urgency": "Routine",
        "first_aid": "Rest, hydrate, and use throat lozenges or salt water gargles for throat pain.",
        "follow_up_questions": ["How long have you felt fatigued?", "Is there any abdominal pain?", "Have you had recent contact with anyone who was sick?"],
        "condition": "Viral Infection"
    }
]

# A separate database for doctor profiles
doctor_profiles = [
    {
        "name": "Dr. Aruna Reddy",
        "specialty": "Cardiology",
        "contact": "+91-9876543210",
        "location": "Apolo Hospital, Delhi"
    },
    {
        "name": "Dr. Sameer Khan",
        "specialty": "Gastroenterology",
        "contact": "+91-9988776655",
        "location": "Max Healthcare, Delhi"
    },
    {
        "name": "Dr. Priya Sharma",
        "specialty": "Dermatology",
        "contact": "+91-9012345678",
        "location": "Fortis Hospital, Mumbai"
    },
    {
        "name": "Dr. Rajiv Gupta",
        "specialty": "Neurology",
        "contact": "+91-9765432109",
        "location": "AIIMS, Delhi"
    },
    {
        "name": "Dr. Pooja Singh",
        "specialty": "General Practice",
        "contact": "+91-9123456789",
        "location": "City Clinic, Delhi"
    },
    {
        "name": "Dr. Vikram Malhotra",
        "specialty": "Infectious Disease",
        "contact": "+91-9234567890",
        "location": "Medanta Hospital, Delhi"
    },
    {
        "name": "Dr. Anjali Mehta",
        "specialty": "Emergency Medicine",
        "contact": "+91-9345678901",
        "location": "Fortis Hospital, Mumbai"
    },
    {
        "name": "Dr. Mohammad Rahman",
        "specialty": "Infectious Disease",
        "contact": "+880-1712345678",
        "location": "Dhaka Medical College Hospital, Dhaka"
    },
    {
        "name": "Dr. Fatima Begum",
        "specialty": "General Practice",
        "contact": "+880-1812345678",
        "location": "Popular Diagnostic Centre, Dhaka"
    },
    {
        "name": "Dr. Abdul Hamid",
        "specialty": "Emergency Medicine",
        "contact": "+880-1912345678",
        "location": "Ibn Sina Hospital, Dhaka"
    }
]

def create_knowledge_base():
    """
    Creates a JSON file containing the medical knowledge data.
    """
    data = {
        "medical_data": medical_data,
        "doctor_profiles": doctor_profiles,
        "symptom_weights": symptom_weights,
        "symptom_mapping": symptom_mapping,
        "dengue_prone_areas": dengue_prone_areas,
        "chikungunya_prone_areas": chikungunya_prone_areas,
        "malaria_prone_areas": malaria_prone_areas
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(KNOWLEDGE_BASE_PATH), exist_ok=True)
    
    with open(KNOWLEDGE_BASE_PATH, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"✅ Knowledge base created successfully at '{KNOWLEDGE_BASE_PATH}' with {len(medical_data)} medical entries and {len(doctor_profiles)} doctor profiles.")

if __name__ == "__main__":
    create_knowledge_base()