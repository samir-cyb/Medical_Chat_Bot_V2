import os
from django.conf import settings

# Define paths relative to the medical_ai directory
MEDICAL_AI_DIR = os.path.join(settings.BASE_DIR, 'chatbot_app', 'medical_ai')
KNOWLEDGE_BASE_PATH = os.path.join(MEDICAL_AI_DIR, "medical_knowledge.json")
PERSIST_DIRECTORY = os.path.join(MEDICAL_AI_DIR, "chroma_medical_db")