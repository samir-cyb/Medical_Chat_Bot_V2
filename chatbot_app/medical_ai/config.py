import os
import sys

try:
    from django.conf import settings
    # Django context
    BASE_DIR = settings.BASE_DIR
    print(f"✅ Config loaded in Django context. BASE_DIR: {BASE_DIR}")
except ImportError:
    # Standalone context (for create_knowledge_base.py and train_model.py)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"✅ Config loaded in standalone context. BASE_DIR: {BASE_DIR}")

# Define paths relative to the medical_ai directory
MEDICAL_AI_DIR = os.path.join(BASE_DIR, 'chatbot_app', 'medical_ai')
KNOWLEDGE_BASE_PATH = os.path.join(MEDICAL_AI_DIR, "medical_knowledge.json")
PERSIST_DIRECTORY = os.path.join(MEDICAL_AI_DIR, "chroma_medical_db")

# Debug information
print(f"🔍 DEBUG - MEDICAL_AI_DIR: {MEDICAL_AI_DIR}")
print(f"🔍 DEBUG - KNOWLEDGE_BASE_PATH: {KNOWLEDGE_BASE_PATH}")
print(f"🔍 DEBUG - KNOWLEDGE_BASE_PATH exists: {os.path.exists(KNOWLEDGE_BASE_PATH) if os.path.exists(MEDICAL_AI_DIR) else 'MEDICAL_AI_DIR does not exist'}")
print(f"🔍 DEBUG - PERSIST_DIRECTORY: {PERSIST_DIRECTORY}")
print(f"🔍 DEBUG - PERSIST_DIRECTORY exists: {os.path.exists(PERSIST_DIRECTORY) if os.path.exists(MEDICAL_AI_DIR) else 'MEDICAL_AI_DIR does not exist'}")

if os.path.exists(MEDICAL_AI_DIR):
    print(f"📁 Contents of MEDICAL_AI_DIR: {os.listdir(MEDICAL_AI_DIR)}")
    if os.path.exists(PERSIST_DIRECTORY):
        print(f"📁 Contents of PERSIST_DIRECTORY: {os.listdir(PERSIST_DIRECTORY)}")