# train_model.py (UPDATED)
import json
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from .config import KNOWLEDGE_BASE_PATH, PERSIST_DIRECTORY

def create_vector_database():
    """
    Loads the medical knowledge JSON and stores it in a Chroma vector database.
    Uses HuggingFace embeddings compatible with LangChain.
    """
    # 1. Load the JSON file
    with open(KNOWLEDGE_BASE_PATH, 'r') as f:
        full_data = json.load(f)
    
    medical_data = full_data["medical_data"]
    doctor_profiles = full_data["doctor_profiles"]
    symptom_weights = full_data["symptom_weights"]
    symptom_mapping = full_data["symptom_mapping"]
    dengue_prone_areas = full_data["dengue_prone_areas"]
    chikungunya_prone_areas = full_data["chikungunya_prone_areas"]
    
    print(f"Loaded {len(medical_data)} medical entries and {len(doctor_profiles)} doctor profiles.")
    print(f"Loaded {len(symptom_weights)} symptom weights and {len(symptom_mapping)} symptom mappings.")
    print(f"Loaded {len(dengue_prone_areas)} dengue-prone areas and {len(chikungunya_prone_areas)} chikungunya-prone areas.")

    # 2. Create embedding function with error handling
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        print("Falling back to default sentence transformer model...")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False}
            )
        except Exception as e2:
            print(f"Failed to create embeddings: {e2}")
            return

    # 3. Prepare documents and metadata for medical data
    documents = []
    
    for i, entry in enumerate(medical_data):
        # Convert the list of questions to a single string
        follow_up_questions_str = ', '.join(entry.get('follow_up_questions', []))

        document_text = (
            f"Symptoms: {', '.join(entry['symptoms'])}. "
            f"Specialty: {entry['specialty']}. "
            f"Suggested Action: {entry['suggested_action']}. "
            f"Urgency: {entry['urgency']}. "
            f"First Aid: {entry.get('first_aid', 'Not specified')}. "
            f"Follow-up Questions: {follow_up_questions_str}. "
        )
        
        # Add stage and condition information if available
        if 'stage' in entry:
            document_text += f"Stage: {entry['stage']}. "
        if 'condition' in entry:
            document_text += f"Condition: {entry['condition']}. "
        
        # Calculate symptom weights for this condition
        condition_weights = sum(symptom_weights.get(symptom, 5) for symptom in entry['symptoms'])
        
        doc = Document(page_content=document_text, metadata={
            "symptoms": str(entry['symptoms']),
            "specialty": entry['specialty'],
            "suggested_action": entry['suggested_action'],
            "urgency": entry['urgency'],
            "first_aid": entry.get('first_aid', 'Not specified'),
            "follow_up_questions": follow_up_questions_str,
            "stage": entry.get('stage', 'Not specified'),
            "condition": entry.get('condition', 'Not specified'),
            "symptom_weight_score": condition_weights
        })
        documents.append(doc)

    # 4. Prepare documents for doctor profiles
    for i, profile in enumerate(doctor_profiles):
        document_text = (
            f"Doctor Profile: Dr. {profile['name']}. "
            f"Specialty: {profile['specialty']}. "
            f"Contact: {profile['contact']}. "
            f"Location: {profile['location']}."
        )
        
        doc = Document(page_content=document_text, metadata={
            "name": profile['name'],
            "specialty": profile['specialty'],
            "contact": profile['contact'],
            "location": profile['location']
        })
        documents.append(doc)

    # 5. Create and persist the vector database
    # Create directory if it doesn't exist
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    
    try:
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY,
            collection_name="medical_knowledge"
        )

        print(f"✅ Added {len(documents)} documents to vector database.")
        print(f"✅ Vector database created and persisted at '{PERSIST_DIRECTORY}'.")
    
    except Exception as e:
        print(f"Error creating vector database: {e}")
        print("This might be due to version incompatibilities.")
        print("Please try: pip install numpy==1.24.3 scikit-learn==1.3.2")

if __name__ == "__main__":
    create_vector_database()