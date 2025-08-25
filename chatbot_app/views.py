import json
import uuid
import sys
import os
import threading
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from django.conf import settings

# Add the medical_ai directory to the Python path
MEDICAL_AI_DIR = os.path.join(settings.BASE_DIR, 'chatbot_app', 'medical_ai')
sys.path.insert(0, MEDICAL_AI_DIR)

print(f"✅ Added MEDICAL_AI_DIR to Python path: {MEDICAL_AI_DIR}")
print(f"✅ Current Python path: {sys.path}")

from .models import ChatSession, ChatMessage

# Thread-safe chatbot instance
chatbot_instance = None
chatbot_lock = threading.Lock()

def get_chatbot():
    global chatbot_instance
    with chatbot_lock:
        if chatbot_instance is None:
            print("🔄 Initializing chatbot...")
            print(f"🔍 DEBUG - Current working directory: {os.getcwd()}")
            print(f"🔍 DEBUG - MEDICAL_AI_DIR: {MEDICAL_AI_DIR}")
            print(f"🔍 DEBUG - MEDICAL_AI_DIR exists: {os.path.exists(MEDICAL_AI_DIR)}")
            
            if os.path.exists(MEDICAL_AI_DIR):
                print(f"📁 Contents of MEDICAL_AI_DIR: {os.listdir(MEDICAL_AI_DIR)}")
            
            try:
                # Import the MedicalChatBot class
                print("🔍 Attempting to import MedicalChatBot...")
                from .medical_ai.chat_bot import MedicalChatBot
                print("✅ Successfully imported MedicalChatBot")
                
                # Create chatbot instance
                chatbot_instance = MedicalChatBot()
                print("✅ Created MedicalChatBot instance")
                
                # Initialize the bot with detailed debugging
                print("🔄 Initializing bot...")
                initialized = chatbot_instance.initialize_bot()
                
                if not initialized:
                    print("❌ Failed to initialize chatbot")
                    
                    # Detailed debugging
                    print("🔍 DEBUG - Checking vector database...")
                    if hasattr(chatbot_instance, 'vectordb') and chatbot_instance.vectordb:
                        try:
                            test_results = chatbot_instance.vectordb.similarity_search("fever", k=1)
                            print(f"✅ Vector DB test query returned {len(test_results)} results")
                            if test_results:
                                print(f"📄 First result: {test_results[0].page_content[:100]}...")
                        except Exception as e:
                            print(f"❌ Vector DB test failed: {e}")
                    else:
                        print("❌ Vector DB not initialized")
                    
                    print("🔍 DEBUG - Checking LLM...")
                    if hasattr(chatbot_instance, 'llm') and chatbot_instance.llm:
                        try:
                            test_response = chatbot_instance.llm("Hello")
                            print(f"✅ LLM test response: {test_response}")
                        except Exception as e:
                            print(f"❌ LLM test failed: {e}")
                    else:
                        print("❌ LLM not initialized")
                    
                    print("🔍 DEBUG - Checking knowledge base...")
                    if hasattr(chatbot_instance, 'medical_data'):
                        print(f"✅ Medical data loaded: {len(chatbot_instance.medical_data)} entries")
                    else:
                        print("❌ Medical data not loaded")
                    
                    return None
                
                print("✅ Chatbot initialized successfully")
                print(f"🔍 DEBUG - Chatbot attributes: {[attr for attr in dir(chatbot_instance) if not attr.startswith('_')]}")
                
            except ImportError as e:
                print(f"❌ Import error: {e}")
                print(f"🔍 DEBUG - Python path: {sys.path}")
                print(f"🔍 DEBUG - Files in medical_ai: {os.listdir(MEDICAL_AI_DIR) if os.path.exists(MEDICAL_AI_DIR) else 'Directory not found'}")
                import traceback
                traceback.print_exc()
                return None
            except Exception as e:
                print(f"❌ Error creating chatbot instance: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            print(f"✅ Using existing chatbot instance")
            print(f"🔍 Current chatbot state - is_medical_chat: {chatbot_instance.is_medical_chat}, last_question: '{chatbot_instance.last_question}'")
        
        
    
    return chatbot_instance

def index(request):
    print("📄 Rendering index page")
    return render(request, 'chatbot_app/index.html')

@csrf_exempt
def debug_chatbot_state(request):
    """Debug endpoint to check chatbot state"""
    chatbot = get_chatbot()
    if chatbot:
        state = {
            'last_question': chatbot.last_question,
            'is_medical_chat': chatbot.is_medical_chat,
            'chat_history_length': len(chatbot.chat_history),
            'user_symptoms': chatbot.user_symptoms,
            'asked_questions': list(chatbot.asked_questions)
        }
        return JsonResponse(state)
    return JsonResponse({'error': 'Chatbot not initialized'})

def chat(request):
    print("💬 Chat view called")
    # Create or get session
    session_id = request.session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id
        
        # Create new chat session
        chat_session = ChatSession.objects.create(
            session_id=session_id,
            user=request.user if request.user.is_authenticated else None
        )
        print(f"🆕 Created new chat session: {session_id}")
    else:
        try:
            chat_session = ChatSession.objects.get(session_id=session_id)
            print(f"📖 Found existing chat session: {session_id}")
        except ChatSession.DoesNotExist:
            session_id = str(uuid.uuid4())
            request.session['chat_session_id'] = session_id
            chat_session = ChatSession.objects.create(
                session_id=session_id,
                user=request.user if request.user.is_authenticated else None
            )
            print(f"🆕 Recreated chat session: {session_id}")
    
    # Get chat history
    messages = chat_session.messages.all()
    print(f"📋 Loaded {messages.count()} messages for session {session_id}")
    
    return render(request, 'chatbot_app/chat.html', {
        'messages': messages,
        'session_id': session_id
    })

@csrf_exempt
@require_http_methods(["POST"])
def send_message(request):
    print("📨 send_message endpoint called")
    try:
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', '')
        
        print(f"🔍 Received message: '{user_message}' for session: {session_id}")
        
        if not user_message or not session_id:
            print("❌ Invalid request - missing message or session_id")
            return JsonResponse({'error': 'Invalid request'}, status=400)
        
        try:
            chat_session = ChatSession.objects.get(session_id=session_id)
            print(f"✅ Found chat session: {session_id}")
        except ChatSession.DoesNotExist:
            print(f"❌ Session not found: {session_id}")
            return JsonResponse({'error': 'Session not found'}, status=404)
        
        # Save user message
        user_msg = ChatMessage.objects.create(
            session=chat_session,
            message=user_message,
            is_bot=False
        )
        print(f"💾 Saved user message: {user_message[:50]}...")
        
        # Get bot response
        chatbot = get_chatbot()
        if chatbot is None:
            error_msg = "Chatbot is not available. Please try again later."
            print("❌ Chatbot is None")
            bot_msg = ChatMessage.objects.create(
                session=chat_session,
                message=error_msg,
                is_bot=True
            )
            return JsonResponse({
                'user_message': user_message,
                'bot_response': error_msg,
                'should_continue': False
            })
        
        print("🤖 Processing message with chatbot...")
        try:
            # Determine query type first
            print(f"🔍 Determining query type for: '{user_message}'")
            query_type = chatbot.get_query_type(user_message)
            print(f"✅ Query type: {query_type}")
            
            if query_type == 'informational':
                print("📚 Handling informational question")
                bot_response = chatbot.get_informational_response(user_message)
                should_continue = False
                
            elif query_type == 'medical':
                print("🏥 Handling medical query")
                bot_response, should_continue = chatbot.process_medical_query(user_message)
                
            elif query_type in ['greeting', 'casual']:
                print("👋 Handling casual conversation")
                bot_response = chatbot.get_casual_response(user_message)
                should_continue = False
                
            elif query_type == 'exit':
                print("👋 Handling exit command")
                bot_response = "Thank you for chatting. Please take care of your health!"
                should_continue = False
                
            else:
                print("❓ Handling unknown query type")
                bot_response = "I'm here to help with medical concerns. Please describe any symptoms or health issues you're experiencing."
                should_continue = False
                
            print(f"✅ Bot response generated: {bot_response[:100]}...")
            print(f"✅ Should continue: {should_continue}")
            
        except Exception as e:
            error_msg = f"Error processing your message: {str(e)}"
            bot_response = error_msg
            should_continue = False
            print(f"❌ Error in message processing: {e}")
            import traceback
            traceback.print_exc()
        
        # Save bot response
        bot_msg = ChatMessage.objects.create(
            session=chat_session,
            message=bot_response,
            is_bot=True
        )
        print(f"💾 Saved bot response")
        
        # Update session timestamp
        chat_session.updated_at = timezone.now()
        chat_session.save()
        
        return JsonResponse({
            'user_message': user_message,
            'bot_response': bot_response,
            'should_continue': should_continue
        })
        
    except Exception as e:
        print(f"❌ Unexpected error in send_message: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': 'Internal server error'}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def clear_chat(request):
    print("🗑️ clear_chat endpoint called")
    session_id = request.session.get('chat_session_id')
    if session_id:
        try:
            chat_session = ChatSession.objects.get(session_id=session_id)
            message_count = chat_session.messages.count()
            chat_session.messages.all().delete()
            print(f"🧹 Cleared {message_count} messages from session {session_id}")
            
            # Reset the chatbot state
            chatbot = get_chatbot()
            if chatbot and hasattr(chatbot, 'reset_conversation'):
                chatbot.reset_conversation()
                print("🔄 Reset chatbot conversation")
        except ChatSession.DoesNotExist:
            print(f"❌ Session not found for clearing: {session_id}")
            pass
    
    return JsonResponse({'status': 'success'})


def debug_chatbot_status(request):
    """Debug endpoint to check chatbot status"""
    chatbot = get_chatbot()
    
    status = {
        'chatbot_available': chatbot is not None,
        'chatbot_initialized': chatbot.initialized if chatbot else False,
        'vector_db_exists': os.path.exists(PERSIST_DIRECTORY),
        'knowledge_base_exists': os.path.exists(KNOWLEDGE_BASE_PATH),
        'medical_ai_dir_exists': os.path.exists(MEDICAL_AI_DIR),
        'medical_data_loaded': len(chatbot.medical_data) if chatbot and hasattr(chatbot, 'medical_data') else 0,
        'vector_db_working': False,
        'llm_working': False
    }
    
    if chatbot:
        # Test vector DB
        try:
            test_results = chatbot.vectordb.similarity_search("fever", k=1)
            status['vector_db_working'] = len(test_results) > 0
        except:
            status['vector_db_working'] = False
        
        # Test LLM
        try:
            test_response = chatbot.llm("Hello")
            status['llm_working'] = True
        except:
            status['llm_working'] = False
    
    return JsonResponse(status)