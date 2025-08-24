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

from .models import ChatSession, ChatMessage

# Thread-safe chatbot instance
chatbot_instance = None
chatbot_lock = threading.Lock()

def get_chatbot():
    global chatbot_instance
    with chatbot_lock:
        if chatbot_instance is None:
            print("Initializing chatbot...")
            
            try:
                # Import the MedicalChatBot class
                from .medical_ai.chat_bot import MedicalChatBot
                
                # Create chatbot instance
                chatbot_instance = MedicalChatBot()
                
                # Initialize the bot
                initialized = chatbot_instance.initialize_bot()
                if not initialized:
                    print("Failed to initialize chatbot")
                    return None
                print("Chatbot initialized successfully")
                
            except Exception as e:
                print(f"Error importing chatbot: {e}")
                import traceback
                traceback.print_exc()
                return None
    
    return chatbot_instance

def index(request):
    return render(request, 'chatbot_app/index.html')

def chat(request):
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
    else:
        try:
            chat_session = ChatSession.objects.get(session_id=session_id)
        except ChatSession.DoesNotExist:
            session_id = str(uuid.uuid4())
            request.session['chat_session_id'] = session_id
            chat_session = ChatSession.objects.create(
                session_id=session_id,
                user=request.user if request.user.is_authenticated else None
            )
    
    # Get chat history
    messages = chat_session.messages.all()
    
    return render(request, 'chatbot_app/chat.html', {
        'messages': messages,
        'session_id': session_id
    })

@csrf_exempt
@require_http_methods(["POST"])
def send_message(request):
    data = json.loads(request.body)
    user_message = data.get('message', '').strip()
    session_id = data.get('session_id', '')
    
    if not user_message or not session_id:
        return JsonResponse({'error': 'Invalid request'}, status=400)
    
    try:
        chat_session = ChatSession.objects.get(session_id=session_id)
    except ChatSession.DoesNotExist:
        return JsonResponse({'error': 'Session not found'}, status=404)
    
    # Save user message
    user_msg = ChatMessage.objects.create(
        session=chat_session,
        message=user_message,
        is_bot=False
    )
    
    # Get bot response
    chatbot = get_chatbot()
    if chatbot is None:
        error_msg = "Chatbot is not available. Please try again later."
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
    
    try:
        # NEW: Determine query type first
        query_type = chatbot.get_query_type(user_message)
        
        if query_type == 'informational':
            # Handle informational questions
            bot_response = chatbot.get_informational_response(user_message)
            should_continue = False
            
        elif query_type == 'medical':
            # Handle medical symptom descriptions
            bot_response, should_continue = chatbot.process_medical_query(user_message)
            
        elif query_type in ['greeting', 'casual']:
            # Handle casual conversation
            bot_response = chatbot.get_casual_response(user_message)
            should_continue = False
            
        elif query_type == 'exit':
            bot_response = "Thank you for chatting. Please take care of your health!"
            should_continue = False
            
        else:
            # Default fallback
            bot_response = "I'm here to help with medical concerns. Please describe any symptoms or health issues you're experiencing."
            should_continue = False
            
    except Exception as e:
        error_msg = f"Error processing your message: {str(e)}"
        bot_response = error_msg
        should_continue = False
        print(f"Error in message processing: {e}")
        import traceback
        traceback.print_exc()
    
    # Save bot response
    bot_msg = ChatMessage.objects.create(
        session=chat_session,
        message=bot_response,
        is_bot=True
    )
    
    # Update session timestamp
    chat_session.updated_at = timezone.now()
    chat_session.save()
    
    return JsonResponse({
        'user_message': user_message,
        'bot_response': bot_response,
        'should_continue': should_continue
    })

@csrf_exempt
@require_http_methods(["POST"])
def clear_chat(request):
    session_id = request.session.get('chat_session_id')
    if session_id:
        try:
            chat_session = ChatSession.objects.get(session_id=session_id)
            chat_session.messages.all().delete()
            # Reset the chatbot state
            chatbot = get_chatbot()
            if chatbot and hasattr(chatbot, 'reset_conversation'):
                chatbot.reset_conversation()
        except ChatSession.DoesNotExist:
            pass
    
    return JsonResponse({'status': 'success'})