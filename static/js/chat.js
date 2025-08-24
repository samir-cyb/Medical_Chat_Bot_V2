document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const clearChatButton = document.getElementById('clear-chat');
    
    // Scroll to bottom of chat
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    scrollToBottom();
    
    // Send message function
    function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        addMessageToChat(message, false);
        userInput.value = '';
        
        // Send to server
        fetch('/api/send_message/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify({
                message: message,
                session_id: sessionId
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                addMessageToChat('Sorry, I encountered an error. Please try again.', true);
            } else {
                addMessageToChat(data.bot_response, true);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            addMessageToChat('Sorry, I encountered an error. Please try again.', true);
        });
    }
    
    // Add message to chat UI
    function addMessageToChat(message, isBot) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isBot ? 'bot-message' : 'user-message'}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Convert line breaks to HTML
        contentDiv.innerHTML = message.replace(/\n/g, '<br>');
        
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        
        scrollToBottom();
    }
    
    // Clear chat function
    function clearChat() {
        if (confirm('Are you sure you want to clear the chat?')) {
            fetch('/api/clear_chat/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken
                },
                body: JSON.stringify({})
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    chatMessages.innerHTML = '';
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
        }
    }
    
    // Event listeners
    sendButton.addEventListener('click', sendMessage);
    
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    clearChatButton.addEventListener('click', clearChat);
});