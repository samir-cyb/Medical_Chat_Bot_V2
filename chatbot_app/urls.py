from django.urls import path
from . import views

app_name = 'chatbot_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('chat/', views.chat, name='chat'),
    path('api/send_message/', views.send_message, name='send_message'),
    path('api/clear_chat/', views.clear_chat, name='clear_chat'),
]