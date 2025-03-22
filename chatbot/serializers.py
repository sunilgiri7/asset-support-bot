# chatbot/serializers.py
from rest_framework import serializers
from .models import Conversation, Message

class MessageSerializer(serializers.ModelSerializer):
    """Serializer for chat messages"""
    class Meta:
        model = Message
        fields = ['id', 'is_user', 'content', 'created_at']
        read_only_fields = ['id', 'created_at']

class ConversationSerializer(serializers.ModelSerializer):
    """Serializer for conversations"""
    messages = MessageSerializer(many=True, read_only=True)
    
    class Meta:
        model = Conversation
        fields = ['id', 'asset_id', 'messages', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']

class QuerySerializer(serializers.Serializer):
    """Serializer for chat queries"""
    asset_id = serializers.CharField(required=True)
    message = serializers.CharField(required=True)
    conversation_id = serializers.UUIDField(required=False)