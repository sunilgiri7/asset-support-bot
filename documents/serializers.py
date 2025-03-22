# documents/serializers.py
from rest_framework import serializers
from .models import Document

class DocumentSerializer(serializers.ModelSerializer):
    """Serializer for Document model"""
    class Meta:
        model = Document
        fields = [
            'id', 'asset_id', 'title', 'file', 'content_type', 
            'file_size', 'is_processed', 'processing_error', 
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'is_processed', 'processing_error', 'created_at', 'updated_at']
    
    def validate_file(self, value):
        """Validate file size and type"""
        from django.conf import settings
        
        # Check file size
        if value.size > settings.MAX_UPLOAD_SIZE:
            raise serializers.ValidationError(
                f"File size exceeds the limit of {settings.MAX_UPLOAD_SIZE / (1024 * 1024)} MB"
            )
        
        # Check file type (optional)
        allowed_types = [
            'application/pdf', 
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'text/plain'
        ]
        
        if value.content_type not in allowed_types:
            raise serializers.ValidationError(
                "Unsupported file type. Please upload PDF, DOCX, DOC, or TXT files."
            )
        
        return value
    
    def create(self, validated_data):
        """Handle file upload data"""
        file = validated_data.get('file')
        
        # Set content type and file size
        if file:
            validated_data['content_type'] = file.content_type
            validated_data['file_size'] = file.size
        
        return super().create(validated_data)