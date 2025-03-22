# documents/models.py
from django.db import models
import uuid

class Document(models.Model):
    """Model to store document metadata"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    asset_id = models.CharField(max_length=100, db_index=True, help_text="Unique ID of the asset this document belongs to")
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='documents/')
    content_type = models.CharField(max_length=100, blank=True)
    file_size = models.IntegerField(default=0)
    is_processed = models.BooleanField(default=False, help_text="Whether document has been processed for embeddings")
    processing_error = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.title} - Asset: {self.asset_id}"