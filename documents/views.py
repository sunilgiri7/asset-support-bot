# documents/views.py
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Document
from .serializers import DocumentSerializer
from .tasks import process_document
from asset_support_bot.utils.pinecone_client import PineconeClient
from rest_framework.permissions import AllowAny

class DocumentViewSet(viewsets.ModelViewSet):
    permission_classes = [AllowAny]
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    
    def get_queryset(self):
        """Filter documents by asset_id if provided"""
        queryset = Document.objects.all()
        asset_id = self.request.query_params.get('asset_id')
        if asset_id:
            queryset = queryset.filter(asset_id=asset_id)
        return queryset
    
    def perform_create(self, serializer):
        """Handle file upload and trigger document processing"""
        # Save the document
        document = serializer.save()
        print("documents-------->", document)
        
        # Trigger asynchronous processing
        process_document.delay(str(document.id))
        
        return document
    
    def perform_destroy(self, instance):
        """Delete document from database and vector store"""
        try:
            # Delete from vector store first
            pinecone_client = PineconeClient()
            pinecone_client.delete_document(str(instance.id))
            
            # Then delete the model instance
            instance.delete()
        except Exception as e:
            return Response(
                {"error": f"Failed to delete document: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['get'])
    def by_asset(self, request):
        """Get all documents for a specific asset"""
        asset_id = request.query_params.get('asset_id')
        if not asset_id:
            return Response(
                {"error": "asset_id query parameter is required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        documents = Document.objects.filter(asset_id=asset_id)
        serializer = self.get_serializer(documents, many=True)
        return Response(serializer.data)