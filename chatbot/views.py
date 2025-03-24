# chatbot/views.py
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Conversation, Message
from .serializers import ConversationSerializer, MessageSerializer, QuerySerializer
from .utils.llm_client import MistralLLMClient
from asset_support_bot.utils.pinecone_client import PineconeClient
import logging
from rest_framework.permissions import AllowAny

logger = logging.getLogger(__name__)

class ConversationViewSet(viewsets.ModelViewSet):
    queryset = Conversation.objects.all()
    serializer_class = ConversationSerializer
    
    def get_queryset(self):
        """Filter conversations by asset_id if provided"""
        queryset = Conversation.objects.all()
        asset_id = self.request.query_params.get('asset_id')
        if asset_id:
            queryset = queryset.filter(asset_id=asset_id)
        return queryset

class ChatbotViewSet(viewsets.ViewSet):
    
    @action(detail=False, methods=['post'])
    def query(self, request):
        """Handle user queries"""
        serializer = QuerySerializer(data=request.data)
        print("chabot", serializer)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        # Extract data
        asset_id = serializer.validated_data['asset_id']
        message_content = serializer.validated_data['message']
        conversation_id = serializer.validated_data.get('conversation_id')
        
        try:
            # Get or create conversation
            if conversation_id:
                try:
                    conversation = Conversation.objects.get(id=conversation_id)
                    print("conversation", conversation)
                except Conversation.DoesNotExist:
                    return Response(
                        {"error": f"Conversation with ID {conversation_id} not found"},
                        status=status.HTTP_404_NOT_FOUND
                    )
            else:
                conversation = Conversation.objects.create(asset_id=asset_id)
            
            # Save user message
            user_message = Message.objects.create(
                conversation=conversation,
                is_user=True,
                content=message_content
            )
            print("user_message", user_message)
            
            # Get relevant context from Pinecone
            pinecone_client = PineconeClient()
            relevant_chunks = pinecone_client.query_similar_chunks(
                query_text=message_content,
                asset_id=asset_id,
                top_k=3
            )
            
            # Prepare context for LLM
            context = ""
            if relevant_chunks:
                context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
                logger.info(f"Found {len(relevant_chunks)} relevant chunks for query")
            else:
                logger.info("No relevant chunks found for query")
            
            # Generate response using LLM
            llm_client = MistralLLMClient()
            response_content = llm_client.generate_response(
                prompt=message_content,
                context=context
            )
            
            # Save assistant response
            system_message = Message.objects.create(
                conversation=conversation,
                is_user=False,
                content=response_content
            )
            
            # Update conversation's updated_at timestamp
            conversation.save()
            
            # Return the messages
            return Response({
                "conversation_id": conversation.id,
                "user_message": MessageSerializer(user_message).data,
                "assistant_message": MessageSerializer(system_message).data
            })
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return Response(
                {"error": f"Failed to process query: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )