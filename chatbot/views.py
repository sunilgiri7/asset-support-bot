# chatbot/views.py
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from chatbot.models import Conversation, Message
from chatbot.serializers import ConversationSerializer, MessageSerializer, QuerySerializer
from asset_support_bot.utils.pinecone_client import PineconeClient
from chatbot.utils.llm_client import MistralLLMClient  # or your APIBasedLLMClient if updated
import logging
from rest_framework.permissions import AllowAny

logger = logging.getLogger(__name__)

class ChatbotViewSet(viewsets.ViewSet):
    
    @action(detail=False, methods=['post'])
    def query(self, request):
        serializer = QuerySerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        # Extract data
        asset_id = serializer.validated_data['asset_id']
        message_content = serializer.validated_data['message']
        conversation_id = serializer.validated_data.get('conversation_id')
        
        try:
            # Get or create conversation using the corrected method name
            conversation = self._get_or_create_conversation(conversation_id, asset_id)
            logger.info(f"conversation--------> {conversation}")
            
            # Save user message
            user_message = Message.objects.create(
                conversation=conversation,
                is_user=True,
                content=message_content
            )
            logger.info(f"user_message------> {user_message}")
            
            # Retrieve contextual chunks from Pinecone using corrected method name
            context_chunks = self._retrieve_context_chunks(message_content, asset_id)
            logger.info(f"context_chunks-----> {context_chunks}")
            
            # Prepare context for LLM using corrected method name
            context = self._format_context(context_chunks)
            logger.info(f"context--------> {context}")
            
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
            
            # Return the messages
            return Response({
                "conversation_id": conversation.id,
                "user_message": MessageSerializer(user_message).data,
                "assistant_message": MessageSerializer(system_message).data,
                "context_used": bool(context_chunks)
            })
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return Response(
                {"error": f"Failed to process query: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    # Updated helper methods with underscore prefixes for clarity
    def _get_or_create_conversation(self, conversation_id, asset_id):
        if conversation_id:
            try:
                return Conversation.objects.get(id=conversation_id)
            except Conversation.DoesNotExist:
                logger.warning(f"Conversation {conversation_id} not found")
        return Conversation.objects.create(asset_id=asset_id)
    
    def _retrieve_context_chunks(self, query, asset_id, top_k=3):
        try:
            pinecone_client = PineconeClient()
            
            # Log debugging information
            logger.info(f"Retrieving context for query: {query}")
            logger.info(f"Asset ID: {asset_id}")
            
            # Retrieve similar chunks
            context_chunks = pinecone_client.query_similar_chunks(
                query_text=query,
                asset_id=str(asset_id),  # Ensure asset_id is a string
                top_k=top_k
            )
            
            # Additional logging
            logger.info(f"Retrieved context chunks: {context_chunks}")
            
            return context_chunks
        except Exception as e:
            logger.error(f"Error retrieving context chunks: {str(e)}")
            return []
    
    def _format_context(self, context_chunks):
        if not context_chunks:
            return ""
        formatted_context = "Relevant Context:\n"
        for i, chunk in enumerate(context_chunks, 1):
            formatted_context += f"{i}. {chunk['text']} (Relevance Score: {chunk['score']:.2f})\n"
        return formatted_context
