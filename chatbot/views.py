from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from chatbot.models import Conversation, Message
from chatbot.serializers import ConversationSerializer, MessageSerializer, QuerySerializer
from asset_support_bot.utils.pinecone_client import PineconeClient
from chatbot.utils.llm_client import MistralLLMClient  # or your APIBasedLLMClient if updated
import logging
from rest_framework.permissions import AllowAny
import time

logger = logging.getLogger(__name__)

class ChatbotViewSet(viewsets.ViewSet):
    permission_classes = [AllowAny]
    
    @action(detail=False, methods=['post'])
    def query(self, request):
        overall_start = time.perf_counter()

        serializer = QuerySerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        # Extract data
        asset_id = serializer.validated_data['asset_id']
        message_content = serializer.validated_data['message']
        conversation_id = serializer.validated_data.get('conversation_id')
        
        timings = {}

        try:
            # Get or create conversation and measure time
            conv_start = time.perf_counter()
            conversation = self._get_or_create_conversation(conversation_id, asset_id)
            timings['conversation_time'] = f"{time.perf_counter() - conv_start:.2f} seconds"
            
            # Create user message and measure time
            user_msg_start = time.perf_counter()
            user_message = Message.objects.create(
                conversation=conversation,
                is_user=True,
                content=message_content
            )
            timings['user_message_time'] = f"{time.perf_counter() - user_msg_start:.2f} seconds"

            # Retrieve context chunks from Pinecone and measure time
            context_start = time.perf_counter()
            context_chunks = self._retrieve_context_chunks(message_content, asset_id)
            timings['context_retrieval_time'] = f"{time.perf_counter() - context_start:.2f} seconds"
            
            # Format context for LLM
            context = self._format_context(context_chunks)
            logger.info(f"context--------> {context}")
            
            # Generate response using LLM and measure time
            llm_start = time.perf_counter()
            llm_client = MistralLLMClient()
            response_content = llm_client.generate_response(
                prompt=message_content,
                context=context
            )
            timings['llm_response_time'] = f"{time.perf_counter() - llm_start:.2f} seconds"
            
            # Save assistant response and measure time
            assist_msg_start = time.perf_counter()
            system_message = Message.objects.create(
                conversation=conversation,
                is_user=False,
                content=response_content
            )
            timings['assistant_message_save_time'] = f"{time.perf_counter() - assist_msg_start:.2f} seconds"
            
            overall_elapsed = time.perf_counter() - overall_start
            timings['total_time'] = f"{overall_elapsed:.2f} seconds"

            # Return the conversation messages, timings and unique conversation key
            return Response({
                "conversation_id": conversation.id,
                "user_message": MessageSerializer(user_message).data,
                "assistant_message": MessageSerializer(system_message).data,
                "context_used": bool(context_chunks),
                "response_time": f"{overall_elapsed:.2f} seconds",
                "timings": timings
            })
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return Response(
                {"error": f"Failed to process query: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['get'])
    def history(self, request):
        """
        Retrieve previous conversations.
        You can filter by:
          - conversation_id: to fetch a specific conversation, or
          - asset_id: to fetch all conversations for a specific asset.
        """
        conversation_id = request.query_params.get('conversation_id')
        asset_id = request.query_params.get('asset_id')
        
        if conversation_id:
            conversation = get_object_or_404(Conversation, id=conversation_id)
            serializer = ConversationSerializer(conversation)
            return Response(serializer.data)
        elif asset_id:
            conversations = Conversation.objects.filter(asset_id=asset_id).order_by('-updated_at')
            serializer = ConversationSerializer(conversations, many=True)
            return Response(serializer.data)
        else:
            return Response({"error": "Please provide either conversation_id or asset_id as query parameters."},
                            status=status.HTTP_400_BAD_REQUEST)
    
    def _get_or_create_conversation(self, conversation_id, asset_id):
        if conversation_id:
            try:
                return Conversation.objects.get(id=conversation_id)
            except Conversation.DoesNotExist:
                logger.warning(f"Conversation {conversation_id} not found; creating new one.")
        return Conversation.objects.create(asset_id=asset_id)
    
    def _retrieve_context_chunks(self, query, asset_id, top_k=3):
        try:
            pinecone_client = PineconeClient()
            logger.info(f"Retrieving context for query: {query} for asset_id: {asset_id}")
            context_chunks = pinecone_client.query_similar_chunks(
                query_text=query,
                asset_id=str(asset_id),  # Ensure asset_id is a string
                top_k=top_k
            )
            logger.info(f"Retrieved context chunks: {context_chunks}")
            return context_chunks
        except Exception as e:
            logger.error(f"Error retrieving context chunks: {str(e)}")
            return []
    
    def _format_context(self, context_chunks):
        if not context_chunks:
            return ""
        return "\n".join([
            f"Context Chunk {i+1}: {chunk['text']} (Relevance: {chunk['score']:.2f})" 
            for i, chunk in enumerate(context_chunks)
        ])
