from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from chatbot.models import Conversation, Message
from chatbot.serializers import ConversationSerializer, MessageSerializer, QuerySerializer
from asset_support_bot.utils.pinecone_client import PineconeClient
from chatbot.utils.llm_client import MistralLLMClient
import logging
from rest_framework.permissions import AllowAny
import time
import concurrent.futures
import threading

logger = logging.getLogger(__name__)

# Initialize PineconeClient at module level
try:
    pinecone_client = PineconeClient()
except Exception as e:
    logger.error(f"Failed to initialize PineconeClient: {str(e)}")
    pinecone_client = None

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
            # Get or create conversation immediately (this rarely fails)
            conv_start = time.perf_counter()
            conversation = self._get_or_create_conversation(conversation_id, asset_id)
            timings['conversation_time'] = f"{time.perf_counter() - conv_start:.2f} seconds"
            
            # Create user message immediately (this rarely fails)
            user_msg_start = time.perf_counter()
            user_message = Message.objects.create(
                conversation=conversation,
                is_user=True,
                content=message_content
            )
            timings['user_message_time'] = f"{time.perf_counter() - user_msg_start:.2f} seconds"
            
            # Start context retrieval with proper error handling
            context_start = time.perf_counter()
            context_chunks = []
            context_error = None
            
            try:
                # Only attempt retrieval if the PineconeClient was initialized successfully
                if pinecone_client is not None:
                    context_chunks = self._retrieve_context_chunks(message_content, asset_id)
                else:
                    context_error = "PineconeClient initialization failed"
            except Exception as e:
                context_error = str(e)
                logger.error(f"Error during context retrieval: {context_error}")
            
            timings['context_retrieval_time'] = f"{time.perf_counter() - context_start:.2f} seconds"
            
            # Format context for LLM or explain error
            if context_error:
                context = f"Note: Unable to retrieve context information. Error: {context_error}"
                logger.warning(f"Using empty context due to error: {context_error}")
            else:
                context = self._format_context(context_chunks)
                logger.info(f"Using context with {len(context_chunks)} chunks")
            
            # Generate response using LLM and measure time
            llm_start = time.perf_counter()
            llm_client = MistralLLMClient()
            
            # Append warning to prompt if context retrieval failed
            prompt = message_content
            if context_error:
                prompt += "\n\nNote: I couldn't access some relevant information due to a technical issue."
                
            response_content = llm_client.generate_response(
                prompt=prompt,
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
            
            # Return the conversation messages, timings and diagnostics
            response_data = {
                "conversation_id": conversation.id,
                "user_message": MessageSerializer(user_message).data,
                "assistant_message": MessageSerializer(system_message).data,
                "context_used": bool(context_chunks),
                "response_time": f"{overall_elapsed:.2f} seconds",
                "timings": timings
            }
            
            # Add error info if context retrieval failed
            if context_error:
                response_data["context_error"] = context_error
                
            return Response(response_data)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return Response(
                {"error": f"Failed to process query: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _get_or_create_conversation(self, conversation_id, asset_id):
        if conversation_id:
            try:
                return Conversation.objects.get(id=conversation_id)
            except Conversation.DoesNotExist:
                logger.warning(f"Conversation {conversation_id} not found; creating new one.")
                return Conversation.objects.create(asset_id=asset_id)
        return Conversation.objects.create(asset_id=asset_id)
    
    def _retrieve_context_chunks(self, query, asset_id, top_k=3):
        """Retrieve context chunks with proper error handling."""
        global pinecone_client
        
        if pinecone_client is None:
            logger.error("PineconeClient is not initialized")
            return []
            
        logger.info(f"Retrieving context for query: {query} for asset_id: {asset_id}")
        
        # Apply preprocessing to query text for better results
        query = self._preprocess_query(query)
        
        # Try using vector similarity search
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(pinecone_client.query_similar_chunks, 
                                    query_text=query, 
                                    asset_id=str(asset_id),
                                    top_k=top_k)
                context_chunks = future.result(timeout=5.0)  # 5-second timeout
            if context_chunks:
                logger.info(f"Retrieved {len(context_chunks)} context chunks via vector search")
                return context_chunks
        except concurrent.futures.TimeoutError:
            logger.error("Vector search timed out after 5 seconds")
        
        # If vector search failed or returned empty results, try fallback
        try:
            fallback_chunks = pinecone_client.get_fallback_chunks(asset_id, limit=top_k)
            if fallback_chunks:
                logger.info(f"Using {len(fallback_chunks)} fallback chunks")
                return fallback_chunks
        except Exception as e:
            logger.error(f"Fallback retrieval failed: {str(e)}")
        
        # If all retrieval methods failed
        logger.warning("All context retrieval methods failed")
        return []
    
    def _preprocess_query(self, query):
        """Preprocess query to improve context retrieval."""
        # Remove special characters that might affect semantic meaning
        query = query.strip()
        
        # Truncate very long queries (keep core meaning)
        max_query_length = 200
        if len(query) > max_query_length:
            logger.info(f"Truncating query from {len(query)} to {max_query_length} chars")
            query = query[:max_query_length]
            
        return query
    
    def _format_context(self, context_chunks):
        if not context_chunks:
            return ""
        
        # Sort chunks by relevance score
        sorted_chunks = sorted(context_chunks, key=lambda x: x.get('score', 0), reverse=True)
        
        # Format with more emphasis on score
        return "\n\n".join([
            f"Context Chunk {i+1} (Relevance: {chunk['score']:.2f}):\n{chunk['text']}"
            for i, chunk in enumerate(sorted_chunks)
        ])
    
    # The history action remains unchanged
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