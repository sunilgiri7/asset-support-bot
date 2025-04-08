import json
import sys
import time
import concurrent.futures
import logging
from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from chatbot.models import Conversation, Message
from chatbot.serializers import (
    ConversationSerializer, MessagePairSerializer, MessageSerializer,
    QuerySerializer, VibrationAnalysisInputSerializer
)
from asset_support_bot.utils.pinecone_client import PineconeClient
from chatbot.utils.llm_client import GroqLLMClient
from rest_framework.permissions import AllowAny
from chatbot.utils.web_search import web_search

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # Explicitly use stdout
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

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

        asset_id = serializer.validated_data['asset_id']
        message_content = serializer.validated_data['message']
        conversation_id = serializer.validated_data.get('conversation_id')
        timings = {}

        try:
            # Get or create the conversation
            conv_start = time.perf_counter()
            conversation = self._get_or_create_conversation(conversation_id, asset_id)
            timings['conversation_time'] = f"{time.perf_counter() - conv_start:.2f} seconds"

            # Create and save the user message
            user_msg_start = time.perf_counter()
            user_message = Message.objects.create(
                conversation=conversation,
                is_user=True,
                content=message_content
            )
            timings['user_message_time'] = f"{time.perf_counter() - user_msg_start:.2f} seconds"

            # Check if the incoming message is a basic greeting
            basic_greetings = {"hi", "hii", "hello", "hey", "hlo", "h", "hh", "hiii", "helloo", "helo", "hilo", "hellooo"}
            if message_content.strip().lower() in basic_greetings:
                hardcoded_response = (
                    '<div class="response-container" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                    '<p>Hello! How can I help you today with Presage Insights? I can assist with predictive maintenance, IoT sensor data, or analytics questions.</p>'
                    '</div>'
                )
                system_message = Message.objects.create(
                    conversation=conversation,
                    is_user=False,
                    content=hardcoded_response
                )
                overall_elapsed = time.perf_counter() - overall_start
                timings['total_time'] = f"{overall_elapsed:.2f} seconds"

                response_data = {
                    "conversation_id": conversation.id,
                    "user_message": MessageSerializer(user_message).data,
                    "assistant_message": MessageSerializer(system_message).data,
                    "context_used": False,
                    "response_time": f"{overall_elapsed:.2f} seconds",
                    "timings": timings
                }
                return Response(response_data)

            # Retrieve external document context from Pinecone
            context_start = time.perf_counter()
            context_chunks = []
            context_error = None
            try:
                if pinecone_client is not None:
                    context_chunks = self._retrieve_context_chunks(message_content, asset_id)
                else:
                    context_error = "PineconeClient initialization failed"
            except Exception as e:
                context_error = str(e)
                logger.error(f"Error during context retrieval: {context_error}")
            timings['context_retrieval_time'] = f"{time.perf_counter() - context_start:.2f} seconds"

            if context_error:
                document_context = f"Note: Unable to retrieve document context. Error: {context_error}"
                logger.warning(f"Using empty document context due to error: {context_error}")
            else:
                document_context = self._format_context(context_chunks)
                logger.info(f"Using document context with {len(context_chunks)} chunks")

            # Build conversation context using sliding window and summarization
            llm_client = GroqLLMClient()
            conversation_context = self._build_context_prompt(conversation, llm_client)

            # Check if the user has toggled web search functionality.
            use_search = serializer.validated_data.get('use_search', False)
            web_search_results = ""
            if use_search:
                logger.info(f"Web search enabled for query: '{message_content}'")
                print(f"Web search enabled for query: '{message_content}'")
                # Use the new web_search function instead of duckduckgo_search
                web_search_results = web_search(message_content)
                if web_search_results:
                    logger.info("Web search results found and formatted successfully")
                    print("Web search results found and formatted successfully")
                else:
                    logger.info("No web search results were found or an error occurred")
                    print("No web search results were found or an error occurred")

            # Combine all context: document context, conversation context, and current user message.
            combined_prompt = f"{document_context}\n\nConversation Context:\n{conversation_context}\n\nUser: {message_content}"
            if web_search_results:
                logger.info("Adding web search results to combined prompt")
                combined_prompt += f"{web_search_results}\n\n"

            # Generate the assistant's response using the combined prompt and context.
            llm_start = time.perf_counter()
            response_content = llm_client.generate_response(
                prompt=message_content,  # original prompt if needed for LLM reference
                context=combined_prompt
            )
            timings['llm_response_time'] = f"{time.perf_counter() - llm_start:.2f} seconds"

            # Save the assistant's response
            assist_msg_start = time.perf_counter()
            system_message = Message.objects.create(
                conversation=conversation,
                is_user=False,
                content=response_content
            )
            timings['assistant_message_save_time'] = f"{time.perf_counter() - assist_msg_start:.2f} seconds"

            # --- New: Summarization step for conversation history ---
            summary_prompt = (
                "Summarize the following conversation in 2-3 lines, capturing the key points:\n\n"
                f"User: {message_content}\n"
                f"Assistant: {response_content}"
            )
            new_summary = llm_client.generate_response(prompt=summary_prompt, context="")

            # Update the conversation's summary.
            # Here, you can choose whether to replace or append.
            if conversation.summary:
                conversation.summary += "\n" + new_summary
            else:
                conversation.summary = new_summary
            conversation.save()

            overall_elapsed = time.perf_counter() - overall_start
            timings['total_time'] = f"{overall_elapsed:.2f} seconds"

            # Build and return the response data
            response_data = {
                "conversation_id": conversation.id,
                "user_message": MessageSerializer(user_message).data,
                "assistant_message": MessageSerializer(system_message).data,
                "context_used": bool(context_chunks),
                "response_time": f"{overall_elapsed:.2f} seconds",
                "timings": timings
            }
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
        else:
            # Try to fetch the latest conversation for the asset
            conversation = Conversation.objects.filter(asset_id=asset_id).order_by('-updated_at').first()
            if conversation:
                return conversation
            return Conversation.objects.create(asset_id=asset_id)

    def _retrieve_context_chunks(self, query, asset_id, top_k=3):
        global pinecone_client
        if pinecone_client is None:
            logger.error("PineconeClient is not initialized")
            return []

        logger.info(f"Retrieving context for query: {query} for asset_id: {asset_id}")
        query = self._preprocess_query(query)

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    pinecone_client.query_similar_chunks,
                    query_text=query,
                    asset_id=str(asset_id),
                    top_k=top_k
                )
                context_chunks = future.result(timeout=5.0)
            if context_chunks:
                logger.info(f"Retrieved {len(context_chunks)} context chunks via vector search")
                return context_chunks
        except concurrent.futures.TimeoutError:
            logger.error("Vector search timed out after 5 seconds")

        try:
            fallback_chunks = pinecone_client.get_fallback_chunks(asset_id, limit=top_k)
            if fallback_chunks:
                logger.info(f"Using {len(fallback_chunks)} fallback chunks")
                return fallback_chunks
        except Exception as e:
            logger.error(f"Fallback retrieval failed: {str(e)}")

        logger.warning("All context retrieval methods failed")
        return []

    def _preprocess_query(self, query):
        query = query.strip()
        max_query_length = 200
        if len(query) > max_query_length:
            logger.info(f"Truncating query from {len(query)} to {max_query_length} chars")
            query = query[:max_query_length]
        return query

    def _format_context(self, context_chunks):
        if not context_chunks:
            return ""
        sorted_chunks = sorted(context_chunks, key=lambda x: x.get('score', 0), reverse=True)
        return "\n\n".join([
            f"Context Chunk {i+1} (Relevance: {chunk['score']:.2f}):\n{chunk['text']}"
            for i, chunk in enumerate(sorted_chunks)
        ])

    # --- New Helper Functions for Context Management ---

    def _build_conversation_context(self, conversation, max_recent=10):
        """
        Retrieves the most recent messages from the conversation.
        """
        messages = list(
            Message.objects.filter(conversation=conversation).order_by('-created_at')[:max_recent]
        )
        messages = sorted(messages, key=lambda x: x.created_at)
        context_lines = []
        for msg in messages:
            prefix = "User:" if msg.is_user else "Assistant:"
            context_lines.append(f"{prefix} {msg.content}")
        return "\n".join(context_lines)

    def _summarize_conversation_context(self, conversation, llm_client, word_threshold=300):
        """
        If the conversation summary is too long, re-summarize it.
        This uses a basic word count approximation.
        """
        summary = conversation.summary or ""
        if len(summary.split()) > word_threshold:
            prompt = (
                "Please summarize the following conversation history into a concise summary (2-3 lines):\n\n"
                f"{summary}"
            )
            new_summary = llm_client.generate_response(prompt=prompt, context="")
            conversation.summary = new_summary
            conversation.save()
            return new_summary
        return summary

    def _build_context_prompt(self, conversation, llm_client, max_recent=10, word_threshold=300):
        """
        Combines a sliding window of recent messages with the (possibly summarized) conversation summary.
        """
        summarized_context = self._summarize_conversation_context(conversation, llm_client, word_threshold)
        recent_context = self._build_conversation_context(conversation, max_recent)
        if summarized_context:
            return f"Conversation Summary:\n{summarized_context}\n\nRecent Conversation:\n{recent_context}"
        else:
            return recent_context

    @action(detail=False, methods=['get'])
    def history(self, request):
        conversation_id = request.query_params.get('conversation_id')
        asset_id = request.query_params.get('asset_id')

        if conversation_id:
            conversation = get_object_or_404(Conversation, id=conversation_id)
            messages = Message.objects.filter(conversation=conversation).order_by('created_at')
            message_pairs = []
            user_message = None

            for message in messages:
                if message.is_user:
                    if user_message:
                        message_pairs.append({
                            'conversation': conversation,
                            'user_message': user_message,
                            'system_message': None
                        })
                    user_message = message
                else:
                    if user_message:
                        message_pairs.append({
                            'conversation': conversation,
                            'user_message': user_message,
                            'system_message': message
                        })
                        user_message = None
                    else:
                        message_pairs.append({
                            'conversation': conversation,
                            'user_message': None,
                            'system_message': message
                        })
            if user_message:
                message_pairs.append({
                    'conversation': conversation,
                    'user_message': user_message,
                    'system_message': None
                })
            serializer = MessagePairSerializer(message_pairs, many=True)
            return Response(serializer.data)

        elif asset_id:
            conversations = Conversation.objects.filter(asset_id=asset_id).order_by('-updated_at')
            all_message_pairs = []
            for conversation in conversations:
                messages = Message.objects.filter(conversation=conversation).order_by('created_at')
                user_message = None
                for message in messages:
                    if message.is_user:
                        if user_message:
                            all_message_pairs.append({
                                'conversation': conversation,
                                'user_message': user_message,
                                'system_message': None
                            })
                        user_message = message
                    else:
                        if user_message:
                            all_message_pairs.append({
                                'conversation': conversation,
                                'user_message': user_message,
                                'system_message': message
                            })
                            user_message = None
                        else:
                            all_message_pairs.append({
                                'conversation': conversation,
                                'user_message': None,
                                'system_message': message
                            })
                if user_message:
                    all_message_pairs.append({
                        'conversation': conversation,
                        'user_message': user_message,
                        'system_message': None
                    })
            serializer = MessagePairSerializer(all_message_pairs, many=True)
            return Response(serializer.data)
        else:
            return Response(
                {"error": "Please provide either conversation_id or asset_id as query parameters."},
                status=status.HTTP_400_BAD_REQUEST
            )


@api_view(['POST'])
def analyze_vibration(request):
    serializer = VibrationAnalysisInputSerializer(data=request.data)
    if serializer.is_valid():
        data = serializer.validated_data

        prompt = f"""
You are a level 3 vibration analyst.
Perform a comprehensive analysis of the asset's condition using the full set of provided data.
Return your analysis as a structured JSON object with the following keys:
- "overview": A brief summary of the asset's condition.
- "time_domain_analysis": Detailed analysis of the acceleration and velocity time waveforms.
- "frequency_domain_analysis": Analysis of the harmonics and cross PSD data.
- "bearing_faults": Analysis of the bearing fault frequencies.
- "recommendations": A list of actionable maintenance recommendations.

Here is the data:
{{
  "asset_type": "{data['asset_type']}",
  "running_RPM": {data['running_RPM']},
  "bearing_fault_frequencies": {data['bearing_fault_frequencies']},
  "acceleration_time_waveform": {data['acceleration_time_waveform']},
  "velocity_time_waveform": {data['velocity_time_waveform']},
  "harmonics": {data['harmonics']},
  "cross_PSD": {data['cross_PSD']}
}}

**Instructions**:
- Provide a concise "overview" of the overall condition.
- In "time_domain_analysis", include metrics and any concerning trends from the acceleration and velocity time waveforms.
- In "frequency_domain_analysis", detail the implications of the harmonics and cross PSD data.
- In "bearing_faults", mention whether there are any signs of bearing damage.
- In "recommendations", list clear maintenance actions.
- **Return only valid JSON** without any additional markdown or HTML.
"""
        client = GroqLLMClient()
        response_text = client.query_llm([
            {"role": "user", "content": prompt}
        ])

        try:
            analysis_data = json.loads(response_text)
        except Exception as e:
            analysis_data = {
                "error": "Failed to parse LLM response into JSON.",
                "raw_response": response_text
            }
        return Response({"analysis": analysis_data})
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
