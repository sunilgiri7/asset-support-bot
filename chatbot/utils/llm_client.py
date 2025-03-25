import os
import requests
from django.conf import settings
import logging
import re

logger = logging.getLogger(__name__)

class MistralLLMClient:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MistralLLMClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize Mistral API client with configuration"""
        self.api_key = os.getenv('MISTRAL_API_KEY', settings.MISTRAL_API_KEY)
        self.base_url = "https://api.mistral.ai/v1/chat/completions"
        if not self.api_key:
            logger.error("Mistral API key is not configured")
            raise ValueError("Mistral API key is required")
    
    def _clean_response(self, text):
        """
        Clean and format the response
        - Remove excessive whitespace
        - Remove markdown-style formatting
        - Normalize line breaks
        """
        # Remove markdown formatting
        text = re.sub(r'([*_`])', '', text)
        
        # Replace multiple consecutive newlines with a single newline
        text = re.sub(r'\n{2,}', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def generate_response(self, prompt, context=None, max_length=1024):
        """
        Generate a response using Mistral's Chat Completions API
        
        Args:
            prompt (str): User's query
            context (str, optional): Relevant context from Pinecone
            max_length (int, optional): Maximum token length for response
        
        Returns:
            str: Cleaned and formatted generated response
        """
        try:
            # Prepare messages payload with carefully crafted system prompts
            messages = [
                {
                    "role": "system",
                    "content": "You are a precise, professional technical support assistant. "
                               "Provide clear, concise, and structured responses. "
                               "Use plain language and avoid unnecessary technical jargon. "
                               "If the context is insufficient, clearly state what additional information is needed."
                },
                {
                    "role": "system",
                    "content": "Response Guidelines:\n"
                               "1. Use clear, professional language\n"
                               "2. Provide structured information\n"
                               "3. Focus on clarity and direct communication\n"
                               "4. Avoid markdown or special formatting\n"
                               "5. Respond directly to the specific query"
                }
            ]
            
            # Add context-aware system message if context is available
            if context:
                messages.append({
                    "role": "system",
                    "content": f"Relevant Context: {context}\n\n"
                               "Use the provided context to inform your response. "
                               "If the context does not fully answer the query, "
                               "explain what additional information would be helpful."
                })
            
            # Add user prompt
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Prepare API request payload
            payload = {
                "model": "mistral-small-latest",
                "messages": messages,
                "temperature": 0.6,  # Slightly reduced for more consistent responses
                "max_tokens": max_length,
                "top_p": 0.8,
                "response_format": {"type": "text"}
            }
            
            # Make API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers
            )
            
            # Check for successful response
            response.raise_for_status()
            
            # Extract and clean the response
            result = response.json()
            assistant_response = result['choices'][0]['message']['content'].strip()
            
            # Clean the response
            cleaned_response = self._clean_response(assistant_response)
            
            return cleaned_response
        
        except requests.RequestException as e:
            logger.error(f"Mistral API request failed: {str(e)}")
            return "I apologize, but I'm having trouble processing your request. Please try again later."
        
        except KeyError as e:
            logger.error(f"Unexpected response format from Mistral API: {str(e)}")
            return "I encountered an error while generating a response. Please try again."
        
        except Exception as e:
            logger.error(f"Unexpected error in generate_response: {str(e)}")
            return "An unexpected error occurred. Please try again."