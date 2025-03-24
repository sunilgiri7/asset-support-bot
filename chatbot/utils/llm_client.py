import os
import requests
from django.conf import settings
import logging

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

    def generate_response(self, prompt, context=None, max_length=1024):
        """
        Generate a response using Mistral's Chat Completions API
        
        Args:
            prompt (str): User's query
            context (str, optional): Relevant context from Pinecone
            max_length (int, optional): Maximum token length for response
        
        Returns:
            str: Generated response
        """
        try:
            # Prepare messages payload
            messages = []
            
            # Add system context
            messages.append({
                "role": "system", 
                "content": "You are a helpful and friendly support assistant. Answer questions precisely and concisely."
            })
            
            # Add context if available
            if context:
                messages.append({
                    "role": "system", 
                    "content": f"""Context: {context}
        
                    Based on the provided context, answer the following query:
                    {prompt}
                    
                    If the context does not contain sufficient information to answer the query, 
                    please explain what information is missing."""
                })
            
            # Add user prompt
            messages.append({
                "role": "user", 
                "content": prompt
            })
        
            # Prepare API request payload
            payload = {
                "model": "mistral-small-latest",  # or another Mistral model
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": max_length,
                "top_p": 0.9
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
            
            # Extract and return the response
            result = response.json()
            assistant_response = result['choices'][0]['message']['content'].strip()
            
            return assistant_response
        
        except requests.RequestException as e:
            logger.error(f"Mistral API request failed: {str(e)}")
            return "I apologize, but I'm having trouble processing your request. Please try again later."
        except KeyError as e:
            logger.error(f"Unexpected response format from Mistral API: {str(e)}")
            return "I encountered an error while generating a response. Please try again."
        except Exception as e:
            logger.error(f"Unexpected error in generate_response: {str(e)}")
            return "An unexpected error occurred. Please try again."