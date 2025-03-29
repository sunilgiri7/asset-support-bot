import os
import requests
import logging
import re
import time
from django.conf import settings

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
        Clean and format the response:
        - Remove excessive whitespace
        - Remove markdown-style formatting
        - Normalize line breaks
        """
        text = re.sub(r'([*_`])', '', text)
        text = re.sub(r'\n{2,}', '\n\n', text)
        return text.strip()
    
    def generate_response(self, prompt, context=None, max_length=500):
        overall_start = time.perf_counter()
        try:
            # Combine system messages into a single prompt
            system_content = (
                "You are a precise, professional technical support assistant. "
                "Provide clear, concise, and structured responses using plain language. "
                "Avoid unnecessary technical jargon and special formatting. "
                "If context is insufficient, state what additional information is needed. "
                "Follow these guidelines: "
                "1. Use clear, professional language; "
                "2. Provide structured information; "
                "3. Focus on clarity and direct communication; "
                "4. Respond directly to the query."
            )
            if context:
                system_content += f"\n\nRelevant Context: {context}\n\nUse this context to inform your response."
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
            
            payload = {
                "model": "mistral-small-latest",
                "messages": messages,
                "temperature": 0.6,
                "max_tokens": max_length,
                "top_p": 0.8,
                "response_format": {"type": "text"}
            }
            logger.info("Prepared payload for Mistral API.")
            
            # Log API call start time
            api_start = time.perf_counter()
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=10  # 10-second timeout
            )
            api_elapsed = time.perf_counter() - api_start
            logger.info(f"Mistral API call completed in {api_elapsed:.2f} seconds.")
            
            response.raise_for_status()
            result = response.json()
            assistant_response = result['choices'][0]['message']['content'].strip()
            
            # Clean the response
            cleaned_response = self._clean_response(assistant_response)
            overall_elapsed = time.perf_counter() - overall_start
            logger.info(f"Total generate_response time: {overall_elapsed:.2f} seconds.")
            
            return cleaned_response
        
        except requests.Timeout:
            logger.error("Mistral API request timed out after 10 seconds")
            return "Sorry, the response is taking too long. Please try again later."
        
        except requests.RequestException as e:
            logger.error(f"Mistral API request failed: {str(e)}")
            return "I apologize, but I'm having trouble processing your request. Please try again later."
        
        except KeyError as e:
            logger.error(f"Unexpected response format from Mistral API: {str(e)}")
            return "I encountered an error while generating a response. Please try again."
        
        except Exception as e:
            logger.error(f"Unexpected error in generate_response: {str(e)}")
            return "An unexpected error occurred. Please try again."