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
    
    def generate_response(self, prompt, context=None, max_length=500):
        overall_start = time.perf_counter()
        
        # Check for basic greetings and return a hardcoded response if applicable.
        basic_greetings = {"hi", "hii", "hello", "hey"}
        normalized_prompt = prompt.strip().lower()
        if normalized_prompt in basic_greetings:
            hardcoded_response = (
                '<div class="response-container" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                '<p>Hello! How can I help you today?</p>'
                '</div>'
            )
            logger.info("Returning hardcoded greeting response.")
            return hardcoded_response
        
        try:
            # Domain expert instructions for the Presage Insights platform.
            domain_expert_instructions = (
                "You are a domain expert AI assistant specialized in predictive maintenance and asset performance management for industrial environments, specifically for the Presage Insights platform. "
                "Your role is to provide detailed, technical, and actionable responses based on real‑time IoT sensor data and AI‑driven predictive analytics. "
                "When addressing user queries, ensure that your response includes the following elements: "
                "1. Technical Context: Explain the role of IoT sensors (e.g., vibration, temperature, and acoustic sensors) in monitoring machine health, detailing their contribution to real‑time data acquisition and anomaly detection. "
                "2. Predictive Analytics: Discuss the use of AI algorithms for trend analysis and fault prediction, emphasizing the significance of early detection in preventing equipment failures and reducing maintenance costs. "
                "3. Operational Insights: Offer best practices for implementing predictive maintenance strategies in harsh industrial environments, including sensor calibration, data governance, and integration with existing asset management systems. "
                "4. Actionable Recommendations: Provide clear, step‑by‑step guidance for troubleshooting common issues, optimizing sensor placement, and leveraging the platform’s customizable dashboards and alerts. "
                "5. Industry-Specific Considerations: Tailor your answers to reflect unique challenges and opportunities in sectors such as manufacturing, automotive, and FMCG, and reference relevant standards and regulatory requirements where applicable. "
                "Your responses should be clear, concise, and supported by technical reasoning, ensuring that maintenance engineers, reliability experts, and asset managers can make informed decisions to optimize operational efficiency. "
                "For example, if a user asks about improving vibration analysis on a critical machine, include recommendations on sensor calibration, data smoothing techniques, and integration of historical performance data to fine-tune predictive models. "
                "Remember: Always align your explanations with the advanced capabilities of the Presage Insights platform, ensuring that the conversation remains highly technical and context‑specific. "
            )
            
            # Instruct the model to directly output HTML
            system_content = (
                "You are a precise, professional technical support assistant. "
                "Provide clear, concise, and structured responses. "
                "IMPORTANT: Format your ENTIRE response as clean HTML with NO markdown. Follow these guidelines: "
                "1. Start with <div class='response-container'> and end with </div> "
                "2. Use appropriate HTML tags: <p> for paragraphs, <h3> for headings, <strong> for emphasis "
                "3. Use proper HTML lists: <ul><li> for bullet points and <ol><li> for numbered lists "
                "4. For nested lists, place the entire <ul> or <ol> inside the parent <li> element "
                "5. DO NOT include any markdown formatting like **, #, or - for lists "
                "6. DO NOT include any \\n characters or extra whitespace between tags "
                "7. Keep your HTML clean, valid, and properly nested. "
                "Respond directly to the query with relevant information. "
            )
            # Append domain expert instructions to the system message.
            system_content += domain_expert_instructions
            
            # Only include additional context if provided.
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
            
            # Log API call start time.
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
            html_response = result['choices'][0]['message']['content'].strip()
            
            # Clean up any remaining issues (strip newlines, fix common HTML issues)
            html_response = self._clean_html(html_response)
            
            overall_elapsed = time.perf_counter() - overall_start
            logger.info(f"Total generate_response time: {overall_elapsed:.2f} seconds.")
            
            return html_response
        
        except requests.Timeout:
            logger.error("Mistral API request timed out after 10 seconds")
            return '<div class="response-container error" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"><p>Sorry, the response is taking too long. Please try again later.</p></div>'
        
        except requests.RequestException as e:
            logger.error(f"Mistral API request failed: {str(e)}")
            return '<div class="response-container error" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"><p>I apologize, but I\'m having trouble processing your request. Please try again later.</p></div>'
        
        except KeyError as e:
            logger.error(f"Unexpected response format from Mistral API: {str(e)}")
            return '<div class="response-container error" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"><p>I encountered an error while generating a response. Please try again.</p></div>'
        
        except Exception as e:
            logger.error(f"Unexpected error in generate_response: {str(e)}")
            return '<div class="response-container error" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"><p>An unexpected error occurred. Please try again.</p></div>'
    
    def _clean_html(self, html):
        """Clean HTML by removing newlines and extra spaces between tags"""
        # Remove newlines and tabs.
        html = re.sub(r'[\n\t]+', ' ', html)
        # Remove extra spaces between tags.
        html = re.sub(r'>\s+<', '><', html)
        # Remove spaces at the beginning and end of the HTML.
        html = html.strip()
        # Ensure response is wrapped in our container div.
        if not (html.startswith('<div class="response-container"') or html.startswith("<div class='response-container'")):
            html = f'<div class="response-container">{html}</div>'
        # Add CSS styling to ensure a consistent look.
        html = html.replace('class="response-container"', 'class="response-container" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"')
        html = html.replace("class='response-container'", 'class="response-container" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"')
        return html
