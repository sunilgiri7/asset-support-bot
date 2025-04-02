import json
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
    
    def _is_html_complete(self, html):
        """
        A check to determine if the HTML has balanced key tags.
        It compares the number of opening and closing tags for tags that are used in our formatted responses.
        Also checks for incomplete last paragraph.
        """
        # First check if the response has proper container div
        if not html.strip().startswith('<div class="response-container"'):
            logger.warning("Response missing proper container div")
            return False
        
        if not html.strip().endswith('</div>'):
            logger.warning("Response missing closing div tag")
            return False
        
        # Check for specific tags that should be balanced
        tags_to_check = ['div', 'p', 'h3', 'ul', 'ol', 'li', 'strong']
        for tag in tags_to_check:
            open_count = len(re.findall(f'<{tag}[^>]*>', html))
            close_count = len(re.findall(f'</{tag}>', html))
            if open_count != close_count:
                logger.warning(f"Incomplete HTML detected for <{tag}>: {open_count} opening vs {close_count} closing tags.")
                return False
        
        # Check if the last paragraph appears cut off (ends without proper punctuation)
        content_text = re.sub(r'<[^>]+>', ' ', html).strip()
        if content_text and len(content_text) > 20:
            last_char = content_text[-1]
            if last_char not in ['.', '!', '?', ':', ';', '"', ')', ']', '}']:
                logger.warning(f"Content appears to be cut off, last character: '{last_char}'")
                return False
                
        return True

    def _clean_html(self, html):
        """
        Clean HTML by removing newlines/extra spaces, ensure exactly ONE
        correctly styled top-level container div.
        """
        # Remove newlines and tabs.
        html = re.sub(r'[\n\t]+', ' ', html)
        # Remove extra spaces between tags.
        html = re.sub(r'>\s+<', '><', html)
        # Remove spaces at the beginning and end of the HTML.
        html = html.strip()

        # --- Refined Container Logic ---
        # Check if we already have a properly formatted response container
        if html.startswith('<div class="response-container"') and html.endswith('</div>'):
            # Already has container, just ensure it has our style
            if 'style=' not in html[:100]:  # Check the opening tag
                # Add style to existing container
                html = re.sub(r'^<div class="response-container"', 
                            '<div class="response-container" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"', 
                            html)
            return html
            
        # Remove any existing top-level response container divs (case-insensitive)
        html = re.sub(r'^<div\s+class=["\']response-container["\'].*?>', '', html, flags=re.IGNORECASE | re.DOTALL).strip()
        # Remove closing div if at the end
        if html.endswith('</div>'):
            html = html[:-len('</div>')].strip()

        # Wrap the cleaned content with a styled container div
        style_attr = 'style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"'
        html_response = f'<div class="response-container" {style_attr}>{html}</div>'

        return html_response

    def _repair_html(self, html):
        """
        Attempt to repair incomplete HTML.
        This ensures any unbalanced tags are properly closed.
        """
        # Start with our cleaned HTML
        repaired = html
        
        # Make sure we have a container div
        if not repaired.startswith('<div class="response-container"'):
            style_attr = 'style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"'
            repaired = f'<div class="response-container" {style_attr}>{repaired}'
        
        # Make sure we have a closing div
        if not repaired.endswith('</div>'):
            repaired = f'{repaired}</div>'
        
        # Check and close common tags
        tags_to_check = ['p', 'h3', 'ul', 'ol', 'li', 'strong']
        for tag in tags_to_check:
            # Count opening and closing tags
            open_tags = re.findall(f'<{tag}[^>]*>', repaired)
            close_tags = re.findall(f'</{tag}>', repaired)
            
            # If more opening than closing tags, add the needed closing tags
            if len(open_tags) > len(close_tags):
                for _ in range(len(open_tags) - len(close_tags)):
                    # Add closing tag before the final </div>
                    repaired = repaired[:-6] + f'</{tag}>' + repaired[-6:]
        
        logger.info("Repaired incomplete HTML response")
        return repaired

    def generate_response(self, prompt, context=None, max_length=800):
        overall_start = time.perf_counter()
        
        # Check for basic greetings and return a hardcoded response if applicable.
        basic_greetings = {"hi", "hii", "hello", "hey"}
        normalized_prompt = prompt.strip().lower()
        if normalized_prompt in basic_greetings:
            hardcoded_response = (
                '<div class="response-container" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                '<p>Hello! How can I help you today with Presage Insights? I can assist with predictive maintenance, IoT sensor data, or analytics questions.</p>'
                '</div>'
            )
            logger.info("Returning hardcoded greeting response.")
            return hardcoded_response
        
        try:
            # STRUCTURED RESPONSE APPROACH
            # Instead of trying to get the entire response at once, we'll use a structured approach
            # to ensure we get a complete, well-formed response
            
            # First get an outline of the response structure
            outline_response = self._get_outline(prompt, context)
            
            # Then get the full response with the outline as a guide
            full_response = self._get_full_response(prompt, outline_response, context, max_length)
            
            # Clean up and check the HTML structure
            html_response = self._clean_html(full_response)
            
            # Check if the HTML is complete, if not, attempt to repair it
            if not self._is_html_complete(html_response):
                logger.warning("Incomplete HTML structure detected. Attempting to repair...")
                html_response = self._repair_html(html_response)
            
            overall_elapsed = time.perf_counter() - overall_start
            logger.info(f"Total generate_response time: {overall_elapsed:.2f} seconds.")
            
            return html_response
        
        except requests.Timeout:
            logger.error("Mistral API request timed out")
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

    def _get_outline(self, prompt, context=None):
        """
        Get a structured outline of the response to ensure completeness.
        This outline will serve as a guide for generating the full response.
        """
        system_content = (
            "You are a technical planning assistant. Create a brief outline for a response about the Presage Insights platform. "
            "The outline should include 3-5 main sections with 2-3 bullet points each. "
            "Format as a simple HTML list with <h3> for main topics and <ul><li> for bullet points. "
            "Keep it concise - this is just an outline structure, not the full content."
        )
        
        if context:
            system_content += f"\n\nRelevant Context: {context}\n\nUse this context to inform your outline structure."
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Create an outline for responding to this query: {prompt}"}
        ]
        
        payload = {
            "model": "mistral-small-latest",
            "messages": messages,
            "temperature": 0.3,  # Lower temperature for more consistent outline structure
            "max_tokens": 300,   # Should be enough for an outline
            "top_p": 0.9,
            "response_format": {"type": "text"}
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        logger.info("Fetching response outline structure...")
        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            outline = result['choices'][0]['message']['content'].strip()
            logger.info("Successfully generated response outline.")
            return outline
        except Exception as e:
            logger.error(f"Error generating outline: {str(e)}")
            # Return a basic outline if outline generation fails
            return "<h3>Topic Overview</h3><ul><li>Key points</li></ul><h3>Details</h3><ul><li>Important details</li></ul><h3>Conclusion</h3><ul><li>Summary points</li></ul>"

    def _get_full_response(self, prompt, outline, context=None, max_length=800):
        """
        Generate a complete response using the outline as a guide.
        This helps ensure the response follows a structure and is complete.
        """
        # Domain expert instructions for the Presage Insights platform
        domain_expert_instructions = (
            "You are a domain expert AI assistant specialized in predictive maintenance and asset performance management for industrial environments, specifically for the Presage Insights platform. "
            "Your role is to provide detailed, technical, and actionable responses based on real‑time IoT sensor data and AI‑driven predictive analytics. "
            "When addressing user queries, ensure that your response includes the following elements: "
            "1. Technical Context: Explain the role of IoT sensors (e.g., vibration, temperature, and acoustic sensors) in monitoring machine health, detailing their contribution to real‑time data acquisition and anomaly detection. "
            "2. Predictive Analytics: Discuss the use of AI algorithms for trend analysis and fault prediction, emphasizing the significance of early detection in preventing equipment failures and reducing maintenance costs. "
            "3. Operational Insights: Offer best practices for implementing predictive maintenance strategies in harsh industrial environments, including sensor calibration, data governance, and integration with existing asset management systems. "
            "4. Actionable Recommendations: Provide clear, step‑by‑step guidance for troubleshooting common issues, optimizing sensor placement, and leveraging the platform's customizable dashboards and alerts. "
            "5. Industry-Specific Considerations: Tailor your answers to reflect unique challenges and opportunities in sectors such as manufacturing, automotive, and FMCG, and reference relevant standards and regulatory requirements where applicable. "
        )
        
        # Enhanced system prompt with structured response instructions
        system_content = (
            "You are a precise, professional technical support assistant for the Presage Insights platform. "
            f"Follow this outline for your response structure: \n\n{outline}\n\n"
            "CRITICAL FORMATTING REQUIREMENTS:\n"
            "1. Format your ENTIRE response as clean HTML with NO markdown.\n"
            "2. Start with <div class='response-container'> and end with </div>\n"
            "3. Use appropriate HTML tags: <p> for paragraphs, <h3> for headings, <strong> for emphasis\n"
            "4. Use proper HTML lists: <ul><li> for bullet points and <ol><li> for numbered lists\n"
            "5. For nested lists, place the entire <ul> or <ol> inside the parent <li> element\n"
            "6. DO NOT exceed the token limit - prioritize completing all sections over verbosity\n"
            "7. ALWAYS end with a proper conclusion paragraph\n"
            "8. NEVER leave a tag unclosed or a sentence unfinished\n"
            "9. ALWAYS ensure your response follows a clear logical structure with a beginning, middle, and end\n"
        )
        
        # Append domain expert instructions
        system_content += domain_expert_instructions
        
        # Include context if provided
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
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        logger.info("Generating full response with outline guidance...")
        response = requests.post(
            self.base_url,
            json=payload,
            headers=headers,
            timeout=15
        )
        response.raise_for_status()
        
        result = response.json()
        full_response = result['choices'][0]['message']['content'].strip()
        logger.info("Successfully generated full response.")
        
        return full_response