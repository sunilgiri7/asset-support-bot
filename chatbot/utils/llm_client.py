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
    
    def _fetch_continuation(self, prompt, context=None, max_length=200):
        """
        Fetch additional response from the LLM using the same Mistral API model
        to continue an incomplete response.
        If any error occurs, simply return an empty string so that the original response remains unchanged.
        """
        # Prepare system message for continuation
        system_content = (
            "You are a technical support assistant for Presage Insights platform. "
            "Please continue your previous answer without repeating the content already provided. "
            "Format your response as clean HTML with NO markdown, starting directly with the continuation content. "
            "Ensure all HTML tags opened in the previous part or this part are properly closed."
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
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            logger.info(f"Attempting Mistral continuation call to {self.base_url} with model {payload['model']}")
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=15  # Slightly increased timeout
            )
            logger.info(f"Mistral continuation response status code: {response.status_code}")
            response.raise_for_status()
            
            try:
                result = response.json()
                continuation_html = result['choices'][0]['message']['content'].strip()
                # Clean the continuation HTML (but don't wrap it in a new container)
                continuation_html = self._clean_continuation_html(continuation_html)
                logger.info("Successfully fetched continuation from Mistral.")
                return continuation_html
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse JSON from Mistral continuation response: {json_err}. Raw response text: {response.text[:500]}...")
                return ""
            except (KeyError, IndexError) as e:
                logger.error(f"Unexpected JSON structure in Mistral continuation response: {e}. Response: {result}")
                return ""
            
        except requests.Timeout:
            logger.error("Mistral continuation API request timed out.")
            return ""
        except requests.RequestException as e:
            logger.error(f"Failed to fetch continuation using Mistral: {str(e)}")
            if e.response is not None:
                logger.error(f"Mistral continuation error response status: {e.response.status_code}, Body: {e.response.text[:500]}...")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error during Mistral continuation fetch: {str(e)}")
            return ""

    def _is_html_complete(self, html):
        """
        A more robust check to determine if the HTML has balanced key tags.
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

    def _clean_continuation_html(self, html):
        """
        Cleans HTML snippet without adding the main container div or style.
        Also handles special cases where the continuation might start with a closing tag
        or end with an opening tag.
        """
        # Remove newlines and tabs.
        html = re.sub(r'[\n\t]+', ' ', html)
        # Remove extra spaces between tags.
        html = re.sub(r'>\s+<', '><', html)
        # Remove spaces at the beginning and end.
        html = html.strip()
        
        # Remove any container divs the model might have accidentally added
        html = re.sub(r'^<div class=["\']response-container["\'].*?>', '', html, flags=re.IGNORECASE | re.DOTALL)
        html = re.sub(r'</div>$', '', html, flags=re.IGNORECASE)
        
        # Handle special case: if continuation starts with a closing tag that's not paired
        # (This happens when the original response has an unclosed tag)
        if html.startswith('</'):
            tag_match = re.match(r'^</([a-z0-9]+)>', html, re.IGNORECASE)
            if tag_match:
                tag_name = tag_match.group(1)
                # Check if this closing tag has a matching opening tag
                if not re.search(f'<{tag_name}[^>]*>', html):
                    # No matching opening tag, remove the orphaned closing tag
                    html = re.sub(r'^</[a-z0-9]+>', '', html, flags=re.IGNORECASE)
        
        # Handle special case: if continuation ends with an opening tag
        if re.search(r'<[a-z0-9]+[^>]*>$', html, re.IGNORECASE):
            tag_match = re.search(r'<([a-z0-9]+)[^>]*>$', html, re.IGNORECASE)
            if tag_match:
                tag_name = tag_match.group(1)
                # Check if this opening tag has a matching closing tag
                if not re.search(rf'</\s*{tag_name}\s*>', html, re.IGNORECASE):
                    # Remove the orphaned opening tag
                    html = re.sub(r'<[a-z0-9]+[^>]*>$', '', html, flags=re.IGNORECASE)
        
        return html.strip()

    def generate_response(self, prompt, context=None, max_length=500):
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
            # Domain expert instructions for the Presage Insights platform.
            domain_expert_instructions = (
                "You are a domain expert AI assistant specialized in predictive maintenance and asset performance management for industrial environments, specifically for the Presage Insights platform. "
                "Your role is to provide detailed, technical, and actionable responses based on real‑time IoT sensor data and AI‑driven predictive analytics. "
                "When addressing user queries, ensure that your response includes the following elements: "
                "1. Technical Context: Explain the role of IoT sensors (e.g., vibration, temperature, and acoustic sensors) in monitoring machine health, detailing their contribution to real‑time data acquisition and anomaly detection. "
                "2. Predictive Analytics: Discuss the use of AI algorithms for trend analysis and fault prediction, emphasizing the significance of early detection in preventing equipment failures and reducing maintenance costs. "
                "3. Operational Insights: Offer best practices for implementing predictive maintenance strategies in harsh industrial environments, including sensor calibration, data governance, and integration with existing asset management systems. "
                "4. Actionable Recommendations: Provide clear, step‑by‑step guidance for troubleshooting common issues, optimizing sensor placement, and leveraging the platform's customizable dashboards and alerts. "
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
                "IMPORTANT: Make sure your response is complete and does not end mid-sentence or with incomplete HTML."
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
            
            # Check for incomplete HTML structure using enhanced tag balance check
            if not self._is_html_complete(html_response):
                logger.warning("Incomplete HTML structure detected. Attempting to fetch continuation...")
                
                # Extract the last full paragraph or element to provide context
                last_content = html_response
                continuation_prompt = (
                    "Please continue your previous response from where it ended. "
                    "Ensure you properly close any open HTML tags. "
                    "The response should be a complete and professional answer about Presage Insights platform."
                )
                
                # Get continuation
                continuation_html = self._fetch_continuation(continuation_prompt, context=context, max_length=200)
                
                if continuation_html:
                    # Check if the original response ends with a partial tag
                    match_end_tag = re.search(r'<([a-z0-9]+)[^>]*>([^<]*)$', html_response, re.IGNORECASE)
                    if match_end_tag and not continuation_html.startswith('</'):
                        # If original ends with opening tag and some content, but continuation doesn't start with closing tag
                        tag_name = match_end_tag.group(1)
                        # Add closing tag for the last opened tag
                        continuation_html = f"</{tag_name}>" + continuation_html
                    
                    # Remove any container div tags from the continuation
                    continuation_html = re.sub(r'^<div class="response-container".*?>', '', continuation_html)
                    continuation_html = re.sub(r'</div>$', '', continuation_html)
                    
                    # Join original and continuation, making sure we keep response container structure
                    html_response = html_response.rstrip("</div>") + continuation_html + "</div>"
                    
                    # Do a final clean to ensure proper structure
                    html_response = self._clean_html(html_response)
                    logger.warning("Successfully appended continuation to complete the response.")
                else:
                    logger.warning("Continuation failed. Attempting to repair the original response.")
                    # If we couldn't get a continuation, try to repair the original
                    html_response = self._repair_html(html_response)
            
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

    def _repair_html(self, html):
        """
        Attempt to repair incomplete HTML when continuation fails.
        This is a last resort method when we can't get a proper continuation.
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