# chatbot/utils/llm_client.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from django.conf import settings
import logging
from accelerate import init_empty_weights, infer_auto_device_map, disk_offload

logger = logging.getLogger(__name__)

class MistralLLMClient:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MistralLLMClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the Mistral model and tokenizer"""
        try:
            logger.info(f"Loading LLM model: {settings.LLM_MODEL_ID}")
            self.tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL_ID)

            # Check if GPU is available
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            device = "cpu"
            logger.info(f"Using device: {device}")

            # Load model configuration
            config = AutoConfig.from_pretrained(settings.LLM_MODEL_ID)

            with init_empty_weights():
                # Initialize model with empty weights
                model = AutoModelForCausalLM.from_config(config)

            # Infer device map for model layers
            device_map = infer_auto_device_map(
                model,
                max_memory={device: "48GiB"} if device == "cuda" else {"cpu": "48GiB"},
                no_split_module_classes=["GPTNeoXLayer"]
            )

            # Load the model with the device map
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.LLM_MODEL_ID,
                device_map=device_map,
                torch_dtype=torch.float16,
                offload_folder="offload"  # Ensure this folder exists
            )

            # Apply disk offload
            disk_offload(self.model, offload_folder="offload")

            logger.info("LLM model loaded successfully with disk offloading.")
        except Exception as e:
            logger.error(f"Error loading LLM model: {str(e)}")
            raise
    
    def generate_response(self, prompt, context=None, max_length=1024):
        """
        Generate a response from the LLM
        
        Args:
            prompt (str): The user's question
            context (str, optional): Relevant context retrieved from documents
            max_length (int): Maximum length of the generated response
            
        Returns:
            str: The generated response
        """
        try:
            # Construct full prompt with context if available
            if context:
                full_prompt = f"""
                <s>[INST] You are a helpful and friendly support assistant. Use the following context to answer the user's question. If the context doesn't contain the answer, say that you don't have enough information and suggest what the user could ask instead.

                Context:
                {context}

                User Question: {prompt} [/INST]
                """
            else:
                full_prompt = f"""
                <s>[INST] You are a helpful and friendly support assistant. Answer the following question to the best of your ability. If you don't know the answer, be honest about it.

                User Question: {prompt} [/INST]
                """
            
            # Generate response
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and return the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated response (remove the prompt)
            response = response.split('[/INST]')[-1].strip()
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I'm sorry, I'm having trouble processing your request. Please try again."