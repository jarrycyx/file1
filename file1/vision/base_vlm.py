"""
Base class for Visual Language Model (VLM) implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import json
from datetime import datetime
from loguru import logger

from ..config import File1Config


class BaseVLM(ABC):
    """
    Abstract base class for Visual Language Model implementations.
    """
    
    def __init__(self, config: File1Config):
        """
        Initialize the VLM with configuration.
        
        Args:
            config: File1Configuration object
        """
        self.config = config
        self.model = config.llm.vision.model
        self.base_url = config.llm.vision.base_url
        self.api_key = config.llm.vision.api_key
    
    @abstractmethod
    def _initialize_model(self):
        """
        Initialize the specific VLM model implementation.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def _format_message(self, prompt: str, image_base64: str) -> Any:
        """
        Format the message for the specific VLM API.
        Must be implemented by subclasses.
        
        Args:
            prompt: Text prompt
            image_base64: Base64 encoded image
            
        Returns:
            Formatted message for the specific VLM API
        """
        pass
    
    @abstractmethod
    def _call_model(self, message: Any) -> str:
        """
        Call the VLM model with the formatted message.
        Must be implemented by subclasses.
        
        Args:
            message: Formatted message for the VLM API
            
        Returns:
            Model response text
        """
        pass
    
    def _save_llm_call(self, messages: list):
        """
        Save the LLM call to a JSON file for debugging.
        
        Args:
            messages: List of messages in the conversation
        """
        time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = os.path.join(self.config.save_path, "llm_calls", f"{time_stamp}.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w") as f:
            # Convert messages to JSON-serializable format
            json_messages = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    json_messages.append({"content": msg.content})
                else:
                    json_messages.append({"content": str(msg)})
            
            json.dump(json_messages, f, indent=4, ensure_ascii=False)
    
    def call_vlm(self, image_base64: str, prompt: str, max_retries: int = 3) -> str:
        """
        Generic VLM calling function for handling image-related requests.
        
        Args:
            image_base64: Base64 encoded image data
            prompt: Prompt text
            max_retries: Maximum number of retries on failure
            
        Returns:
            VLM response content
        """
        self._initialize_model()
        
        error_msg = ""
        for attempt in range(max_retries):
            try:
                # Format the message for the specific VLM API
                message = self._format_message(prompt, image_base64)
                
                # Call the VLM model
                response = self._call_model(message)
                
                # Save the conversation for debugging
                self._save_llm_call([message, response])
                
                logger.info(f"Vision response: {response}")
                return response
            except Exception as e:
                error_msg = f"Error when calling VLM: {e}"
                logger.warning(error_msg)
                logger.warning("Retrying...")
                import time
                time.sleep(5)
        
        return error_msg
    
    def _load_prompt(self, prompt_filename: str) -> str:
        """
        Load prompt text from file.
        
        Args:
            prompt_filename: Prompt file name
            
        Returns:
            Prompt text
        """
        # Load prompt based on domain configuration
        domain = getattr(self.config, "domain", "general")
        prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", domain, prompt_filename)
        
        # Fallback to general prompts if domain-specific doesn't exist
        if not os.path.exists(prompt_path):
            prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "general", prompt_filename)
        
        with open(prompt_path, "r") as f:
            from ..config import get_lang_prompt
            prompt = f.read() + get_lang_prompt(self.config.llm.language)
        return prompt
    
    def get_vision_feedback(self, image_base64: str) -> str:
        """
        Get general image feedback.
        
        Args:
            image_base64: Base64 encoded image
            
        Returns:
            VLM feedback on the image
        """
        prompt = self._load_prompt("vision_feedback.md")
        return self.call_vlm(image_base64, prompt)
    
    def get_latex_vision_feedback(self, image_base64: str) -> str:
        """
        Get LaTeX image feedback.
        
        Args:
            image_base64: Base64 encoded image
            
        Returns:
            VLM feedback on the LaTeX image
        """
        prompt = self._load_prompt("vision_latex_feedback.md")
        return self.call_vlm(image_base64, prompt)
    
    def get_vision_classification(self, image_base64: str) -> str:
        """
        Get image classification.
        
        Args:
            image_base64: Base64 encoded image
            
        Returns:
            VLM classification of the image
        """
        prompt = self._load_prompt("vision_classify.md")
        return self.call_vlm(image_base64, prompt)
