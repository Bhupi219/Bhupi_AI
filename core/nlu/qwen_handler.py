"""
Qwen2-7B-Instruct Handler for Friend AI Assistant
Handles model loading, inference, and basic conversation management
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import Optional, Dict, List
import gc

class QwenHandler:
    def __init__(self, model_path: str = None, device: str = "auto", config=None):
        """
        Initialize Qwen2-7B handler
        
        Args:
            model_path: Path to model (local path or HuggingFace name)
            device: Device to run model on ("auto", "cuda", "cpu")
            config: NLUConfig instance for configuration
        """
        # Import config here to avoid circular imports
        if config is None:
            try:
                from config import get_quick_config
                config = get_quick_config()
            except ImportError:
                config = None
        
        # Determine model path
        if model_path is None and config:
            self.model_path = config.get_model_path()
        elif model_path is None:
            self.model_path = "Qwen/Qwen2-7B-Instruct"  # Default fallback
        else:
            self.model_path = model_path
            
        self.device = self._setup_device(device)
        self.config = config
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        self.max_history_length = 10  # Keep last 10 exchanges
        
        # Model configuration
        self.generation_config = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": None,  # Will be set after tokenizer loading
        }
        
        self.logger = logging.getLogger(__name__)
        
    def _setup_device(self, device: str) -> str:
        """Setup optimal device for inference"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def load_model(self) -> bool:
        """
        Load Qwen2-7B model and tokenizer
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info(f"Loading Qwen2-7B from {self.model_path}")
            
            # Check if it's a local path
            from pathlib import Path
            model_path_obj = Path(self.model_path)
            
            if model_path_obj.exists() and model_path_obj.is_dir():
                self.logger.info(f"✅ Found local model at: {self.model_path}")
                model_source = self.model_path
            else:
                self.logger.info(f"🔥 Local model not found, will download from HuggingFace: {self.model_path}")
                model_source = self.model_path
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                trust_remote_code=True,
                padding_side='left'
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id
            
            # Load model with optimal settings
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_source,
                **model_kwargs
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.logger.info("✅ Qwen2-7B loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load Qwen2-7B: {str(e)}")
            return False
    
    def generate_response(self, user_input: str, system_prompt: str = None) -> str:
        """
        Generate response from user input
        
        Args:
            user_input: User's message
            system_prompt: System prompt for personality/behavior
            
        Returns:
            Generated response string
        """
        if not self.model or not self.tokenizer:
            return "❌ Model not loaded. Please load the model first."
        
        try:
            # Build conversation prompt
            prompt = self._build_conversation_prompt(user_input, system_prompt)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.generation_config,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Update conversation history
            self._update_history(user_input, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Generation failed: {str(e)}")
            return "Sorry, I had trouble processing that. Can you try again?"
    
    def _build_conversation_prompt(self, user_input: str, system_prompt: str = None) -> str:
        """Build conversation prompt with history and system prompt"""
        
        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = """You are a close friend having a casual conversation. 
            Be supportive, wise, playful, and humorous. Respond naturally like a real friend would."""
        
        # Build prompt with conversation history
        prompt_parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>"]
        
        # Add conversation history (last few exchanges)
        for exchange in self.conversation_history[-self.max_history_length:]:
            prompt_parts.append(f"<|im_start|>user\n{exchange['user']}<|im_end|>")
            prompt_parts.append(f"<|im_start|>assistant\n{exchange['assistant']}<|im_end|>")
        
        # Add current user input
        prompt_parts.append(f"<|im_start|>user\n{user_input}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        
        return "\n".join(prompt_parts)
    
    def _update_history(self, user_input: str, response: str):
        """Update conversation history"""
        self.conversation_history.append({
            "user": user_input,
            "assistant": response
        })
        
        # Keep history within limits
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.logger.info("🔥 Conversation history cleared")
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "loaded": self.model is not None,
            "history_length": len(self.conversation_history),
            "max_history": self.max_history_length
        }
    
    def update_generation_config(self, **kwargs):
        """Update generation configuration"""
        self.generation_config.update(kwargs)
        self.logger.info(f"🔧 Generation config updated: {kwargs}")
    
    def cleanup(self):
        """Clean up model from memory"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        self.logger.info("🧹 Model cleaned up from memory")


# Quick test function
def test_qwen_handler():
    """Test the QwenHandler basic functionality"""
    handler = QwenHandler()
    
    print("Loading model...")
    if handler.load_model():
        print("✅ Model loaded successfully!")
        
        # Test basic conversation
        response = handler.generate_response("Hey, how are you doing?")
        print(f"Response: {response}")
        
        # Test follow-up
        response2 = handler.generate_response("That's great! What should we talk about?")
        print(f"Response 2: {response2}")
        
        # Show model info
        print(f"Model Info: {handler.get_model_info()}")
        
    else:
        print("❌ Failed to load model")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run test
    test_qwen_handler()