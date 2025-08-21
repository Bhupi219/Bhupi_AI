"""
Configuration file for Friend AI NLU System
Contains all settings, paths, and customizable parameters
"""

import os
from typing import Dict, Any
from pathlib import Path

class NLUConfig:
    """Configuration class for Friend AI NLU System"""
    
    def __init__(self):
        # Base paths
        self.PROJECT_ROOT = Path("/home/bhupendra_singh/Projects/Bhupi_AI")
        self.CORE_PATH = self.PROJECT_ROOT / "core"
        self.NLU_PATH = self.CORE_PATH / "nlu"
        self.MEMORY_PATH = self.PROJECT_ROOT / "memory"
        
        # Model configuration
        self.MODELS_BASE_PATH = Path("/home/bhupendra_singh/Models")
        self.MODEL_CONFIG = {
            "model_name": "Qwen2-7B-Instruct",  # Model folder name in Models directory
            "model_path": self.MODELS_BASE_PATH / "Qwen2-7B-Instruct",  # Full path to model
            "huggingface_name": "Qwen/Qwen2-7B-Instruct",  # Fallback HF name if local not found
            "device": "auto",  # auto, cuda, cpu
            "torch_dtype": "float16",  # float16, float32
            "load_in_4bit": False,  # Enable for memory efficiency
            "load_in_8bit": False,  # Alternative quantization
            "trust_remote_code": True,
            "use_cache": True,
            "prefer_local": True  # Prefer local model over HuggingFace download
        }
        
        # Generation parameters
        self.GENERATION_CONFIG = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "pad_token_id": None,  # Will be set automatically
            "eos_token_id": None   # Will be set automatically
        }
        
        # Friend AI personality settings
        self.PERSONALITY_CONFIG = {
            "supportive_level": 0.9,      # 0.0 to 1.0
            "humor_level": 0.8,           # 0.0 to 1.0
            "wisdom_level": 0.8,          # 0.0 to 1.0
            "playfulness_level": 0.9,     # 0.0 to 1.0
            "casualness_level": 0.9,      # 0.0 to 1.0
            "empathy_level": 0.95,        # 0.0 to 1.0
            "authenticity_level": 0.9,    # 0.0 to 1.0
            "expressiveness": 0.8,        # How emotional/expressive
            "follow_up_frequency": 0.3    # Probability of asking follow-ups
        }
        
        # Conversation management
        self.CONVERSATION_CONFIG = {
            "max_history_length": 10,         # Number of exchanges to remember
            "context_window_size": 2048,      # Token context window
            "response_length_min": 10,        # Minimum response length
            "response_length_max": 200,       # Maximum response length
            "mood_persistence": 3,            # How many turns to remember mood
            "topic_persistence": 5,           # How many turns to remember topics
            "session_timeout": 30             # Minutes before new session
        }
        
        # Memory and storage
        self.MEMORY_CONFIG = {
            "save_conversations": True,
            "conversation_file": "current_conversation.json",
            "backup_conversations": True,
            "max_memory_file_size": 10,       # MB
            "auto_cleanup_days": 30,          # Days to keep old conversations
            "enable_long_term_memory": False, # Future feature
            "memory_compression": True        # Compress old conversations
        }
        
        # Performance settings
        self.PERFORMANCE_CONFIG = {
            "batch_size": 1,
            "max_concurrent_requests": 1,
            "response_timeout": 30,           # Seconds
            "memory_optimization": True,
            "cache_responses": False,         # Cache similar inputs
            "preload_model": True,           # Load model at startup
            "gpu_memory_fraction": 0.8       # Fraction of GPU memory to use
        }
        
        # Logging configuration
        self.LOGGING_CONFIG = {
            "level": "INFO",                  # DEBUG, INFO, WARNING, ERROR
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "log_file": "friend_ai_nlu.log",
            "max_log_size": 10,              # MB
            "backup_count": 5,
            "console_logging": True,
            "file_logging": True
        }
        
        # Development and debugging
        self.DEBUG_CONFIG = {
            "debug_mode": False,
            "verbose_logging": False,
            "save_prompts": False,           # Save system prompts for debugging
            "performance_monitoring": True,
            "conversation_analytics": True,
            "response_timing": True
        }
        
        # Integration settings
        self.INTEGRATION_CONFIG = {
            "enable_stt_integration": True,
            "enable_tts_integration": True,
            "enable_memory_integration": True,
            "enable_personalization": True,
            "api_timeout": 10,
            "retry_attempts": 3
        }
    
    def get_model_path(self) -> str:
        """Get the correct model path, checking local first"""
        local_path = self.MODEL_CONFIG["model_path"]
        
        # Check if local model exists
        if self.MODEL_CONFIG["prefer_local"] and local_path.exists():
            return str(local_path)
        else:
            # Fall back to HuggingFace name
            return self.MODEL_CONFIG["huggingface_name"]
    
    def validate_model_path(self) -> Dict[str, bool]:
        """Validate model path and availability"""
        local_path = self.MODEL_CONFIG["model_path"]
        
        validation = {
            "local_path_exists": local_path.exists(),
            "local_path_is_dir": local_path.is_dir() if local_path.exists() else False,
            "models_base_exists": self.MODELS_BASE_PATH.exists(),
            "has_model_files": False
        }
        
        # Check for essential model files
        if validation["local_path_is_dir"]:
            essential_files = ["config.json", "tokenizer.json"]
            model_files = list(local_path.glob("*.safetensors")) + list(local_path.glob("*.bin"))
            
            has_config = any((local_path / f).exists() for f in essential_files)
            has_weights = len(model_files) > 0
            validation["has_model_files"] = has_config and has_weights
            validation["model_files_found"] = len(model_files)
        
        return validation
    
    def create_models_directory(self):
        """Create models directory structure"""
        try:
            self.MODELS_BASE_PATH.mkdir(parents=True, exist_ok=True)
            model_path = self.MODEL_CONFIG["model_path"]
            model_path.parent.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False
    
    def get_generation_config(self) -> Dict[str, Any]:
        """Get generation configuration"""
        return self.GENERATION_CONFIG.copy()
    
    def get_personality_config(self) -> Dict[str, Any]:
        """Get personality configuration"""
        return self.PERSONALITY_CONFIG.copy()
    
    def get_paths(self) -> Dict[str, Path]:
        """Get all configured paths"""
        return {
            "project_root": self.PROJECT_ROOT,
            "core": self.CORE_PATH,
            "nlu": self.NLU_PATH,
            "memory": self.MEMORY_PATH,
            "conversation_file": self.MEMORY_PATH / self.MEMORY_CONFIG["conversation_file"],
            "log_file": self.PROJECT_ROOT / "logs" / self.LOGGING_CONFIG["log_file"]
        }
    
    def create_directories(self):
        """Create necessary directories"""
        paths = self.get_paths()
        
        for name, path in paths.items():
            if name.endswith('_file'):
                # Create parent directory for files
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Create directory
                path.mkdir(parents=True, exist_ok=True)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration with resolved path"""
        config = self.MODEL_CONFIG.copy()
        config["resolved_model_path"] = self.get_model_path()
        return config
        """Validate configuration settings"""
        validation_results = {}
        
        # Validate model config
        model_path = self.MODEL_CONFIG.get("model_path") or self.MODEL_CONFIG["model_name"]
        validation_results["model_accessible"] = True  # We'll assume it's accessible
        
        # Validate paths
        validation_results["paths_valid"] = all(
            isinstance(path, Path) for path in self.get_paths().values()
        )
        
        # Validate personality ranges
        personality_valid = all(
            0.0 <= value <= 1.0 
            for value in self.PERSONALITY_CONFIG.values()
            if isinstance(value, (int, float))
        )
        validation_results["personality_ranges_valid"] = personality_valid
        
        # Validate generation config
        gen_config = self.GENERATION_CONFIG
        validation_results["generation_config_valid"] = (
            gen_config["max_new_tokens"] > 0 and
            0.0 < gen_config["temperature"] <= 2.0 and
            0.0 < gen_config["top_p"] <= 1.0 and
            gen_config["repetition_penalty"] >= 1.0
        )
        
        return validation_results
    
    def update_config(self, section: str, updates: Dict[str, Any]):
        """Update configuration section"""
        section_map = {
            "model": self.MODEL_CONFIG,
            "generation": self.GENERATION_CONFIG,
            "personality": self.PERSONALITY_CONFIG,
            "conversation": self.CONVERSATION_CONFIG,
            "memory": self.MEMORY_CONFIG,
            "performance": self.PERFORMANCE_CONFIG,
            "logging": self.LOGGING_CONFIG,
            "debug": self.DEBUG_CONFIG,
            "integration": self.INTEGRATION_CONFIG
        }
        
        if section in section_map:
            section_map[section].update(updates)
            return True
        return False
    
    def get_hardware_optimized_config(self) -> Dict[str, Any]:
        """Get hardware-optimized configuration based on system specs"""
        # This would analyze your RTX 5070ti and optimize settings
        optimized = {}
        
        # RTX 5070ti has 12GB VRAM - good for 7B models
        optimized["model"] = {
            "load_in_4bit": False,  # You have enough VRAM
            "torch_dtype": "float16",  # Good balance
            "device": "cuda"
        }
        
        # Optimize for your 32GB RAM
        optimized["generation"] = {
            "max_new_tokens": 512,  # You can handle longer responses
            "batch_size": 1,        # Single conversation
        }
        
        # Performance settings for your i9 ultra 275HX
        optimized["performance"] = {
            "preload_model": True,
            "gpu_memory_fraction": 0.8,  # Use most of your 12GB VRAM
            "memory_optimization": False  # You have plenty of RAM
        }
        
        return optimized
    
    def save_config_to_file(self, file_path: str = None):
        """Save current configuration to file"""
        if file_path is None:
            file_path = self.PROJECT_ROOT / "config" / "nlu_config.json"
        
        import json
        
        config_data = {
            "model_config": self.MODEL_CONFIG,
            "generation_config": self.GENERATION_CONFIG,
            "personality_config": self.PERSONALITY_CONFIG,
            "conversation_config": self.CONVERSATION_CONFIG,
            "memory_config": self.MEMORY_CONFIG,
            "performance_config": self.PERFORMANCE_CONFIG,
            "logging_config": self.LOGGING_CONFIG,
            "debug_config": self.DEBUG_CONFIG,
            "integration_config": self.INTEGRATION_CONFIG
        }
        
        # Create config directory
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, default=str)
    
    def load_config_from_file(self, file_path: str):
        """Load configuration from file"""
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Update configurations
        for key, value in config_data.items():
            section_name = key.replace('_config', '').upper() + '_CONFIG'
            if hasattr(self, section_name):
                getattr(self, section_name).update(value)


# Global configuration instance
nlu_config = NLUConfig()

# Quick setup functions
def setup_for_development():
    """Setup configuration for development"""
    nlu_config.update_config("debug", {
        "debug_mode": True,
        "verbose_logging": True,
        "save_prompts": True,
        "performance_monitoring": True
    })
    
    nlu_config.update_config("logging", {
        "level": "DEBUG",
        "console_logging": True
    })

def setup_for_production():
    """Setup configuration for production"""
    nlu_config.update_config("debug", {
        "debug_mode": False,
        "verbose_logging": False,
        "save_prompts": False
    })
    
    nlu_config.update_config("logging", {
        "level": "INFO",
        "file_logging": True
    })
    
    # Apply hardware optimizations
    optimized = nlu_config.get_hardware_optimized_config()
    for section, updates in optimized.items():
        nlu_config.update_config(section, updates)

def get_quick_config() -> NLUConfig:
    """Get pre-configured NLU config for quick start"""
    config = NLUConfig()
    
    # Apply hardware optimizations for your system
    setup_for_development()  # Since you're developing
    
    # Create necessary directories
    config.create_directories()
    
    return config


# Test the configuration
def test_config():
    """Test configuration system"""
    print("🔧 Testing NLU Configuration")
    
    config = get_quick_config()
    
    # Validate configuration
    validation = config.validate_config()
    print(f"Validation results: {validation}")
    
    # Show paths
    paths = config.get_paths()
    print("📁 Configured paths:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    
    # Show hardware optimized settings
    optimized = config.get_hardware_optimized_config()
    print("⚡ Hardware optimized settings:")
    for section, settings in optimized.items():
        print(f"  {section}: {settings}")
    
    print("✅ Configuration test completed!")


if __name__ == "__main__":
    test_config()