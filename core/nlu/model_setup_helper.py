"""
Model Setup Helper for Friend AI NLU System
Helps organize and verify models in the dedicated Models directory
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import json

class ModelSetupHelper:
    """Helper class to setup and manage models"""
    
    def __init__(self, models_base_path: str = "/home/bhupendra_singh/Models"):
        self.models_base_path = Path(models_base_path)
        self.qwen_model_path = self.models_base_path / "Qwen2-7B-Instruct"
        
    def create_models_directory(self) -> bool:
        """Create the models directory structure"""
        try:
            self.models_base_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created models directory: {self.models_base_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to create models directory: {e}")
            return False
    
    def check_model_status(self) -> Dict:
        """Check the status of Qwen2-7B-Instruct model"""
        status = {
            "models_dir_exists": self.models_base_path.exists(),
            "model_dir_exists": self.qwen_model_path.exists(),
            "model_files_found": [],
            "essential_files_present": False,
            "model_size_gb": 0,
            "download_needed": True
        }
        
        if not status["models_dir_exists"]:
            return status
        
        if status["model_dir_exists"]:
            # Check for model files
            model_files = []
            model_files.extend(list(self.qwen_model_path.glob("*.safetensors")))
            model_files.extend(list(self.qwen_model_path.glob("*.bin")))
            
            status["model_files_found"] = [f.name for f in model_files]
            
            # Check for essential files
            essential_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
            essential_present = [
                (self.qwen_model_path / f).exists() for f in essential_files
            ]
            status["essential_files_present"] = all(essential_present)
            
            # Calculate model size
            total_size = 0
            for f in self.qwen_model_path.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size
            
            status["model_size_gb"] = round(total_size / (1024**3), 2)
            status["download_needed"] = not (len(model_files) > 0 and status["essential_files_present"])
        
        return status
    
    def download_model_with_transformers(self) -> bool:
        """Download model using transformers library"""
        try:
            print("📥 Downloading Qwen2-7B-Instruct using transformers...")
            print("This may take a while (model is ~7GB)")
            
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Create model directory
            self.qwen_model_path.mkdir(parents=True, exist_ok=True)
            
            # Download tokenizer
            print("📄 Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2-7B-Instruct",
                trust_remote_code=True
            )
            tokenizer.save_pretrained(self.qwen_model_path)
            print("✅ Tokenizer downloaded")
            
            # Download model
            print("🧠 Downloading model weights...")
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2-7B-Instruct",
                trust_remote_code=True,
                torch_dtype="auto"
            )
            model.save_pretrained(self.qwen_model_path)
            print("✅ Model downloaded")
            
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def download_model_with_huggingface_hub(self) -> bool:
        """Download model using huggingface_hub"""
        try:
            print("📥 Downloading using huggingface_hub...")
            
            from huggingface_hub import snapshot_download
            
            snapshot_download(
                repo_id="Qwen/Qwen2-7B-Instruct",
                local_dir=str(self.qwen_model_path),
                local_dir_use_symlinks=False
            )
            
            print("✅ Model downloaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def verify_model_integrity(self) -> bool:
        """Verify that the downloaded model is complete and working"""
        try:
            print("🔍 Verifying model integrity...")
            
            status = self.check_model_status()
            
            if not status["essential_files_present"]:
                print("❌ Essential files missing")
                return False
            
            if len(status["model_files_found"]) == 0:
                print("❌ No model weight files found")
                return False
            
            # Try to load tokenizer
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                str(self.qwen_model_path),
                trust_remote_code=True
            )
            print("✅ Tokenizer loads successfully")
            
            # Try to load model config
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                str(self.qwen_model_path),
                trust_remote_code=True
            )
            print("✅ Model config loads successfully")
            
            print(f"✅ Model verification complete ({status['model_size_gb']} GB)")
            return True
            
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False
    
    def show_model_info(self):
        """Display detailed model information"""
        status = self.check_model_status()
        
        print("\n📊 Model Status Report")
        print("=" * 50)
        print(f"Models Directory: {self.models_base_path}")
        print(f"  Exists: {'✅' if status['models_dir_exists'] else '❌'}")
        
        print(f"\nQwen2-7B-Instruct Model: {self.qwen_model_path}")
        print(f"  Directory exists: {'✅' if status['model_dir_exists'] else '❌'}")
        print(f"  Essential files: {'✅' if status['essential_files_present'] else '❌'}")
        print(f"  Model files: {len(status['model_files_found'])}")
        print(f"  Total size: {status['model_size_gb']} GB")
        print(f"  Download needed: {'Yes' if status['download_needed'] else 'No'}")
        
        if status['model_files_found']:
            print(f"\n📁 Model Files Found:")
            for file in status['model_files_found']:
                print(f"  • {file}")
        
        print("=" * 50)
    
    def setup_model_complete(self) -> bool:
        """Complete model setup process"""
        print("🔧 Starting complete model setup...")
        
        # Create directories
        if not self.create_models_directory():
            return False
        
        # Check current status
        status = self.check_model_status()
        
        if not status["download_needed"]:
            print("✅ Model already downloaded and ready!")
            return self.verify_model_integrity()
        
        # Try different download methods
        print("📥 Model download needed...")
        
        # Method 1: transformers
        if self.download_model_with_transformers():
            return self.verify_model_integrity()
        
        # Method 2: huggingface_hub
        print("Trying alternative download method...")
        if self.download_model_with_huggingface_hub():
            return self.verify_model_integrity()
        
        print("❌ All download methods failed")
        return False
    
    def create_model_symlink(self, source_path: str) -> bool:
        """Create symlink if model exists elsewhere"""
        try:
            source = Path(source_path)
            if not source.exists():
                print(f"❌ Source model not found: {source}")
                return False
            
            # Remove existing directory if it exists
            if self.qwen_model_path.exists():
                import shutil
                shutil.rmtree(self.qwen_model_path)
            
            # Create symlink
            self.qwen_model_path.symlink_to(source)
            print(f"✅ Created symlink: {self.qwen_model_path} -> {source}")
            
            return self.verify_model_integrity()
            
        except Exception as e:
            print(f"❌ Failed to create symlink: {e}")
            return False


def main():
    """Main setup function"""
    print("🤖 Friend AI Model Setup Helper")
    print("=" * 50)
    
    helper = ModelSetupHelper()
    
    while True:
        print("\n🎯 Options:")
        print("1. Check model status")
        print("2. Download model")
        print("3. Verify model")
        print("4. Complete setup (recommended)")
        print("5. Create symlink from existing model")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            helper.show_model_info()
        
        elif choice == '2':
            success = helper.download_model_with_transformers()
            if success:
                print("✅ Download completed!")
            else:
                print("❌ Download failed!")
        
        elif choice == '3':
            success = helper.verify_model_integrity()
            if success:
                print("✅ Model verification passed!")
            else:
                print("❌ Model verification failed!")
        
        elif choice == '4':
            success = helper.setup_model_complete()
            if success:
                print("✅ Complete setup successful!")
                print("Your Friend AI NLU system is ready to use!")
            else:
                print("❌ Setup failed!")
        
        elif choice == '5':
            source_path = input("Enter path to existing Qwen2-7B-Instruct model: ").strip()
            success = helper.create_model_symlink(source_path)
            if success:
                print("✅ Symlink created successfully!")
        
        elif choice == '6':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-6.")


if __name__ == "__main__":
    main()