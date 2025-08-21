"""
Demo script for Friend AI NLU System
Complete demonstration of conversational AI capabilities
"""

import asyncio
import logging
import sys
from pathlib import Path
import time
from typing import List, Dict

# Add the current directory to Python path for imports
# Direct imports from same directory
try:
    from config import get_quick_config, setup_for_development
    from nlu_main import FriendAINLU
    from friend_persona import ConversationMood
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all NLU files are in the same directory")
    print("Required files: config.py, nlu_main.py, friend_persona.py, qwen_handler.py")
    sys.exit(1)

class FriendAIDemo:
    """Demo application for Friend AI NLU system"""
    
    def __init__(self):
        self.config = get_quick_config()
        self.nlu_system = None
        self.demo_conversations = self._get_demo_conversations()
        self.interactive_mode = True
        
    def _get_demo_conversations(self) -> List[Dict]:
        """Pre-defined conversations for demo"""
        return [
            {
                "name": "Casual Greeting",
                "inputs": [
                    "Hey! How's it going?",
                    "Not much, just chilling. What about you?",
                    "That sounds cool! I love lazy days."
                ]
            },
            {
                "name": "Exciting News",
                "inputs": [
                    "Dude, I just got the job I applied for!",
                    "Thanks! I'm so excited to start next week.",
                    "Yeah, it's exactly what I wanted to do!"
                ]
            },
            {
                "name": "Seeking Support",
                "inputs": [
                    "I'm feeling really stressed about my presentation tomorrow...",
                    "You're right, I have prepared well. Just nervous I guess.",
                    "Thanks for the encouragement, you always know what to say."
                ]
            },
            {
                "name": "Asking for Advice",
                "inputs": [
                    "I need some advice about asking someone out.",
                    "We've been friends for a while, but I'm not sure if they like me.",
                    "You think I should just go for it?"
                ]
            },
            {
                "name": "Sharing Something Funny",
                "inputs": [
                    "You won't believe what happened to me today!",
                    "I was in a video meeting and my cat jumped on my keyboard.",
                    "Right in the middle of my presentation! Everyone was cracking up."
                ]
            }
        ]
    
    async def initialize_system(self) -> bool:
        """Initialize the Friend AI NLU system"""
        print("🚀 Initializing Friend AI NLU System...")
        print(f"📁 Project path: {self.config.PROJECT_ROOT}")
        print(f"🤖 Models path: {self.config.MODELS_BASE_PATH}")
        
        # Validate model path
        model_validation = self.config.validate_model_path()
        print(f"🧠 Model validation:")
        
        if model_validation["local_path_exists"]:
            print(f"  ✅ Local model found: {self.config.MODEL_CONFIG['model_path']}")
            if model_validation["has_model_files"]:
                print(f"  ✅ Model files verified ({model_validation.get('model_files_found', 0)} weight files)")
            else:
                print(f"  ⚠️  Model directory exists but missing essential files")
        else:
            print(f"  ❌ Local model not found: {self.config.MODEL_CONFIG['model_path']}")
            print(f"  🔥 Will download from HuggingFace: {self.config.MODEL_CONFIG['huggingface_name']}")
        
        print(f"💾 Memory path: {self.config.MEMORY_PATH}")
        
        try:
            # Setup for development
            setup_for_development()
            
            # Initialize NLU system with config
            self.nlu_system = FriendAINLU(
                config=self.config,
                memory_folder=str(self.config.MEMORY_PATH)
            )
            
            # Initialize the system
            success = await self.nlu_system.initialize()
            
            if success:
                print("✅ Friend AI NLU System initialized successfully!")
                self._show_system_info()
                return True
            else:
                print("❌ Failed to initialize Friend AI NLU System")
                return False
                
        except Exception as e:
            print(f"❌ Error during initialization: {str(e)}")
            return False
    
    def _show_system_info(self):
        """Display system information"""
        if not self.nlu_system:
            return
            
        info = self.nlu_system.get_system_info()
        print("\n📊 System Information:")
        print(f"  Session ID: {info['nlu_system']['session_id']}")
        print(f"  Model Device: {info['qwen_model']['device']}")
        print(f"  Model Loaded: {info['qwen_model']['loaded']}")
        print(f"  Memory Folder: {info['nlu_system']['memory_folder']}")
        print(f"  Current Mood: {info['nlu_system']['current_mood']}")
    
    async def run_demo_conversations(self):
        """Run predefined demo conversations"""
        print("\n🎬 Running Demo Conversations")
        print("=" * 50)
        
        for i, conversation in enumerate(self.demo_conversations, 1):
            print(f"\n--- Demo {i}: {conversation['name']} ---")
            
            for j, user_input in enumerate(conversation['inputs'], 1):
                print(f"\n💬 Exchange {j}:")
                print(f"You: {user_input}")
                
                # Process input
                start_time = time.time()
                response = self.nlu_system.process_input(user_input)
                end_time = time.time()
                
                # Display response
                print(f"Friend AI: {response.text}")
                print(f"Mood: {response.mood.value} | Confidence: {response.confidence:.2f} | Time: {end_time-start_time:.1f}s")
                
                # Small delay for natural flow
                await asyncio.sleep(1)
            
            print("\n" + "-" * 40)
            await asyncio.sleep(2)
    
    async def run_interactive_mode(self):
        """Run interactive conversation mode"""
        print("\n💬 Interactive Mode Started")
        print("=" * 50)
        print("Start chatting with your Friend AI! (Type 'quit', 'exit', or 'bye' to stop)")
        print("Commands: 'stats' for statistics, 'clear' to clear history, 'config' for settings")
        print()
        
        conversation_count = 0
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                # Handle exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    response = self.nlu_system.process_input(user_input)
                    print(f"Friend AI: {response.text}")
                    break
                
                # Handle special commands
                if user_input.lower() == 'stats':
                    self._show_conversation_stats()
                    continue
                elif user_input.lower() == 'clear':
                    self.nlu_system.clear_conversation_history()
                    print("🔥 Conversation history cleared!")
                    conversation_count = 0
                    continue
                elif user_input.lower() == 'config':
                    self._show_config_info()
                    continue
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                # Skip empty inputs
                if not user_input:
                    continue
                
                # Process input and get response
                start_time = time.time()
                response = self.nlu_system.process_input(user_input)
                end_time = time.time()
                
                # Display response with metadata
                print(f"Friend AI: {response.text}")
                
                # Show additional info in debug mode
                if self.config.DEBUG_CONFIG.get("debug_mode", False):
                    print(f"[Debug] Mood: {response.mood.value} | Confidence: {response.confidence:.2f} | Time: {end_time-start_time:.1f}s")
                
                conversation_count += 1
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for chatting!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                continue
    
    def _show_conversation_stats(self):
        """Display conversation statistics"""
        stats = self.nlu_system.get_conversation_stats()
        
        print("\n📊 Conversation Statistics:")
        print(f"  Total exchanges: {stats.get('total_exchanges', 0)}")
        print(f"  This session: {stats.get('session_exchanges', 0)}")
        print(f"  Current mood: {stats.get('current_mood', 'unknown')}")
        
        if 'mood_distribution' in stats:
            print("  Mood distribution:")
            for mood, count in stats['mood_distribution'].items():
                print(f"    {mood}: {count}")
        
        if 'topic_distribution' in stats:
            print("  Topic distribution:")
            for topic, count in stats['topic_distribution'].items():
                print(f"    {topic}: {count}")
        print()
    
    def _show_config_info(self):
        """Display current configuration"""
        print("\n🔧 Current Configuration:")
        print(f"  Model: {self.config.MODEL_CONFIG['model_name']}")
        print(f"  Device: {self.config.MODEL_CONFIG['device']}")
        print(f"  Temperature: {self.config.GENERATION_CONFIG['temperature']}")
        print(f"  Max tokens: {self.config.GENERATION_CONFIG['max_new_tokens']}")
        print(f"  Debug mode: {self.config.DEBUG_CONFIG['debug_mode']}")
        print(f"  Memory enabled: {self.config.MEMORY_CONFIG['save_conversations']}")
        print()
    
    def _show_help(self):
        """Display help information"""
        print("\n❓ Friend AI Help:")
        print("  Commands:")
        print("    'stats' - Show conversation statistics")
        print("    'clear' - Clear conversation history")
        print("    'config' - Show current configuration")
        print("    'help' - Show this help message")
        print("    'quit', 'exit', 'bye' - End conversation")
        print()
        print("  Tips:")
        print("    - Talk naturally like you would with a close friend")
        print("    - Share your emotions and feelings")
        print("    - Ask for advice or support when needed")
        print("    - The AI remembers your conversation within the session")
        print()
    
    async def run_performance_test(self):
        """Run performance tests"""
        print("\n⚡ Performance Testing")
        print("=" * 50)
        
        test_inputs = [
            "Hi there!",
            "How are you doing today?",
            "I'm having a great day, thanks for asking!",
            "What do you think about the weather?",
            "I love sunny days like this."
        ]
        
        response_times = []
        
        for i, test_input in enumerate(test_inputs, 1):
            print(f"Test {i}: Processing '{test_input}'")
            
            start_time = time.time()
            response = self.nlu_system.process_input(test_input, save_to_memory=False)
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            print(f"  Response time: {response_time:.2f}s")
            print(f"  Response length: {len(response.text)} chars")
            print(f"  Confidence: {response.confidence:.2f}")
            print()
        
        # Performance summary
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        print(f"📊 Performance Summary:")
        print(f"  Average response time: {avg_time:.2f}s")
        print(f"  Fastest response: {min_time:.2f}s")
        print(f"  Slowest response: {max_time:.2f}s")
        print(f"  Total tests: {len(test_inputs)}")
        
    def cleanup(self):
        """Cleanup resources"""
        if self.nlu_system:
            self.nlu_system.shutdown()
            print("🧹 System cleanup completed")


async def main():
    """Main demo function"""
    print("🤖 Friend AI NLU System Demo")
    print("=" * 60)
    
    demo = FriendAIDemo()
    
    try:
        # Initialize the system
        if not await demo.initialize_system():
            print("❌ Failed to initialize system. Exiting.")
            return
        
        # Show menu
        while True:
            print("\n🎯 Demo Options:")
            print("1. Run demo conversations")
            print("2. Interactive chat mode")
            print("3. Performance test")
            print("4. Show system info")
            print("5. Exit")
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == '1':
                await demo.run_demo_conversations()
            elif choice == '2':
                await demo.run_interactive_mode()
            elif choice == '3':
                await demo.run_performance_test()
            elif choice == '4':
                demo._show_system_info()
            elif choice == '5':
                print("👋 Thanks for trying Friend AI!")
                break
            else:
                print("❌ Invalid choice. Please select 1-5.")
    
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
    
    except Exception as e:
        print(f"❌ Demo error: {str(e)}")
        logging.exception("Demo failed with exception")
    
    finally:
        demo.cleanup()


def setup_logging():
    """Setup logging for the demo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('friend_ai_demo.log')
        ]
    )


def check_requirements():
    """Check if all requirements are met"""
    requirements = []
    
    try:
        import torch
        requirements.append(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        requirements.append("❌ PyTorch: Not installed")
        return False
    
    try:
        import transformers
        requirements.append(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        requirements.append("❌ Transformers: Not installed")
        return False
    
    # Check CUDA availability
    if torch.cuda.is_available():
        requirements.append(f"✅ CUDA: Available (GPU: {torch.cuda.get_device_name(0)})")
    else:
        requirements.append("⚠️  CUDA: Not available (will use CPU)")
    
    print("📋 Requirements Check:")
    for req in requirements:
        print(f"  {req}")
    
    return True


if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements not met. Please install missing packages:")
        print("  pip install torch transformers accelerate")
        sys.exit(1)
    
    # Run the demo
    print("\n" + "="*60)
    asyncio.run(main())