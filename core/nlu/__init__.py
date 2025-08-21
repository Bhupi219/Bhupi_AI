"""
Friend AI NLU Package
Natural Language Understanding system for conversational AI assistant
"""

__version__ = "1.0.0"
__author__ = "Bhupendra Singh"
__description__ = "Friend AI NLU System - Conversational AI with personality"

# Import main components
try:
    from nlu_main import FriendAINLU, ConversationResponse
    from friend_persona import FriendPersona, ConversationMood
    from qwen_handler import QwenHandler
    from config import NLUConfig, get_quick_config
    
    __all__ = [
        'FriendAINLU',
        'ConversationResponse', 
        'FriendPersona',
        'ConversationMood',
        'QwenHandler',
        'NLUConfig',
        'get_quick_config'
    ]
    
    # Package is properly imported
    _PACKAGE_AVAILABLE = True
    
except ImportError as e:
    # Handle import errors gracefully
    _PACKAGE_AVAILABLE = False
    _IMPORT_ERROR = str(e)
    
    print(f"⚠️  Warning: Some NLU components couldn't be imported: {e}")
    print("Make sure all required files are present:")
    print("  - nlu_main.py")
    print("  - friend_persona.py") 
    print("  - qwen_handler.py")
    print("  - config.py")


def get_version():
    """Get package version"""
    return __version__


def get_system_info():
    """Get system information"""
    info = {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "package_available": _PACKAGE_AVAILABLE
    }
    
    if not _PACKAGE_AVAILABLE:
        info["import_error"] = _IMPORT_ERROR
    
    return info


def quick_setup():
    """Quick setup function for immediate use"""
    if not _PACKAGE_AVAILABLE:
        print("❌ Package not properly available. Cannot run quick setup.")
        return None
    
    try:
        print("🚀 Setting up Friend AI NLU...")
        
        # Get configuration
        config = get_quick_config()
        
        # Create NLU system
        nlu = FriendAINLU(
            model_path=config.MODEL_CONFIG["model_name"],
            device=config.MODEL_CONFIG["device"],
            memory_folder=str(config.MEMORY_PATH),
            config=config
        )
        
        print("✅ Friend AI NLU setup complete!")
        print("Use: await nlu.initialize() to start the system")
        
        return nlu
        
    except Exception as e:
        print(f"❌ Quick setup failed: {str(e)}")
        return None


def print_welcome():
    """Print welcome message"""
    print("🤖 Friend AI NLU System")
    print("=" * 40)
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print(f"Description: {__description__}")
    print("=" * 40)
    
    if _PACKAGE_AVAILABLE:
        print("✅ All components loaded successfully!")
        print("\nQuick start:")
        print("  from __init__ import quick_setup")
        print("  nlu = quick_setup()")
        print("  await nlu.initialize()")
    else:
        print("❌ Some components missing!")
        print(f"Error: {_IMPORT_ERROR}")


# Print welcome message when package is imported
if __name__ != "__main__":
    print_welcome()


# Example usage function
def example_usage():
    """Show example usage"""
    example_code = '''
# Example: Using Friend AI NLU System
import asyncio
from __init__ import quick_setup

async def main():
    # Quick setup
    nlu = quick_setup()
    
    # Initialize system
    if await nlu.initialize():
        print("Friend AI is ready!")
        
        # Have a conversation
        response = nlu.process_input("Hey! How's it going?")
        print(f"Friend AI: {response.text}")
        
        # Another message
        response = nlu.process_input("I just got a new job!")
        print(f"Friend AI: {response.text}")
        
        # Cleanup
        nlu.shutdown()
    else:
        print("Failed to initialize Friend AI")

# Run the example
if __name__ == "__main__":
    asyncio.run(main())
'''
    
    print("📝 Example Usage:")
    print(example_code)


if __name__ == "__main__":
    # Run when called directly
    print_welcome()
    example_usage()
    
    # Show system info
    info = get_system_info()
    print("\n📊 System Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")