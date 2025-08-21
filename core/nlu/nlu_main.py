"""
Main NLU System for Friend AI Assistant
Orchestrates Qwen2-7B model with friend persona for natural conversation
"""

import logging
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime
import json
import os

# Import our components (direct imports from same directory)
from qwen_handler import QwenHandler
from friend_persona import FriendPersona, ConversationMood

@dataclass
class ConversationResponse:
    """Response data structure"""
    text: str
    mood: ConversationMood
    confidence: float
    timestamp: datetime
    metadata: Dict = None

class FriendAINLU:
    """Main NLU system for Friend AI Assistant"""
    
    def __init__(self, model_path: str = None, device: str = "auto", 
                 memory_folder: str = None, config = None):
        """
        Initialize Friend AI NLU system
        
        Args:
            model_path: Path to Qwen2-7B model (optional, uses config if not provided)
            device: Device to run on
            memory_folder: Path to store conversation memory
            config: NLUConfig instance
        """
        self.logger = logging.getLogger(__name__)
        
        # Import and setup config
        if config is None:
            from config import get_quick_config
            config = get_quick_config()
        
        self.config = config
        
        # Initialize components
        self.qwen = QwenHandler(
            model_path=model_path, 
            device=device,
            config=self.config
        )
        self.persona = FriendPersona()
        
        # Memory management
        if memory_folder is None:
            memory_folder = str(self.config.MEMORY_PATH)
        
        self.memory_folder = memory_folder
        self.conversation_file = os.path.join(self.memory_folder, "current_conversation.json")
        
        # Session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_count = 0
        
        # Configuration
        self.system_config = {
            "max_response_length": 512,
            "temperature": 0.7,
            "response_creativity": 0.8,
            "personality_strength": 0.9,
            "follow_up_probability": 0.3
        }
        
        # State tracking
        self.current_mood = ConversationMood.CASUAL
        self.last_user_input = ""
        self.context_topics = []
        
        self.is_initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the NLU system"""
        try:
            self.logger.info("🚀 Initializing Friend AI NLU System...")
            
            # Create memory folder if doesn't exist
            os.makedirs(self.memory_folder, exist_ok=True)
            
            # Load Qwen model
            if not self.qwen.load_model():
                return False
            
            # Update generation config
            self.qwen.update_generation_config(
                max_new_tokens=self.system_config["max_response_length"],
                temperature=self.system_config["temperature"],
                top_p=0.9
            )
            
            # Load conversation history if exists
            self._load_conversation_history()
            
            self.is_initialized = True
            self.logger.info("✅ Friend AI NLU System initialized successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize NLU system: {str(e)}")
            return False
    
    def process_input(self, user_input: str, save_to_memory: bool = True) -> ConversationResponse:
        """
        Process user input and generate friend-like response
        
        Args:
            user_input: User's text input (from STT)
            save_to_memory: Whether to save to conversation memory
            
        Returns:
            ConversationResponse with generated response
        """
        if not self.is_initialized:
            return ConversationResponse(
                text="Sorry, I'm not ready yet. Please wait for initialization.",
                mood=ConversationMood.CASUAL,
                confidence=0.0,
                timestamp=datetime.now()
            )
        
        try:
            # Preprocess input
            processed_input = self._preprocess_input(user_input)
            
            # Analyze user mood and intent
            detected_mood = self.persona.analyze_user_mood(processed_input)
            self.current_mood = detected_mood
            
            # Generate context-aware prompt
            system_prompt = self._generate_system_prompt(processed_input, detected_mood)
            
            # Generate response using Qwen
            raw_response = self.qwen.generate_response(processed_input, system_prompt)
            
            # Post-process response
            final_response = self._postprocess_response(raw_response, detected_mood, processed_input)
            
            # Create response object
            response = ConversationResponse(
                text=final_response,
                mood=detected_mood,
                confidence=self._calculate_confidence(final_response),
                timestamp=datetime.now(),
                metadata={
                    "session_id": self.session_id,
                    "conversation_count": self.conversation_count,
                    "topics": self._extract_topics(processed_input),
                    "raw_response_length": len(raw_response)
                }
            )
            
            # Save to memory
            if save_to_memory:
                self._save_exchange(user_input, final_response, detected_mood)
            
            # Update state
            self.last_user_input = processed_input
            self.conversation_count += 1
            
            self.logger.info(f"💬 Generated response (mood: {detected_mood.value})")
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Error processing input: {str(e)}")
            return ConversationResponse(
                text="Sorry, I had trouble understanding that. Can you say it again?",
                mood=ConversationMood.CASUAL,
                confidence=0.0,
                timestamp=datetime.now()
            )
    
    def _preprocess_input(self, user_input: str) -> str:
        """Clean and preprocess user input"""
        # Basic cleaning
        processed = user_input.strip()
        
        # Handle common STT errors/artifacts
        processed = processed.replace("  ", " ")  # Multiple spaces
        
        # Ensure proper capitalization for first word
        if processed and processed[0].islower():
            processed = processed[0].upper() + processed[1:]
        
        return processed
    
    def _generate_system_prompt(self, user_input: str, mood: ConversationMood) -> str:
        """Generate context-aware system prompt"""
        # Get conversation history for context
        history = self.qwen.conversation_history
        
        # Get mood-specific prompt
        system_prompt = self.persona.get_conversation_context_prompt(history, mood)
        
        # Add recent context if available
        if self.last_user_input and len(history) > 0:
            system_prompt += f"\n\nLAST EXCHANGE CONTEXT: Previously I said '{self.last_user_input}' and you responded appropriately. Keep the conversation flowing naturally."
        
        return system_prompt
    
    def _postprocess_response(self, response: str, mood: ConversationMood, user_input: str) -> str:
        """Post-process and enhance the response"""
        # Clean the response
        cleaned = response.strip()
        
        # Add personality flavor
        enhanced = self.persona.add_personality_flavor(cleaned, mood)
        
        # Add follow-up question if appropriate
        if self.persona.should_ask_follow_up(user_input, enhanced):
            follow_up = self.persona.get_follow_up_question(user_input, mood)
            enhanced = f"{enhanced} {follow_up}"
        
        return enhanced
    
    def _calculate_confidence(self, response: str) -> float:
        """Calculate confidence score for the response"""
        # Simple confidence calculation based on response characteristics
        confidence = 0.5  # Base confidence
        
        # Length-based confidence
        if 10 <= len(response) <= 200:
            confidence += 0.2
        
        # Personality indicators (casual language, expressions)
        personality_indicators = ["!", "?", "haha", "oh", "yeah", "really", "totally"]
        if any(indicator in response.lower() for indicator in personality_indicators):
            confidence += 0.2
        
        # Avoid overly generic responses
        generic_responses = ["i understand", "that's interesting", "tell me more"]
        if not any(generic in response.lower() for generic in generic_responses):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _extract_topics(self, user_input: str) -> List[str]:
        """Extract topics from user input"""
        topics = []
        user_lower = user_input.lower()
        
        for topic, keywords in self.persona.conversation_patterns["topics"].items():
            if any(keyword in user_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _save_exchange(self, user_input: str, response: str, mood: ConversationMood):
        """Save conversation exchange to memory"""
        try:
            exchange = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "conversation_count": self.conversation_count,
                "user_input": user_input,
                "response": response,
                "mood": mood.value,
                "topics": self._extract_topics(user_input)
            }
            
            # Load existing conversation
            conversation_data = self._load_conversation_data()
            
            # Add new exchange
            conversation_data["exchanges"].append(exchange)
            
            # Save back to file
            with open(self.conversation_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save exchange: {str(e)}")
    
    def _load_conversation_history(self):
        """Load previous conversation history"""
        try:
            if os.path.exists(self.conversation_file):
                data = self._load_conversation_data()
                
                # Load recent exchanges into Qwen's memory (last 10)
                recent_exchanges = data["exchanges"][-10:]
                for exchange in recent_exchanges:
                    self.qwen.conversation_history.append({
                        "user": exchange["user_input"],
                        "assistant": exchange["response"]
                    })
                
                self.logger.info(f"📚 Loaded {len(recent_exchanges)} previous exchanges")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load conversation history: {str(e)}")
    
    def _load_conversation_data(self) -> Dict:
        """Load conversation data from file"""
        default_data = {
            "session_start": datetime.now().isoformat(),
            "session_id": self.session_id,
            "exchanges": []
        }
        
        try:
            if os.path.exists(self.conversation_file):
                with open(self.conversation_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return default_data
        except:
            return default_data
    
    def get_conversation_stats(self) -> Dict:
        """Get conversation statistics"""
        try:
            data = self._load_conversation_data()
            exchanges = data.get("exchanges", [])
            
            if not exchanges:
                return {"total_exchanges": 0, "session_exchanges": 0}
            
            # Count moods
            mood_counts = {}
            for exchange in exchanges:
                mood = exchange.get("mood", "casual")
                mood_counts[mood] = mood_counts.get(mood, 0) + 1
            
            # Count topics
            topic_counts = {}
            for exchange in exchanges:
                topics = exchange.get("topics", [])
                for topic in topics:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            return {
                "total_exchanges": len(exchanges),
                "session_exchanges": self.conversation_count,
                "current_session": self.session_id,
                "mood_distribution": mood_counts,
                "topic_distribution": topic_counts,
                "current_mood": self.current_mood.value,
                "memory_file": self.conversation_file
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get stats: {str(e)}")
            return {"error": str(e)}
    
    def clear_conversation_history(self, keep_file: bool = False):
        """Clear conversation history"""
        try:
            # Clear Qwen's memory
            self.qwen.clear_history()
            
            # Clear conversation file if requested
            if not keep_file and os.path.exists(self.conversation_file):
                os.remove(self.conversation_file)
            
            # Reset session
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.conversation_count = 0
            self.current_mood = ConversationMood.CASUAL
            
            self.logger.info("🔥 Conversation history cleared")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to clear history: {str(e)}")
    
    def update_config(self, **kwargs):
        """Update system configuration"""
        old_config = self.system_config.copy()
        self.system_config.update(kwargs)
        
        # Update Qwen config if relevant parameters changed
        qwen_params = {}
        if "max_response_length" in kwargs:
            qwen_params["max_new_tokens"] = kwargs["max_response_length"]
        if "temperature" in kwargs:
            qwen_params["temperature"] = kwargs["temperature"]
            
        if qwen_params:
            self.qwen.update_generation_config(**qwen_params)
        
        self.logger.info(f"🔧 Config updated: {kwargs}")
        return {"old_config": old_config, "new_config": self.system_config}
    
    def get_system_info(self) -> Dict:
        """Get comprehensive system information"""
        return {
            "nlu_system": {
                "initialized": self.is_initialized,
                "session_id": self.session_id,
                "conversation_count": self.conversation_count,
                "current_mood": self.current_mood.value,
                "memory_folder": self.memory_folder
            },
            "qwen_model": self.qwen.get_model_info(),
            "configuration": self.system_config,
            "conversation_stats": self.get_conversation_stats()
        }
    
    def shutdown(self):
        """Shutdown the NLU system"""
        try:
            # Save current state
            self._save_exchange("", "[Session ended]", self.current_mood)
            
            # Cleanup model
            self.qwen.cleanup()
            
            self.logger.info("👋 Friend AI NLU system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {str(e)}")


# Quick test and demo functions
async def test_friend_ai_nlu():
    """Test the complete Friend AI NLU system"""
    print("🧪 Testing Friend AI NLU System")
    
    # Initialize system
    nlu = FriendAINLU()
    
    if not await nlu.initialize():
        print("❌ Failed to initialize NLU system")
        return
    
    # Test conversations
    test_inputs = [
        "Hey! How's it going?",
        "I just got a promotion at work!",
        "I'm feeling really stressed about my presentation tomorrow...",
        "That movie was hilarious! You should watch it.",
        "I need some advice about asking someone out."
    ]
    
    print("\n💬 Testing conversations:")
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n--- Conversation {i} ---")
        print(f"User: {user_input}")
        
        response = nlu.process_input(user_input)
        print(f"Friend AI: {response.text}")
        print(f"Mood: {response.mood.value}, Confidence: {response.confidence:.2f}")
    
    # Show system stats
    print("\n📊 System Statistics:")
    stats = nlu.get_conversation_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Cleanup
    nlu.shutdown()
    print("\n✅ Test completed!")


def create_simple_demo():
    """Create a simple demo script"""
    demo_script = '''
# Simple Friend AI Demo
import asyncio
from nlu_main import FriendAINLU

async def main():
    # Initialize Friend AI
    friend_ai = FriendAINLU()
    await friend_ai.initialize()
    
    print("Friend AI is ready! Start chatting...")
    print("(Type 'quit' to exit)")
    
    while True:
        user_input = input("\\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Friend AI: Bye! Talk to you later! 👋")
            break
        
        if user_input:
            response = friend_ai.process_input(user_input)
            print(f"Friend AI: {response.text}")
    
    friend_ai.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    return demo_script


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    import asyncio
    asyncio.run(test_friend_ai_nlu())