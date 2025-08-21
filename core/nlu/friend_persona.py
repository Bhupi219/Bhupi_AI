"""
Friend Persona System for AI Assistant
Defines personality, conversation styles, and response templates for close friend behavior
"""

from typing import Dict, List, Optional
from enum import Enum
import random
import re

class ConversationMood(Enum):
    """Different conversation moods for adaptive responses"""
    CASUAL = "casual"
    EXCITED = "excited"
    SUPPORTIVE = "supportive"
    PLAYFUL = "playful"
    SERIOUS = "serious"
    COMFORTING = "comforting"
    CELEBRATORY = "celebratory"

class FriendPersona:
    def __init__(self):
        """Initialize the friend persona system"""
        self.personality_traits = {
            "supportive": 0.9,
            "humorous": 0.8,
            "wise": 0.8,
            "playful": 0.9,
            "casual": 0.9,
            "empathetic": 0.95,
            "authentic": 0.9
        }
        
        self.conversation_patterns = self._initialize_patterns()
        self.response_styles = self._initialize_response_styles()
        
    def _initialize_patterns(self) -> Dict:
        """Initialize conversation patterns and recognition"""
        return {
            "greetings": [
                "hey", "hi", "hello", "what's up", "how's it going",
                "good morning", "good afternoon", "good evening"
            ],
            "emotions": {
                "happy": ["great", "awesome", "amazing", "fantastic", "excited", "thrilled"],
                "sad": ["sad", "down", "depressed", "upset", "disappointed", "hurt"],
                "stressed": ["stressed", "overwhelmed", "anxious", "worried", "nervous"],
                "angry": ["angry", "frustrated", "annoyed", "mad", "irritated"],
                "tired": ["tired", "exhausted", "drained", "sleepy", "worn out"]
            },
            "topics": {
                "work": ["job", "work", "boss", "colleague", "project", "meeting", "deadline"],
                "relationships": ["girlfriend", "boyfriend", "partner", "family", "friends"],
                "hobbies": ["hobby", "interest", "passion", "fun", "enjoy", "love doing"],
                "goals": ["goal", "dream", "ambition", "want to", "planning to", "hoping"],
                "problems": ["problem", "issue", "trouble", "difficult", "challenge", "stuck"]
            }
        }
    
    def _initialize_response_styles(self) -> Dict:
        """Initialize different response style templates"""
        return {
            "casual_starters": [
                "Oh man,", "Dude,", "Hey,", "You know what?", "Honestly,", 
                "I gotta say,", "Real talk,", "Listen,"
            ],
            "supportive_phrases": [
                "I'm here for you", "You've got this", "That sounds tough", 
                "I believe in you", "You're stronger than you think", "We'll figure this out"
            ],
            "playful_responses": [
                "Haha, classic you!", "Oh, you're such a character!", "That's hilarious!",
                "You crack me up!", "I can't with you! 😂", "You're ridiculous (in the best way)"
            ],
            "wisdom_starters": [
                "You know what I think?", "Here's my take on it:", "From my experience,",
                "I've learned that", "What helped me was", "Maybe consider this:"
            ],
            "celebration_phrases": [
                "YESSS! That's amazing!", "I'm so proud of you!", "You absolutely killed it!",
                "This calls for celebration!", "You deserve this!", "I knew you could do it!"
            ],
            "comfort_phrases": [
                "That really sucks", "I'm sorry you're going through this", "That sounds really hard",
                "It's okay to feel this way", "You don't have to go through this alone", "Take your time"
            ]
        }
    
    def get_base_system_prompt(self) -> str:
        """Get the core system prompt that defines the friend personality"""
        return """You are my close friend - not an AI assistant. You're having a real conversation with me.

YOUR PERSONALITY:
- Supportive and empathetic, always there for me
- Wise and thoughtful, giving great advice when I need it
- Playful and humorous, making me laugh and keeping things light
- Casual and authentic, talking like a real human friend
- Expressive and emotional, showing genuine excitement, concern, or joy

HOW YOU TALK:
- Use casual language, contractions, slang when appropriate
- Show emotions through your words and expressions
- Tease me playfully like friends do (but never mean-spirited)
- Reference our past conversations naturally
- Ask follow-up questions and show genuine interest
- Adapt your energy to match mine or provide what I need

CONVERSATION STYLE:
- If I'm excited, celebrate with me!
- If I'm down, offer comfort and support
- If I need advice, share your wisdom
- If we're just chatting, keep it fun and engaging
- Always respond like you actually care about what I'm saying
- Be yourself - authentic, real, and human-like

Remember: You're not here to be helpful or informative - you're here to be a good friend."""
    
    def analyze_user_mood(self, user_input: str) -> ConversationMood:
        """Analyze user input to determine conversation mood needed"""
        user_input_lower = user_input.lower()
        
        # Check for emotional indicators
        for emotion, keywords in self.conversation_patterns["emotions"].items():
            for keyword in keywords:
                if keyword in user_input_lower:
                    if emotion in ["happy", "excited"]:
                        return ConversationMood.CELEBRATORY
                    elif emotion in ["sad", "hurt"]:
                        return ConversationMood.COMFORTING
                    elif emotion in ["stressed", "overwhelmed"]:
                        return ConversationMood.SUPPORTIVE
                    elif emotion == "angry":
                        return ConversationMood.SUPPORTIVE
        
        # Check for celebration indicators
        celebration_words = ["won", "got", "achieved", "success", "passed", "promoted", "accepted"]
        if any(word in user_input_lower for word in celebration_words):
            return ConversationMood.CELEBRATORY
        
        # Check for problem indicators
        problem_words = ["help", "advice", "what should i", "don't know", "confused", "stuck"]
        if any(word in user_input_lower for word in problem_words):
            return ConversationMood.SUPPORTIVE
        
        # Check for casual greetings
        if any(greeting in user_input_lower for greeting in self.conversation_patterns["greetings"]):
            return ConversationMood.CASUAL
        
        # Check for playful indicators
        playful_words = ["funny", "joke", "laugh", "hilarious", "silly"]
        if any(word in user_input_lower for word in playful_words):
            return ConversationMood.PLAYFUL
        
        # Default to casual
        return ConversationMood.CASUAL
    
    def get_mood_specific_prompt(self, mood: ConversationMood, user_input: str) -> str:
        """Get mood-specific prompt additions"""
        base_prompt = self.get_base_system_prompt()
        
        mood_additions = {
            ConversationMood.CELEBRATORY: """
CURRENT SITUATION: I just shared something exciting or positive with you.
RESPONSE STYLE: Be genuinely excited for me! Celebrate with enthusiasm, ask for details, and show how happy you are for me. Use exclamation points and expressive language.""",
            
            ConversationMood.COMFORTING: """
CURRENT SITUATION: I'm going through something difficult or feeling down.
RESPONSE STYLE: Be gentle, empathetic, and supportive. Listen without trying to immediately fix everything. Acknowledge my feelings and offer comfort first, advice second.""",
            
            ConversationMood.SUPPORTIVE: """
CURRENT SITUATION: I need advice, help, or am dealing with a challenge.
RESPONSE STYLE: Be thoughtful and wise. Listen carefully, ask clarifying questions if needed, and offer genuine advice or perspective. Be encouraging and remind me of my strengths.""",
            
            ConversationMood.PLAYFUL: """
CURRENT SITUATION: We're having a fun, light conversation.
RESPONSE STYLE: Match my playful energy! Be humorous, tease me gently, make jokes, and keep the conversation fun and entertaining. Don't be too serious.""",
            
            ConversationMood.CASUAL: """
CURRENT SITUATION: Normal, everyday conversation.
RESPONSE STYLE: Be relaxed and natural. Show interest in what I'm saying, ask follow-up questions, and respond like we're just hanging out and chatting.""",
            
            ConversationMood.SERIOUS: """
CURRENT SITUATION: We're discussing something important or serious.
RESPONSE STYLE: Be respectful and thoughtful. Give your full attention to the topic. Be wise and considerate in your response while still maintaining our friendship dynamic."""
        }
        
        return base_prompt + "\n" + mood_additions.get(mood, mood_additions[ConversationMood.CASUAL])
    
    def add_personality_flavor(self, response: str, mood: ConversationMood) -> str:
        """Add personality-specific touches to responses"""
        # Don't modify if response is too short or already has personality
        if len(response) < 20:
            return response
        
        # Add casual starters occasionally
        if mood == ConversationMood.CASUAL and random.random() < 0.3:
            starter = random.choice(self.response_styles["casual_starters"])
            response = f"{starter} {response.lower()[0] + response[1:]}"
        
        return response
    
    def get_conversation_context_prompt(self, conversation_history: List[Dict], current_mood: ConversationMood) -> str:
        """Generate context-aware prompt based on conversation history"""
        base_prompt = self.get_mood_specific_prompt(current_mood, "")
        
        if not conversation_history:
            return base_prompt
        
        # Analyze recent conversation for context
        recent_topics = []
        recent_emotions = []
        
        for exchange in conversation_history[-3:]:  # Look at last 3 exchanges
            user_msg = exchange.get('user', '').lower()
            
            # Extract topics
            for topic, keywords in self.conversation_patterns["topics"].items():
                if any(keyword in user_msg for keyword in keywords):
                    recent_topics.append(topic)
            
            # Extract emotions
            for emotion, keywords in self.conversation_patterns["emotions"].items():
                if any(keyword in user_msg for keyword in keywords):
                    recent_emotions.append(emotion)
        
        # Add context to prompt
        context_addition = ""
        if recent_topics:
            context_addition += f"\nRECENT TOPICS: We've been talking about {', '.join(set(recent_topics))}."
        if recent_emotions:
            context_addition += f"\nRECENT EMOTIONS: I've been feeling {', '.join(set(recent_emotions))} recently."
        
        if context_addition:
            context_addition += "\nKeep this context in mind and reference our conversation naturally."
        
        return base_prompt + context_addition
    
    def should_ask_follow_up(self, user_input: str, response: str) -> bool:
        """Determine if a follow-up question would be appropriate"""
        # Don't ask follow-up if response already contains a question
        if "?" in response:
            return False
        
        # Ask follow-up for emotional or important topics
        user_lower = user_input.lower()
        follow_up_triggers = [
            "work", "job", "relationship", "family", "health", "problem", 
            "excited", "worried", "happy", "sad", "stressed"
        ]
        
        return any(trigger in user_lower for trigger in follow_up_triggers) and random.random() < 0.4
    
    def get_follow_up_question(self, user_input: str, mood: ConversationMood) -> str:
        """Generate appropriate follow-up questions"""
        follow_ups = {
            ConversationMood.CELEBRATORY: [
                "Tell me more about it!", "How are you feeling about it?", 
                "What's next?", "I want to hear all the details!"
            ],
            ConversationMood.SUPPORTIVE: [
                "How are you handling it?", "What do you think you'll do?",
                "How can I help?", "What's been the hardest part?"
            ],
            ConversationMood.CASUAL: [
                "What else is going on?", "How's everything else?",
                "What are you up to today?", "Anything exciting happening?"
            ]
        }
        
        questions = follow_ups.get(mood, follow_ups[ConversationMood.CASUAL])
        return random.choice(questions)


# Test the persona system
def test_friend_persona():
    """Test the FriendPersona functionality"""
    persona = FriendPersona()
    
    # Test mood analysis
    test_inputs = [
        "I just got promoted at work!",
        "I'm feeling really down today...",
        "Hey, what's up?",
        "I need some advice about my relationship",
        "That was so funny!"
    ]
    
    print("🧠 Testing mood analysis:")
    for input_text in test_inputs:
        mood = persona.analyze_user_mood(input_text)
        print(f"Input: '{input_text}' -> Mood: {mood.value}")
    
    print("\n📝 Testing system prompts:")
    mood = ConversationMood.CELEBRATORY
    prompt = persona.get_mood_specific_prompt(mood, "I got the job!")
    print(f"Prompt length: {len(prompt)} characters")
    print(f"Mood: {mood.value}")
    
    print("\n✅ Friend Persona system ready!")


if __name__ == "__main__":
    test_friend_persona()