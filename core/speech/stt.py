import whisper
import sounddevice as sd
import numpy as np
import webrtcvad
import noisereduce as nr
import threading
import queue
import time
from collections import deque
import asyncio
import re

class MultilingualRealTimeSTT:
    def __init__(self, model_size="small", sample_rate=16000, chunk_size=320, language_mode="auto"):
        """
        Multilingual Real-time Speech-to-Text engine with code-mixing support.
        
        Args:
            model_size (str): Whisper model size ("tiny", "base", "small", "medium", "large")
            sample_rate (int): Audio sample rate
            chunk_size (int): Audio chunk size for VAD (160, 320, or 640 for 16kHz)
            language_mode (str): Language detection mode:
                - "auto": Auto-detect all languages
                - "hindi": Hindi only
                - "english": English only
                - "hindi_english": Mixed Hindi + English (code-mixing)
        """
        # Use multilingual models (without .en suffix)
        self.model = whisper.load_model(model_size)
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.language_mode = language_mode
        self.vad = webrtcvad.Vad(1)
        
        # Language mapping for Whisper
        self.language_map = {
            "auto": None,
            "hindi": "hi",
            "english": "en",
            "hindi_english": None  # Will be handled specially
        }
        
        # Real-time processing buffers
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_recording = False
        self.is_processing = False
        
        # Audio buffer for accumulating speech segments
        self.audio_buffer = deque(maxlen=50)  # ~3 seconds at 320 chunk size
        self.speech_buffer = []
        self.silence_counter = 0
        self.speech_detected = False
        
        # Code-mixing detection patterns
        self.hindi_patterns = [
            r'[\u0900-\u097F]',  # Devanagari script
            r'\b(hai|hain|ka|ki|ke|ko|se|me|par|aur|ya|nahi|nahin|kya|koi|yeh|woh|jo|jab|kab|kahan|kyun|kaise|phir|abhi|ab|kal|aaj|bahut|thoda|zyada|achha|bura|paani|khana|ghar|naam|kaam|time|phone|money|computer|internet|facebook|whatsapp)\b',
            r'\b(main|mein|hum|tum|aap|woh|yeh|unka|unki|hamara|hamari|tumhara|tumhari|aapka|aapki|mera|meri|tera|teri)\b'
        ]
        
        self.english_patterns = [
            r'\b[a-zA-Z]+\b',  # Basic English words
            r'\b(the|is|are|was|were|have|has|had|will|would|can|could|should|may|might|must|shall|do|does|did|get|got|make|made|take|took|come|came|go|went|see|saw|know|knew|think|thought|want|wanted|like|liked|need|needed|work|worked|time|people|way|day|man|woman|life|world|hand|part|place|case|week|company|system|program|question|government|number|group|problem|fact|money|business|service|computer|internet|phone|email|website|social|media|technology)\b'
        ]
        
        self.setup_audio_device()
        print(f"Initialized with model: {model_size}")
        print(f"Language mode: {self._get_language_mode_description()}")

    def _get_language_mode_description(self):
        """Get description of current language mode"""
        descriptions = {
            "auto": "Auto-detect all languages",
            "hindi": "Hindi only",
            "english": "English only",
            "hindi_english": "Mixed Hindi + English (Code-mixing detection)"
        }
        return descriptions.get(self.language_mode, "Unknown")

    def setup_audio_device(self):
        """Set up audio device with fallback options"""
        try:
            devices = sd.query_devices()
            input_device_found = False
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    if not input_device_found:
                        sd.default.device[0] = i
                        print(f"Using input device: {device['name']}")
                        input_device_found = True
                        break
            
            if not input_device_found:
                sd.default.device[0] = None
                
        except Exception as e:
            print(f"Audio setup error: {e}")
            sd.default.device[0] = None

    def set_language_mode(self, language_mode):
        """
        Set the language detection mode
        
        Args:
            language_mode (str): "auto", "hindi", "english", or "hindi_english"
        """
        if language_mode in ["auto", "hindi", "english", "hindi_english"]:
            self.language_mode = language_mode
            print(f"Language mode set to: {self._get_language_mode_description()}")
        else:
            print(f"Invalid language mode: {language_mode}")
            print("Valid options: 'auto', 'hindi', 'english', 'hindi_english'")

    def _detect_code_mixing(self, text):
        """
        Detect if text contains code-mixing (Hindi + English)
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Analysis results with language breakdown
        """
        text_lower = text.lower()
        
        # Count Hindi indicators
        hindi_score = 0
        for pattern in self.hindi_patterns:
            hindi_matches = len(re.findall(pattern, text_lower))
            hindi_score += hindi_matches
        
        # Count English indicators  
        english_score = 0
        for pattern in self.english_patterns:
            english_matches = len(re.findall(pattern, text_lower))
            english_score += english_matches
        
        total_words = len(text.split())
        
        # Determine language composition
        if hindi_score > 0 and english_score > 0:
            language_type = "mixed"
            hindi_ratio = hindi_score / (hindi_score + english_score)
            english_ratio = english_score / (hindi_score + english_score)
        elif hindi_score > english_score:
            language_type = "hindi"
            hindi_ratio = 1.0
            english_ratio = 0.0
        elif english_score > hindi_score:
            language_type = "english"
            hindi_ratio = 0.0
            english_ratio = 1.0
        else:
            language_type = "unknown"
            hindi_ratio = 0.5
            english_ratio = 0.5
        
        return {
            "type": language_type,
            "hindi_score": hindi_score,
            "english_score": english_score,
            "hindi_ratio": hindi_ratio,
            "english_ratio": english_ratio,
            "total_words": total_words,
            "is_code_mixed": language_type == "mixed"
        }

    def _preprocess_audio(self, audio_data):
        """Preprocess audio data"""
        if audio_data.size == 0:
            return audio_data
        
        # Ensure consistent data type (float32)
        audio_data = audio_data.astype(np.float32)
            
        # Reduce noise
        try:
            reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=self.sample_rate)
            reduced_noise_audio = reduced_noise_audio.astype(np.float32)
        except Exception as e:
            print(f"Noise reduction failed: {e}, using original audio")
            reduced_noise_audio = audio_data
        
        # Normalize
        max_val = np.max(np.abs(reduced_noise_audio))
        if max_val > 0:
            normalized_audio = (reduced_noise_audio / max_val).astype(np.float32)
        else:
            normalized_audio = reduced_noise_audio
        
        return normalized_audio

    def audio_callback(self, indata, frames, time, status):
        """Callback function for continuous audio stream"""
        if status:
            print(f"Audio callback status: {status}")
        
        self.audio_queue.put(indata.copy())

    def process_audio_stream(self, silence_threshold=30):
        """Process audio stream in real-time with language mode support"""
        print(f"Real-time processing started in {self._get_language_mode_description()} mode...")
        
        while self.is_processing:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                
                # Convert to int16 for VAD
                chunk_int16 = (chunk * 32767).astype(np.int16)
                
                # Voice Activity Detection
                if len(chunk_int16) == self.chunk_size:
                    try:
                        is_speech = self.vad.is_speech(chunk_int16.tobytes(), self.sample_rate)
                    except:
                        is_speech = False
                else:
                    is_speech = False
                
                if is_speech:
                    self.speech_buffer.append(chunk_int16)
                    self.silence_counter = 0
                    self.speech_detected = True
                    print("*", end="", flush=True)
                else:
                    if self.speech_detected:
                        self.silence_counter += 1
                        
                        if self.silence_counter >= silence_threshold:
                            if self.speech_buffer:
                                self._process_speech_segment()
                            self._reset_speech_buffer()
                    print(".", end="", flush=True)
                
                self.audio_buffer.append(chunk_int16)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")

    def _process_speech_segment(self):
        """Process accumulated speech segment with language mode support"""
        if not self.speech_buffer:
            return
        
        try:
            # Combine speech chunks
            audio_data = np.concatenate(self.speech_buffer, axis=0).flatten()
            preprocessed_audio = self._preprocess_audio(audio_data.astype(np.float32))
            
            if preprocessed_audio.size > 0:
                # Prepare transcription options
                transcribe_options = {
                    'fp16': False,
                    'no_speech_threshold': 0.6
                }
                
                # Set language based on mode
                if self.language_mode == "hindi":
                    transcribe_options['language'] = 'hi'
                elif self.language_mode == "english":
                    transcribe_options['language'] = 'en'
                # For auto and hindi_english modes, let Whisper auto-detect
                
                # Transcribe
                result = self.model.transcribe(preprocessed_audio, **transcribe_options)
                text = result['text'].strip()
                detected_language = result.get('language', 'unknown')
                
                if text:
                    # Process based on language mode
                    processed_result = self._process_transcription_result(
                        text, detected_language, result
                    )
                    
                    if processed_result:
                        self._display_result(processed_result)
                        self.result_queue.put(processed_result)
                
        except Exception as e:
            print(f"Transcription error: {e}")

    def _process_transcription_result(self, text, detected_language, raw_result):
        """Process transcription result based on language mode"""
        timestamp = time.time()
        
        if self.language_mode == "hindi_english":
            # Perform code-mixing analysis
            mixing_analysis = self._detect_code_mixing(text)
            
            return {
                'text': text,
                'detected_language': detected_language,
                'language_mode': self.language_mode,
                'timestamp': timestamp,
                'code_mixing': mixing_analysis,
                'is_code_mixed': mixing_analysis['is_code_mixed'],
                'raw_result': raw_result
            }
        
        elif self.language_mode in ["hindi", "english"]:
            # Validate if result matches expected language
            expected_lang = "hi" if self.language_mode == "hindi" else "en"
            is_expected = detected_language == expected_lang
            
            return {
                'text': text,
                'detected_language': detected_language,
                'language_mode': self.language_mode,
                'timestamp': timestamp,
                'is_expected_language': is_expected,
                'raw_result': raw_result
            }
        
        else:  # auto mode
            return {
                'text': text,
                'detected_language': detected_language,
                'language_mode': self.language_mode,
                'timestamp': timestamp,
                'raw_result': raw_result
            }

    def _display_result(self, result):
        """Display transcription result with appropriate formatting"""
        text = result['text']
        detected_lang = result['detected_language']
        mode = result['language_mode']
        
        if mode == "hindi_english" and result.get('is_code_mixed'):
            mixing = result['code_mixing']
            hindi_pct = int(mixing['hindi_ratio'] * 100)
            english_pct = int(mixing['english_ratio'] * 100)
            print(f"\n[CODE-MIX] Hindi:{hindi_pct}% English:{english_pct}% → {text}")
            
        elif mode == "hindi_english" and not result.get('is_code_mixed'):
            mixing = result['code_mixing']
            dominant = "Hindi" if mixing['type'] == "hindi" else "English"
            print(f"\n[{dominant.upper()}] → {text}")
            
        elif mode in ["hindi", "english"]:
            expected = mode.upper()
            actual = detected_lang.upper()
            if result.get('is_expected_language', True):
                print(f"\n[{expected}] → {text}")
            else:
                print(f"\n[{expected}*] (detected: {actual}) → {text}")
                
        else:  # auto mode
            print(f"\n[{detected_lang.upper()}] → {text}")

    def _reset_speech_buffer(self):
        """Reset speech processing state"""
        self.speech_buffer = []
        self.silence_counter = 0
        self.speech_detected = False

    def start_realtime_transcription(self, callback=None):
        """Start real-time transcription with language mode support"""
        self.is_recording = True
        self.is_processing = True
        
        # Start processing thread
        processing_thread = threading.Thread(target=self.process_audio_stream)
        processing_thread.daemon = True
        processing_thread.start()
        
        # Start result monitoring thread
        if callback:
            result_thread = threading.Thread(target=self._monitor_results, args=(callback,))
            result_thread.daemon = True
            result_thread.start()
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self.chunk_size,
                callback=self.audio_callback
            ):
                print(f"\nReal-time transcription active.")
                print(f"Mode: {self._get_language_mode_description()}")
                print("Press Ctrl+C to stop.")
                while self.is_recording:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nStopping real-time transcription...")
        finally:
            self.stop_transcription()

    def _monitor_results(self, callback):
        """Monitor results queue and call callback"""
        while self.is_processing:
            try:
                result = self.result_queue.get(timeout=0.1)
                callback(result)
            except queue.Empty:
                continue

    def stop_transcription(self):
        """Stop real-time transcription"""
        self.is_recording = False
        self.is_processing = False

    def get_latest_results(self):
        """Get all pending results"""
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def start_streaming_transcription(self, window_duration=3.0, overlap=1.0):
        """Start streaming transcription with language mode support"""
        window_samples = int(window_duration * self.sample_rate)
        overlap_samples = int(overlap * self.sample_rate)
        step_samples = window_samples - overlap_samples
        
        audio_buffer = np.array([])
        
        print(f"Streaming transcription started (window: {window_duration}s, overlap: {overlap}s)")
        print(f"Mode: {self._get_language_mode_description()}")
        
        def audio_callback(indata, frames, time, status):
            nonlocal audio_buffer
            
            if status:
                print(f"Status: {status}")
            
            new_audio = indata[:, 0]
            audio_buffer = np.append(audio_buffer, new_audio)
            
            if len(audio_buffer) >= window_samples:
                window = audio_buffer[:window_samples]
                
                threading.Thread(
                    target=self._process_streaming_window,
                    args=(window.copy(),),
                    daemon=True
                ).start()
                
                audio_buffer = audio_buffer[step_samples:]
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=audio_callback,
                dtype='float32'
            ):
                print("Streaming transcription active. Press Ctrl+C to stop.")
                while True:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nStopping streaming transcription...")

    def _process_streaming_window(self, audio_window):
        """Process a streaming window with language mode support"""
        try:
            audio_window = audio_window.astype(np.float32)
            preprocessed = self._preprocess_audio(audio_window)
            
            if preprocessed.size > 0:
                transcribe_options = {
                    'fp16': False,
                    'no_speech_threshold': 0.7
                }
                
                # Set language based on mode
                if self.language_mode == "hindi":
                    transcribe_options['language'] = 'hi'
                elif self.language_mode == "english":
                    transcribe_options['language'] = 'en'
                
                result = self.model.transcribe(preprocessed, **transcribe_options)
                text = result['text'].strip()
                detected_language = result.get('language', 'unknown')
                
                if text and len(text) > 3:
                    processed_result = self._process_transcription_result(
                        text, detected_language, result
                    )
                    
                    if processed_result:
                        timestamp = time.strftime("%H:%M:%S")
                        self._display_streaming_result(processed_result, timestamp)
        
        except Exception as e:
            print(f"Error processing audio window: {e}")

    def _display_streaming_result(self, result, timestamp):
        """Display streaming result with timestamp"""
        text = result['text']
        mode = result['language_mode']
        
        if mode == "hindi_english" and result.get('is_code_mixed'):
            mixing = result['code_mixing']
            hindi_pct = int(mixing['hindi_ratio'] * 100)
            english_pct = int(mixing['english_ratio'] * 100)
            print(f"\n[{timestamp}] [CODE-MIX H:{hindi_pct}% E:{english_pct}%]: {text}")
        else:
            detected_lang = result['detected_language']
            print(f"\n[{timestamp}] [{detected_lang.upper()}]: {text}")

# Enhanced callback functions
def enhanced_transcription_callback(result):
    """Enhanced callback for handling transcribed text with code-mixing support"""
    if isinstance(result, dict):
        text = result['text']
        mode = result['language_mode']
        
        if mode == "hindi_english" and result.get('is_code_mixed'):
            mixing = result['code_mixing']
            print(f"CALLBACK - CODE-MIXED TEXT: {text}")
            print(f"  Hindi ratio: {mixing['hindi_ratio']:.2f}")
            print(f"  English ratio: {mixing['english_ratio']:.2f}")
        else:
            lang = result['detected_language']
            print(f"CALLBACK - [{lang.upper()}]: {text}")
    else:
        print(f"CALLBACK: {result}")

def main():
    print("Enhanced Multilingual Real-Time Speech-to-Text with Code-Mixing Support")
    print("=" * 70)
    
    # Model selection
    print("\nSelect Whisper model:")
    print("1. tiny (fastest, least accurate)")
    print("2. base (good balance)")
    print("3. small (recommended for real-time)")
    print("4. medium (more accurate, slower)")
    print("5. large (most accurate, slowest)")
    
    model_choice = input("Enter choice (1-5, default=3): ").strip() or "3"
    model_map = {"1": "tiny", "2": "base", "3": "small", "4": "medium", "5": "large"}
    model_size = model_map.get(model_choice, "small")
    
    # Language mode selection
    print("\nSelect language detection mode:")
    print("1. Auto-detect all languages")
    print("2. Hindi only")
    print("3. English only")
    print("4. Mixed Hindi + English (Code-mixing detection)")
    
    lang_choice = input("Enter choice (1-4, default=4): ").strip() or "4"
    language_modes = {
        "1": "auto",
        "2": "hindi", 
        "3": "english",
        "4": "hindi_english"
    }
    language_mode = language_modes.get(lang_choice, "hindi_english")
    
    # Initialize STT with selected options
    stt = MultilingualRealTimeSTT(model_size=model_size, language_mode=language_mode)
    
    print(f"\nInitialized with:")
    print(f"  Model: {model_size}")
    print(f"  Language mode: {stt._get_language_mode_description()}")
    
    print("\nChoose transcription method:")
    print("1. Real-time VAD-based transcription")
    print("2. Streaming with overlapping windows")
    
    mode_choice = input("Enter choice (1/2, default=1): ").strip() or "1"
    
    if mode_choice == "1":
        stt.start_realtime_transcription(callback=enhanced_transcription_callback)
    elif mode_choice == "2":
        stt.start_streaming_transcription(window_duration=4.0, overlap=1.5)
    else:
        print("Invalid choice")

def transcribe_audio(audio_path: str) -> str:
    """
    Simple wrapper to transcribe a full audio file with Whisper.
    """
    import whisper
    model = whisper.load_model("small")  # change to "base"/"medium"/"large" if you prefer
    result = model.transcribe(audio_path)
    return result["text"].strip()

if __name__ == '__main__':
    main()