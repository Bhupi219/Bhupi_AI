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

class MultilingualRealTimeSTT:
    def __init__(self, model_size="small", sample_rate=16000, chunk_size=320, language=None):
        """
        Multilingual Real-time Speech-to-Text engine.
        
        Args:
            model_size (str): Whisper model size ("tiny", "base", "small", "medium", "large")
                             Note: Remove .en suffix for multilingual support
            sample_rate (int): Audio sample rate
            chunk_size (int): Audio chunk size for VAD (160, 320, or 640 for 16kHz)
            language (str): Target language code (None for auto-detection, 'hi' for Hindi, 'en' for English)
        """
        # Use multilingual models (without .en suffix)
        self.model = whisper.load_model(model_size)
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.language = language  # None for auto-detect, 'hi' for Hindi, 'en' for English
        self.vad = webrtcvad.Vad(1)
        
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
        
        self.setup_audio_device()
        print(f"Initialized with model: {model_size}")
        print(f"Language setting: {'Auto-detect' if language is None else language}")

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

    def set_language(self, language_code):
        """
        Set the target language for transcription
        
        Args:
            language_code (str): Language code ('hi' for Hindi, 'en' for English, None for auto-detect)
                               Common codes: 'hi', 'en', 'es', 'fr', 'de', 'zh', 'ja', 'ko', etc.
        """
        self.language = language_code
        print(f"Language set to: {'Auto-detect' if language_code is None else language_code}")

    def _preprocess_audio(self, audio_data):
        """Preprocess audio data"""
        if audio_data.size == 0:
            return audio_data
        
        # Ensure consistent data type (float32)
        audio_data = audio_data.astype(np.float32)
            
        # Reduce noise
        try:
            reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=self.sample_rate)
            # Ensure the result is also float32
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
        
        # Add audio chunk to queue for processing
        self.audio_queue.put(indata.copy())

    def process_audio_stream(self, silence_threshold=30):
        """
        Process audio stream in real-time with VAD
        
        Args:
            silence_threshold (int): Number of silent chunks before processing speech
        """
        print("Real-time multilingual processing started. Speak in any supported language...")
        
        while self.is_processing:
            try:
                # Get audio chunk with timeout
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
                        
                        # If we've been silent long enough, process the speech
                        if self.silence_counter >= silence_threshold:
                            if self.speech_buffer:
                                self._process_speech_segment()
                            self._reset_speech_buffer()
                    print(".", end="", flush=True)
                
                # Add to rolling buffer (for continuous context)
                self.audio_buffer.append(chunk_int16)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")

    def _process_speech_segment(self):
        """Process accumulated speech segment with multilingual support"""
        if not self.speech_buffer:
            return
        
        try:
            # Combine speech chunks
            audio_data = np.concatenate(self.speech_buffer, axis=0).flatten()
            
            # Preprocess
            preprocessed_audio = self._preprocess_audio(audio_data.astype(np.float32))
            
            if preprocessed_audio.size > 0:
                # Prepare transcription options
                transcribe_options = {
                    'fp16': False,
                    'no_speech_threshold': 0.6
                }
                
                # Add language parameter if specified
                if self.language is not None:
                    transcribe_options['language'] = self.language
                
                # Transcribe with language support
                result = self.model.transcribe(preprocessed_audio, **transcribe_options)
                
                text = result['text'].strip()
                detected_language = result.get('language', 'unknown')
                
                if text:
                    print(f"\n[{detected_language.upper()}]: {text}")
                    self.result_queue.put({
                        'text': text,
                        'language': detected_language,
                        'timestamp': time.time()
                    })
                
        except Exception as e:
            print(f"Transcription error: {e}")

    def _reset_speech_buffer(self):
        """Reset speech processing state"""
        self.speech_buffer = []
        self.silence_counter = 0
        self.speech_detected = False

    def start_realtime_transcription(self, callback=None):
        """
        Start real-time multilingual transcription
        
        Args:
            callback (function): Optional callback function for transcribed text
        """
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
            # Start audio stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self.chunk_size,
                callback=self.audio_callback
            ):
                print(f"\nMultilingual real-time transcription active.")
                print(f"Language mode: {'Auto-detect' if self.language is None else self.language}")
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

    # Method 2: Streaming with overlapping windows (multilingual)
    def start_streaming_transcription(self, window_duration=3.0, overlap=1.0):
        """
        Start multilingual streaming transcription with overlapping windows
        
        Args:
            window_duration (float): Duration of each transcription window in seconds
            overlap (float): Overlap between windows in seconds
        """
        window_samples = int(window_duration * self.sample_rate)
        overlap_samples = int(overlap * self.sample_rate)
        step_samples = window_samples - overlap_samples
        
        audio_buffer = np.array([])
        
        print(f"Multilingual streaming transcription started (window: {window_duration}s, overlap: {overlap}s)")
        print(f"Language mode: {'Auto-detect' if self.language is None else self.language}")
        
        def audio_callback(indata, frames, time, status):
            nonlocal audio_buffer
            
            if status:
                print(f"Status: {status}")
            
            # Append new audio
            new_audio = indata[:, 0]  # mono
            audio_buffer = np.append(audio_buffer, new_audio)
            
            # Process when we have enough audio
            if len(audio_buffer) >= window_samples:
                # Extract window
                window = audio_buffer[:window_samples]
                
                # Process in separate thread to avoid blocking
                threading.Thread(
                    target=self._process_streaming_window,
                    args=(window.copy(),),
                    daemon=True
                ).start()
                
                # Slide window
                audio_buffer = audio_buffer[step_samples:]
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=audio_callback,
                dtype='float32'
            ):
                print("Multilingual streaming transcription active. Press Ctrl+C to stop.")
                while True:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nStopping streaming transcription...")

    def _process_streaming_window(self, audio_window):
        """Process a streaming window with multilingual support"""
        try:
            # Ensure audio window is float32
            audio_window = audio_window.astype(np.float32)
            
            # Preprocess
            preprocessed = self._preprocess_audio(audio_window)
            
            if preprocessed.size > 0:
                # Prepare transcription options
                transcribe_options = {
                    'fp16': False,
                    'no_speech_threshold': 0.7
                }
                
                # Add language parameter if specified
                if self.language is not None:
                    transcribe_options['language'] = self.language
                
                # Quick transcription
                result = self.model.transcribe(preprocessed, **transcribe_options)
                
                text = result['text'].strip()
                detected_language = result.get('language', 'unknown')
                
                if text and len(text) > 3:  # Filter out short/noise transcriptions
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"\n[{timestamp}] ({detected_language.upper()}): {text}")
        
        except Exception as e:
            print(f"Error processing audio window: {e}")
            return None

# Enhanced callback functions
def multilingual_transcription_callback(result):
    """Enhanced callback for handling multilingual transcribed text"""
    if isinstance(result, dict):
        text = result['text']
        language = result['language']
        print(f"CALLBACK RECEIVED [{language.upper()}]: {text}")
    else:
        # Fallback for simple text
        print(f"CALLBACK RECEIVED: {result}")
    
    # Here you could:
    # - Handle different languages differently
    # - Send to language-specific AI systems
    # - Translate to a common language
    # - Log with language metadata

def main():
    print("Multilingual Real-Time Speech-to-Text")
    print("Supported languages include: Hindi (hi), English (en), Spanish (es), French (fr), German (de), Chinese (zh), Japanese (ja), Korean (ko), and many others")
    
    # Model selection
    print("\nSelect model (larger = more accurate but slower):")
    print("1. tiny (fastest, least accurate)")
    print("2. base (good balance)")
    print("3. small (recommended for real-time)")
    print("4. medium (more accurate, slower)")
    print("5. large (most accurate, slowest)")
    
    model_choice = input("Enter choice (1-5, default=3): ").strip() or "3"
    model_map = {"1": "tiny", "2": "base", "3": "small", "4": "medium", "5": "large"}
    model_size = model_map.get(model_choice, "small")
    
    # Language selection
    print("\nLanguage mode:")
    print("1. Auto-detect (recommended)")
    print("2. Hindi only")
    print("3. English only")
    print("4. Custom language code")
    
    lang_choice = input("Enter choice (1-4, default=1): ").strip() or "1"
    language = None
    
    if lang_choice == "2":
        language = "hi"
    elif lang_choice == "3":
        language = "en"
    elif lang_choice == "4":
        custom_lang = input("Enter language code (e.g., 'hi', 'es', 'fr'): ").strip()
        if custom_lang:
            language = custom_lang
    
    # Initialize with multilingual model
    stt = MultilingualRealTimeSTT(model_size=model_size, language=language)
    
    print("\nChoose transcription mode:")
    print("1. Real-time VAD-based transcription")
    print("2. Streaming with overlapping windows")
    
    mode_choice = input("Enter choice (1/2, default=1): ").strip() or "1"
    
    if mode_choice == "1":
        # Method 1: VAD-based real-time transcription
        stt.start_realtime_transcription(callback=multilingual_transcription_callback)
    elif mode_choice == "2":
        # Method 2: Streaming transcription
        stt.start_streaming_transcription(window_duration=4.0, overlap=1.5)
    else:
        print("Invalid choice")

if __name__ == '__main__':
    main()