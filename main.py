# main.py
from core.pipeline import RealTimeVoicePipeline

def main():
    print("🎤 Starting AI Assistant (Real-time STT + NLU)")
    print("=" * 60)

    pipeline = RealTimeVoicePipeline(model_size="small", language_mode="auto")
    pipeline.start()   # will run until Ctrl+C

if __name__ == "__main__":
    main()
