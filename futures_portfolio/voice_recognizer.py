"""
Voice message recognizer for Telegram bot.
Uses Vosk (offline, ~50 MB RAM) + ffmpeg (OGG -> WAV conversion).

Usage:
    from voice_recognizer import recognize_voice
    text = recognize_voice("path/to/voice.ogg")
"""

import json
import subprocess
import wave
import os
from vosk import Model, KaldiRecognizer

# Paths
MODEL_PATH = "C:/Python/Prosperous_Bot/vosk-model-small-ru-0.22"
FFMPEG_PATH = "C:/Python/Prosperous_Bot/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe"

# Load model once (shared across calls)
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = Model(MODEL_PATH)
    return _model


def ogg_to_wav(ogg_path: str, wav_path: str) -> str:
    """Convert OGG/Opus (Telegram voice) to WAV 16kHz 16-bit mono."""
    cmd = [
        FFMPEG_PATH,
        "-i", ogg_path,
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        "-y",  # overwrite
        wav_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")
    return wav_path


def recognize_wav(wav_path: str) -> str:
    """Recognize speech from WAV file using Vosk."""
    model = _get_model()
    recognizer = KaldiRecognizer(model, 16000)
    recognizer.SetWords(True)

    with wave.open(wav_path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            raise ValueError(f"WAV must be 16kHz 16-bit mono, got {wf.getframerate()}Hz {wf.getsampwidth()*8}-bit {wf.getnchannels()}ch")

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            recognizer.AcceptWaveData(data)

    result = json.loads(recognizer.FinalResult())
    return result.get("text", "").strip()


def recognize_voice(ogg_path: str, wav_path: str = None) -> str:
    """
    Full pipeline: OGG -> WAV -> text.
    
    Args:
        ogg_path: Path to OGG/Opus file from Telegram
        wav_path: Optional temp WAV path (auto-generated if None)
    
    Returns:
        Recognized text (empty string if nothing recognized)
    """
    if wav_path is None:
        wav_path = ogg_path.replace(".ogg", "_converted.wav")

    try:
        ogg_to_wav(ogg_path, wav_path)
        text = recognize_wav(wav_path)
        return text
    finally:
        # Clean up temp WAV
        if os.path.exists(wav_path):
            os.remove(wav_path)


if __name__ == "__main__":
    # Quick test with a file
    import sys
    if len(sys.argv) > 1:
        ogg_file = sys.argv[1]
        print(f"Recognizing: {ogg_file}")
        text = recognize_voice(ogg_file)
        print(f"Result: '{text}'")
    else:
        print("Usage: python voice_recognizer.py <file.ogg>")
        print(f"Model path: {MODEL_PATH}")
        print(f"ffmpeg path: {FFMPEG_PATH}")
        print(f"Model exists: {os.path.exists(MODEL_PATH)}")
        print(f"ffmpeg exists: {os.path.exists(FFMPEG_PATH)}")
