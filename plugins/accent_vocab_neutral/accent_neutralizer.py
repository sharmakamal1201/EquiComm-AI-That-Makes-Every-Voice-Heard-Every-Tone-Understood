from utils.print_helpers import print_colored

"""
Module for accent neutralization (audio adaptation)
"""
def neutralize_accent(audio_file, target_accent):
    """
    Adapts audio to target accent using transcription and TTS API.
    Steps:
    1. Transcribe audio to text using transcribe_media utility.
    2. Synthesize text to audio in target accent using TTS API (pyttsx3 as example).
    Returns: path to neutral-accent audio file (or None if failed)
    """
    from utils.transcribe_media import transcribe_media

    segments = transcribe_media(audio_file)
    full_text = " ".join([seg["text"] for seg in segments])
    try:
        import pyttsx3
        tts_output = f"neutralized_{target_accent}.wav"
        engine = pyttsx3.init()
        # Set voice/accent (pyttsx3 voices are system-dependent)
        voices = engine.getProperty('voices')
        # Try to select a voice matching target_accent
        selected_voice = None
        for v in voices:
            if target_accent.lower() in v.name.lower() or target_accent.lower() in v.id.lower():
                selected_voice = v.id
                break
        if selected_voice:
            engine.setProperty('voice', selected_voice)
        engine.save_to_file(full_text, tts_output)
        engine.runAndWait()
        print_colored(f"[INFO] Synthesized '{full_text}' to '{tts_output}' in accent '{target_accent}'", "green")
        return tts_output
    except Exception as e:
        print_colored(f"Accent neutralization error: {e}", "red")
        return None
