import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["SPEECHBRAIN_LOGLEVEL"] = "CRITICAL"

import logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("speechbrain").propagate = False

import warnings
warnings.filterwarnings("ignore")

from utils.print_helpers import log_debug, log_output, log_error, log_info
#from config import hf_token, vosk_model_path, pyannote_model_name

from faster_whisper import WhisperModel
from deepmultilingualpunctuation import PunctuationModel


# ====================================
# GLOBAL MODELS (load once at startup)
# ====================================
# Whisper base model (medium.en on CPU, int8 quantization for speed/memory)
# time taking (my laptop):
# small-en: 0.3 x audio length (85% accuracy) + 4-5% for clean audio
# medium.en: 1 x audio length (92% accuracy) + 3-4% for clean audio
# large-v2: 3 x audio length (96% accuracy) + 2-3% for clean audio

whisper_model = WhisperModel("medium.en", device="cpu", compute_type="int8")

# Punctuation restoration model (loaded once, reused)
punct_model = PunctuationModel()


"""
Transcribe audio/video with Whisper + optional Pyannote speaker diarization + improved punctuation
"""
def transcribe_media(file_path):
    import time
    import soundfile as sf
    import numpy as np
    import librosa
    import re

    t0 = time.time()

    # ==============================
    # Audio preprocessing
    # ==============================
    data, sample_rate = sf.read(file_path)

    # Convert stereo -> mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        data = librosa.resample(data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    # Ensure float32 in [-1.0, 1.0]
    data = data.astype(np.float32)

    log_debug(f"Audio preprocessing done! took {time.time() - t0:.2f} seconds")

    # ==============================
    # Whisper transcription
    # ==============================
    whisper_trans_start = time.time()

    # VAD = Voice Activity Detection settings
    vad_params = {
        "threshold": 0.5,                  # min confidence to treat as speech
        "min_speech_duration_ms": 250,     # ignore very short blips
        "max_speech_duration_s": float("inf"),
        "min_silence_duration_ms": 400,    # required silence between segments
    }

    segments, info = whisper_model.transcribe(
        data,
        language="en",                     # force English
        beam_size=5,                       # beam search decoding (reduces hallucination)
        temperature=0.0,                   # deterministic decoding
        best_of=1,                         # single best output
        vad_filter=True,                   # remove non-speech automatically
        vad_parameters=vad_params,
        no_speech_threshold=0.75,          # conservative non-speech cutoff
        condition_on_previous_text=False,  # avoid leaking across segments
        word_timestamps=True               # get per-word timing for realignment
    )

    segments = list(segments)
    if not segments:
        return []

    # ==============================
    # Collect all words
    # ==============================
    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append({
                    "word": w.word.strip(),
                    "start": w.start,
                    "end": w.end
                })

    if not words:
        return []

    raw_text = " ".join([w["word"] for w in words])
    log_debug("Whisper text: " + raw_text.strip())
    whisper_trans_end = time.time()
    log_debug(f"Whisper transcription done! took {whisper_trans_end - whisper_trans_start:.2f} seconds")
    
    # ==============================
    # Global punctuation restoration
    # ==============================
    punct_start = time.time()
    punctuated_text = punct_model.restore_punctuation(raw_text.strip())
    punct_end = time.time()
    log_debug(f"Punctuation addition took {punct_end - punct_start:.2f} seconds")
    log_debug("Full punctuated text: " + punctuated_text)

    # ==============================
    # Regex-based sentence segmentation
    # ==============================
    # Split text on sentence-ending punctuation
    punct_sentences = re.split(r'(?<=[.!?]) +', punctuated_text)

    # Map sentences back to word timestamps
    segments_out = []
    word_idx = 0

    for sentence in punct_sentences:
        sent_words = []
        for _ in sentence.split():
            if word_idx < len(words):
                sent_words.append(words[word_idx])
                word_idx += 1
        if sent_words:
            segment = {
                "start_time": str(sent_words[0]["start"]),
                "end_time": str(sent_words[-1]["end"]),
                "text": sentence.strip(),
                "speaker": "Speaker id currently disabled",  # placeholder, enable Pyannote later
            }
            segments_out.append(segment)
            log_info(f"[{segment['speaker']}] {sentence.strip()}")

    return segments_out
