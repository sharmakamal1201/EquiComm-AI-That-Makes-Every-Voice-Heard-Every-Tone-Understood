import time
import os
import queue
import sounddevice as sd
import numpy as np
import re
import threading
from collections import deque
import torchaudio
import torch
import uuid

# Light imports first
from utils.print_helpers import log_debug, log_info, log_error, log_output

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")


# ==============================
# File logging setup
# ==============================
SENTENCES_FILE = "sentences_output.txt"
ANNOTATIONS_FILE = "annotations_output.txt"

def init_output_files():
    """Initialize output files with headers and timestamps"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    with open(SENTENCES_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n\n=== Session Started: {timestamp} ===\n")
    
    with open(ANNOTATIONS_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n\n=== Session Started: {timestamp} ===\n")

def log_sentence_to_file(sentence, start_time, end_time):
    """Append sentence to sentences file"""
    try:
        with open(SENTENCES_FILE, "a", encoding="utf-8") as f:
            timestamp = time.strftime("%H:%M:%S")
            f.write(f"[{timestamp}] ({start_time:.2f}s-{end_time:.2f}s): {sentence}\n")
    except Exception as e:
        log_error(f"Failed to write sentence to file: {e}")

def log_annotation_to_file(annotation, start_time, end_time):
    """Append annotation to annotations file"""
    try:
        with open(ANNOTATIONS_FILE, "a", encoding="utf-8") as f:
            timestamp = time.strftime("%H:%M:%S")
            f.write(f"[{timestamp}] ({start_time:.2f}s-{end_time:.2f}s): {annotation}\n")
    except Exception as e:
        log_error(f"Failed to write annotation to file: {e}")

# ==============================
# Global variables for lazy loading + performance caching
# ==============================
model = None
punct_model = None
gender_model = None
gender_processor = None
audio_classifier = None
text_classifier = None
sarcasm_classifier = None

# Performance optimizations - cache imported modules after models load
_annotate_transcript = None

# Thread-safe loading flags
_models_loading = threading.Event()
_models_loaded = threading.Event()

# ==============================
# Lazy model loaders
# ==============================
def load_whisper_model():
    """Load Whisper model lazily"""
    global model
    if model is None:
        #log_debug("Loading Whisper model...")
        t0 = time.time()
        from faster_whisper import WhisperModel
        model = WhisperModel("small.en", device="cpu", compute_type="int8")
        log_debug("Whisper model loaded in {:.2f} sec".format(time.time() - t0))
    return model

def load_punctuation_model():
    """Load punctuation model lazily"""
    global punct_model
    if punct_model is None:
        #log_debug("Loading punctuation model...")
        t0 = time.time()
        from deepmultilingualpunctuation import PunctuationModel
        punct_model = PunctuationModel()
        log_debug("Punctuation model loaded in {:.2f} sec".format(time.time() - t0))
    return punct_model

def load_emotion_models():
    """Load emotion models lazily"""
    global audio_classifier, text_classifier, sarcasm_classifier, _annotate_transcript
    if audio_classifier is None:
        #log_debug("Loading emotion models...")
        t0 = time.time()
        from plugins.emotion_transcript.load_emotion_models import audio_classifier as ac, text_classifier as tc, sarcasm_classifier as sc
        from plugins.emotion_transcript.transcript_annotator import annotate_transcript
        audio_classifier = ac
        text_classifier = tc
        sarcasm_classifier = sc
        _annotate_transcript = annotate_transcript  # Cache the function
        log_debug("Emotion models loaded in {:.2f} sec".format(time.time() - t0))
    return audio_classifier, text_classifier, sarcasm_classifier

def load_gender_models():
    """Load gender models lazily"""
    global gender_model, gender_processor
    if gender_model is None:
        #log_debug("Loading gender models...")
        t0 = time.time()
        from plugins.voice_equity.load_voice_equity_models import gender_model as gm, gender_processor as gp
        gender_model = gm
        gender_processor = gp
        log_debug("Gender models loaded in {:.2f} sec".format(time.time() - t0))
    return gender_model, gender_processor

def load_all_models_async():
    """Load all models in background thread"""
    if _models_loading.is_set():
        return  # Already loading
    
    _models_loading.set()
    
    def _load():
        try:
            # Load models in order of likely usage
            load_whisper_model()
            # load_punctuation_model()  # Commented out for now - may be overkill
            load_emotion_models()
            #load_gender_models()
            _models_loaded.set()
            log_info("All models loaded successfully!")
        except Exception as e:
            log_error(f"Error loading models: {e}")
            _models_loaded.set()  # Set even on error to prevent hanging
    
    thread = threading.Thread(target=_load, daemon=True)
    thread.start()

# ==============================
# Mic streaming setup
# ==============================
mic_queue = queue.Queue()
def mic_callback(indata, frames, time_info, status):
    if status:
        print(status, flush=True)
    mic_queue.put(indata.copy())

# ==============================
# Anti-hallucination helpers
# ==============================
EMIT_DEDUP_WINDOW_SEC = 15
MIN_SENT_CHARS = 6
last_emitted = deque(maxlen=50)
PUNC_FALLBACK_SEC = 20

_word_re = re.compile(r"\b\w+(?:'\w+)?\b")

def norm_text(s: str) -> str:
    return re.sub(r"[^\w]+", "", s.lower()).strip()

def should_emit(sent: str, now: float) -> bool:
    if len(sent.strip()) < MIN_SENT_CHARS:
        return False
    n = norm_text(sent)
    if not n:
        return False
    for prev_n, t_prev in list(last_emitted):
        if n == prev_n and (now - t_prev) <= EMIT_DEDUP_WINDOW_SEC:
            return False
    last_emitted.append((n, now))
    return True

# ==============================
# Sentence segmentation helper (Refactor #5)
# ==============================
def segment_to_sentences(words, raw_text):
    """
    Align raw_text (punctuated) with word timestamps → sentences.
    Returns list of (sentence_text, sent_words, start, end).
    """
    sentences = []
    punct_sents = re.split(r'(?<=[.!?])\s+', raw_text)
    idx = 0
    for sent in punct_sents:
        sent = sent.strip()
        if not sent:
            continue
        n_words = len(_word_re.findall(sent))
        if n_words == 0:
            continue
        if idx + n_words > len(words):
            n_words = len(words) - idx
            if n_words <= 0:
                break
        sent_words = words[idx: idx + n_words]
        idx += n_words
        sent_start = sent_words[0]["start"]
        sent_end = sent_words[-1]["end"]
        sentences.append((sent, sent_words, sent_start, sent_end))
    return sentences, idx

# ==============================
# SentenceProcessor class (Refactor #4)
# ==============================
class SentenceProcessor:
    def __init__(self, proc_queue):
        self.proc_queue = proc_queue
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def enqueue(self, audio, text, sr, seg_meta):
        try:
            self.proc_queue.put_nowait((audio.copy(), text, sr, seg_meta))
        except queue.Full:
            log_error("Processing queue full — dropping sentence")

    def shutdown(self):
        self.proc_queue.put(None)  # sentinel
        self.thread.join()

    def _worker(self):
        while True:
            task = self.proc_queue.get()
            if task is None:
                self.proc_queue.task_done()
                break
            audio_chunk, text, sr, seg_meta = task
            try:
                self._process(audio_chunk, text, sr, seg_meta)
            except Exception as e:
                log_error(f"Processing worker error: {e}")
            self.proc_queue.task_done()

    def _process(self, audio_chunk, text, sr, seg_meta):
        if audio_chunk is None or len(audio_chunk) == 0:
            log_error("Empty audio chunk, skipping.")
            return

        # Fast path - avoid repeated type conversions
        if not isinstance(audio_chunk, np.ndarray):
            audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
        audio_tensor = torch.from_numpy(audio_chunk).unsqueeze(0)

        segment_path = f"temp_seg_{uuid.uuid4().hex}.wav"
        try:
            torchaudio.save(segment_path, audio_tensor, sr, encoding="PCM_S", bits_per_sample=16)
        except Exception as e:
            log_error(f"ERROR writing temp audio: {e}")
            return

        # Use cached function and models (already loaded at startup)
        try:
            # Models are guaranteed to be loaded since we wait for _models_loaded
            annotation = _annotate_transcript(segment_path, text, audio_classifier, text_classifier, sarcasm_classifier)
        except Exception as e:
            log_error(f"Annotation failed: {e}")
            annotation = {"annotation": text}

        result = {
            "speaker": seg_meta.get("speaker", "Unknown"),
            "start_time": seg_meta.get("start_time", 0),
            "end_time": seg_meta.get("end_time", 0),
            "annotation": annotation.get("annotation", text)
        }

        log_output(f"{result['annotation']}")
        # Log annotation to file
        log_annotation_to_file(result['annotation'], result['start_time'], result['end_time'])

        # Fast cleanup
        try:
            os.remove(segment_path)
        except Exception:
            pass

# ==============================
# Word timestamp helper
# ==============================
def extract_word_timestamps(seg_list):
    words = []
    for seg in seg_list:
        seg_text = (seg.text or "").strip()
        seg_words = getattr(seg, "words", None)
        if seg_words:
            for w in seg_words:
                if w is None:
                    continue
                start = getattr(w, "start", None)
                end = getattr(w, "end", None)
                tok = (getattr(w, "word", "") or "").strip()
                if not tok:
                    continue
                if start is None or end is None:
                    seg_words = None
                    break
                words.append({"word": tok, "start": float(start), "end": float(end)})
            if seg_words is not None:
                continue
        toks = _word_re.findall(seg_text)
        if not toks:
            continue
        seg_dur = max(1e-6, (seg.end - seg.start))
        per = seg_dur / len(toks)
        for i, tok in enumerate(toks):
            start = float(seg.start + i * per)
            end = float(start + per)
            words.append({"word": tok, "start": start, "end": end})
    return words

# ==============================
# Main loop with graceful shutdown (Refactor #7)
# ==============================
def listen_and_transcribe():
    samplerate = 16000
    blocksize = 8000  # Increased back for more stable chunks
    chunk_seconds = 5  # Increased for longer context
    audio_buffer = np.zeros(0, dtype=np.float32)
    text_buffer  = ""
    last_punct_time = time.time()

    # Balanced VAD parameters for natural speech flow
    vad_params = {
        "threshold": 0.5,
        "min_speech_duration_ms": 300,  # Increased for longer segments
        "max_speech_duration_s": float("inf"),
        "min_silence_duration_ms": 400
    }

    proc_q = queue.Queue(maxsize=256)
    processor = SentenceProcessor(proc_q)

    # Load all models first before starting microphone
    log_info("Loading all AI models... Please wait.")
    load_all_models_async()
    
    # Initialize output files
    init_output_files()
    log_info("Output files initialized: sentences_output.txt and annotations_output.txt")
    
    # Wait for all models to be loaded
    log_info("Waiting for models to load completely...")
    _models_loaded.wait()  # Block until all models are loaded
    log_info("All models loaded successfully! Starting microphone now.")

    log_debug("Starting stream...")
    with sd.InputStream(samplerate=samplerate, blocksize=blocksize,
                        channels=1, dtype="float32", callback=mic_callback):
        log_info("Microphone listening (Ctrl+C to stop)...")

        try:
            while True:
                indata = mic_queue.get()
                audio = indata.flatten()
                audio_buffer = np.concatenate((audio_buffer, audio))

                if len(audio_buffer) >= samplerate * chunk_seconds:
                    # Use cached model reference (no function call overhead)
                    segments, info = model.transcribe(
                        audio_buffer,
                        language="en",
                        beam_size=5,
                        temperature=0.0,
                        best_of=1,
                        vad_filter=True,
                        vad_parameters=vad_params,
                        no_speech_threshold=0.75,
                        condition_on_previous_text=False,
                        word_timestamps=True
                    )

                    seg_list = list(segments)
                    words = extract_word_timestamps(seg_list)
                    raw_text = " ".join(w["word"] for w in words).strip()
                    text_buffer += " " + raw_text if raw_text else ""

                    #if raw_text:
                    #    log_debug("Chunk raw text: " + raw_text)

                    # Only try to segment if we have actual sentence-ending punctuation
                    word_count = len(text_buffer.split())
                    
                    if re.search(r'[.!?]', raw_text):
                        # We have punctuation - combine accumulated text with current chunk and segment
                        if text_buffer:
                            full_text = text_buffer + " " + raw_text
                        else:
                            full_text = raw_text
                        
                        sentences, idx = segment_to_sentences(words, full_text)
                        now = time.time()
                        for sent, sent_words, sent_start, sent_end in sentences:
                            sample_start = max(0, int(round(sent_start * samplerate)))
                            sample_end   = min(len(audio_buffer), int(round(sent_end * samplerate)))
                            if sample_end <= sample_start:
                                sample_end = min(len(audio_buffer), sample_start + 1)
                            sentence_audio = audio_buffer[sample_start:sample_end]

                            if should_emit(sent, now):
                                log_info(f"Sentence: {sent}")
                                # Log sentence to file
                                log_sentence_to_file(sent, sent_start, sent_end)
                                seg_meta = {"text": sent, "start_time": sent_start, "end_time": sent_end, "speaker": "Unknown"}
                                processor.enqueue(sentence_audio, sent, samplerate, seg_meta)
                                if re.search(r'[.!?]$', sent):
                                    last_punct_time = now

                        remaining_words = words[idx:]
                        text_buffer = " ".join(w["word"] for w in remaining_words).strip()
                    elif word_count > 50:
                        # Safety valve: if we have 50+ words without punctuation, force a segment
                        if text_buffer:
                            full_text = text_buffer + " " + raw_text
                        else:
                            full_text = raw_text
                            
                        log_info(f"Long speech detected ({len(full_text.split())} words), forcing segmentation")
                        sentences, idx = segment_to_sentences(words, full_text)
                        now = time.time()
                        if sentences:
                            sent, sent_words, sent_start, sent_end = sentences[0]
                            sample_start = max(0, int(round(sent_start * samplerate)))
                            sample_end   = min(len(audio_buffer), int(round(sent_end * samplerate)))
                            if sample_end <= sample_start:
                                sample_end = min(len(audio_buffer), sample_start + 1)
                            sentence_audio = audio_buffer[sample_start:sample_end]

                            if should_emit(sent, now):
                                log_info(f"Long sentence: {sent}")
                                # Log sentence to file
                                log_sentence_to_file(sent, sent_start, sent_end)
                                seg_meta = {"text": sent, "start_time": sent_start, "end_time": sent_end, "speaker": "Unknown"}
                                processor.enqueue(sentence_audio, sent, samplerate, seg_meta)
                        
                        remaining_words = words[1:] if len(words) > 1 else []
                        text_buffer = " ".join(w["word"] for w in remaining_words).strip()
                    else:
                        # No punctuation and under word limit - just accumulate text
                        # IMPORTANT: Update text_buffer to include the new words from this chunk
                        if text_buffer:
                            text_buffer = text_buffer + " " + raw_text
                        else:
                            text_buffer = raw_text
                        #log_debug(f"No punctuation in chunk, accumulating: {len(text_buffer.split())} words total")

                    if (time.time() - last_punct_time) > PUNC_FALLBACK_SEC and text_buffer:
                        # Use cached model reference
                        # fallback_punct = punct_model.restore_punctuation(text_buffer)  # Commented out - may be overkill
                        fallback_punct = text_buffer  # Use raw text without punctuation for now
                        log_info("Fallback (no punctuation): " + fallback_punct)

                        if remaining_words:
                            sent_start = remaining_words[0]["start"]
                            sent_end   = remaining_words[-1]["end"]
                            sample_start = max(0, int(round(sent_start * samplerate)))
                            sample_end   = min(len(audio_buffer), int(round(sent_end * samplerate)))
                            fallback_audio = audio_buffer[sample_start:sample_end]
                        else:
                            fallback_audio = audio_buffer.copy()
                            sent_start = 0.0
                            sent_end = len(audio_buffer) / samplerate

                        seg_meta = {"text": fallback_punct, "start_time": sent_start, "end_time": sent_end, "speaker": "Unknown"}
                        # Log fallback sentence to file
                        log_sentence_to_file(fallback_punct, sent_start, sent_end)
                        processor.enqueue(fallback_audio, fallback_punct, samplerate, seg_meta)

                        text_buffer = ""
                        last_punct_time = time.time()

                    audio_buffer = np.zeros(0, dtype=np.float32)

        except KeyboardInterrupt:
            log_debug("\nKeyboardInterrupt caught, flushing buffer...")
            final_raw = text_buffer.strip()
            if final_raw:
                # Use cached model reference
                # final_punct = punct_model.restore_punctuation(final_raw)  # Commented out - may be overkill
                final_punct = final_raw  # Use raw text without punctuation for now
                log_debug("Final (no punctuation): " + final_punct)
                seg_meta = {"text": final_punct, "start_time": 0.0,
                            "end_time": len(audio_buffer)/samplerate if samplerate else 0.0,
                            "speaker": "Unknown"}
                # Log final sentence to file
                log_sentence_to_file(final_punct, 0.0, len(audio_buffer)/samplerate if samplerate else 0.0)
                processor.enqueue(audio_buffer.copy(), final_punct, samplerate, seg_meta)
            log_debug("Exiting cleanly.")
            processor.shutdown()

if __name__ == "__main__":
    listen_and_transcribe()
