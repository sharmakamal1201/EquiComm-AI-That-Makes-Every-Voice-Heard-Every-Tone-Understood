import time
import os
import queue
import sounddevice as sd
import numpy as np
import re
import threading
from collections import deque
from faster_whisper import WhisperModel
from deepmultilingualpunctuation import PunctuationModel
from utils.print_helpers import log_debug, log_info, log_error, log_output
from plugins.voice_equity.speaker_demographics import predict_gender
from plugins.emotion_transcript.transcript_annotator import annotate_transcript
import torchaudio
import torch
import uuid

# ----------------------------
# optional model loaders you already had
from plugins.voice_equity.load_voice_equity_models import gender_model, gender_processor
from plugins.emotion_transcript.load_emotion_models import audio_classifier, text_classifier, sarcasm_classifier
# ----------------------------

# ==============================
# Init models (global, loaded once)
# ==============================
t0 = time.time()
log_debug("Loading Whisper model...")
model = WhisperModel("medium.en", device="cpu", compute_type="int8")
log_debug("Whisper model loaded in {:.2f} sec".format(time.time() - t0))

punct_model = PunctuationModel()
log_debug("Punctuation model loaded.")

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
PUNC_FALLBACK_SEC = 200

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
    def __init__(self, proc_queue, gender_model, gender_proc, classifiers):
        self.proc_queue = proc_queue
        self.gender_model = gender_model
        self.gender_proc = gender_proc
        self.audio_classifier, self.text_classifier, self.sarcasm_classifier = classifiers
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

        audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
        audio_tensor = torch.from_numpy(audio_chunk).unsqueeze(0)

        segment_path = f"temp_seg_{uuid.uuid4().hex}.wav"
        try:
            torchaudio.save(segment_path, audio_tensor, sr, encoding="PCM_S", bits_per_sample=16)
        except Exception as e:
            log_error(f"ERROR writing temp audio: {e}")
            return

        result = {}
        try:
            predicted_gender = predict_gender(segment_path, self.gender_model, self.gender_proc)
        except Exception as e:
            log_error(f"Gender prediction failed: {e}")
            predicted_gender = "Unknown"
        result["gender"] = predicted_gender
        log_info("Predicted Gender: " + predicted_gender)

        try:
            annotation = annotate_transcript(segment_path, text,
                                             self.audio_classifier,
                                             self.text_classifier,
                                             self.sarcasm_classifier)
        except Exception as e:
            log_error(f"Annotation failed: {e}")
            annotation = {"annotation": text}

        result.update({
            "speaker": seg_meta.get("speaker", "Unknown"),
            "start_time": seg_meta.get("start_time", 0),
            "end_time": seg_meta.get("end_time", 0),
            "annotation": annotation.get("annotation", text)
        })

        log_output(f"{result['annotation']}")

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
    blocksize = 8000
    chunk_seconds = 5
    audio_buffer = np.zeros(0, dtype=np.float32)
    text_buffer  = ""
    last_punct_time = time.time()

    vad_params = {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": float("inf"),
        "min_silence_duration_ms": 400
    }

    proc_q = queue.Queue(maxsize=256)
    processor = SentenceProcessor(proc_q, gender_model, gender_processor,
                                  (audio_classifier, text_classifier, sarcasm_classifier))

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

                    if raw_text:
                        log_debug("Chunk raw text: " + raw_text)

                    sentences, idx = segment_to_sentences(words, raw_text)
                    now = time.time()
                    for sent, sent_words, sent_start, sent_end in sentences:
                        sample_start = max(0, int(round(sent_start * samplerate)))
                        sample_end   = min(len(audio_buffer), int(round(sent_end * samplerate)))
                        if sample_end <= sample_start:
                            sample_end = min(len(audio_buffer), sample_start + 1)
                        sentence_audio = audio_buffer[sample_start:sample_end]

                        if should_emit(sent, now):
                            log_info(f"Sentence: {sent}")
                            seg_meta = {"text": sent, "start_time": sent_start, "end_time": sent_end, "speaker": "Unknown"}
                            processor.enqueue(sentence_audio, sent, samplerate, seg_meta)
                            if re.search(r'[.!?]$', sent):
                                last_punct_time = now

                    remaining_words = words[idx:]
                    text_buffer = " ".join(w["word"] for w in remaining_words).strip()

                    if (time.time() - last_punct_time) > PUNC_FALLBACK_SEC and text_buffer:
                        fallback_punct = punct_model.restore_punctuation(text_buffer)
                        log_info("Punctuated Fallback: " + fallback_punct)

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
                        processor.enqueue(fallback_audio, fallback_punct, samplerate, seg_meta)

                        text_buffer = ""
                        last_punct_time = time.time()

                    audio_buffer = np.zeros(0, dtype=np.float32)

        except KeyboardInterrupt:
            log_debug("\nKeyboardInterrupt caught, flushing buffer...")
            final_raw = text_buffer.strip()
            if final_raw:
                final_punct = punct_model.restore_punctuation(final_raw)
                log_debug("Final punctuated (flush): " + final_punct)
                seg_meta = {"text": final_punct, "start_time": 0.0,
                            "end_time": len(audio_buffer)/samplerate if samplerate else 0.0,
                            "speaker": "Unknown"}
                processor.enqueue(audio_buffer.copy(), final_punct, samplerate, seg_meta)
            log_debug("Exiting cleanly.")
            processor.shutdown()

if __name__ == "__main__":
    listen_and_transcribe()
