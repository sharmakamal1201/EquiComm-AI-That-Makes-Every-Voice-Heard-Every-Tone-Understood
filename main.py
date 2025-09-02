"""
EquiComm Main Entry Point
"""

import os
import torchaudio
from utils.print_helpers import print_colored, log_debug, log_info, log_error, log_output
from utils.transcribe_media import transcribe_media
from plugins.voice_equity.speaker_demographics import predict_gender
from plugins.emotion_transcript.transcript_annotator import annotate_transcript

import time
t0 = time.time()
from plugins.voice_equity.load_voice_equity_models import gender_model, gender_processor
from plugins.emotion_transcript.load_emotion_models import audio_classifier, text_classifier, sarcasm_classifier
t1 = time.time()
log_debug(f"Emotion and gender detection models loading took {t1 - t0:.2f} seconds")

AUDIO_FILE = os.path.abspath("Weekly Meeting Example.mp3")
segments = transcribe_media(AUDIO_FILE)
signal, sample_rate = torchaudio.load(AUDIO_FILE)

annotated_results = []

total_segment_creation = 0.0
total_emoji_annotation = 0.0
total_gender_detection = 0.0

for idx, seg in enumerate(segments):
    annotation = {}
    start_sample = int(float(seg["start_time"]) * sample_rate)
    end_sample = int(float(seg["end_time"]) * sample_rate)
    segment_signal = signal[:, start_sample:end_sample]
    segment_path = f"temp_segment_{idx}.wav"
    t_seg_start = time.time()
    torchaudio.save(segment_path, segment_signal, sample_rate)
    t_seg_end = time.time()
    total_segment_creation += (t_seg_end - t_seg_start)

    result = {}

    t_gender_start = time.time()
    predicted_gender = predict_gender(segment_path, gender_model, gender_processor)
    t_gender_end = time.time()
    total_gender_detection += (t_gender_end - t_gender_start)
    result["gender"] = predicted_gender
    log_info("Predicted Gender: " + predicted_gender)

    t_emoji_start = time.time()
    annotation = annotate_transcript(segment_path, seg.get("text", ""), audio_classifier, text_classifier, sarcasm_classifier)
    t_emoji_end = time.time()
    total_emoji_annotation += (t_emoji_end - t_emoji_start)

    result["speaker"] = seg.get("speaker", "Unknown")
    result["start_time"] = seg.get("start_time", 0)
    result["end_time"] = seg.get("end_time", 0)
    result["annotation"] = annotation.get("annotation", seg.get("text", ""))

    log_output(f"{result['annotation']}")

    annotated_results.append(result)

    if os.path.exists(segment_path):
        os.remove(segment_path)

log_info(f"{annotated_results}")

log_debug(f"Temp audio segment creation took {total_segment_creation:.2f} seconds")
log_debug(f"Emoji annotation took {total_emoji_annotation:.2f} seconds")
log_debug(f"Gender detection took {total_gender_detection:.2f} seconds")

from plugins.voice_equity.analytics import analyze_participation
from plugins.voice_equity.dashboard import generate_dashboard

analytics = analyze_participation(annotated_results)
log_output(f"{analytics}")
generate_dashboard(analytics)