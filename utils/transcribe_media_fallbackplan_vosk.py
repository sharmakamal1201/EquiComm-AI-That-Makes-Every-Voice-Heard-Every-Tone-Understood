

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

from utils.print_helpers import print_colored

from config import hf_token, vosk_model_path, pyannote_model_name
"""
Transcribe audio/video with Vosk + Pyannote speaker diarization + improved punctuation
"""

def transcribe_media(file_path, vosk_model_path=vosk_model_path):

    print_colored("DEBUG COMMENT: inside transcribe_media", "yellow")

    import time
    t0 = time.time()

    import os
    import wave
    import json
    import soundfile as sf
    import numpy as np

    import vosk
    from vosk import Model, KaldiRecognizer
    vosk.SetLogLevel(-1)

    from deepmultilingualpunctuation import PunctuationModel
    from pyannote.audio import Pipeline

    print_colored(f"DEBUG COMMENT: imports done! took {time.time() - t0:.2f} seconds", "yellow")

    # uncomment for finding speaker in multi-speaker environment
    # diarization = None
    # pipeline = Pipeline.from_pretrained(pyannote_model_name, use_auth_token=hf_token)
    # print_colored("DEBUG COMMENT: pipeline initialization with pyannote done!", "yellow")
    # diarization = pipeline(file_path)
    # print("DEBUG COMMENT: pipeline processing with pyannote done!")

    # Temporary PCM file for Vosk
    pcm_temp_file = "_temp_pcm_for_vosk.wav"
    PCM_FILE = pcm_temp_file

    if not os.path.exists(file_path):
        print_colored(f"Audio file not found: {file_path}", "red")
        return []

    pcm_start = time.time()
    # Convert to mono 16-bit PCM
    data, sample_rate = sf.read(file_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    data_int16 = (data * 32767).astype(np.int16)
    sf.write(PCM_FILE, data_int16, sample_rate, subtype='PCM_16')
    wf = wave.open(PCM_FILE, "rb")
    pcm_end = time.time()
    print_colored(f"DEBUG COMMENT: Writing PCM file done! took {pcm_end - pcm_start:.2f} seconds", "yellow")


    # Vosk transcription
    vosk_trans_start = time.time()
    if not os.path.exists(vosk_model_path):
        print_colored(f"Vosk model not found: {vosk_model_path}", "red")
        return []
    
    model = Model(vosk_model_path)
    vosk_init = time.time()
    print_colored(f"DEBUG COMMENT: Vosk recognizer initialization done! took {vosk_init - vosk_trans_start:.2f} seconds", "yellow")

    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)    
    results = []
    while True:
        buf = wf.readframes(4000)
        if len(buf) == 0:
            break
        if rec.AcceptWaveform(buf):
            results.append(rec.Result())
    results.append(rec.FinalResult())
    wf.close()
    if os.path.exists(PCM_FILE):
        os.remove(PCM_FILE)
    vosk_trans_end = time.time()

    # Collect words
    words = []
    for res in results:
        jres = json.loads(res)
        if "result" in jres:
            words.extend(jres["result"])
    if not words:
        return []

    print_colored(f"DEBUG COMMENT: Vosk transcription done! took {vosk_trans_end - vosk_init:.2f} seconds", "yellow")

    # Punctuation restoration
    punct_start = time.time()
    punct_model = PunctuationModel()
    raw_text = ' '.join([w['word'] for w in words])
    punctuated_text = punct_model.restore_punctuation(raw_text)
    punct_end = time.time()
    print_colored(f"DEBUG COMMENT: Punctuation addition took {punct_end - punct_start:.2f} seconds", "yellow")

    # def get_speaker_for_time(t):
    #     for turn, _, speaker in diarization.itertracks(yield_label=True):
    #         if t >= turn.start and t <= turn.end:
    #             return speaker
    #     return "Unknown"

    # Split into sentences using punctuation restoration
    import re
    print_colored("DEBUG COMMENT: Punctauated text: " + punctuated_text, "yellow")
    punct_sentences = re.split(r'(?<=[.!?]) +', punctuated_text)

    segments = []
    word_idx = 0

    for sentence in punct_sentences:
        sent_words = []
        sentence_words = sentence.split()
        for _ in sentence_words:
            if word_idx < len(words):
                sent_words.append(words[word_idx])
                word_idx += 1
        if sent_words:
            speaker = "Speaker id currently disabled" #get_speaker_for_time(sent_words[0]['start'])
            segment = {
                "start_time": str(sent_words[0]["start"]),
                "end_time": str(sent_words[-1]["end"]),
                "text": sentence.strip(),
                "speaker": speaker
            }
            segments.append(segment)
            print_colored(f"[{speaker}] {sentence.strip()}", "cyan")

    return segments
