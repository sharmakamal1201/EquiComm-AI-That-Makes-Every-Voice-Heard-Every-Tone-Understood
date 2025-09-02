import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import logging
logging.basicConfig(level=logging.ERROR)

from transformers import pipeline

audio_classifier = pipeline("audio-classification", model="superb/hubert-large-superb-er", device=-1)

text_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True, device=-1)

sarcasm_classifier = pipeline("text-classification", model="mrm8488/t5-base-finetuned-sarcasm-twitter", return_all_scores=True, device=-1)