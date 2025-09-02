"""
Module for annotating transcripts with detected emotions
"""
import os
from utils.print_helpers import log_debug, log_error
from plugins.emotion_transcript.emotion_audio import detect_emotion_audio
from plugins.emotion_transcript.emotion_text import detect_emotion_text
from plugins.emotion_transcript.emoji_config import emoji_map, emoji_groups

def merge_and_average(dict1, dict2):
    merged = {}
    keys = set(dict1.keys()) | set(dict2.keys())
    for k in keys:
        v1 = dict1.get(k)
        v2 = dict2.get(k)
        if v1 is not None and v2 is not None:
            merged[k] = round(float((v1 + v2) / 2), 2)
        elif v1 is not None:
            merged[k] = round(float(v1), 2)
        else:
            merged[k] = round(float(v2), 2)
    return merged


def get_emoji_for_emotion(emotion, score, emoji_map):
    """Select emoji based on intensity score [0-1]."""
    emojis = emoji_map.get(emotion, [""])
    if not emojis:
        return ""
    index = min(int(score * len(emojis)), len(emojis) - 1)
    return emojis[index]


def annotate_transcript(AUDIO_FILE, text, audio_classifier, text_classifier, sarcasm_classifier):
    if not os.path.exists(AUDIO_FILE):
        log_error(f"Audio file not found: {AUDIO_FILE}")
        return None

    # Prepare quick lookup for combos
    combo_emoji_map = {}
    for emoji, pairs in emoji_groups.items():
        for pair in pairs:
            combo_emoji_map[pair] = emoji

    annotation = {}

    audio_emotion_result = detect_emotion_audio(AUDIO_FILE, audio_classifier)

    text_emotion_result = detect_emotion_text(text, text_classifier, sarcasm_classifier)

    emotions = (
        merge_and_average(audio_emotion_result, text_emotion_result)
        if audio_emotion_result and text_emotion_result
        else audio_emotion_result or text_emotion_result
    )

    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)

    # Use second emotion if score is within 0.4 of top emotion
    if len(sorted_emotions) > 1 and sorted_emotions[1][1] >= sorted_emotions[0][1] - 0.4:
        top_emotions = [sorted_emotions[0][0], sorted_emotions[1][0]]
        top_scores = [sorted_emotions[0][1], sorted_emotions[1][1]]
    else:
        top_emotions = [sorted_emotions[0][0]]
        top_scores = [sorted_emotions[0][1]]

    # Emoji selection
    emoji = ""
    if len(top_emotions) == 2:
        combo_key = tuple(top_emotions)
        emoji = combo_emoji_map.get(combo_key, None)
        if not emoji:
            emoji = get_emoji_for_emotion(top_emotions[0], top_scores[0], emoji_map)
    elif top_emotions:
        emoji = get_emoji_for_emotion(top_emotions[0], top_scores[0], emoji_map)
    
    annotation["annotation"] = f"{text} {emoji}"
    annotation["emotion"] = ','.join(top_emotions)
    annotation["confidence"] = {e: round(float(emotions[e]), 2) for e in top_emotions}
    log_debug("Transcript annotation completed: " + str(annotation))

    return annotation