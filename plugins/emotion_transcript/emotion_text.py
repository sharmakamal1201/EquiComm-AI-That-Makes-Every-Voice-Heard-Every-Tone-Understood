from utils.print_helpers import log_debug, log_error
"""
Module for text-based emotion detection using pre-trained models
"""
def detect_emotion_text(text, text_classifier, sarcasm_classifier):
    """
    Detects emotion and sarcasm from text using HuggingFace transformers.
    Returns a tuple (emotion, confidence) if confidence > threshold, else None.
    """
    sarcasm_threshold = 0.65
    try:
        # Use a pre-trained emotion model (e.g., "j-hartmann/emotion-english-distilroberta-base")
        emotion_results = text_classifier(text)[0]
        # Collect all emotions with score > 0.3
        emotion_dict = {e['label'].lower(): round(float(e['score']), 2) for e in emotion_results}
        # Down-weight neutral by 0.7
        if 'neutral' in emotion_dict:
            emotion_dict['neutral'] = round(emotion_dict['neutral'] * 0.7, 2)

        # Sarcasm detection (using a sarcasm model if available)
        sarcasm_results = sarcasm_classifier(text)[0]
        # Map LABEL_0 and LABEL_1 to actual labels
        label_map = {"LABEL_0": "not_sarcasm", "LABEL_1": "sarcasm"}
        for sr in sarcasm_results:
            sr["label"] = label_map.get(sr["label"], sr["label"])
        top_sarcasm = max(sarcasm_results, key=lambda x: x['score'])

        if top_sarcasm['label'].lower() == 'sarcasm' and top_sarcasm['score'] > sarcasm_threshold:
            emotion_dict['sarcasm'] = round(float(top_sarcasm['score']*0.8), 2) # slightly down-weighting
        log_debug("Text Classification results: " + str(emotion_dict))
        if emotion_dict:
            return emotion_dict
        else:
            return None
    except Exception as e:
        log_error(f"Text emotion detection error: {e}")
        return None
