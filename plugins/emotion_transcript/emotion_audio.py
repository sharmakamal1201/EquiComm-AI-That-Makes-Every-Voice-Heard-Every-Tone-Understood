from utils.print_helpers import log_debug, log_error

"""
Module for audio-based emotion detection using Hugging Face audio-classification pipeline
"""
def detect_emotion_audio(audio_file, audio_classifier):
    """
    Detects emotion from audio using Hugging Face's superb/hubert-large-superb-er model.
    Returns a dictionary: {label: score, ...} for all detected emotions, else None.
    """
    import warnings
    warnings.filterwarnings("ignore")

    try:
        # log_debug(f"Classifying audio file: {audio_file}")
        results = audio_classifier(audio_file)
        # Map model labels to standard emotion names
        label_map = {
            "ang": "anger",
            "neu": "neutral",
            "hap": "joy",
            "sad": "sadness"
        }
        mapped_results = [
            {'score': round(r['score'], 2), 'label': label_map.get(r['label'], r['label'])}
            for r in results
        ]
        if mapped_results:
            # Downweight 'neutral' score
            for r in mapped_results:
                if r['label'] == 'neutral':
                    r['score'] = round(r['score'] * 0.7, 2)

            log_debug(f"Audio Classification results: {mapped_results}")
            return {r['label']: r['score'] for r in mapped_results}
        else:
            return None
    except Exception as e:
        log_error(f"Audio emotion detection error: {e}")
        return None
