"""
Module for vocabulary normalization
"""
def normalize_vocabulary(text, audience_profile):
    """
    Suggests neutral terms for uncommon/localized phrases in text.
    Uses a language model (HuggingFace transformers) for dynamic detection and suggestion.
    Returns: list of dicts [{"original": str, "suggestion": str, "explanation": str}]
    """
    from transformers import pipeline
    # Use a zero-shot classification pipeline to detect uncommon/localized terms
    # Candidate labels can be extended for more regions/languages
    candidate_labels = [
        "Indian English", "British English", "American English", "localized term", "global term", "neutral term"
    ]
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    results = []
    # Split text into sentences/phrases
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    for sent in sentences:
        output = classifier(sent, candidate_labels)
        # If highest score is for a localized label, suggest neutral alternative
        top_label = output['labels'][0]
        score = output['scores'][0]
        if top_label in ["Indian English", "localized term"] and score > 0.6:
            # For demo, use a simple replacement (in production, use a mapping or external API)
            suggestion = "[Suggest neutral/global alternative]"
            explanation = f"Detected '{top_label}' with confidence {score:.2f}. Consider using a more neutral/global term."
            results.append({
                "original": sent,
                "suggestion": suggestion,
                "explanation": explanation
            })
    return results
