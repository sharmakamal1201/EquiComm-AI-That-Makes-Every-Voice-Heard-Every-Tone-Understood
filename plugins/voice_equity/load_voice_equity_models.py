from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor


# Load model & processor
model_name = "prithivMLmods/Common-Voice-Geneder-Detection"
gender_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
gender_processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
