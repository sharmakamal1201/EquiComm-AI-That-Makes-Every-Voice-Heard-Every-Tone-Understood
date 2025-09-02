import torch
import librosa


# Label mapping
id2label = {0: "female", 1: "male"}

def predict_gender(audio_path, model, processor):
    waveform, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_label = logits.argmax(-1).item()
    return id2label[pred_label]