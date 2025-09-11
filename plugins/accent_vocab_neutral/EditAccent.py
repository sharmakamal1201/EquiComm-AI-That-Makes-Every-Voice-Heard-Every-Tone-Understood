import os
import torch
import librosa
import numpy as np
import soundfile as sf
import tempfile
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

# ============================ SETUP ============================
# This needs to be done only once to download the models.
# Make sure you have PyTorch installed.
# pip install git+https://github.com/myshell-ai/OpenVoice.git
# ===============================================================

def reduce_noise(audio_data, sample_rate, noise_factor=0.15):
    """
    Apply gentle noise reduction to audio while preserving voice quality.
    Uses spectral gating to reduce background noise without over-compressing.
    """
    # Convert to frequency domain
    stft = librosa.stft(audio_data, hop_length=512, n_fft=2048)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate noise floor from the first and last 10% of the audio
    start_frames = int(0.1 * magnitude.shape[1])
    end_frames = int(0.9 * magnitude.shape[1])
    
    noise_profile = np.mean(np.concatenate([
        magnitude[:, :start_frames], 
        magnitude[:, end_frames:]
    ], axis=1), axis=1, keepdims=True)
    
    # Create spectral gate
    # Reduce noise but preserve frequencies above the threshold
    threshold = noise_profile * (1 + noise_factor)
    mask = magnitude > threshold
    
    # Apply gentle smoothing to avoid artifacts
    from scipy.ndimage import gaussian_filter1d
    mask = gaussian_filter1d(mask.astype(float), sigma=1.0, axis=1)
    
    # Apply mask with minimum gain to preserve voice
    min_gain = 0.1  # Don't reduce below 10% to keep voice audible
    mask = np.maximum(mask, min_gain)
    
    # Apply the mask
    cleaned_magnitude = magnitude * mask
    
    # Convert back to time domain
    cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
    cleaned_audio = librosa.istft(cleaned_stft, hop_length=512)
    
    return cleaned_audio

def change_accent_ai(input_path, reference_path, output_path):
    """
    Converts the accent of an input audio file to match a reference audio file
    using the OpenVoice AI model.
    """
    print("Loading AI models... (This may take a moment)")
    
    # Check for GPU availability
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Load the Tone Color Converter model
    ckpt_base = 'checkpoints/base_speakers/EN'
    ckpt_converter = 'checkpoints/converter'
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = 'outputs'

    #base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
    #base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    os.makedirs(output_dir, exist_ok=True)

    print("Models loaded.")

    # --- 1. Extract Target Timbre/Accent ---
    print(f"Extracting target accent from {reference_path}...")
    # This creates a 'speaker embedding' file that captures the vocal characteristics.
    target_se, audio_name = se_extractor.get_se(reference_path, tone_color_converter, target_dir='processed', vad=True)

    # --- 2. Convert the Voice ---
    print(f"Converting accent for {input_path}...")
    
    # Settings for the conversion
    source_se = None # Use the original speaker's timbre from the source file
    
    # Create temporary file for initial conversion
    temp_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_output.close()
    
    try:
        # Run the conversion to temporary file first
        tone_color_converter.convert(
            audio_src_path=input_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=temp_output.name,
            message="@MyShell" # This is a placeholder and doesn't affect the output
        )
        
        # --- 3. Apply Noise Reduction ---
        print("Applying noise reduction...")
        
        # Load the converted audio
        audio_data, sample_rate = librosa.load(temp_output.name, sr=None)
        
        # Apply gentle noise reduction
        cleaned_audio = reduce_noise(audio_data, sample_rate, noise_factor=0.15)
        
        # Save the final cleaned output
        sf.write(output_path, cleaned_audio, sample_rate)
        
        print(f"AI accent conversion with noise reduction successful! Saved to {output_path}")
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_output.name):
            os.remove(temp_output.name)

if __name__ == '__main__':
    # You need three files:
    # 1. The script itself.
    # 2. The source audio with the original accent.
    # 3. A reference audio with the desired target accent.

    INPUT_FILE_AI = 'D:\Global Hackathon - 2025\EquiComm\Kamal_testdata_trim30s.mp3'
    REFERENCE_ACCENT_FILE = 'D:\Global Hackathon - 2025\EquiComm\AccentsDataset\Accentsrecordings\english21.mp3' # A 5-15 second clip of a clear American voice
    OUTPUT_FILE_AI = 'output_converted_ai.wav'
    
    try:
        change_accent_ai(INPUT_FILE_AI, REFERENCE_ACCENT_FILE, OUTPUT_FILE_AI)
    except Exception as e:
        print("\n--- AI Model Error ---")
        print(f"An error occurred: {e}")