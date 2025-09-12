"""
Audio processing utilities for noise reduction and file management.
"""
import os
import librosa
import numpy as np
import soundfile as sf
import tempfile
from typing import List
from utils.print_helpers import log_debug, log_error, log_info


def reduce_noise(audio_data: np.ndarray, sample_rate: int, noise_factor: float = 0.15, min_gain: float = 0.1) -> np.ndarray:
    """
    Apply gentle noise reduction to audio while preserving voice quality.
    Uses spectral gating to reduce background noise without over-compressing.
    
    Args:
        audio_data: Input audio as numpy array
        sample_rate: Sample rate of the audio
        noise_factor: Factor for noise threshold (higher = more aggressive)
        min_gain: Minimum gain to preserve voice (0.1 = don't reduce below 10%)
    
    Returns:
        Cleaned audio as numpy array
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
    threshold = noise_profile * (1 + noise_factor)
    mask = magnitude > threshold
    
    # Apply gentle smoothing to avoid artifacts
    from scipy.ndimage import gaussian_filter1d
    mask = gaussian_filter1d(mask.astype(float), sigma=1.0, axis=1)
    
    # Apply mask with minimum gain to preserve voice
    mask = np.maximum(mask, min_gain)
    
    # Apply the mask
    cleaned_magnitude = magnitude * mask
    
    # Convert back to time domain
    cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
    cleaned_audio = librosa.istft(cleaned_stft, hop_length=512)
    
    return cleaned_audio


def process_audio_file(input_path: str, output_path: str, noise_factor: float = 0.15) -> None:
    """
    Load audio file, apply noise reduction, and save to output path.
    
    Args:
        input_path: Path to input audio file
        output_path: Path where cleaned audio will be saved
        noise_factor: Noise reduction factor
    """
    audio_data, sample_rate = librosa.load(input_path, sr=None)
    log_info(f"Applying noise reduction: {input_path}")
    cleaned_audio = reduce_noise(audio_data, sample_rate, noise_factor)
    sf.write(output_path, cleaned_audio, sample_rate)


def create_temp_file(suffix: str = '.wav') -> str:
    """
    Create a temporary file and return its path.
    
    Args:
        suffix: File extension for the temporary file
        
    Returns:
        Path to the created temporary file
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_file.close()
    return temp_file.name


def cleanup_temp_files(temp_files: List[str]) -> None:
    """
    Remove a list of temporary files.
    
    Args:
        temp_files: List of file paths to remove
    """
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                log_debug(f"Cleaned up temporary file: {os.path.basename(temp_file)}")
        except Exception as e:
            log_error(f"Could not remove temporary file {temp_file}: {e}")
