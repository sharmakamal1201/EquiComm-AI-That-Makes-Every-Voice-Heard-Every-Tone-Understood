import os
import warnings
import torch

# Suppress various warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress specific PyTorch warnings
warnings.filterwarnings("ignore", message="stft with return_complex=False is deprecated")
warnings.filterwarnings("ignore", message=".*weight_norm.*is deprecated.*")

from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from utils.print_helpers import log_debug, log_error, log_info, log_output
from utils.audio_processing import process_audio_file, create_temp_file, cleanup_temp_files, analyze_pitch

def load_accent_models(ckpt_converter: str):
    """Load the OpenVoice models and return the converter."""
    log_info("Loading AI models... (This may take a moment)")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tone_color_converter = ToneColorConverter(
        f'{ckpt_converter}/config.json', 
        device=device
    )
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
    
    log_debug("Models loaded.")
    return tone_color_converter


def extract_speaker_embedding(reference_path: str, tone_color_converter, target_dir: str = 'processed'):
    """Extract speaker embedding from reference audio."""
    log_info(f"Extracting target accent from {reference_path}...")
    
    target_se, audio_name = se_extractor.get_se(
        reference_path, 
        tone_color_converter, 
        target_dir=target_dir, 
        vad=True
    )
    return target_se, audio_name


def convert_accent(input_path: str, target_se, output_path: str, tone_color_converter) -> None:
    """Convert accent of input audio to match target speaker embedding."""
    log_info(f"Converting accent for {input_path}...")
    
    tone_color_converter.convert(
        audio_src_path=input_path,
        src_se=None,  # Use original speaker's timbre
        tgt_se=target_se,
        output_path=output_path,
        message="@MyShell"
    )

def change_accent_ai(input_path, reference_path, output_path, ckpt_base, ckpt_converter):
    """
    Converts the accent of an input audio file to match a reference audio file
    using the OpenVoice AI model.
    """
    # Track temporary files for cleanup
    temp_files = []
    
    try:
        log_info("Applying noise reduction on input and reference files...")
        
        # Create temporary files for cleaned audio
        temp_input = create_temp_file()
        temp_reference = create_temp_file()
        temp_files.extend([temp_input, temp_reference])
        
        # Apply noise reduction to input and reference files
        process_audio_file(input_path, temp_input)
        process_audio_file(reference_path, temp_reference)

        #analyze_pitch(temp_input)
        #analyze_pitch(temp_reference)

        log_info("Noise reduction applied.")
        
        # Load models
        tone_color_converter = load_accent_models(ckpt_converter)
        
        # Create output directory
        os.makedirs('outputs', exist_ok=True)
        
        # Extract target speaker embedding from cleaned reference
        target_se, _ = extract_speaker_embedding(temp_reference, tone_color_converter)
        
        # Convert accent using cleaned input
        convert_accent(temp_input, target_se, output_path, tone_color_converter)
        
        log_output(f"AI accent conversion successful! Saved to {output_path}")
        
    finally:
        # Clean up temporary files
        cleanup_temp_files(temp_files)


if __name__ == '__main__':
    # Configuration
    INPUT_FILE_AI = 'D:\\Global Hackathon - 2025\\EquiComm\\WhatsApp Audio 2025-09-24 at 17.53.53_f3cf26ab.mp3'
    REFERENCE_ACCENT_FILE = 'D:\\Global Hackathon - 2025\\EquiComm\\AccentsDataset\\Accentsrecordings\\russian1.mp3'
    OUTPUT_FILE_AI = 'output_converted_ai.wav'

    ckpt_base = 'D:\\Global Hackathon - 2025\\EquiComm\\plugins\\accent_vocab_neutral\\checkpoints_v2\\base_speakers\\ses\\en-default.pth'
    ckpt_converter = 'D:\\Global Hackathon - 2025\\EquiComm\\plugins\\accent_vocab_neutral\\checkpoints\\converter'
    
    try:
        #analyze_pitch(INPUT_FILE_AI)
        #analyze_pitch(REFERENCE_ACCENT_FILE)
        change_accent_ai(INPUT_FILE_AI, REFERENCE_ACCENT_FILE, OUTPUT_FILE_AI, ckpt_base, ckpt_converter)
    except Exception as e:
        log_error(f"An error occurred: {e}")