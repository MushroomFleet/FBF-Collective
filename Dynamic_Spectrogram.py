import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt
from PIL import Image
import io
import sys
from typing import Dict, Tuple, Any, Type, List

class DynamicSpectrogram:
    @classmethod
    def INPUT_TYPES(cls: Type["DynamicSpectrogram"]) -> Dict[str, Any]:
        cmaps: Dict[str, str] = {
            "Cosmic Flow": "cubehelix",  # Replaced with cubehelix
            "Orange-Purple Gradient": "plasma",
            "Red-Black Intensity": "inferno",
            "Purple-Red Spectrum": "magma",
            "Blue Temperature": "cool",
            "Red Temperature": "hot",
            "Ocean Depths": "ocean",
            "Rainbow Spectrum": "jet",
            "Thermal Imaging": "gnuplot2",
            "Electric Blue": "cool",
            "Fiery Red": "afmhot",
            "Forest Green": "Greens",
        }
        window_functions: List[str] = ["hann", "hamming", "kaiser", "blackman"]
        spectrogram_types: List[str] = ["magnitude", "power", "mel"]
        return {
            "required": {
                "audio_input": ("AUDIO",),
                "image_width": ("INT", {"default": 256, "min": 64, "max": 1024}),
                "image_height": ("INT", {"default": 256, "min": 64, "max": 1024}),
                "frames_per_second": ("INT", {"default": 25, "min": 1, "max": 60, "step": 1}),
                "color_map": (list(cmaps.keys()), {"default": "Fiery Red"}),
                "window_function": (window_functions, {"default": "hann"}),
                "spectrogram_type": (spectrogram_types, {"default": "mel"}),
                "temporal_resolution": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            },
            "optional": {
                "n_fft": ("INT", {"default": 2048, "min": 512, "max": 16384, "step": 256}),
                "hop_length": ("INT", {"default": 8, "min": 8, "max": 4096, "step": 8}),
                "n_mels": ("INT", {"default": 256, "min": 16, "max": 512, "step": 16}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "normalize": ("BOOLEAN", {"default": True}),
                "min_frequency": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20000.0, "step": 100.0}),
                "max_frequency": ("FLOAT", {"default": 20000.0, "min": 0.0, "max": 20000.0, "step": 100.0}),
                "detail_level": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("spectrograms",)
    FUNCTION = "generate_spectrograms_from_audio"
    CATEGORY = "Sonification node"

    def generate_spectrograms_from_audio(
        self, 
        audio_input: Dict[str, torch.Tensor],
        image_width: int,
        image_height: int,
        frames_per_second: int,
        color_map: str,
        window_function: str,
        spectrogram_type: str,
        temporal_resolution: int = 1,
        n_fft: int = 2048,
        hop_length: int = 64,
        n_mels: int = 256,
        contrast: float = 1.5,
        normalize: bool = True,
        min_frequency: float = 0.0,
        max_frequency: float = 20000.0,
        detail_level: float = 1.0
    ) -> Tuple[torch.Tensor]:
        # Mapping of user-friendly names to matplotlib colormaps
        cmap_mapping = {
            "Cosmic Flow": "cubehelix",  # Updated mapping
            "Orange-Purple Gradient": "plasma",
            "Red-Black Intensity": "inferno",
            "Purple-Red Spectrum": "magma",
            "Blue Temperature": "cool",
            "Red Temperature": "hot",
            "Ocean Depths": "ocean",
            "Rainbow Spectrum": "jet",
            "Thermal Imaging": "gnuplot2",
            "Electric Blue": "cool",
            "Fiery Red": "afmhot",
            "Forest Green": "Greens",
        }
        
        # Convert user-friendly name to actual colormap
        actual_colormap = cmap_mapping.get(color_map, 'viridis')

        # Rest of the implementation remains the same
        try:
            waveform = audio_input['waveform'].cpu().numpy().flatten()
            sample_rate = audio_input['sample_rate']
        except Exception:
            blank_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
            return (torch.from_numpy(np.array([blank_image])).to(torch.float32),)

        # Calculate number of frames based on audio length and desired frames per second
        audio_duration = len(waveform) / sample_rate
        num_frames = max(1, int(audio_duration / 2 * frames_per_second))

        # Adjust parameters based on detail level and temporal resolution
        adjusted_hop_length = int(hop_length / (detail_level * temporal_resolution))
        adjusted_n_mels = int(n_mels * detail_level)

        # Ensure sufficient audio length
        if len(waveform) < adjusted_hop_length * 2:
            blank_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
            return (torch.from_numpy(np.array([blank_image])).to(torch.float32),)

        # Select window function
        window_funcs = {
            'hann': np.hanning,
            'hamming': np.hamming,
            'kaiser': lambda n: np.kaiser(n, 14),
            'blackman': np.blackman
        }
        window = window_funcs.get(window_function, np.hanning)

        generated_spectrograms = []

        # Progress tracking
        print(f"Generating {num_frames} spectrograms...")
        
        # Compute spectrograms
        for i in range(num_frames):
            # Progress indicator
            progress = (i + 1) / num_frames * 100
            sys.stdout.write(f"\rProgress: [{int(progress/10)*'#'}{(10-int(progress/10))*'-'}] {progress:.1f}%")
            sys.stdout.flush()

            # Divide audio into frames
            start = int(i * len(waveform) / num_frames)
            end = int((i + 1) * len(waveform) / num_frames)
            frame_audio = waveform[start:end]

            try:
                # Compute spectrogram based on type
                if spectrogram_type == "mel":
                    spectrogram = librosa.feature.melspectrogram(
                        y=frame_audio, 
                        sr=sample_rate, 
                        n_fft=n_fft, 
                        hop_length=adjusted_hop_length,
                        n_mels=adjusted_n_mels,
                        fmin=min_frequency,
                        fmax=max_frequency
                    )
                elif spectrogram_type == "power":
                    spectrogram = np.abs(librosa.stft(
                        frame_audio, 
                        n_fft=n_fft, 
                        hop_length=adjusted_hop_length, 
                        window=window
                    ))**2
                else:  # magnitude
                    spectrogram = np.abs(librosa.stft(
                        frame_audio, 
                        n_fft=n_fft, 
                        hop_length=adjusted_hop_length, 
                        window=window
                    ))

                # Convert to decibels
                spectrogram_db = librosa.amplitude_to_db(
                    spectrogram, 
                    ref=np.max
                )

                # Normalize if requested
                if normalize:
                    spectrogram_db = (spectrogram_db - spectrogram_db.min()) / (spectrogram_db.max() - spectrogram_db.min())

                # Ensure non-empty spectrogram
                if spectrogram_db.size == 0:
                    continue

                # Create image with black background
                plt.figure(figsize=(image_width/100, image_height/100), dpi=100, facecolor='black')
                plt.gca().set_facecolor('black')
                plt.axis('off')
                plt.tight_layout(pad=0)
                
                # Plot spectrogram from bottom to top
                plt.imshow(
                    spectrogram_db ** contrast, 
                    cmap=actual_colormap, 
                    aspect='auto', 
                    origin='lower'  # Ensures bottom-to-top orientation
                )

                # Save to buffer
                buf = io.BytesIO()
                plt.savefig(
                    buf, 
                    format='png', 
                    bbox_inches='tight', 
                    pad_inches=0, 
                    facecolor='black',
                    edgecolor='none'
                )
                plt.close()

                # Convert to image
                buf.seek(0)
                spectrogram_image = Image.open(buf).convert("RGB")
                spectrogram_image_resized = spectrogram_image.resize(
                    (image_width, image_height), 
                    Image.LANCZOS
                )

                generated_spectrograms.append(np.array(spectrogram_image_resized))

            except Exception:
                # Skip problematic frames
                continue

        # Clear progress line
        sys.stdout.write("\n")
        sys.stdout.flush()

        # Fallback to blank image if no spectrograms generated
        if not generated_spectrograms:
            blank_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
            generated_spectrograms = [blank_image]

        return (torch.from_numpy(np.array(generated_spectrograms)).to(torch.float32),)

NODE_CLASS_MAPPINGS = {
    "DynamicSpectrogram": DynamicSpectrogram
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DynamicSpectrogram": "Dynamic Spectrogram"
}