import numpy as np
import torch
from PIL import Image
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm
import io

class AudioToFrames:
    @classmethod
    def INPUT_TYPES(cls):
        cmaps = ["viridis", "plasma", "inferno", "magma", "cividis"]
        return {
            "required": {
                "audio": ("AUDIO",),
                "image_width": ("INT", {"default": 80, "min": 1}),
                "image_height": ("INT", {"default": 80, "min": 1}),
                "num_frames": ("INT", {"default": 100, "min": 1}),
                "color_map": (cmaps,),
                "n_fft": ("INT", {"default": 512, "min": 512, "max": 8192, "step": 256}),
                "hop_length": ("INT", {"default": 64, "min": 64, "max": 4096, "step": 128}),
                "n_mels": ("INT", {"default": 32, "min": 32, "max": 256, "step": 32}),
                "top_db": ("FLOAT", {"default": 80.0, "min": 10.0, "max": 100.0, "step": 5.0}),
            },
            "optional": {
                # Exposing RGB saturation values as user-friendly fields for all presets
                "red_saturation": ("INT", {"default": 127, "min": 0, "max": 255}),   # Red saturation parameter
                "green_saturation": ("INT", {"default": 121, "min": 0, "max": 255}), # Green saturation parameter
                "blue_saturation": ("INT", {"default": 95, "min": 0, "max": 255}), # Blue saturation parameter
                # New lightness parameter
                "lightness": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 10.0, "step": 0.01}),    # Lightness control (1.0 = normal)
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("Images", "Spectrograms")
    FUNCTION = "generate_images_from_audio"
    CATEGORY = "audio/generation"

    def generate_images_from_audio(self, audio, image_width, image_height, num_frames,
                                    color_map,
                                    n_fft, hop_length, n_mels,
                                    top_db,
                                    red_saturation,
                                    green_saturation,
                                    blue_saturation,
                                    lightness):
        waveform = audio['waveform'].cpu().numpy().flatten()
        sample_rate = audio['sample_rate']

        generated_images = []
        generated_spectrograms = []

        frame_length_samples = len(waveform) // num_frames
        if frame_length_samples == 0:
            return [], []

        total_steps = num_frames * 2
        
        with tqdm(total=total_steps, desc="Generating Images and Spectrograms") as progress_bar:
            waveform_tensor = torch.tensor(waveform).unsqueeze(0)
            mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
                power=2.0, norm="slaney", mel_scale="htk"
            )
            amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=top_db)

            # Precompute brightness values for the entire waveform
            max_value = np.max(np.abs(waveform))
            brightness_values = np.abs(waveform) / max_value if max_value > 0 else np.zeros_like(waveform)

            for i in range(num_frames):
                start_sample = i * frame_length_samples
                end_sample = min(start_sample + frame_length_samples, len(waveform))
                
                if start_sample >= len(waveform): 
                    break
                
                frame_waveform = waveform[start_sample:end_sample]

                mel_spectrogram = mel_spectrogram_transform(torch.tensor(frame_waveform).unsqueeze(0)).squeeze().cpu().numpy()
                mel_spectrogram_db = amplitude_to_db(torch.from_numpy(mel_spectrogram)).numpy()

                fig, ax = plt.subplots(figsize=(mel_spectrogram_db.shape[1]/100, mel_spectrogram_db.shape[0]/100), dpi=25)
                ax.set_facecolor("black")
                im = ax.imshow(mel_spectrogram_db, origin='lower', aspect='auto', cmap=color_map, interpolation='nearest')
                ax.axis('off')
                fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                
                buf.seek(0)
                spectrogram_image = Image.open(buf).convert("RGB")

                # Resize the spectrogram image to fit specified dimensions before appending
                spectrogram_image_resized = spectrogram_image.resize((image_width, image_height), Image.LANCZOS)
                
                # Append each resized spectrogram image directly to the output
                generated_spectrograms.append(np.array(spectrogram_image_resized))
                
                progress_bar.update(1)

            for i in range(num_frames):
                start_sample = i * frame_length_samples
                end_sample = min(start_sample + frame_length_samples, len(waveform))
                
                if start_sample >= len(waveform): 
                    break
                
                frame_waveform = waveform[start_sample:end_sample]

                # Create image for current frame with a black background
                image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

                for h in range(image_height):
                    for w in range(image_width):
                        index = (h * image_width + w) % len(frame_waveform)
                        value = brightness_values[start_sample + index] 

                        # Assign colors based on brightness levels and saturation settings
                        red_value = int(value * red_saturation * lightness) if red_saturation > 0 else int(value * lightness * 255)
                        green_value = int(value * green_saturation * lightness) if green_saturation > 0 else int(value * lightness * 255)
                        blue_value = int(value * blue_saturation * lightness) if blue_saturation > 0 else int(value * lightness * 255)

                        image[h, w] = [red_value if red_saturation > 0 else int(value * lightness * 255),
                                       green_value if green_saturation > 0 else int(value * lightness * 255),
                                       blue_value if blue_saturation > 0 else int(value * lightness * 255)]

                final_image = Image.fromarray(image).resize((image_width, image_height), Image.LANCZOS)
                
                generated_images.append(np.array(final_image))
                
                progress_bar.update(1)

        return [torch.from_numpy(np.array(generated_images)).to(torch.float32),
                 torch.from_numpy(np.array(generated_spectrograms)).to(torch.float32)]

NODE_CLASS_MAPPINGS = {"AudioToFrames": AudioToFrames}
NODE_DISPLAY_NAME_MAPPINGS = {"AudioToFrames": "Audio to Frames"}
