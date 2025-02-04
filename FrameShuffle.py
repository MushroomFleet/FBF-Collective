import numpy as np
import torch
import random

class AudioPad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "pad_left": ("INT", {"default": 0, "min": 0, "max": 44100, "step": 1}),
                "pad_right": ("INT", {"default": 0, "min": 0, "max": 44100, "step": 1}),
                "pad_mode": (["constant", "reflect", "replicate", "circular"],),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "pad_audio_node"

    def pad_audio_node(self, audio, pad_left, pad_right, pad_mode):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        padded_waveform = self.pad_audio(waveform, pad_left, pad_right, pad_mode)
        return ({"waveform": padded_waveform, "sample_rate": sample_rate},)

    def pad_audio(self, waveform, pad_left, pad_right, pad_mode):
        # Implement padding logic here based on pad_mode
        if pad_mode == 'constant':
            # Pad with zeros (constant value)
            padded_waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right), mode='constant', value=0)
        elif pad_mode == 'reflect':
            # Reflect padding
            padded_waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right), mode='reflect')
        elif pad_mode == 'replicate':
            # Replicate padding
            padded_waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right), mode='replicate')
        elif pad_mode == 'circular':
            # Circular padding
            padded_waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right), mode='circular')
        
        return padded_waveform

class FrameShuffle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "shuffle_range": ("INT", {"default": 2, "min": 1}),
                "audio_shuffle_mode": (["Pass Through", "Random", "Sequential", "Reverse"],),
                "audio_segment_duration": ("INT", {"default": 0, "min": 0}),  # Duration in frames
                # Add parameters for audio padding
                "pad_left": ("INT", {"default": 0, "min": 0}),
                "pad_right": ("INT", {"default": 0, "min": 0}),
                "pad_mode": (["constant", "reflect", "replicate", "circular"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "shuffle_frames"
    CATEGORY = "PGFX & Electrolab"

    def shuffle_frames(self, images, audio, shuffle_range, audio_shuffle_mode,
                       audio_segment_duration, pad_left=0, pad_right=0, pad_mode="constant"):
        
        # Apply audio padding before processing
        audio_pad = AudioPad()
        padded_audio = audio_pad.pad_audio_node(audio, pad_left, pad_right, pad_mode)[0]

        # Convert images to NumPy array
        batch_numpy = images.cpu().numpy()
        batch_size = batch_numpy.shape[0]

        # Prepare shuffled images
        shuffled_images = np.copy(batch_numpy)

        # Shuffle frames in groups defined by shuffle_range
        for i in range(0, batch_size, shuffle_range):
            end_index = min(i + shuffle_range, batch_size)
            group = batch_numpy[i:end_index].copy()
            shuffled_images[i:end_index] = group[::-1]

        shuffled_images_tensor = torch.from_numpy(shuffled_images).to(images.device)

        if audio_shuffle_mode == "Pass Through":
            print("Audio Pass Through Enabled")
            return (shuffled_images_tensor, padded_audio)

        if isinstance(padded_audio, dict) and 'waveform' in padded_audio and 'sample_rate' in padded_audio:
            waveform = padded_audio['waveform'].squeeze(0)  # Remove unnecessary dimensions
            sample_rate = padded_audio['sample_rate']
            audio_np = waveform.cpu().numpy()

            if audio_np.ndim > 1:
                audio_np = audio_np[0]  # Use the first channel if stereo

            samples_per_frame = len(audio_np) / batch_size

            if samples_per_frame <= 0:
                print("Warning: Not enough audio samples to shuffle after padding.")
                return (shuffled_images_tensor, padded_audio)

            # Determine samples per segment based on frame duration
            if audio_segment_duration > 0:
                samples_per_segment = int(audio_segment_duration * samples_per_frame)
            else:
                samples_per_segment = int(samples_per_frame)  # Default to one frame

            num_segments = int(np.ceil(len(audio_np) / samples_per_segment))
            audio_segments = [
                audio_np[i * samples_per_segment:min((i + 1) * samples_per_segment, len(audio_np))]
                for i in range(num_segments)
            ]
            
            shuffled_audio_segments = []

            # Shuffle or reverse segments based on mode
            for i in range(0, num_segments, shuffle_range):
                end_index = min(i + shuffle_range, num_segments)
                group = audio_segments[i:end_index].copy()
                
                if audio_shuffle_mode == "Random":
                    random.shuffle(group)
                    # Randomly reverse content of individual segments with 50% probability
                    for j in range(len(group)):
                        if random.random() < 0.5:  # 50% chance to reverse
                            group[j] = group[j][::-1]  # Reverse content of the segment
                elif audio_shuffle_mode == "Sequential":
                    group = group[::-1]  # Reverse within the group
                elif audio_shuffle_mode == "Reverse":
                    # Reverse each segment in the group individually
                    group = [segment[::-1] for segment in group]  # Reverse content of each segment
                
                shuffled_audio_segments.extend(group)

            try:
                shuffled_audio = np.concatenate(shuffled_audio_segments)
                shuffled_audio_tensor = torch.from_numpy(shuffled_audio).to(waveform.device).unsqueeze(0).unsqueeze(0)
                
                shuffled_audio_dict = {'waveform': shuffled_audio_tensor, 'sample_rate': sample_rate}
                return (shuffled_images_tensor, shuffled_audio_dict)
                
            except ValueError as e:
                print(f"Error concatenating audio segments: {e}. Returning original audio.")
                return (shuffled_images_tensor, padded_audio)

NODE_CLASS_MAPPINGS = {
    "FrameShuffle": FrameShuffle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FrameShuffle": "Frame Shuffle",
}