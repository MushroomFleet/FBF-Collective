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
        if pad_mode == 'constant':
            padded_waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right), mode='constant', value=0)
        elif pad_mode == 'reflect':
            padded_waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right), mode='reflect')
        elif pad_mode == 'replicate':
            padded_waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right), mode='replicate')
        elif pad_mode == 'circular':
            padded_waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right), mode='circular')
        
        return padded_waveform

class FrameBufferGlitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "freeze_frame_duration": ("INT", {"default": 1, "min": 1}),  # Duration in frames to freeze
                "number_of_random_glitches": ("INT", {"default": 1, "min": 1}),  # Total number of random glitches
                # Parameters for audio padding
                "pad_left": ("INT", {"default": 0, "min": 0}),
                "pad_right": ("INT", {"default": 0, "min": 0}),
                "pad_mode": (["constant", "reflect", "replicate", "circular"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "buffer_glitch"
    CATEGORY = "PGFX & Electrolab"

    def buffer_glitch(self, images, audio, freeze_frame_duration,
                      number_of_random_glitches=1, pad_left=0, pad_right=0, pad_mode="constant"):
        
        # Apply audio padding before processing
        audio_pad = AudioPad()
        padded_audio = audio_pad.pad_audio_node(audio, pad_left, pad_right, pad_mode)[0]

        # Convert images to NumPy array
        batch_numpy = images.cpu().numpy()
        batch_size = batch_numpy.shape[0]

        # Prepare output arrays
        output_images = []
        output_audio_segments = []

        # Calculate number of samples per frame for synchronization
        if isinstance(padded_audio, dict) and 'waveform' in padded_audio and 'sample_rate' in padded_audio:
            waveform = padded_audio['waveform'].squeeze(0)  # Remove unnecessary dimensions
            sample_rate = padded_audio['sample_rate']
            audio_np = waveform.cpu().numpy()

            if audio_np.ndim > 1:
                audio_np = audio_np[0]  # Use first channel if stereo

            segment_length = len(audio_np) / batch_size

        # Generate random glitch indices based on the specified number of glitches
        glitch_indices = sorted(random.sample(range(batch_size), min(number_of_random_glitches, batch_size)))

        i = 0
        while i < batch_size:
            if i in glitch_indices:  # Determine if we should freeze the frame at this index
                freeze_start_index = i
                freeze_end_index = min(i + freeze_frame_duration - 1, batch_size - 1)

                # Freeze the current frame and its corresponding audio frame for freeze_frame_duration
                frozen_frame_image = batch_numpy[freeze_start_index]
                frozen_frame_audio_segment = audio_np[int(i * segment_length):int((i + 1) * segment_length)]

                for j in range(freeze_start_index, freeze_end_index + 1):
                    output_images.append(frozen_frame_image)  # Append frozen image
                    
                    # Repeat only the corresponding single audio frame for this glitch effect
                    output_audio_segments.append(frozen_frame_audio_segment) 

                # Continue processing without skipping ahead.
                
            else:
                # Pass through normal video segments when not freezing frames
                output_images.append(batch_numpy[i])  # Append normal image
                
                # Append normal playback of the current audio segment during normal playback
                start_sample_index = int(i * segment_length)
                end_sample_index = int((i + 1) * segment_length)

                output_audio_segments.append(audio_np[start_sample_index:end_sample_index])
                
            i += 1

        try:
            # Concatenate repeated audio segments with pauses during glitches
            shuffled_audio = np.concatenate(output_audio_segments)
            shuffled_audio_tensor = torch.from_numpy(shuffled_audio).to(padded_audio['waveform'].device).unsqueeze(0).unsqueeze(0)
            
            shuffled_audio_dict = {'waveform': shuffled_audio_tensor, 'sample_rate': sample_rate}
            return (torch.from_numpy(np.array(output_images)).to(images.device), shuffled_audio_dict)
            
        except ValueError as e:
            print(f"Error concatenating audio segments: {e}. Returning original audio.")
            return (torch.from_numpy(np.array(output_images)).to(images.device), padded_audio)

NODE_CLASS_MAPPINGS = {
    "FrameBufferGlitch": FrameBufferGlitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FrameBufferGlitch": "Frame Buffer Glitch",
}
