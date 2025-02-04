import numpy as np
import torch
import librosa

class AudioScratchV2:
    def __init__(self, target_sample_rate=44100):
        self.target_sample_rate = target_sample_rate
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Audio Driver": ("AUDIO", {"forceInput": True}),
                "Audio IN": ("AUDIO", {"forceInput": True}),
                "envelope_type": (["amplitude", "pitch", "density", "spectral_centroid", "rms", "spectral_flatness"], {
                    "default": "amplitude"
                }),
                "base_stretch": ("FLOAT", {
                    "default": 0.1, 
                    "min": -2.0, 
                    "max": 2.0, 
                    "step": 0.01,
                    "display": "number"
                }),
                "stretch_intensity": ("FLOAT", {
                    "default": 1.9, 
                    "min": 0.0, 
                    "max": 5.0, 
                    "step": 0.01,
                    "display": "number"
                }),
                "stretch_mode": (["multiply", "add", "clip"], {
                    "default": "multiply"
                }),
                "maintain_pitch": ("BOOLEAN", {
                    "default": False
                })
            },
        }
 
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("Audio OUT",)
    FUNCTION = "modulate_audio_reactively"
    CATEGORY = "Sonification"

    def calculate_amplitude_envelope(self, waveform, frame_size=1024):
        # Convert to numpy for processing
        if torch.is_tensor(waveform):
            waveform = waveform.cpu().numpy()
            
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=0)
        
        num_frames = max(1, waveform.shape[0] // frame_size)
        envelope = np.zeros(num_frames)
        
        for i in range(num_frames):
            start = i * frame_size
            end = min((i + 1) * frame_size, waveform.shape[0])
            frame = waveform[start:end]
            envelope[i] = np.sqrt(np.mean(frame ** 2))
        
        envelope = envelope / (np.max(envelope) + 1e-6)
        
        x_orig = np.linspace(0, 1, num_frames)
        x_new = np.linspace(0, 1, waveform.shape[0])
        envelope = np.interp(x_new, x_orig, envelope)
        
        return envelope

    def calculate_pitch_envelope(self, waveform, frame_size=2048):
        """Calculate pitch-based envelope"""
        if torch.is_tensor(waveform):
            waveform = waveform.cpu().numpy()
            
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=0)
        
        # Calculate pitch using zero-crossing rate as a simple pitch detection
        hop_length = frame_size // 4
        zcr = librosa.feature.zero_crossing_rate(y=waveform, frame_length=frame_size, hop_length=hop_length)[0]
        
        # Normalize and stretch to match audio length
        envelope = zcr / (np.max(zcr) + 1e-6)
        x_orig = np.linspace(0, 1, len(envelope))
        x_new = np.linspace(0, 1, len(waveform))
        envelope = np.interp(x_new, x_orig, envelope)
        
        return envelope

    def calculate_density_envelope(self, waveform, frame_size=1024):
        """Calculate signal density envelope based on spectral flux"""
        if torch.is_tensor(waveform):
            waveform = waveform.cpu().numpy()
            
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=0)
        
        hop_length = frame_size // 4
        
        # Calculate spectral flux using librosa
        spec = np.abs(librosa.stft(waveform, n_fft=frame_size, hop_length=hop_length))
        spec_flux = np.sum(np.diff(spec, axis=1) ** 2, axis=0)
        
        # Normalize and stretch to match audio length
        envelope = spec_flux / (np.max(spec_flux) + 1e-6)
        x_orig = np.linspace(0, 1, len(envelope))
        x_new = np.linspace(0, 1, len(waveform))
        envelope = np.interp(x_new, x_orig, envelope)
        
        return envelope

    def calculate_spectral_centroid_envelope(self, waveform, frame_size=2048):
        """Calculate spectral centroid envelope for brightness/timbre tracking"""
        if torch.is_tensor(waveform):
            waveform = waveform.cpu().numpy()
            
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=0)
        
        hop_length = frame_size // 4
        
        # Calculate spectral centroid using librosa
        centroid = librosa.feature.spectral_centroid(y=waveform, sr=self.target_sample_rate, 
                                                   n_fft=frame_size, hop_length=hop_length)[0]
        
        # Normalize and stretch to match audio length
        envelope = centroid / (np.max(centroid) + 1e-6)
        x_orig = np.linspace(0, 1, len(envelope))
        x_new = np.linspace(0, 1, len(waveform))
        envelope = np.interp(x_new, x_orig, envelope)
        
        return envelope

    def calculate_rms_envelope(self, waveform, frame_size=1024):
        """Calculate RMS (Root Mean Square) envelope"""
        if torch.is_tensor(waveform):
            waveform = waveform.cpu().numpy()
            
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=0)
        
        hop_length = frame_size // 4
        
        # Calculate RMS using librosa
        rms = librosa.feature.rms(y=waveform, frame_length=frame_size, hop_length=hop_length)[0]
        
        # Normalize and stretch to match audio length
        envelope = rms / (np.max(rms) + 1e-6)
        x_orig = np.linspace(0, 1, len(envelope))
        x_new = np.linspace(0, 1, len(waveform))
        envelope = np.interp(x_new, x_orig, envelope)
        
        return envelope

    def calculate_spectral_flatness_envelope(self, waveform, frame_size=2048):
        """Calculate spectral flatness envelope for tonal vs. noisy content"""
        if torch.is_tensor(waveform):
            waveform = waveform.cpu().numpy()
            
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=0)
        
        hop_length = frame_size // 4
        
        # Calculate spectral flatness using librosa
        flatness = librosa.feature.spectral_flatness(y=waveform, n_fft=frame_size, hop_length=hop_length)[0]
        
        # Normalize and stretch to match audio length
        envelope = flatness / (np.max(flatness) + 1e-6)
        x_orig = np.linspace(0, 1, len(envelope))
        x_new = np.linspace(0, 1, len(waveform))
        envelope = np.interp(x_new, x_orig, envelope)
        
        return envelope

    def apply_variable_stretch(self, audio, stretch_factors):
        # Convert to numpy for processing
        if torch.is_tensor(audio):
            audio = audio.cpu().numpy()
            
        audio_length = audio.shape[-1]
        
        stretch_positions = np.cumsum(stretch_factors)
        stretch_positions = stretch_positions / stretch_positions[-1] * (audio_length - 1)
        
        output_positions = np.linspace(0, audio_length - 1, audio_length)
        indices = np.searchsorted(stretch_positions, output_positions)
        indices = np.clip(indices, 0, audio_length - 1)
        
        if len(audio.shape) == 1:
            output = audio[indices]
        else:
            output = audio[:, indices]
            
        return output

    def create_hann_window(self, window_size):
        """Create a Hann window using NumPy"""
        return 0.5 * (1 - np.cos(2 * np.pi * np.linspace(0, 1, window_size, endpoint=False)))

    def phase_vocoder_stretch(self, audio, time_stretch_factors):
        """Pitch-preserving time stretching using basic phase vocoder technique"""
        if len(audio.shape) > 1:
            audio = audio[0]
        
        window_size = 2048
        hop_length = window_size // 4
        window = self.create_hann_window(window_size)
        output_length = int(len(audio) / time_stretch_factors[0])
        output = np.zeros(output_length)
        
        for t in range(output_length // hop_length):
            stretch_index = min(t, len(time_stretch_factors) - 1)
            time_stretch_factor = time_stretch_factors[stretch_index]
            
            input_start = int(t * hop_length * time_stretch_factor)
            input_end = input_start + window_size
            
            if input_end > len(audio):
                break
            
            input_frame = audio[input_start:input_end] * window
            output_start = t * hop_length
            output_end = output_start + window_size
            
            if output_end > len(output):
                break
            
            output[output_start:output_end] += input_frame
        
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val
        
        return output

    def modulate_audio_reactively(self, **kwargs):
        audio_driver = kwargs.get("Audio Driver")
        audio_in = kwargs.get("Audio IN")
        envelope_type = kwargs.get("envelope_type", "amplitude")
        base_stretch = kwargs.get("base_stretch", 0.1)
        stretch_intensity = kwargs.get("stretch_intensity", 1.9)
        stretch_mode = kwargs.get("stretch_mode", "multiply")
        maintain_pitch = kwargs.get("maintain_pitch", False)

        # Extract waveform and convert to numpy for processing
        driver_waveform = audio_driver['waveform'].squeeze(0)
        if driver_waveform.ndim > 1:
            driver_waveform = driver_waveform[0]
        driver_waveform = driver_waveform.cpu().numpy()

        in_waveform = audio_in['waveform'].squeeze(0)
        if in_waveform.ndim > 1:
            in_waveform = in_waveform[0]
        in_waveform = in_waveform.cpu().numpy()
        
        # Select envelope type
        if envelope_type == "amplitude":
            envelope = self.calculate_amplitude_envelope(driver_waveform)
        elif envelope_type == "pitch":
            envelope = self.calculate_pitch_envelope(driver_waveform)
        elif envelope_type == "density":
            envelope = self.calculate_density_envelope(driver_waveform)
        elif envelope_type == "spectral_centroid":
            envelope = self.calculate_spectral_centroid_envelope(driver_waveform)
        elif envelope_type == "rms":
            envelope = self.calculate_rms_envelope(driver_waveform)
        else:  # spectral_flatness
            envelope = self.calculate_spectral_flatness_envelope(driver_waveform)
        
        # Apply different stretch modes
        if stretch_mode == "multiply":
            stretch_factors = base_stretch + stretch_intensity * envelope
        elif stretch_mode == "add":
            stretch_factors = base_stretch * np.ones_like(envelope) + stretch_intensity * envelope
        else:  # clip mode
            stretch_factors = np.clip(base_stretch + stretch_intensity * envelope, 0.01, 5.0)
        
        # Apply stretching
        if maintain_pitch:
            output_waveform = self.phase_vocoder_stretch(in_waveform, time_stretch_factors=stretch_factors)
        else:
            output_waveform = self.apply_variable_stretch(in_waveform, stretch_factors)
        
        # Convert back to tensor with correct format
        output_tensor = torch.from_numpy(output_waveform).to(audio_in['waveform'].device)
        output_tensor = output_tensor.unsqueeze(0).unsqueeze(0)

        return ({
            'waveform': output_tensor,
            'sample_rate': audio_in['sample_rate']
        },)

NODE_CLASS_MAPPINGS = {
    "Sonification:_Audio_Scratch_V2": AudioScratchV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Sonif_2": "AudioScratchV2"
}