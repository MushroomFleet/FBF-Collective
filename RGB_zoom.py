import numpy as np
import torch
from tqdm import tqdm
from scipy.signal import windows
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter

class RGB_zoom:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "frames_per_second": ("INT", {"default": 25, "min": 1, "step": 1}),
                "window_type": (["hann", "hamming", "blackman", "kaiser"], {"default": "hann"}),
                "Simple_Zoom": ("BOOLEAN", {"default": True}),
                "X_center": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1}),
                "Y_center": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "red_zoom_intensity": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 20.0, "step": 0.01}),
                "green_zoom_intensity": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 20.0, "step": 0.01}),
                "blue_zoom_intensity": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 20.0, "step": 0.01}),
                "frequency_smoothing": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "zoom_persistence": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "zoom_range": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 10.0, "step": 0.05}),
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_images"
    CATEGORY = "Sonification"

    def create_window(self, window_type, window_length):
        if window_type == "hann":
            return windows.hann(window_length)
        elif window_type == "hamming":
            return windows.hamming(window_length)
        elif window_type == "blackman":
            return windows.blackman(window_length)
        elif window_type == "kaiser":
            return windows.kaiser(window_length, beta=14)
        return None

    def analyze_audio_frame(self, frame_data, window_function, sample_rate, sensitivity, simple_zoom=False):
        """Extract frequency information from audio frame with enhanced sensitivity"""
        # Apply window function
        windowed_data = frame_data * window_function
        
        if simple_zoom:
            # For simple zoom, calculate total amplitude more robustly
            amplitude = np.mean(np.abs(windowed_data))
            max_amplitude = np.max(np.abs(frame_data))
            
            # Prevent division by zero and ensure valid normalization
            if max_amplitude > 0:
                normalized_amplitude = np.power(amplitude / max_amplitude, 1/sensitivity)
            else:
                normalized_amplitude = 0.0
                
            # Ensure the value is finite and in valid range
            normalized_amplitude = np.clip(normalized_amplitude, 0.0, 1.0)
            return np.array([normalized_amplitude, normalized_amplitude, normalized_amplitude])
        
        # Regular frequency band analysis for RGB zoom
        spectrum = np.fft.rfft(windowed_data)
        magnitudes = np.abs(spectrum)
        
        freq_resolution = sample_rate / len(frame_data)
        frequencies = np.linspace(0, sample_rate/2, len(magnitudes))
        
        freq_max = frequencies[-1]
        freq_bounds = [
            (0, freq_max/3),
            (freq_max/3, 2*freq_max/3),
            (2*freq_max/3, freq_max)
        ]
        
        band_magnitudes = []
        for low, high in freq_bounds:
            band_mask = (frequencies >= low) & (frequencies < high)
            
            if np.any(band_mask):
                band_data = magnitudes[band_mask]
                mean_magnitude = np.mean(band_data)
                peak_magnitude = np.max(band_data)
                
                weight = np.clip(sensitivity - 1, 0, 9) / 9
                enhanced_magnitude = (mean_magnitude * (1 - weight) + peak_magnitude * weight)
                enhanced_magnitude = np.power(enhanced_magnitude, 1/sensitivity)
            else:
                enhanced_magnitude = 0
                
            band_magnitudes.append(enhanced_magnitude)
        
        band_magnitudes = np.array(band_magnitudes)
        
        if np.sum(band_magnitudes) == 0:
            return np.zeros(3)
        
        normalized = band_magnitudes / np.max(band_magnitudes)
        return np.power(normalized, 1/sensitivity)

    def apply_zoom_to_image(self, image, zoom_factor, x_center=0.0, y_center=0.0, use_center=True):
        """Apply zoom to image with optional center point usage"""
        height, width = image.shape[:2]
        
        # Ensure zoom_factor is valid
        zoom_factor = np.clip(float(zoom_factor), 0.1, 20.0)
        
        if zoom_factor == 1.0:
            return image
        
        # Zoom the image
        zoomed = zoom(image, (zoom_factor, zoom_factor, 1), order=1)
        
        # Get new dimensions
        new_height, new_width = zoomed.shape[:2]
        
        if use_center:
            # Use provided center point
            center_x = int(width * (x_center + 1) / 2)
            center_y = int(height * (y_center + 1) / 2)
        else:
            # Use image center
            center_x = width // 2
            center_y = height // 2
        
        # Calculate offsets
        x_offset = int((new_width - width) * (center_x / width))
        y_offset = int((new_height - height) * (center_y / height))
        
        # Ensure valid offset ranges
        x_offset = np.clip(x_offset, 0, new_width - width)
        y_offset = np.clip(y_offset, 0, new_height - height)
        
        # Crop around the center point
        cropped = zoomed[
            y_offset:y_offset + height,
            x_offset:x_offset + width
        ]
        
        return cropped

    def apply_channel_zoom(self, image, audio_bands, intensities, zoom_range, x_center, y_center, simple_zoom=False):
        """Apply audio-driven zoom to individual RGB channels with separate coordinate zooming"""
        if simple_zoom:
            # For simple zoom, apply the same zoom factor to the entire image
            zoom_factor = 1.0 + (float(audio_bands[0]) * float(zoom_range) * float(np.mean(intensities)))
            zoom_factor = np.clip(zoom_factor, 0.1, 20.0)
            return self.apply_zoom_to_image(image, zoom_factor, x_center, y_center, use_center=True)
        
        # First, create a copy of the original image
        zoomed_image = np.zeros_like(image)
        
        # Apply independent zoom to each channel from center
        for channel in range(3):
            # Create a single-channel image
            channel_image = np.zeros_like(image)
            channel_image[..., channel] = image[..., channel]
            
            # Calculate zoom factor for this channel
            zoom_factor = 1 + (audio_bands[channel] * zoom_range * intensities[channel])
            
            # Zoom the channel using the image center (not user-specified center)
            if zoom_factor != 1.0:
                channel_zoomed = self.apply_zoom_to_image(channel_image, zoom_factor, 0.0, 0.0, use_center=False)
            else:
                channel_zoomed = channel_image
            
            # Update the corresponding channel in the zoomed image
            zoomed_image[..., channel] = channel_zoomed[..., channel]
        
        # Now apply the user-specified center coordinates to the RGB-zoomed image
        return self.apply_zoom_to_image(zoomed_image, 1.1, x_center, y_center, use_center=True)

    def process_images(self, images, audio, frames_per_second, window_type="hann",
                      Simple_Zoom=False, X_center=0.0, Y_center=0.0,
                      red_zoom_intensity=1.0, green_zoom_intensity=1.0, 
                      blue_zoom_intensity=1.0, frequency_smoothing=0.0,
                      zoom_persistence=0.0, zoom_range=1.0, sensitivity=1.0):
        
        images_np = images.cpu().numpy()
        batch_size = images_np.shape[0]
        
        waveform = audio['waveform'].cpu().numpy()
        sample_rate = audio['sample_rate']
        
        if len(waveform.shape) > 1 and waveform.shape[0] == 2:
            waveform = np.mean(waveform, axis=0)
        else:
            waveform = waveform.flatten()
        
        samples_per_frame = len(waveform) // batch_size
        window_function = self.create_window(window_type, samples_per_frame)
        
        intensities = np.array([red_zoom_intensity, green_zoom_intensity, blue_zoom_intensity])
        
        processed_frames = []
        previous_bands = None
        
        with tqdm(total=batch_size, desc="Processing Frames") as progress_bar:
            for frame_idx in range(batch_size):
                frame_start = frame_idx * samples_per_frame
                frame_end = min(frame_start + samples_per_frame, len(waveform))
                
                if frame_end - frame_start < samples_per_frame:
                    frame_data = np.pad(waveform[frame_start:frame_end],
                                      (0, samples_per_frame - (frame_end - frame_start)))
                else:
                    frame_data = waveform[frame_start:frame_end]
                
                current_bands = self.analyze_audio_frame(frame_data, window_function, 
                                                       sample_rate, sensitivity, Simple_Zoom)
                
                if frequency_smoothing > 0:
                    current_bands = gaussian_filter(current_bands, sigma=frequency_smoothing)
                
                if previous_bands is not None:
                    current_bands = (zoom_persistence * previous_bands + 
                                   (1 - zoom_persistence) * current_bands)
                previous_bands = current_bands
                
                modified_frame = self.apply_channel_zoom(
                    images_np[frame_idx],
                    current_bands,
                    intensities,
                    zoom_range,
                    X_center,
                    Y_center,
                    Simple_Zoom
                )
                
                processed_frames.append(modified_frame)
                progress_bar.update(1)
        
        return (torch.from_numpy(np.array(processed_frames)).to(images.device),)

# Node registration
NODE_CLASS_MAPPINGS = {
    "RGB_zoom": RGB_zoom
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RGB_zoom": "Audio-Driven RGB Zoom"
}