"""
Signal to Frames node for ComfyUI
Converts audio signals to microscope-like surface visualization frames
"""

import numpy as np
import torch
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.signal import windows
from scipy.interpolate import griddata

class SignalToFrames:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "image_width": ("INT", {"default": 512, "min": 1}),
                "image_height": ("INT", {"default": 512, "min": 1}),
                "frames_per_second": ("INT", {"default": 25, "min": 1, "step": 1}),
                "window_type": (["hann", "hamming", "blackman", "kaiser"], {"default": "hann"}),
            },
            "optional": {
                # Surface controls
                "surface_detail": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 69.0, "step": 0.1}),
                "height_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 69.0, "step": 0.1}),
                "surface_smoothing": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 69.0, "step": 0.1}),
                # Color mapping fields
                "low_intensity_red": ("INT", {"default": 255, "min": 0, "max": 255}),
                "low_intensity_green": ("INT", {"default": 50, "min": 0, "max": 255}),
                "low_intensity_blue": ("INT", {"default": 50, "min": 0, "max": 255}),
                "mid_intensity_red": ("INT", {"default": 50, "min": 0, "max": 255}),
                "mid_intensity_green": ("INT", {"default": 255, "min": 0, "max": 255}),
                "mid_intensity_blue": ("INT", {"default": 50, "min": 0, "max": 255}),
                "high_intensity_red": ("INT", {"default": 50, "min": 0, "max": 255}),
                "high_intensity_green": ("INT", {"default": 50, "min": 0, "max": 255}),
                "high_intensity_blue": ("INT", {"default": 255, "min": 0, "max": 255}),
                "intensity_contrast": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 3.0, "step": 0.1}),
                # Light and shadow
                "light_angle": ("FLOAT", {"default": 45.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "light_intensity": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "ambient_light": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_images_from_signal"
    CATEGORY = "audio"

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

    def calculate_surface_normals(self, height_map):
        """Calculate surface normals for lighting calculations"""
        gradient_y, gradient_x = np.gradient(height_map)
        norm = np.sqrt(gradient_x**2 + gradient_y**2 + 1)
        
        normals = np.dstack((-gradient_x/norm, -gradient_y/norm, 1/norm))
        return normals

    def apply_lighting(self, surface, normals, light_angle, light_intensity, ambient_light):
        """Apply directional and ambient lighting to the surface"""
        # Convert light angle to direction vector
        light_rad = np.radians(light_angle)
        light_dir = np.array([np.cos(light_rad), np.sin(light_rad), 1.0])
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        # Calculate diffuse lighting
        diffuse = np.clip(np.sum(normals * light_dir, axis=2), 0, 1)
        lighting = ambient_light + light_intensity * diffuse
        
        # Apply lighting to surface colors
        return surface * lighting[..., np.newaxis]

    def generate_surface_data(self, frame_data, window_function, image_width, image_height, surface_detail):
        """Generate 3D surface data from audio frame"""
        # Process frame with window function
        windowed_data = frame_data * window_function
        spectrum = np.fft.rfft(windowed_data)
        magnitudes = np.abs(spectrum)
        
        # Create 2D grid points
        x = np.linspace(0, 1, image_width)
        y = np.linspace(0, 1, image_height)
        X, Y = np.meshgrid(x, y)
        
        # Generate surface height data
        points = np.random.rand(len(magnitudes), 2)
        values = np.log1p(magnitudes)
        values = values / np.max(values) if np.max(values) > 0 else values
        
        # Interpolate to create surface
        grid_z = griddata(points, values, (X, Y), method='cubic', fill_value=0)
        grid_z = gaussian_filter(grid_z, sigma=1.0/surface_detail)
        
        return grid_z

    def color_surface(self, height_map, low_color, mid_color, high_color, contrast):
        """Map height values to colors with enhanced contrast"""
        # Normalize height map
        height_norm = (height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map))
        height_norm = np.power(height_norm, contrast)  # Apply contrast adjustment
        
        # Create color map
        colored_surface = np.zeros((*height_map.shape, 3))
        
        # Low intensity regions
        mask_low = height_norm < 0.33
        colored_surface[mask_low] = np.array(low_color) / 255.0
        
        # Mid intensity regions
        mask_mid = (height_norm >= 0.33) & (height_norm < 0.66)
        colored_surface[mask_mid] = np.array(mid_color) / 255.0
        
        # High intensity regions
        mask_high = height_norm >= 0.66
        colored_surface[mask_high] = np.array(high_color) / 255.0
        
        return colored_surface

    def generate_images_from_signal(self, audio, image_width, image_height, frames_per_second,
                                  window_type="hann", surface_detail=1.0, height_scale=1.0,
                                  surface_smoothing=0.5, low_intensity_red=255, low_intensity_green=50,
                                  low_intensity_blue=50, mid_intensity_red=50, mid_intensity_green=255,
                                  mid_intensity_blue=50, high_intensity_red=50, high_intensity_green=50,
                                  high_intensity_blue=255, intensity_contrast=1.2, light_angle=45.0,
                                  light_intensity=0.6, ambient_light=0.4):
        
        waveform = audio['waveform'].cpu().numpy()
        sample_rate = audio['sample_rate']
        
        # Check if stereo and handle appropriately
        if len(waveform.shape) > 1 and waveform.shape[0] == 2:
            # For stereo, take mean of channels before flattening
            waveform = np.mean(waveform, axis=0)
        else:
            waveform = waveform.flatten()
        
        # Calculate frame parameters
        duration = len(waveform) / sample_rate  # Duration in seconds
        total_frames = int(duration * frames_per_second/2)  # Total frames based on duration and FPS
        samples_per_frame = len(waveform) // total_frames
        
        # Create color tuples from individual RGB components
        low_color = (low_intensity_red, low_intensity_green, low_intensity_blue)
        mid_color = (mid_intensity_red, mid_intensity_green, mid_intensity_blue)
        high_color = (high_intensity_red, high_intensity_green, high_intensity_blue)
        
        window_function = self.create_window(window_type, samples_per_frame)
        generated_frames = []
        
        with tqdm(total=total_frames, desc="Generating Surface Frames") as progress_bar:
            for frame_idx in range(total_frames):
                # Extract frame data
                frame_start = frame_idx * samples_per_frame
                frame_end = min(frame_start + samples_per_frame, len(waveform))
                
                if frame_end - frame_start < samples_per_frame:
                    frame_data = np.pad(waveform[frame_start:frame_end],
                                      (0, samples_per_frame - (frame_end - frame_start)))
                else:
                    frame_data = waveform[frame_start:frame_end]
                
                # Generate surface data
                surface_height = self.generate_surface_data(
                    frame_data, window_function, image_width, image_height, surface_detail
                )
                surface_height *= height_scale
                
                if surface_smoothing > 0:
                    surface_height = gaussian_filter(surface_height, sigma=surface_smoothing)
                
                # Calculate surface normals for lighting
                normals = self.calculate_surface_normals(surface_height)
                
                # Generate colored surface
                colored_surface = self.color_surface(
                    surface_height,
                    low_color,
                    mid_color,
                    high_color,
                    intensity_contrast
                )
                
                # Apply lighting
                final_surface = self.apply_lighting(
                    colored_surface,
                    normals,
                    light_angle,
                    light_intensity,
                    ambient_light
                )
                
                # Convert to image format
                frame = np.clip(final_surface * 255, 0, 255).astype(np.uint8)
                generated_frames.append(frame)
                progress_bar.update(1)
        
        return (torch.from_numpy(np.array(generated_frames)).to(torch.float32) / 255.0,)

# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "SignalToFrames": SignalToFrames
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SignalToFrames": "Signal to Frames"
}