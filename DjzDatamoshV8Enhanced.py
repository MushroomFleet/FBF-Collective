import torch
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import scipy.ndimage
import cv2

class DjzDatamoshV8Enhanced:
    def __init__(self):
        self.type = "DjzDatamoshV8Enhanced"
        self.output_node = True
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "sort_mode": (["luminance", "hue", "saturation", "laplacian"],),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "rotation": ("INT", {
                    "default": -90,
                    "min": -180,
                    "max": 180,
                    "step": 90
                }),
                "multi_pass": ("BOOLEAN", {"default": False}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "grow": ("INT", {"default": 4, "min": -999, "max": 999, "step": 1}),
                "blur": ("INT", {"default": 4, "min": 0, "max": 999, "step": 1}),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xFFFFFFFF,
                    "step": 1
                }),
                "use_mask": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "MASK",)
    RETURN_NAMES = ("image", "mask_out", "inverted_mask_out",)
    FUNCTION = "pixel_sort"
    CATEGORY = "image/effects"

    def calculate_hue(self, pixels):
        """Calculates the hue values for each pixel based on RGB channels"""
        if isinstance(pixels, torch.Tensor):
            pixels = pixels.cpu().numpy()
            
        # Ensure consistent dtype
        pixels = pixels.astype(np.float32)
            
        r, g, b = np.split(pixels, 3, axis=-1)
        hue = np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)[..., 0]
        # Normalize to 0-1 range
        hue = (hue + np.pi) / (2 * np.pi)
        return hue

    def calculate_saturation(self, pixels):
        """Calculates the saturation values for each pixel"""
        if isinstance(pixels, torch.Tensor):
            pixels = pixels.cpu().numpy()
            
        # Ensure consistent dtype
        pixels = pixels.astype(np.float32)
            
        r, g, b = np.split(pixels, 3, axis=-1)
        maximum = np.maximum(r, np.maximum(g, b))
        minimum = np.minimum(r, np.minimum(g, b))
        # Add epsilon to avoid division by zero
        denominator = np.maximum(maximum, 1e-7)
        return ((maximum - minimum) / denominator)[..., 0]

    def calculate_laplacian(self, pixels):
        """Calculates the Laplacian values for each pixel"""
        if isinstance(pixels, torch.Tensor):
            pixels = pixels.cpu().numpy()
            
        # Ensure consistent dtype
        pixels = pixels.astype(np.float32)
            
        lum = np.average(pixels, axis=-1)
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        laplacian = np.abs(convolve2d(lum, kernel, 'same', boundary='symm'))
        # Normalize to 0-1 range
        return (laplacian - np.min(laplacian)) / (np.max(laplacian) - np.min(laplacian) + 1e-7)

    def calculate_luminance(self, pixels):
        """Calculates luminance values for each pixel"""
        if isinstance(pixels, torch.Tensor):
            pixels = pixels.cpu().numpy()
            
        # Ensure consistent dtype
        pixels = pixels.astype(np.float32)
            
        # Use fixed coefficients for RGB to luminance conversion
        coefficients = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        return np.dot(pixels[..., :3], coefficients)

    def sort_interval(self, interval, interval_indices):
        """Sort pixels within an interval"""
        # Ensure stable sorting
        return np.argsort(interval, kind='stable') + interval_indices

    def process_row(self, row, row_values, edges, pixels, row_mask=None):
        """Process a single row of pixels with optional mask"""
        # Find indices where edges occur
        interval_indices = np.flatnonzero(edges[row])
        
        # Handle empty intervals case
        if len(interval_indices) == 0:
            return pixels
            
        # Split row values at edge points
        split_values = np.split(row_values, interval_indices)
        original_pixels = pixels[row].copy()
        
        # Process intervals
        for index, interval in enumerate(split_values[1:]):
            if len(interval) > 0:  # Only process non-empty intervals
                start_idx = interval_indices[index]
                end_idx = start_idx + len(interval)
                
                # Sort the interval
                sorted_indices = self.sort_interval(interval, start_idx)
                
                # If mask is provided, blend between sorted and unsorted based on mask values
                if row_mask is not None:
                    mask_interval = row_mask[start_idx:end_idx]
                    # Only apply sorting where mask is active (>0)
                    for channel in range(pixels.shape[-1]):
                        sorted_values = pixels[row, sorted_indices, channel]
                        original_values = original_pixels[start_idx:end_idx, channel]
                        pixels[row, start_idx:end_idx, channel] = mask_interval * sorted_values + \
                            (1 - mask_interval) * original_values
                else:
                    # No mask, apply sorting directly
                    for channel in range(pixels.shape[-1]):
                        pixels[row, start_idx:end_idx, channel] = pixels[row, sorted_indices, channel]
        
        return pixels

    def expand_mask(self, mask, grow, blur):
        """Apply grow and blur operations to the mask"""
        # grow
        c = 0
        kernel = np.array([[c, 1, c],
                          [1, 1, 1],
                          [c, 1, c]])
        output = mask.numpy()
        for _ in range(abs(grow)):
            if grow < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=kernel)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
        
        # blur
        if blur > 0:
            output = cv2.GaussianBlur(output, (blur * 2 + 1, blur * 2 + 1), 0)
        
        return torch.from_numpy(output)

    def apply_pixel_sorting(self, image, mask, calculate_value_fn, threshold, rotation):
        """Apply pixel sorting effect to an image with mask"""
        # Convert image and mask to numpy array if they're tensors
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
            
        # Ensure consistent dtype
        image = image.astype(np.float32)
        if mask is not None:
            mask = mask.astype(np.float32)

        # Rotate pixels and mask based on specified angle
        k_rotations = (rotation // 90) % 4  # Normalize rotation to 0-3 range
        rotated = np.rot90(image, k_rotations)
        if mask is not None:
            rotated_mask = np.rot90(mask, k_rotations)
        else:
            rotated_mask = None
        
        # Calculate values for sorting
        values = calculate_value_fn(rotated)
        
        # Normalize values to 0-1 range
        values_min = np.min(values)
        values_max = np.max(values)
        if values_max > values_min:
            values = (values - values_min) / (values_max - values_min)
        else:
            values = np.zeros_like(values)
        
        # Create mask based on threshold
        threshold_mask = values > threshold
        
        # Compute edges using the mask
        edges = np.zeros_like(threshold_mask)
        edges[:, 1:] = threshold_mask[:, 1:] != threshold_mask[:, :-1]  # Detect changes in mask
        
        # Process each row
        for row in range(rotated.shape[0]):
            row_mask = rotated_mask[row] if rotated_mask is not None else None
            rotated = self.process_row(row, values[row], edges, rotated, row_mask)
        
        # Rotate back
        result = np.rot90(rotated, -k_rotations)
        return result

    def pixel_sort(self, images, sort_mode, threshold, rotation, multi_pass, invert_mask, grow, blur, seed, use_mask, mask=None):
        """Main pixel sorting function with enhanced mask support
        
        Arguments:
            images: Batch of input images (BHWC format)
            mask: Mask to control where sorting is applied (1 = sort, 0 = keep original)
            sort_mode: Sorting method to use (luminance/hue/saturation/laplacian)
            threshold: Value between 0-1 controlling segment creation
            rotation: Angle to rotate sorting direction
            multi_pass: Whether to apply all sorting modes sequentially
            invert_mask: Whether to invert the mask
            grow: Amount to grow/shrink the mask
            blur: Amount to blur the mask
            seed: Random seed for reproducibility
        """
        print(f"Starting DjzDatamoshV8Enhanced pixel sorting with mode: {sort_mode}")
        print(f"Input batch shape: {images.shape}")
        print(f"Using random seed: {seed}")
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        if len(images.shape) != 4:
            print("Warning: DjzDatamoshV8Enhanced requires batch of images in BHWC format")
            return (images,)
            
        try:
            # Select value calculation function based on mode
            mode_functions = {
                "luminance": self.calculate_luminance,
                "hue": self.calculate_hue,
                "saturation": self.calculate_saturation,
                "laplacian": self.calculate_laplacian
            }
            
            calculate_value_fn = mode_functions[sort_mode]
            
            # Process each image in batch
            batch_sorted = []
            for idx in range(len(images)):
                current_image = images[idx].cpu().numpy()
                # Handle mask based on use_mask toggle
                current_mask = None
                if use_mask and mask is not None:
                    current_mask = mask[idx].cpu().numpy()
                    
                # Process mask if provided and enabled
                if current_mask is not None:
                    # Invert mask if requested
                    if invert_mask:
                        current_mask = 1 - current_mask
                    
                    # Apply grow and blur operations
                    if grow != 0 or blur > 0:
                        current_mask = self.expand_mask(torch.from_numpy(current_mask), grow, blur).numpy()
                
                if multi_pass:
                    # Apply multiple sorting passes with different modes in fixed order
                    for mode_name in ["luminance", "hue", "saturation", "laplacian"]:
                        current_image = self.apply_pixel_sorting(
                            current_image, 
                            current_mask,
                            mode_functions[mode_name],
                            threshold,
                            rotation
                        )
                else:
                    # Single pass with selected mode
                    current_image = self.apply_pixel_sorting(
                        current_image,
                        current_mask,
                        calculate_value_fn,
                        threshold,
                        rotation
                    )
                
                batch_sorted.append(current_image)
            
            # Convert back to torch tensor
            result = torch.from_numpy(np.stack(batch_sorted))
            
            # Always generate mask outputs regardless of use_mask setting
            processed_masks = []
            for idx in range(len(images)):
                current_mask = mask[idx].cpu().numpy() if mask is not None else np.ones_like(images[idx][..., 0])
                
                # Process mask for output
                if invert_mask:
                    current_mask = 1 - current_mask
                
                # Apply grow and blur operations
                if grow != 0 or blur > 0:
                    current_mask = self.expand_mask(torch.from_numpy(current_mask), grow, blur).numpy()
                
                processed_masks.append(current_mask)
            
            mask_out = torch.from_numpy(np.stack(processed_masks))
            
            # Create inverted mask output
            inverted_mask_out = 1 - mask_out
            
            print(f"Processing complete. Output shape: {result.shape}")
            return (result, mask_out, inverted_mask_out)
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            return (images, mask if mask is not None else torch.ones_like(images[..., 0]), torch.zeros_like(images[..., 0]))

# Register the node with ComfyUI
NODE_CLASS_MAPPINGS = {
    "DjzDatamoshV8Enhanced": DjzDatamoshV8Enhanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DjzDatamoshV8Enhanced": "Djz Pixel Sort V8 Enhanced"
}
