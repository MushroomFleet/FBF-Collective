import numpy as np
import torch
import cv2

class RGBStream:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",) # Single IMAGE output
    RETURN_NAMES = ("RGB_Stream_Output",) # Single output name
    FUNCTION = "create_rgb_stream"
    CATEGORY = "Sonification"

    def create_rgb_stream(self, images):
        batch_numpy = images.cpu().numpy()
        batch_size = batch_numpy.shape[0]

        rgb_stream_output_list = [] # Single list to hold all output channel images

        for i in range(batch_size):
            frame = batch_numpy[i]
            r, g, b = cv2.split(frame)

            # Output color-tinted RGB images for each channel
            red_tinted = np.zeros_like(frame)
            green_tinted = np.zeros_like(frame)
            blue_tinted = np.zeros_like(frame)

            red_tinted[..., 0] = r  # Grayscale r into Red channel
            green_tinted[..., 1] = g  # Grayscale g into Green channel
            blue_tinted[..., 2] = b  # Grayscale b into Blue channel

            rgb_stream_output_list.append(red_tinted) # Append Red channel image
            rgb_stream_output_list.append(green_tinted) # Append Green channel image
            rgb_stream_output_list.append(blue_tinted) # Append Blue channel image

        rgb_stream_output_np = np.array(rgb_stream_output_list) # Convert single list to numpy array

        return (torch.from_numpy(rgb_stream_output_np).to(images.device),) # Return as tuple with SINGLE tensor

NODE_CLASS_MAPPINGS = {
    "RGBStream": RGBStream,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RGBStream": "RGB Stream",
}