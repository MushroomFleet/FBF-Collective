import numpy as np
import torch
import cv2

class VideoRGBSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("red_channel", "green_channel", "blue_channel")
    FUNCTION = "split_rgb_channels"
    CATEGORY = "Split RGB Channels Image"

    def split_rgb_channels(self, images):
        batch_numpy = images.cpu().numpy()
        batch_size = batch_numpy.shape[0]
        
        red_channel = []
        green_channel = []
        blue_channel = []

        for i in range(batch_size):
            frame = batch_numpy[i]
            r, g, b = cv2.split(frame)
            red_channel.append(r)
            green_channel.append(g)
            blue_channel.append(b)

        red_channel = np.array(red_channel)
        green_channel = np.array(green_channel)
        blue_channel = np.array(blue_channel)

        return (
            torch.from_numpy(red_channel).to(images.device),
            torch.from_numpy(green_channel).to(images.device),
            torch.from_numpy(blue_channel).to(images.device),
        )

NODE_CLASS_MAPPINGS = {
    "VideoRGBSplitter": VideoRGBSplitter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoRGBSplitter": "Video RGB Splitter",
}
