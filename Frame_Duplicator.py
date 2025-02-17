import numpy as np
import torch
from tqdm import tqdm

class Frame_Duplicator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "Number_of_frames": ("INT", {"default": 1, "min": 1, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "duplicate_frames"
    CATEGORY = "Sonification"

    def duplicate_frames(self, images, Number_of_frames):
        """
        Duplicates the input frame 'Number_of_frames' times.
        """
        image_np = images.cpu().numpy()
        single_frame = image_np[0]

        duplicated_frames_list = []

        with tqdm(total=Number_of_frames, desc="Duplicating Frames") as progress_bar:
            for _ in range(Number_of_frames):
                duplicated_frames_list.append(single_frame)
                progress_bar.update(1)

        duplicated_frames_np = np.array(duplicated_frames_list)
        duplicated_frames_torch = torch.from_numpy(duplicated_frames_np).to(images.device)

        return (duplicated_frames_torch,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "Frame_Duplicator": Frame_Duplicator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Frame_Duplicator": "Frame Duplicator"
}