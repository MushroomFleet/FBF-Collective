import numpy as np
import torch
import random
import cv2

class ToadVideoFrameManipulator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),

                # Color to BW settings
                "color_to_bw_on": ("BOOLEAN", {"default": True}),
                "color_to_bw_probability": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bw_duration": ("INT", {"default": 5, "min": 1}),

                # Flip settings
                "flip_on": ("BOOLEAN", {"default": True}),
                "flip_probability": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "flip_duration": ("INT", {"default": 3, "min": 1}),

                # Mirror settings
                "mirror_on": ("BOOLEAN", {"default": True}),
                "mirror_probability": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mirror_duration": ("INT", {"default": 4, "min": 1}),

                # Saturation settings
                "saturation_on": ("BOOLEAN", {"default": True}),
                "saturation_probability": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "saturation_min": ("FLOAT", {"default": 0.5, "min": 0.00, "max": 5.0, "step": 0.01}),
                "saturation_max": ("FLOAT", {"default": 5.0, "min": 0.00, "max": 5.0, "step": 0.01}),
                "saturation_duration": ("INT", {"default": 5, "min": 1}),

                # Tearing settings
                "tearing_on": ("BOOLEAN", {"default": True}),
                "tearing_probability": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tearing_duration": ("INT", {"default": 5, "min": 1}),

                # Melting settings
                "melting_on": ("BOOLEAN", {"default": True}),
                "melting_probability": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "melting_duration": ("INT", {"default": 5, "min": 1}),

                # Tiling settings
                "tiling_on": ("BOOLEAN", {"default": True}),
                "tiling_probability": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tiling_factor": ("INT", {"default": 4, "min": 2, "max": 16}),

                # Color separation settings
                "color_separation_on": ("BOOLEAN", {"default": True}),
                "color_separation_probability": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color_separation_distance": ("INT", {"default": 50, "min": 5, "max": 1000}),

                # Pixelate settings
                "pixelate_on": ("BOOLEAN", {"default": True}),
                "pixelate_probability": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pixelate_factor": ("INT", {"default": 4, "min": 2, "max": 50}),

            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process_frames"
    CATEGORY = "Video Manipulation"

    def __init__(self):
        self.bw_counter = 0
        self.flip_counter = 0
        self.mirror_counter = 0
        self.saturation_counter = 0
        self.tearing_counter = 0
        self.melting_counter = 0
        self.tiling_counter = 0
        self.color_separation_counter = 0
        self.pixelate_counter = 0

    def process_frames(self, images, color_to_bw_probability, bw_duration, flip_probability, flip_duration, mirror_probability, mirror_duration, saturation_probability, saturation_min, saturation_max, saturation_duration, tearing_probability, tearing_duration, melting_probability, melting_duration, tiling_probability, tiling_factor, color_separation_probability, color_separation_distance, pixelate_probability, pixelate_factor, color_to_bw_on, flip_on, mirror_on, saturation_on, tearing_on, melting_on, tiling_on, color_separation_on, pixelate_on):
        batch_numpy = images.cpu().numpy()
        batch_size = batch_numpy.shape[0]
        output_images = []

        # Ensure all frames have the same shape
        frame_shape = batch_numpy[0].shape
        for i in range(batch_size):
            frame = batch_numpy[i]

            if frame.shape != frame_shape:
                frame = cv2.resize(frame, (frame_shape[1], frame_shape[0]))

            if color_to_bw_on:
                if self.bw_counter > 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels
                    self.bw_counter -= 1
                elif random.random() < color_to_bw_probability:
                    self.bw_counter = bw_duration

            if flip_on:
                if self.flip_counter > 0:
                    frame = cv2.flip(frame, 0)
                    self.flip_counter -= 1
                elif random.random() < flip_probability:
                    self.flip_counter = flip_duration

            if mirror_on:
                if self.mirror_counter > 0:
                    frame = cv2.flip(frame, 1)
                    self.mirror_counter -= 1
                elif random.random() < mirror_probability:
                    self.mirror_counter = mirror_duration

            if saturation_on:
                if self.saturation_counter > 0:
                    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    saturation_change = random.uniform(saturation_min, saturation_max)
                    hsv_frame[..., 1] = np.clip(hsv_frame[..., 1] * saturation_change, 0, 255)
                    frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
                    self.saturation_counter -= 1
                elif random.random() < saturation_probability:
                    self.saturation_counter = saturation_duration

            if tearing_on:
                if self.tearing_counter > 0:
                    tear_line = random.randint(0, frame.shape[0] - 1)
                    frame[tear_line:] = np.roll(frame[tear_line:], random.randint(-10, 10), axis=1)
                    self.tearing_counter -= 1
                elif random.random() < tearing_probability:
                    self.tearing_counter = tearing_duration

            if melting_on:
                if self.melting_counter > 0:
                    num_strips = random.randint(1, 4)
                    strip_width = frame.shape[1] // num_strips
                    for j in range(num_strips):
                        strip_start = j * strip_width
                        strip_end = (j + 1) * strip_width
                        strip_shift = random.randint(0, 15)
                        frame[:, strip_start:strip_end] = np.roll(frame[:, strip_start:strip_end], strip_shift, axis=0)
                    self.melting_counter -= 1
                elif random.random() < melting_probability:
                    self.melting_counter = melting_duration

            if tiling_on:
                if random.random() < tiling_probability:
                    random_tiling_factor = random.randint(2, tiling_factor)
                    tile_size = frame_shape[0] // random_tiling_factor
                    frame = cv2.resize(frame, (tile_size, tile_size))
                    frame = np.tile(frame, (random_tiling_factor, random_tiling_factor, 1))
                    frame = cv2.resize(frame, (frame_shape[1], frame_shape[0]))

            if color_separation_on:
                if random.random() < color_separation_probability:
                    b, g, r = cv2.split(frame)
                    frame = cv2.merge([np.roll(b, color_separation_distance, axis=1), np.roll(g, 0, axis=1), np.roll(r, -color_separation_distance, axis=1)])

            if pixelate_on:
                if self.pixelate_counter > 0:
                    small_frame = cv2.resize(frame, (frame_shape[1] // pixelate_factor, frame_shape[0] // pixelate_factor))
                    frame = cv2.resize(small_frame, (frame_shape[1], frame_shape[0]))
                    self.pixelate_counter -= 1
                elif random.random() < pixelate_probability:
                    self.pixelate_counter = pixelate_factor

            output_images.append(frame)

        return (torch.from_numpy(np.array(output_images)).to(images.device),)

NODE_CLASS_MAPPINGS = {
    "ToadVideoFrameManipulator": ToadVideoFrameManipulator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ToadVideoFrameManipulator": "Toad Video Frame Manipulator",
}
