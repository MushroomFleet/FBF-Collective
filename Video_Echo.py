import numpy as np
import torch
import cv2

class Video_Echo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "blend_mode": (["Normal", "Additive", "Multiply", "Screen", "Overlay", "Soft Light", "Hard Light", "Difference", "Color Burn", "Color Dodge"],),
                "trail_strength": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.1,
                    "max": 0.99,
                    "step": 0.01,
                    "display": "number"
                }),
                "decay_rate": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "display": "number"
                }),
                "color_bleed": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "number"
                }),
                "color_bleed_edge_mode": (["Clamp", "Zero", "Wrap", "Reflect"], {"default": "Clamp"}),
                "blur_amount": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "motion_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "display": "number"
                }),
                "motion_based_trails": ("BOOLEAN", {"default": True, "label": "Motion Based Trails"}),
                "red_bleed_direction": (["Right", "Left", "Up", "Down", "None"], {"default": "Right"}),
                "green_bleed_direction": (["Right", "Left", "Up", "Down", "None"], {"default": "None"}),
                "blue_bleed_direction": (["Right", "Left", "Up", "Down", "None"], {"default": "Left"}),
                "full_frame_trails": ("BOOLEAN", {"default": False, "label": "Full Frame Trails"}),
                "trails_transparency": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 1.0, "step": 0.01, "label": "Trails Transparency"}), # New: Trails Transparency input
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_trails_v2"
    CATEGORY = "Sonification"

    def detect_motion_v2(self, current, previous, threshold):
        """Detect motion using frame differencing."""
        curr_frame = (current * 255).astype(np.uint8)
        prev_frame = (previous * 255).astype(np.uint8)
        diff_r = cv2.absdiff(curr_frame[..., 0], prev_frame[..., 0])
        diff_g = cv2.absdiff(curr_frame[..., 1], prev_frame[..., 1])
        diff_b = cv2.absdiff(curr_frame[..., 2], prev_frame[..., 2])
        thresh = int(threshold * 255)
        mask_r = (diff_r > thresh).astype(np.float32)
        mask_g = (diff_g > thresh).astype(np.float32)
        mask_b = (diff_b > thresh).astype(np.float32)
        return np.stack([mask_r, mask_g, mask_b], axis=-1)

    def apply_gaussian_blur(self, image, strength):
        """Apply Gaussian blur."""
        if strength <= 0:
            return image
        kernel_size = max(3, int(strength * 10) | 1)
        sigma = strength * 2
        blurred = cv2.GaussianBlur(
            (image * 255).astype(np.uint8),
            (kernel_size, kernel_size),
            sigma
        )
        return blurred.astype(np.float32) / 255.0

    def apply_color_bleed(self, image, amount, edge_mode, red_bleed_direction, green_bleed_direction, blue_bleed_direction):
        """Apply directional color bleeding effect."""
        if amount <= 0:
            return image

        offset = int(amount * 4)
        if offset == 0:
            return image

        height, width = image.shape[:2]
        result = np.zeros_like(image)

        # Offset red channel based on direction
        if red_bleed_direction == "Right":
            result[..., 0] = np.roll(image[..., 0], offset, axis=1)
        elif red_bleed_direction == "Left":
            result[..., 0] = np.roll(image[..., 0], -offset, axis=1)
        elif red_bleed_direction == "Up":
            result[..., 0] = np.roll(image[..., 0], -offset, axis=0) # Up is negative offset in axis 0
        elif red_bleed_direction == "Down":
            result[..., 0] = np.roll(image[..., 0], offset, axis=0)
        elif red_bleed_direction == "None":
            result[..., 0] = image[..., 0] # No bleed for red

        # Offset green channel based on direction
        if green_bleed_direction == "Right":
            result[..., 1] = np.roll(image[..., 1], offset, axis=1)
        elif green_bleed_direction == "Left":
            result[..., 1] = np.roll(image[..., 1], -offset, axis=1)
        elif green_bleed_direction == "Up":
            result[..., 1] = np.roll(image[..., 1], -offset, axis=0)
        elif green_bleed_direction == "Down":
            result[..., 1] = np.roll(image[..., 1], offset, axis=0)
        elif green_bleed_direction == "None":
            result[..., 1] = image[..., 1] # No bleed for green


        # Offset blue channel based on direction
        if blue_bleed_direction == "Right":
            result[..., 2] = np.roll(image[..., 2], offset, axis=1)
        elif blue_bleed_direction == "Left":
            result[..., 2] = np.roll(image[..., 2], -offset, axis=1)
        elif blue_bleed_direction == "Up":
            result[..., 2] = np.roll(image[..., 2], -offset, axis=0)
        elif blue_bleed_direction == "Down":
            result[..., 2] = np.roll(image[..., 2], offset, axis=0)
        elif blue_bleed_direction == "None":
            result[..., 2] = image[..., 2] # No bleed for blue


        # Handle edges based on edge_mode - applies to ALL channels
        if edge_mode == "Clamp":
            if red_bleed_direction in ["Right", "Left"]: # Apply clamp only for horizontal shifts, else it might clamp vertical edges incorrectly
                if red_bleed_direction == "Right":
                    result[:, :offset, 0] = image[:, :offset, 0]
                elif red_bleed_direction == "Left":
                    result[:, -offset:, 0] = image[:, -offset:, 0]
            if green_bleed_direction in ["Right", "Left"]: # Apply clamp only for horizontal shifts, else it might clamp vertical edges incorrectly
                if green_bleed_direction == "Right":
                    result[:, :offset, 1] = image[:, :offset, 1]
                elif green_bleed_direction == "Left":
                    result[:, -offset:, 1] = image[:, -offset:, 1]
            if blue_bleed_direction in ["Right", "Left"]: # Apply clamp only for horizontal shifts, else it might clamp vertical edges incorrectly
                if blue_bleed_direction == "Right":
                    result[:, :offset, 2] = image[:, :offset, 2]
                elif blue_bleed_direction == "Left":
                    result[:, -offset:, 2] = image[:, -offset:, 2]

        elif edge_mode == "Zero":
            if red_bleed_direction in ["Right", "Left"]: # Apply zero fill only for horizontal shifts
                if red_bleed_direction == "Right":
                    result[:, :offset, 0] = 0
                elif red_bleed_direction == "Left":
                    result[:, :offset, 0] = 0
            if green_bleed_direction in ["Right", "Left"]: # Apply zero fill only for horizontal shifts
                if green_bleed_direction == "Right":
                    result[:, :offset, 1] = 0
                elif green_bleed_direction == "Left":
                    result[:, :offset, 1] = 0
            if blue_bleed_direction in ["Right", "Left"]: # Apply zero fill only for horizontal shifts
                if blue_bleed_direction == "Right":
                    result[:, :offset, 2] = 0
                elif blue_bleed_direction == "Left":
                    result[:, :offset, 2] = 0

        elif edge_mode == "Wrap": # Wrap mode should work for both horizontal and vertical shifts
            if red_bleed_direction == "Right":
                result[:, :offset, 0] = np.roll(image[:, :, 0], -width + offset, axis=1)[:, :offset]
            elif red_bleed_direction == "Left":
                result[:, -offset:, 0] = np.roll(image[:, :, 0], width - offset, axis=1)[:, -offset:]
            elif red_bleed_direction == "Up":
                result[:offset, :, 0] = np.roll(image[:, :, 0], height - offset, axis=0)[:offset, :]
            elif red_bleed_direction == "Down":
                result[-offset:, :, 0] = np.roll(image[:, :, 0], -height + offset, axis=0)[-offset:, :]

            if green_bleed_direction == "Right":
                result[:, :offset, 1] = np.roll(image[:, :, 1], -width + offset, axis=1)[:, :offset]
            elif green_bleed_direction == "Left":
                result[:, -offset:, 1] = np.roll(image[:, :, 1], width - offset, axis=1)[:, -offset:]
            elif green_bleed_direction == "Up":
                result[:offset, :, 1] = np.roll(image[:, :, 1], height - offset, axis=0)[:offset, :]
            elif green_bleed_direction == "Down":
                result[-offset:, :, 1] = np.roll(image[:, :, 1], -height + offset, axis=0)[-offset:, :]

            if blue_bleed_direction == "Right":
                result[:, :offset, 2] = np.roll(image[:, :, 2], -width + offset, axis=1)[:, :offset]
            elif blue_bleed_direction == "Left":
                result[:, -offset:, 2] = np.roll(image[:, :, 2], width - offset, axis=1)[:, -offset:]
            elif blue_bleed_direction == "Up":
                result[:offset, :, 2] = np.roll(image[:, :, 2], height - offset, axis=0)[:offset, :]
            elif blue_bleed_direction == "Down":
                result[-offset:, :, 2] = np.roll(image[:, :, 2], -height + offset, axis=0)[-offset:, :]


        elif edge_mode == "Reflect": # Reflect mode should work for both horizontal and vertical shifts
            if red_bleed_direction in ["Right", "Left"]: # Reflect only for horizontal shifts, else might reflect vertical edges incorrectly
                if red_bleed_direction == "Right":
                    result[:, :offset, 0] = np.fliplr(image[:, :offset, 0])
                elif red_bleed_direction == "Left":
                    result[:, -offset:, 0] = np.fliplr(image[:, -offset:, 0])
            if green_bleed_direction in ["Right", "Left"]: # Reflect only for horizontal shifts, else might reflect vertical edges incorrectly
                if green_bleed_direction == "Right":
                    result[:, :offset, 1] = np.fliplr(image[:, :offset, 1])
                elif green_bleed_direction == "Left":
                    result[:, -offset:, 1] = np.fliplr(image[:, -offset:, 1])
            if blue_bleed_direction in ["Right", "Left"]: # Reflect only for horizontal shifts, else might reflect vertical edges incorrectly
                if blue_bleed_direction == "Right":
                    result[:, :offset, 2] = np.fliplr(image[:, :offset, 2])
                elif blue_bleed_direction == "Left":
                    result[:, -offset:, 2] = np.fliplr(image[:, -offset:, 2])


        return result


    def create_echo_video(self, current_frame_np, blend_mode, trail_strength, decay_rate, color_bleed, color_bleed_edge_mode, blur_amount, motion_threshold, red_bleed_direction, green_bleed_direction, blue_bleed_direction):
        if self.trail_buffer is None:
            self.trail_buffer = np.zeros_like(current_frame_np, dtype=np.float32)

        if blur_amount > 0:
            current_frame_np = self.apply_gaussian_blur(current_frame_np, blur_amount)

        if self.initialized:
            motion_mask = self.detect_motion_v2(
                current_frame_np,
                self.previous_frame,
                motion_threshold
            )
            self.trail_buffer *= np.exp(-decay_rate)
            self.trail_buffer = np.where(
                motion_mask > 0,
                current_frame_np + self.trail_buffer * trail_strength,
                self.trail_buffer
            )
            if color_bleed > 0:
                self.trail_buffer = self.apply_color_bleed(self.trail_buffer, color_bleed, color_bleed_edge_mode, red_bleed_direction, green_bleed_direction, blue_bleed_direction)
        else:
            self.trail_buffer = current_frame_np.copy()
            self.previous_frame = current_frame_np.copy()
            self.initialized = True

        self.trail_buffer = np.clip(self.trail_buffer, 0, 1).astype(np.float32)
        self.previous_frame = current_frame_np.copy()

        if blend_mode == "Normal":
            blended_frame = self.trail_buffer.copy()
        elif blend_mode == "Additive":
            blended_frame = np.clip(current_frame_np + self.trail_buffer, 0, 1)
        elif blend_mode == "Multiply":
            blended_frame = np.clip(current_frame_np * self.trail_buffer, 0, 1)
        elif blend_mode == "Screen":
            blended_frame = np.clip(1.0 - (1.0 - current_frame_np) * (1.0 - self.trail_buffer), 0, 1)
        elif blend_mode == "Overlay":
            blended_frame = np.where(self.trail_buffer < 0.5,
                                     2.0 * current_frame_np * self.trail_buffer,
                                     1.0 - 2.0 * (1.0 - current_frame_np) * (1.0 - self.trail_buffer))
            blended_frame = np.clip(blended_frame, 0, 1)
        elif blend_mode == "Soft Light":
            blended_frame = np.where(current_frame_np < 0.5,
                                     2.0 * current_frame_np * self.trail_buffer + self.trail_buffer**2 * (1.0 - 2.0 * current_frame_np),
                                     np.sqrt(current_frame_np) * (2.0 * self.trail_buffer - 1.0) + (2.0 * current_frame_np) * (1.0 - self.trail_buffer) + self.trail_buffer)
            blended_frame = np.clip(blended_frame, 0, 1)
        elif blend_mode == "Hard Light":
            blended_frame = np.where(current_frame_np < 0.5,
                                     2.0 * current_frame_np * self.trail_buffer,
                                     1.0 - 2.0 * (1.0 - current_frame_np) * (1.0 - self.trail_buffer))
            blended_frame = np.clip(blended_frame, 0, 1)
        elif blend_mode == "Difference":
            blended_frame = np.clip(np.abs(current_frame_np - self.trail_buffer), 0, 1)
        elif blend_mode == "Color Burn":
            blended_frame = np.clip(np.where(current_frame_np == 0.0, 0.0, 1.0 - (1.0 - self.trail_buffer) / current_frame_np), 0, 1)
        elif blend_mode == "Color Dodge":
            blended_frame = np.clip(np.where(current_frame_np == 1.0, 1.0, self.trail_buffer / (1.0 - current_frame_np)), 0, 1)
        else:
            blended_frame = self.trail_buffer.copy()

        return blended_frame

    def apply_trails_v2(self, images, blend_mode, trail_strength, decay_rate, color_bleed, color_bleed_edge_mode, blur_amount, motion_threshold, motion_based_trails, red_bleed_direction, green_bleed_direction, blue_bleed_direction, full_frame_trails, trails_transparency): # Added trails_transparency parameter

        print("\nStarting VideoEcho effect processing...")

        batch_numpy = images.cpu().numpy()
        batch_size = batch_numpy.shape[0]
        processed_batch = np.zeros_like(batch_numpy, dtype=np.float32)
        trail_buffer = np.zeros_like(batch_numpy[0], dtype=np.float32)

        print(f"Processing {batch_size} frames...")

        self.initialized = False
        self.trail_buffer = None
        self.previous_frame = None

        for i in range(batch_size):
            current_frame = batch_numpy[i].copy()

            if blur_amount > 0:
                current_frame = self.apply_gaussian_blur(current_frame, blur_amount)

            if i > 0:
                transparent_frame = current_frame * trails_transparency # Apply transparency here

                if motion_based_trails: # Motion-Based Trails Logic
                    motion_mask = self.detect_motion_v2(
                        current_frame, # Use original current_frame for motion detection
                        batch_numpy[i-1],
                        motion_threshold
                    )
                    trail_buffer *= np.exp(-decay_rate)
                    trail_buffer = np.where(
                        motion_mask > 0,
                        transparent_frame + trail_buffer * trail_strength, # Use transparent_frame for trail update
                        trail_buffer
                    )


                elif full_frame_trails: # Full Frame Trails Logic
                    trail_buffer *= np.exp(-decay_rate)
                    trail_buffer += transparent_frame * trail_strength # Use transparent_frame for accumulation


                else: # Smooth Smear Mode
                    trail_buffer *= np.exp(-decay_rate)
                    trail_buffer = transparent_frame + trail_buffer * trail_strength # Use transparent_frame for smear


                if color_bleed > 0:
                    trail_buffer = self.apply_color_bleed(trail_buffer, color_bleed, color_bleed_edge_mode, red_bleed_direction, green_bleed_direction, blue_bleed_direction)
            else:
                trail_buffer = current_frame.copy()
                self.previous_frame = current_frame.copy()
                self.initialized = True

            trail_buffer = np.clip(trail_buffer, 0, 1).astype(np.float32)
            processed_batch[i] = self.create_echo_video(current_frame, blend_mode, trail_strength, decay_rate, color_bleed, color_bleed_edge_mode, blur_amount, motion_threshold, red_bleed_direction, green_bleed_direction, blue_bleed_direction)

        print("VideoEcho processing complete!")
        return (torch.from_numpy(processed_batch).to(images.device),)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Video_Echo": Video_Echo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Video_Echo": "Video Echo"
}