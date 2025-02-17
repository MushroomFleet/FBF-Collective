# Video Echo

## Overview
The **Video Echo** node creates trailing effects on video frames by blending the current frame with processed data from previous frames. This node is designed to simulate echo or trail effects, where visual information from past frames influences the present output. It incorporates motion detection, Gaussian blur, and directional color bleeding to generate a dynamic, artistic echo effect that can enhance video compositing and sonification workflows.

## How It Works
1. **Gaussian Blur:**  
   The node can apply a Gaussian blur to the current frame (controlled by the **blur_amount** parameter) to soften details before further processing.

2. **Motion Detection:**  
   Using the `detect_motion_v2` function, the node detects movement by comparing the current frame with the previous frame. A motion mask is generated based on the **motion_threshold**, which helps decide where to update the trailing effect.

3. **Trail Buffer and Decay:**  
   - A trail buffer is maintained to accumulate the visual "echo" from previous frames.
   - With each new frame, the trail buffer is decayed exponentially using the **decay_rate** parameter.
   - Depending on the mode selected:
     - **Motion Based Trails:** The node updates the trail buffer only in areas where motion is detected.
     - **Full Frame Trails:** The entire frame is blended into the trail buffer.
     - If neither is enabled, a smooth smear mode is applied where the current (possibly blurred) frame is added to the decayed trail buffer.
   - The **trail_strength** parameter controls how much the current frame influences the trail buffer.
   - Additionally, the **trails_transparency** parameter sets the transparency level of the current frame before it is integrated into the trail.

4. **Color Bleed:**  
   When **color_bleed** is greater than zero, a color bleeding effect is applied:
   - Each color channel (red, green, and blue) is shifted directionally based on the corresponding bleed direction parameters (**red_bleed_direction**, **green_bleed_direction**, **blue_bleed_direction**).
   - The **color_bleed_edge_mode** parameter (with options such as Clamp, Zero, Wrap, or Reflect) determines how the edges are handled during the bleed.
   
5. **Blend Modes:**  
   After processing the trail buffer, the node blends it with the current frame using a user-selected **blend_mode**. Supported modes include:
   - **Normal, Additive, Multiply, Screen, Overlay, Soft Light, Hard Light, Difference, Color Burn,** and **Color Dodge.**
   Each mode employs a different mathematical approach to combine the current frame with the echo effect.

## Parameters
- **images** (IMAGE):  
  A tensor containing input video frames.

- **blend_mode** (Choice):  
  Determines the method for combining the current frame with the trail buffer. Options include: "Normal", "Additive", "Multiply", "Screen", "Overlay", "Soft Light", "Hard Light", "Difference", "Color Burn", and "Color Dodge".

- **trail_strength** (FLOAT, default: 0.85, min: 0.1, max: 0.99, step: 0.01):  
  Controls how strongly the trail buffer is updated with the current frame.

- **decay_rate** (FLOAT, default: 0.15, min: 0.01, max: 0.5, step: 0.01):  
  Specifies the exponential decay applied to the trail buffer between frames.

- **color_bleed** (FLOAT, default: 0.3, min: 0.0, max: 1.0, step: 0.05):  
  Sets the amount of directional color bleeding to apply to the trail buffer.

- **color_bleed_edge_mode** (Choice: "Clamp", "Zero", "Wrap", "Reflect"; default: "Clamp"):  
  Determines how image edges are treated during the color bleeding process.

- **blur_amount** (FLOAT, default: 0.5, min: 0.0, max: 2.0, step: 0.1):  
  Adjusts the intensity of the Gaussian blur applied to the current frame.

- **motion_threshold** (FLOAT, default: 0.1, min: 0.01, max: 0.5, step: 0.01):  
  Threshold for detecting motion differences between consecutive frames.

- **motion_based_trails** (BOOLEAN, default: True):  
  If enabled, updates to the trail buffer are based on detected motion areas.

- **red_bleed_direction** (Choice: "Right", "Left", "Up", "Down", "None"; default: "Right"):  
  Direction for the red channel bleed.

- **green_bleed_direction** (Choice: "Right", "Left", "Up", "Down", "None"; default: "None"):  
  Direction for the green channel bleed.

- **blue_bleed_direction** (Choice: "Right", "Left", "Up", "Down", "None"; default: "Left"):  
  Direction for the blue channel bleed.

- **full_frame_trails** (BOOLEAN, default: False):  
  When enabled, applies trail updates across the full frame rather than using motion-based trails.

- **trails_transparency** (FLOAT, default: 0.7, min: 0.01, max: 1.0, step: 0.01):  
  Sets the transparency level of the current frame before it is blended into the trail buffer.

## Return Value
- **IMAGE**:  
  A tensor containing the processed video frames with the echo effect applied.

## Example Usage
1. Feed a sequence of video frames into the node along with appropriate parameters.
2. Select a **blend_mode** (e.g., "Overlay") to achieve the desired echo effect.
3. Adjust **trail_strength** and **decay_rate** to control trail visibility and fading.
4. Configure color bleeding parameters to add dynamic chromatic shifts.
5. Enable **motion_based_trails** to have the effect react to scene movement, or use **full_frame_trails** for a uniform trail across the frame.
6. The node processes the video frame-by-frame, blending the current input with the decayed trail buffer, and outputs a dynamically echoing video.

Happy echoing!
