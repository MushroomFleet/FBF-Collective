# RGB Zoom: Audio-Driven RGB Zoom

## Overview
The **RGB Zoom** node applies dynamic, audio-driven zoom effects to image frames. It analyzes an accompanying audio signal to extract frequency-based information and uses this data to adjust the zoom level for each RGB channel independently. This effect can be used to create visually engaging sonification effects where the image responds to audio characteristics.

## How It Works
1. **Audio Analysis:**  
   - The audio waveform is segmented based on the number of image frames.
   - A chosen window function (e.g., Hann, Hamming, Blackman, or Kaiser) is applied to each audio segment.
   - The node performs either:
     - A **simple zoom** calculation that derives a uniform zoom factor from the overall amplitude, if *Simple_Zoom* is enabled.
     - A regular frequency band analysis where the FFT is computed on the windowed data, and the spectrum is divided into three bands (low, mid, high). Each band's magnitude is enhanced based on its mean and peak values with sensitivity adjustment.
   - The resulting frequency information is normalized and optionally smoothed with a Gaussian filter. A persistence mechanism blends current values with those from the previous frame if desired.

2. **Zoom Application:**  
   - For **simple zoom**, a single zoom factor is computed and applied uniformly across the image.
   - For the detailed method, the node applies independent zoom factors to each color channel. Each channel’s zoom is influenced by its corresponding audio band value, an intensity parameter, and a zoom range.
   - After channel-specific zooms are applied, a final adjustment is made using user-specified center coordinates to control the focal point of the zoomed image.

## Parameters

### Required Parameters
- **images** (IMAGE):  
  A tensor of input image frames to be processed.

- **audio** (AUDIO):  
  Audio input that is synchronized with the image frames. This should be provided as a dictionary with a waveform and sample rate.

- **frames_per_second** (INT, default: 25):  
  The frame rate of the images that determines how the audio is segmented.

- **window_type** (Choice: "hann", "hamming", "blackman", "kaiser"; default: "hann"):  
  The window function applied to each audio segment prior to Fourier analysis.

- **Simple_Zoom** (BOOLEAN, default: False):  
  When set to True, the node uses a simplified zoom calculation based on overall amplitude rather than per-channel frequency analysis.

- **X_center** (FLOAT, default: 0.0, min: -1.0, max: 1.0):  
  The horizontal center for the zoom effect. Values shift the zoom focus left or right.

- **Y_center** (FLOAT, default: 0.0, min: -1.0, max: 1.0):  
  The vertical center for the zoom effect. Values shift the zoom focus up or down.

### Optional Parameters
- **red_zoom_intensity** (FLOAT, default: 1.0, min: 0.01, max: 20.0):  
  Controls the zoom intensity applied to the red channel.

- **green_zoom_intensity** (FLOAT, default: 1.0, min: 0.01, max: 20.0):  
  Controls the zoom intensity applied to the green channel.

- **blue_zoom_intensity** (FLOAT, default: 1.0, min: 0.01, max: 20.0):  
  Controls the zoom intensity applied to the blue channel.

- **frequency_smoothing** (FLOAT, default: 0.0, min: 0.0, max: 2.0):  
  Amount of Gaussian smoothing to apply to the computed frequency bands, reducing sudden fluctuations.

- **zoom_persistence** (FLOAT, default: 0.0, min: 0.0, max: 1.0):  
  Blends the current audio analysis with the previous frame’s values to smooth transitions over time.

- **zoom_range** (FLOAT, default: 1.0, min: 0.0, max: 10.0):  
  Determines the maximum additional zoom factor that can be applied based on the audio analysis.

- **sensitivity** (FLOAT, default: 1.0, min: 0.1, max: 10.0):  
  Adjusts the responsiveness of the audio analysis, influencing how strongly the frequency information affects the zoom factor.

## Return Value
- **IMAGE**:  
  A tensor containing the processed image frames with the audio-driven zoom effect applied.

## Dependencies
- **NumPy:** For numerical operations and array processing.
- **PyTorch:** For tensor manipulation and integration with image data.
- **SciPy:** Provides window functions, the zoom operation, and Gaussian filtering.
- **tqdm:** Displays progress during frame processing.

## Example Usage
1. Provide a tensor of image frames and an audio input (with waveform and sample rate).
2. Set the **frames_per_second** parameter to match the frame rate of your images.
3. Choose a **window_type** suitable for your audio analysis.
4. Enable **Simple_Zoom** for a unified zoom effect or leave it disabled for channel-specific analysis.
5. Adjust **X_center** and **Y_center** to control the focal point of the zoom.
6. Optionally, fine-tune the zoom intensities (**red_zoom_intensity**, **green_zoom_intensity**, **blue_zoom_intensity**), **frequency_smoothing**, **zoom_persistence**, **zoom_range**, and **sensitivity** to achieve the desired visual effect.
7. The node processes each frame, applies the zoom effects driven by the audio signal, and outputs the resulting frames as a tensor ready for further processing or display.

Enjoy creating dynamic, audio-reactive zoom effects with the RGB Zoom node!
