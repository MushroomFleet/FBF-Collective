# Psych: Audio-Driven RGB Modifier

## Overview
The **Psych** node applies dynamic, audio-driven color modulation to image frames. By analyzing a synchronized audio signal, it extracts frequency information that is used to modulate the RGB channels of each frame. This creates an effect where visual elements respond in real time to audio characteristics, making it ideal for music visualizations, live performances, and other interactive multimedia projects.

## How It Works
1. **Audio Analysis:**  
   - The audio waveform is divided into segments corresponding to the number of image frames.
   - A window function (e.g., Hann, Hamming, Blackman, or Kaiser) is applied to each segment.
   - The Fast Fourier Transform (FFT) is used to compute the frequency spectrum.
   - The spectrum is split into three bands (low, mid, high). Each band’s magnitude is enhanced based on both mean and peak values with a sensitivity adjustment.

2. **Frequency Smoothing & Persistence:**  
   - A Gaussian filter smooths the extracted frequency bands to reduce abrupt changes.
   - A color persistence mechanism blends the current frequency bands with those from the previous frame, creating smoother, more cohesive transitions.

3. **Color Modulation:**  
   - The analyzed audio bands drive the modulation of the red, green, and blue channels of each image.
   - The modulation intensity for each channel is adjustable.
   - When enabled, the inversion mode applies an inverted modulation scheme to alter the visual effect.

## Parameters
- **images** (IMAGE):  
  A tensor of input image frames. Each frame is processed individually to apply audio-driven modifications.

- **audio** (AUDIO):  
  Audio input (a dictionary containing a waveform and sample rate) that is analyzed to generate modulation parameters.

- **frames_per_second** (INT, default: 25):  
  The number of frames per second to process. This value determines how the audio is segmented for frequency analysis.

- **window_type** (Choice: "hann", "hamming", "blackman", "kaiser"; default: "hann"):  
  Selects the window function applied to each audio segment before FFT analysis.

- **invert_color_mode** (BOOLEAN, default: False):  
  If set to True, the node applies an inverted modulation scheme, altering the basic color mapping for creative effects.

### Optional Parameters
- **red_intensity** (FLOAT, default: 1.0, min: 0.0, max: 2.0):  
  Adjusts the modulation strength for the red channel.

- **green_intensity** (FLOAT, default: 1.0, min: 0.0, max: 2.0):  
  Adjusts the modulation strength for the green channel.

- **blue_intensity** (FLOAT, default: 1.0, min: 0.0, max: 2.0):  
  Adjusts the modulation strength for the blue channel.

- **frequency_smoothing** (FLOAT, default: 0.5, min: 0.0, max: 2.0):  
  Controls the level of Gaussian smoothing applied to the frequency bands, reducing abrupt fluctuations.

- **color_persistence** (FLOAT, default: 0.3, min: 0.0, max: 1.0):  
  Determines the blend ratio between the frequency bands of the current frame and the previous frame, enhancing visual continuity.

- **modulation_depth** (FLOAT, default: 0.5, min: 0.0, max: 1.0):  
  Specifies the depth of modulation applied to image colors, affecting the overall intensity of the color change.

- **sensitivity** (FLOAT, default: 1.0, min: 0.1, max: 10.0):  
  Sets the sensitivity of the audio analysis, influencing how much weight is given to peak versus mean frequency values.

## Return Value
- **IMAGE**:  
  A tensor containing the processed image frames with audio-driven color modulation applied.

## Dependencies
- **NumPy:** For numerical operations and handling arrays.
- **PyTorch:** For tensor manipulation and integration with image data.
- **SciPy:** Provides window functions and Gaussian filtering.
- **tqdm:** Displays progress during frame processing.

## Example Usage
1. Supply a tensor of image frames along with an audio input containing a waveform and sample rate.
2. Set the **frames_per_second** parameter to match your video's frame rate if needed.
3. Choose a **window_type** based on your preferred spectral analysis method.
4. Optionally, adjust the red, green, and blue intensity parameters, as well as **frequency_smoothing**, **color_persistence**, **modulation_depth**, and **sensitivity** to fine-tune the visual effect.
5. The node processes each frame by analyzing the corresponding audio segment and then modulates the image colors accordingly.

Enjoy creating dynamic, audio-reactive visuals with the Psych node!
