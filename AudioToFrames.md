# AudioToFrames Node

## Description
The AudioToFrames node converts audio signals into visual frames, generating both waveform-based images and spectrograms. It provides extensive customization options for visual representation of audio data.

## Input Parameters

### Required Parameters
- **audio**: Input audio signal
- **image_width**: Width of output images (default: 80, min: 1)
- **image_height**: Height of output images (default: 80, min: 1)
- **num_frames**: Number of frames to generate (default: 100, min: 1)
- **color_map**: Color mapping for spectrograms
  - Options: `viridis`, `plasma`, `inferno`, `magma`, `cividis`
- **n_fft**: FFT window size (default: 512, range: 512-8192, step: 256)
- **hop_length**: Number of samples between successive frames (default: 64, range: 64-4096, step: 128)
- **n_mels**: Number of mel bands (default: 32, range: 32-256, step: 32)
- **top_db**: Maximum decibel value (default: 80.0, range: 10.0-100.0, step: 5.0)

### Optional Parameters
- **red_saturation**: Red channel saturation (default: 127, range: 0-255)
- **green_saturation**: Green channel saturation (default: 121, range: 0-255)
- **blue_saturation**: Blue channel saturation (default: 95, range: 0-255)
- **lightness**: Overall lightness control (default: 0.3, range: 0.0-10.0, step: 0.01)

## Outputs
- **Images**: Generated waveform-based images
- **Spectrograms**: Generated spectrogram images

## Example Usage

### Basic Audio Visualization
```
image_width: 256
image_height: 256
num_frames: 60
color_map: viridis
n_fft: 2048
hop_length: 512
n_mels: 128
top_db: 80.0
```
Creates standard audio visualization frames.

### High-Resolution Spectrograms
```
image_width: 512
image_height: 512
num_frames: 30
color_map: magma
n_fft: 4096
hop_length: 1024
n_mels: 256
top_db: 90.0
```
Generates detailed spectrogram visualizations.

### Custom Color Settings
```
red_saturation: 200
green_saturation: 150
blue_saturation: 100
lightness: 0.5
```
Produces frames with custom color balance and brightness.
