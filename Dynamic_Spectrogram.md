# Dynamic Spectrogram Node

## Description
The Dynamic Spectrogram node creates advanced spectrogram visualizations from audio input with extensive customization options for color mapping, window functions, and spectral analysis parameters.

## Input Parameters

### Required Parameters
- **audio_input**: Input audio signal
- **image_width**: Width of output images (default: 256, range: 64-1024)
- **image_height**: Height of output images (default: 256, range: 64-1024)
- **frames_per_second**: Output frame rate (default: 25, range: 1-60)
- **color_map**: Visual style for the spectrogram
  - Options:
    - Cosmic Flow (cubehelix)
    - Orange-Purple Gradient (plasma)
    - Red-Black Intensity (inferno)
    - Purple-Red Spectrum (magma)
    - Blue Temperature (cool)
    - Red Temperature (hot)
    - Ocean Depths (ocean)
    - Rainbow Spectrum (jet)
    - Thermal Imaging (gnuplot2)
    - Electric Blue (cool)
    - Fiery Red (afmhot)
    - Forest Green (Greens)
- **window_function**: Type of window function for analysis
  - Options: `hann`, `hamming`, `kaiser`, `blackman`
- **spectrogram_type**: Type of spectrogram analysis
  - Options: `magnitude`, `power`, `mel`
- **temporal_resolution**: Time resolution (default: 1, range: 1-100)

### Optional Parameters
- **n_fft**: FFT window size (default: 2048, range: 512-16384, step: 256)
- **hop_length**: Analysis frame hop (default: 8, range: 8-4096, step: 8)
- **n_mels**: Number of mel bands (default: 256, range: 16-512, step: 16)
- **contrast**: Spectrogram contrast (default: 1.0, range: 0.1-3.0)
- **normalize**: Enable normalization (default: True)
- **min_frequency**: Minimum frequency (default: 0.0, range: 0-20000 Hz)
- **max_frequency**: Maximum frequency (default: 20000.0, range: 0-20000 Hz)
- **detail_level**: Detail enhancement (default: 1.0, range: 0.1-2.0)

## Outputs
- **spectrograms**: Generated spectrogram frames

## Example Usage

### High-Resolution Mel Spectrogram
```
image_width: 512
image_height: 512
frames_per_second: 30
color_map: "Fiery Red"
window_function: hann
spectrogram_type: mel
n_fft: 4096
hop_length: 16
n_mels: 512
contrast: 1.5
detail_level: 1.8
```
Creates detailed mel spectrograms with high temporal and frequency resolution.

### Power Spectrogram with Custom Frequency Range
```
spectrogram_type: power
window_function: blackman
min_frequency: 20.0
max_frequency: 8000.0
contrast: 2.0
normalize: True
detail_level: 1.2
```
Generates power spectrograms focused on a specific frequency range.

### Low-Latency Visualization
```
image_width: 256
image_height: 256
frames_per_second: 60
n_fft: 1024
hop_length: 8
temporal_resolution: 2
detail_level: 0.8
```
Optimized for real-time visualization with lower latency.
