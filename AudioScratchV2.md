# AudioScratchV2 Node

## Description
The AudioScratchV2 node provides advanced audio manipulation capabilities through various envelope types and stretching modes. It allows for reactive audio processing based on different audio characteristics.

## Input Parameters

### Required Parameters
- **Audio Driver**: Input audio signal that controls the modulation
- **Audio IN**: Input audio signal to be processed
- **envelope_type**: Type of envelope to use for modulation
  - Options: `amplitude`, `pitch`, `density`, `spectral_centroid`, `rms`, `spectral_flatness`
  - Default: `amplitude`
- **base_stretch**: Base stretching factor
  - Range: -2.0 to 2.0
  - Default: 0.1
  - Step: 0.01
- **stretch_intensity**: Intensity of the stretching effect
  - Range: 0.0 to 5.0
  - Default: 1.9
  - Step: 0.01
- **stretch_mode**: Mode of stretching application
  - Options: `multiply`, `add`, `clip`
  - Default: `multiply`
- **maintain_pitch**: Whether to maintain the original pitch during stretching
  - Default: False

## Outputs
- **Audio OUT**: Processed audio signal

## Envelope Types

### Amplitude
Modulates based on the audio's amplitude envelope, useful for volume-based effects.

### Pitch
Uses zero-crossing rate for pitch-based modulation, effective for frequency-dependent effects.

### Density
Calculates signal density using spectral flux, good for texture-based modulation.

### Spectral Centroid
Tracks brightness/timbre changes in the audio, ideal for spectral-based effects.

### RMS (Root Mean Square)
Uses RMS energy for modulation, provides smooth volume-based effects.

### Spectral Flatness
Measures tonal vs. noisy content, useful for distinguishing between harmonic and noise-based sounds.

## Example Usage

### Basic Time Stretching
```
envelope_type: amplitude
base_stretch: 0.1
stretch_intensity: 1.0
stretch_mode: multiply
maintain_pitch: True
```
Creates a basic time-stretching effect that responds to the audio's amplitude.

### Pitch-Based Modulation
```
envelope_type: pitch
base_stretch: 0.5
stretch_intensity: 2.0
stretch_mode: add
maintain_pitch: False
```
Produces modulation based on pitch changes in the audio.

### Spectral Effects
```
envelope_type: spectral_centroid
base_stretch: 0.2
stretch_intensity: 1.5
stretch_mode: clip
maintain_pitch: True
```
Creates effects that respond to the spectral content of the audio.
