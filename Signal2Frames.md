# Signal2Frames Node

## Description
The Signal2Frames node converts audio signals into microscope-like surface visualization frames. It provides extensive customization options for surface generation, color mapping, and lighting effects.

## Input Parameters

### Required Parameters
- **audio**: Input audio signal
- **image_width**: Width of output images (default: 512, min: 1)
- **image_height**: Height of output images (default: 512, min: 1)
- **frames_per_second**: Output frame rate (default: 25, min: 1)
- **window_type**: Window function for analysis
  - Options: `hann`, `hamming`, `blackman`, `kaiser`

### Optional Parameters

#### Surface Controls
- **surface_detail**: Level of surface detail (default: 1.0, range: 0.1-69.0)
- **height_scale**: Surface height scaling (default: 1.0, range: 0.1-69.0)
- **surface_smoothing**: Surface smoothness (default: 0.5, range: 0.0-69.0)

#### Color Mapping
- **low_intensity_red**: Red value for low intensity (default: 255, range: 0-255)
- **low_intensity_green**: Green value for low intensity (default: 50, range: 0-255)
- **low_intensity_blue**: Blue value for low intensity (default: 50, range: 0-255)
- **mid_intensity_red**: Red value for mid intensity (default: 50, range: 0-255)
- **mid_intensity_green**: Green value for mid intensity (default: 255, range: 0-255)
- **mid_intensity_blue**: Blue value for mid intensity (default: 50, range: 0-255)
- **high_intensity_red**: Red value for high intensity (default: 50, range: 0-255)
- **high_intensity_green**: Green value for high intensity (default: 50, range: 0-255)
- **high_intensity_blue**: Blue value for high intensity (default: 255, range: 0-255)
- **intensity_contrast**: Color contrast adjustment (default: 1.2, range: 0.1-3.0)

#### Lighting
- **light_angle**: Direction of light source (default: 45.0, range: 0.0-360.0)
- **light_intensity**: Strength of directional light (default: 0.6, range: 0.0-1.0)
- **ambient_light**: Level of ambient lighting (default: 0.4, range: 0.0-1.0)

## Outputs
- **IMAGE**: Generated surface visualization frames

## Example Usage

### High Detail Surface
```
surface_detail: 2.0
height_scale: 1.5
surface_smoothing: 0.3
intensity_contrast: 1.5
light_intensity: 0.8
ambient_light: 0.3
```
Creates detailed surface visualization with enhanced contrast and lighting.

### Smooth Terrain Effect
```
surface_detail: 0.8
height_scale: 1.2
surface_smoothing: 1.5
light_angle: 30.0
light_intensity: 0.5
ambient_light: 0.6
```
Produces smooth, terrain-like visualizations with softer lighting.

### Custom Color Mapping
```
low_intensity_red: 200
low_intensity_green: 100
low_intensity_blue: 50
mid_intensity_red: 100
mid_intensity_green: 200
mid_intensity_blue: 100
high_intensity_red: 50
high_intensity_green: 100
high_intensity_blue: 200
intensity_contrast: 1.8
```
Creates visualizations with custom color gradients and enhanced contrast.
