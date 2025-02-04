# ToadVideoFrameManipulator Node

## Description
The ToadVideoFrameManipulator node provides a comprehensive suite of video frame manipulation effects. It includes multiple effect types that can be applied individually or in combination, each with its own probability and duration settings.

## Input Parameters

### Required Parameters

#### Color to BW Settings
- **color_to_bw_on**: Enable color to black & white conversion (default: True)
- **color_to_bw_probability**: Probability of effect (default: 0.1, range: 0.0-1.0)
- **bw_duration**: Duration of effect in frames (default: 5, min: 1)

#### Flip Settings
- **flip_on**: Enable vertical flip effect (default: True)
- **flip_probability**: Probability of effect (default: 0.10, range: 0.0-1.0)
- **flip_duration**: Duration of effect in frames (default: 3, min: 1)

#### Mirror Settings
- **mirror_on**: Enable horizontal mirror effect (default: True)
- **mirror_probability**: Probability of effect (default: 0.10, range: 0.0-1.0)
- **mirror_duration**: Duration of effect in frames (default: 4, min: 1)

#### Saturation Settings
- **saturation_on**: Enable saturation adjustment (default: True)
- **saturation_probability**: Probability of effect (default: 0.10, range: 0.0-1.0)
- **saturation_min**: Minimum saturation value (default: 0.5, range: 0.0-5.0)
- **saturation_max**: Maximum saturation value (default: 5.0, range: 0.0-5.0)
- **saturation_duration**: Duration of effect in frames (default: 5, min: 1)

#### Tearing Settings
- **tearing_on**: Enable frame tearing effect (default: True)
- **tearing_probability**: Probability of effect (default: 0.10, range: 0.0-1.0)
- **tearing_duration**: Duration of effect in frames (default: 5, min: 1)

#### Melting Settings
- **melting_on**: Enable melting effect (default: True)
- **melting_probability**: Probability of effect (default: 0.10, range: 0.0-1.0)
- **melting_duration**: Duration of effect in frames (default: 5, min: 1)

#### Tiling Settings
- **tiling_on**: Enable frame tiling effect (default: True)
- **tiling_probability**: Probability of effect (default: 0.10, range: 0.0-1.0)
- **tiling_factor**: Tiling grid size (default: 4, range: 2-16)

#### Color Separation Settings
- **color_separation_on**: Enable RGB channel separation (default: True)
- **color_separation_probability**: Probability of effect (default: 0.10, range: 0.0-1.0)
- **color_separation_distance**: Pixel distance for separation (default: 50, range: 5-1000)

#### Pixelate Settings
- **pixelate_on**: Enable pixelation effect (default: True)
- **pixelate_probability**: Probability of effect (default: 0.10, range: 0.0-1.0)
- **pixelate_factor**: Pixelation intensity (default: 4, range: 2-50)

## Outputs
- **images**: Processed image frames with applied effects

## Effect Descriptions

### Color to BW
Converts frames to black and white temporarily, creating dramatic contrast shifts.

### Flip & Mirror
Provides vertical and horizontal reflection effects for disorienting visual impact.

### Saturation
Dynamically adjusts color saturation levels for vibrant or muted effects.

### Tearing
Creates horizontal displacement effects simulating video signal disruption.

### Melting
Applies vertical distortion effects that appear to melt the frame contents.

### Tiling
Repeats frame content in a grid pattern with customizable size.

### Color Separation
Splits RGB channels with controllable offset for chromatic aberration effects.

### Pixelate
Reduces frame resolution temporarily for retro-style effects.

## Example Usage

### Subtle Glitch Effects
```
color_to_bw_probability: 0.05
flip_probability: 0.05
mirror_probability: 0.05
tearing_probability: 0.05
color_separation_distance: 20
```
Creates occasional, mild distortion effects.

### Intense Manipulation
```
saturation_min: 0.1
saturation_max: 4.0
melting_probability: 0.2
tiling_factor: 8
pixelate_factor: 10
```
Produces more frequent and dramatic visual effects.

### Color-Focused Effects
```
color_to_bw_probability: 0.15
saturation_probability: 0.2
color_separation_probability: 0.15
color_separation_distance: 100
```
Emphasizes color manipulation effects.
