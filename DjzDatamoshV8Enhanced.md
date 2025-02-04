# DjzDatamoshV8Enhanced Node

## Description
The DjzDatamoshV8Enhanced node provides advanced pixel sorting and datamoshing effects for images. It includes multiple sorting modes, mask support, and various parameters for fine-tuning the visual effects.

## Input Parameters

### Required Parameters
- **images**: Input image batch
- **sort_mode**: Pixel sorting method
  - Options: `luminance`, `hue`, `saturation`, `laplacian`
- **threshold**: Threshold for segment creation (default: 0.5, range: 0.0-1.0, step: 0.05)
- **rotation**: Sorting direction rotation (default: -90, range: -180 to 180, step: 90)
- **multi_pass**: Apply all sorting modes sequentially (default: False)
- **invert_mask**: Invert the effect mask (default: False)
- **grow**: Grow/shrink mask size (default: 4, range: -999 to 999, step: 1)
- **blur**: Blur mask edges (default: 4, range: 0-999, step: 1)
- **seed**: Random seed for reproducibility (default: 42)
- **use_mask**: Enable mask usage (default: True)

### Optional Parameters
- **mask**: Mask to control where sorting is applied (1 = sort, 0 = keep original)

## Outputs
- **image**: Processed image batch
- **mask_out**: Output mask
- **inverted_mask_out**: Inverted output mask

## Sort Modes

### Luminance
Sorts pixels based on their brightness values. Useful for creating smooth gradients and light-based effects.

### Hue
Sorts pixels based on their color hue. Creates interesting color separation effects.

### Saturation
Sorts pixels based on color saturation. Effective for creating vibrant visual effects.

### Laplacian
Sorts pixels based on edge detection using the Laplacian operator. Great for creating edge-based glitch effects.

## Example Usage

### Basic Pixel Sorting
```
sort_mode: luminance
threshold: 0.5
rotation: -90
multi_pass: False
grow: 0
blur: 0
```
Creates a simple pixel sorting effect based on brightness.

### Multi-Pass Color Sorting
```
sort_mode: hue
threshold: 0.3
rotation: 0
multi_pass: True
grow: 2
blur: 2
```
Applies multiple sorting passes with color-based effects.

### Edge-Based Glitch Effect
```
sort_mode: laplacian
threshold: 0.7
rotation: 90
multi_pass: False
grow: 4
blur: 6
```
Creates glitch effects focused on image edges.

### Masked Sorting
```
sort_mode: saturation
threshold: 0.4
use_mask: True
invert_mask: False
grow: 3
blur: 4
```
Applies sorting only in masked areas for precise control.
