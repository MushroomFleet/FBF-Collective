# FrameBufferGlitch Node

## Description
The FrameBufferGlitch node creates frame freezing and audio glitch effects by manipulating video frames and their corresponding audio segments. It includes options for frame duration control and random glitch generation.

## Input Parameters

### Required Parameters
- **images**: Input image batch
- **audio**: Input audio signal
- **freeze_frame_duration**: Duration to freeze frames (default: 1, min: 1)
- **number_of_random_glitches**: Total number of random glitch effects (default: 1, min: 1)
- **pad_left**: Audio padding before glitch (default: 0, min: 0)
- **pad_right**: Audio padding after glitch (default: 0, min: 0)
- **pad_mode**: Type of audio padding
  - Options: `constant`, `reflect`, `replicate`, `circular`

## Outputs
- **images**: Processed image frames with glitch effects
- **audio**: Processed audio with synchronized glitch effects

## Features

### Frame Freezing
- Randomly freezes frames for specified durations
- Maintains synchronization with audio
- Creates stutter-like visual effects

### Audio Padding
- Multiple padding modes for audio manipulation
- Helps prevent audio artifacts at boundaries
- Synchronizes with visual glitches

### Random Glitch Generation
- Generates specified number of random glitches
- Distributes glitches throughout the sequence
- Maintains audio-visual synchronization

## Example Usage

### Basic Glitch Effect
```
freeze_frame_duration: 3
number_of_random_glitches: 2
pad_mode: constant
```
Creates simple frame freezes with basic audio padding.

### Extended Glitch with Audio Reflection
```
freeze_frame_duration: 5
number_of_random_glitches: 4
pad_left: 1000
pad_right: 1000
pad_mode: reflect
```
Produces longer glitch effects with reflected audio padding.

### Rapid Glitch Sequence
```
freeze_frame_duration: 1
number_of_random_glitches: 8
pad_mode: circular
```
Creates rapid succession of short glitches with circular audio padding.
