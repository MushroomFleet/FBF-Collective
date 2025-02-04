# FrameShuffle Node

## Description
The FrameShuffle node provides advanced frame and audio shuffling capabilities with various modes for creative video manipulation. It includes options for both visual and audio shuffling with customizable parameters.

## Input Parameters

### Required Parameters
- **images**: Input image batch
- **audio**: Input audio signal
- **shuffle_range**: Number of frames to shuffle together (default: 2, min: 1)
- **audio_shuffle_mode**: Audio manipulation mode
  - Options: `Pass Through`, `Random`, `Sequential`, `Reverse`
- **audio_segment_duration**: Duration of audio segments in frames (default: 0, min: 0)
- **pad_left**: Audio padding before shuffle (default: 0, min: 0)
- **pad_right**: Audio padding after shuffle (default: 0, min: 0)
- **pad_mode**: Type of audio padding
  - Options: `constant`, `reflect`, `replicate`, `circular`

## Outputs
- **images**: Shuffled image frames
- **audio**: Processed audio with synchronized shuffling

## Audio Shuffle Modes

### Pass Through
- Maintains original audio without shuffling
- Useful when only visual shuffling is desired

### Random
- Randomly shuffles audio segments
- 50% chance to reverse individual segments
- Creates unpredictable audio patterns

### Sequential
- Reverses audio segments within shuffle groups
- Maintains some temporal relationship
- Creates structured audio patterns

### Reverse
- Reverses each audio segment individually
- Creates consistent reverse effects
- Maintains segment boundaries

## Example Usage

### Basic Frame Shuffling
```
shuffle_range: 2
audio_shuffle_mode: Pass Through
audio_segment_duration: 0
```
Simple frame shuffling without audio manipulation.

### Complex Audio-Visual Shuffle
```
shuffle_range: 4
audio_shuffle_mode: Random
audio_segment_duration: 5
pad_mode: reflect
pad_left: 1000
pad_right: 1000
```
Creates complex shuffling with randomized audio segments.

### Reverse Effect
```
shuffle_range: 3
audio_shuffle_mode: Reverse
audio_segment_duration: 3
pad_mode: replicate
```
Produces reversed segments while maintaining synchronization.
