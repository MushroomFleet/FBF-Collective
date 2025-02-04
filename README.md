# ComfyUI Community Nodes Collection

A collection of custom nodes for ComfyUI focusing on audio-visual manipulation, effects, and creative transformations.

## Installation

1. Clone this repository into your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/MushroomFleet/FBF-Collective
```

2. Install the required dependencies:
```bash
cd communityNodes
pip install -r requirements.txt
```
(portable:)
```bash
cd communityNodes
install-requirements.bat
```

3. Restart ComfyUI to load the new nodes.

## Available Nodes

### Audio Processing
- [AudioScratchV2](AudioScratchV2.md) - Advanced audio manipulation with various envelope types and stretching modes, by Sonification
- [AudioToFrames](AudioToFrames.md) - Convert audio signals into visual frames and spectrograms, by Sonification

### Visual Effects
- [DjzDatamoshV8Enhanced](DjzDatamoshV8Enhanced.md) - Advanced pixel sorting and datamoshing effects, by Enjoykaos
- [Dynamic_Spectrogram](Dynamic_Spectrogram.md) - Create dynamic spectrogram visualizations with extensive customization, by Sonification
- [FrameBufferGlitch](FrameBufferGlitch.md) - Frame freezing and audio glitch effects, by Sonification
- [FrameShuffle](FrameShuffle.md) - Frame and audio shuffling with various modes, by Sonification
- [Signal2Frames](Signal2Frames.md) - Convert audio signals into microscope-like surface visualizations, by Sonification
- [ToadVideoFrameManipulator](ToadVideoFrameManipulator.md) - Multiple video frame manipulation effects, from Toad

## Features

### Audio Processing
- Multiple envelope types for audio manipulation
- Time-stretching and pitch preservation
- Audio-reactive effects
- Spectrogram generation
- Audio-to-visual conversion

### Visual Effects
- Pixel sorting with multiple modes
- Frame manipulation and glitching
- Audio-synchronized effects
- Custom color mapping
- Advanced visualization options

### General Features
- Extensive parameter customization
- Real-time processing capabilities
- Audio-visual synchronization
- Multiple output formats

## Dependencies

The following Python packages are required:
- numpy
- torch
- librosa
- matplotlib
- PIL
- scipy
- cv2
- tqdm

These dependencies can be installed using the provided requirements.txt file.

## Usage

1. After installation, launch ComfyUI
2. The new nodes will appear in the node browser under their respective categories
3. Drag nodes onto the workspace to use them
4. Connect nodes according to your workflow requirements
5. Configure node parameters as needed

For detailed information about each node's parameters and usage examples, please refer to their individual documentation files linked above.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.

## Support

If you encounter any issues or need help, please:
1. Check the individual node documentation
2. Look for similar issues in the issue tracker
3. Create a new issue with a detailed description of your problem
