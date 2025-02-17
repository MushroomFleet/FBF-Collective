# MixerSonif

## Overview
The **MixerSonif** node performs audio mixing by blending two audio inputs into a single output. It ensures that both inputs are resampled to a common target sample rate (44,100 Hz) and converted to stereo. The node also provides an option to time-stretch the longer audio to match the duration of the shorter one, or alternatively, pad the shorter audio so that both have the same length before mixing.

## How It Works
1. **Resampling:**  
   Both audio inputs are checked against the target sample rate (44,100 Hz). If an input's sample rate differs, it is resampled using Librosa to maintain consistency.

2. **Stereo Conversion:**  
   The node confirms that both audio waveforms are in stereo. If an audio is mono or has a single channel, it is duplicated to create a stereo pair.

3. **Duration Alignment:**  
   - If **fit_longer** is set to True, the node detects which audio is longer and applies time stretching (using Librosa’s time stretching functionality) to the longer audio so that its length matches the shorter one.
   - If **fit_longer** is False, the node pads the shorter audio with zeros to match the length of the longer audio.  
   This ensures that both audios have the same number of samples before mixing.

4. **Mixing:**  
   The node computes a mix ratio based on the **mix** parameter (ranging from 0.0 to 100.0). A value of 0% outputs exclusively Audio_1, while 100% outputs exclusively Audio_2. The two audios are blended channel-wise using a weighted average:
   - `Output = (1.0 - mix_ratio) * Audio_1 + mix_ratio * Audio_2`
   
5. **Output:**  
   The mixed audio is returned as a stereo PyTorch tensor along with the target sample rate (44,100 Hz).

## Parameters
- **Audio_1** (AUDIO):  
  The first audio input. This parameter is mandatory and ensures the node receives a valid audio waveform.

- **Audio_2** (AUDIO):  
  The second audio input. This parameter is also mandatory.

- **mix** (FLOAT):  
  Specifies the mixing percentage between Audio_1 and Audio_2.  
  - **Default:** 50.0  
  - **Range:** 0.0 to 100.0  
  - **Step:** 0.5  
  A value of 0.0 means the output is entirely Audio_1, while 100.0 means it is entirely Audio_2.

- **fit_longer** (BOOLEAN):  
  Determines the method for duration alignment if the two audio inputs have different lengths.  
  - **True:** The longer audio is time-stretched to match the shorter audio.  
  - **False:** The shorter audio is padded with zeros to match the length of the longer audio.

## Return Value
- **Audio OUT** (AUDIO):  
  The resulting mixed audio, returned as a dictionary containing:
  - **waveform:** A PyTorch tensor representing the mixed stereo audio.
  - **sample_rate:** The sample rate of the output audio (44,100 Hz).

## Dependencies
- **NumPy:** For numerical operations and handling waveform arrays.
- **PyTorch:** For tensor manipulation and storage.
- **Librosa:** For audio processing operations such as resampling and time stretching.

## Example Usage
1. Connect two audio nodes to **Audio_1** and **Audio_2**.
2. Set the **mix** parameter to the desired blend ratio (e.g., 70.0 for a 70% weight on Audio_2).
3. Toggle **fit_longer** according to whether you want the longer audio to be time-stretched (True) or padded (False) to match the shorter audio.
4. The node outputs the mixed audio as a stereo tensor with a sample rate of 44,100 Hz, ready for further processing or playback.

Happy mixing!
