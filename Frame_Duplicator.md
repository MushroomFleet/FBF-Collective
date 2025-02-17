# Frame Duplicator

## Overview
The **Frame Duplicator** node (implemented in Frame_Duplicator_Sonification.py) duplicates the first frame of an input image tensor a specified number of times. This is useful for creating repeated sequences or for scenarios where a single frame needs to be propagated multiple times for further processing.

## How It Works
- The node extracts the first frame from the provided image tensor.
- It then duplicates this frame the number of times specified by the **Number_of_frames** parameter.
- The duplicated frames are assembled into a new image tensor which can then be used in subsequent operations.

## Parameters
- **images** (IMAGE):  
  The input image tensor from which the first frame is selected.
  
- **Number_of_frames** (INT):  
  Specifies the number of times the first frame should be duplicated.  
  - **Default:** 1  
  - **Minimum:** 1

## Return Value
- **IMAGE**:  
  A new image tensor consisting of the duplicated frames.
