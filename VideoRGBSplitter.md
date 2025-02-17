# Video RGB Splitter

## Overview
The **Video RGB Splitter** node splits an input video frame into its constituent red, green, and blue channels. By separating these channels, users can independently manipulate or analyze each color component for creative or processing purposes.

## How It Works
- The node accepts a video frame as an input image tensor.
- It decomposes the frame into three separate channels corresponding to red, green, and blue.
- Each channel can then be processed individually, allowing for targeted effects or further manipulation.

## Parameters
- **images** (IMAGE):  
  The input video frame tensor that will be split into separate color channels.

## Return Value
- **TUPLE (IMAGE, IMAGE, IMAGE)**:  
  A tuple containing three images representing the red, green, and blue channels respectively.
