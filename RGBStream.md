# RGB Stream Node

## Overview
The RGB Stream Node is a custom node designed to split an input image—or a batch of images—into its red, green, and blue channels. For each image, it generates three color-tinted outputs, each emphasizing one of the primary colors.

## Functionality
- **Input Conversion:**  
  The input, provided as a Torch tensor, is converted to a NumPy array for processing.

- **Channel Splitting:**  
  Each image in the batch is processed using OpenCV's `cv2.split` to separate the red, green, and blue components.

- **Color Tinting:**  
  For each image, three new images are created:
  - **Red Tinted:** The red channel is kept while the green and blue channels are set to zero.
  - **Green Tinted:** The green channel is kept while the red and blue channels are set to zero.
  - **Blue Tinted:** The blue channel is kept while the red and green channels are set to zero.

- **Output Assembly:**  
  The tinted images are collected and converted back into a Torch tensor. The node returns a tuple with a single element, named `RGB_Stream_Output`, containing the processed images. For an input batch with N images, the output tensor contains 3×N images.

## Parameters
- **images (IMAGE):**  
  - **Description:** A batch of input images provided as a Torch tensor.  
  - **Requirements:** Must be appropriately normalized and located on the desired device (CPU or GPU) before processing.

## Return Types
- **RGB_Stream_Output (IMAGE):**  
  - **Description:** The processed output containing the red, green, and blue tinted images.  
  - **Structure:** A tuple containing a single Torch tensor where each set of three images corresponds to the separated and tinted channels of each input image.

## Integration and Usage
- **Workflow:**  
  Integrate this node into your ComfyUI workflow to visually analyze individual RGB channels from input images.
  
- **Category:**  
  This node is categorized under "Sonification", which integrates visual processing into audio-visual or multimedia pipelines.

- **Example Code:**
  ```python
  # Example usage within a ComfyUI workflow:
  # Assume 'input_tensor' is a Torch tensor containing a batch of images.
  from your_module import RGBStream

  node = RGBStream()
  output_tuple = node.create_rgb_stream(input_tensor)
  # output_tuple[0] now holds the tinted images tensor.
  ```

## Dependencies
- **NumPy:** For numerical operations.
- **Torch:** For tensor operations and device management.
- **OpenCV (cv2):** For image processing and channel splitting.

## Additional Notes
- **Device Consistency:**  
  The node ensures that the output tensor is on the same device as the input.
- **Batch Processing:**  
  If processing a batch of images, remember that the output will contain three times the number of images (one for each color channel).

This README provides a comprehensive guide on the functionality, parameters, and usage of the RGB Stream Node.
