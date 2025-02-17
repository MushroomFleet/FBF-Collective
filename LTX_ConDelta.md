# LTX ConDelta Nodes Documentation

## Overview
The **LTX ConDelta** module provides a suite of custom nodes for manipulating video conditioning deltas within ComfyUI. These nodes enable advanced video processing operations such as saving, loading, applying, subtracting, and adding conditioning deltas using the safetensors format. They are designed to work with conditioning data (typically in tensor form) and incorporate temporal and scaling adjustments to enhance video conditioning workflows.

---

## Nodes in this Module

### 1. Save LTX Video ConDelta
**Purpose:**  
Saves a video conditioning delta (and its pooled output) as a safetensors file.

**Function:** `save`

**Parameters:**
- **conditioning_delta** (CONDITIONING):  
  The input conditioning delta. This is expected to be a list where the first element contains the tensor (and optionally the pooled output) that will be saved.
  
- **file_name** (STRING):  
  The file name for the output. If the file name does not end with `.safetensors`, the extension will be automatically appended.
  
- **overwrite** (BOOLEAN):  
  If set to `True`, an existing file with the same name will be overwritten. Default is `False`.

**Return:**  
None

**Notes:**  
- The node ensures the target directory exists.
- It handles tensor cloning safely to avoid issues with TeaCache modifications.
- It saves both the main conditioning tensor and the pooled output (or creates a zero tensor if the pooled output is absent).

---

### 2. Load LTX Video ConDelta
**Purpose:**  
Loads a video conditioning delta from a selected safetensors file.

**Function:** `load`

**Parameters:**
- **condelta** (List of filenames):  
  A selection of file names (retrieved from the LTXConDelta directory) from which the conditioning delta will be loaded.

**Return:**  
- **CONDITIONING**:  
  The loaded conditioning delta, structured as a list containing the main tensor and associated pooled output.

**Notes:**  
- The node safely clones the loaded tensor and handles the absence of pooled output by creating a zero tensor if necessary.
- It provides debug information regarding the shape, data type, and device of the loaded tensor.

---

### 3. Apply LTX Video ConDelta
**Purpose:**  
Applies a saved conditioning delta to a given base conditioning using temporal scaling and smoothing techniques.

**Function:** `apply`

**Parameters:**
- **conditioning** (CONDITIONING):  
  The base conditioning data to which the delta will be applied.

- **condelta** (List of filenames):  
  Selects a conditioning delta file from the LTXConDelta directory.

- **strength** (FLOAT):  
  The scalar value determining the influence or weight of the delta when applied. This adjusts how strongly the delta modifies the base conditioning.

- **temporal_scale** (FLOAT, default: 0.8, min: 0.0, max: 10.0):  
  Specifies the degree of temporal scaling to apply, affecting how temporal dimensions are adjusted.

- **frame_window** (INT, default: 3, min: 1, max: 16):  
  Defines the temporal window size used for averaging (temporal smoothing). A larger window smoothes over more frames.

- **scale_type** (Choice: "median", "mean", "max", default: "median"):  
  Determines the statistical method used to compute scaling ratios, which in turn influences the enhancement of the delta's features.

**Return:**  
- **CONDITIONING**:  
  The output conditioning after applying the delta adjustments.

**Notes:**  
- The node performs tensor cloning, padding, cropping, and temporal smoothing to align dimensions.
- It calculates scaling ratios based on the chosen `scale_type` and applies a progressive temporal factor.
- Feature enhancement is applied by emphasizing significant tensor elements.

---

### 4. LTX Video Conditioning Subtract (Create ConDelta)
**Purpose:**  
Creates a conditioning delta by subtracting one conditioning from another, which can be used to derive differences between conditioning inputs.

**Function:** `subtract`

**Parameters:**
- **conditioning_a** (CONDITIONING):  
  The base conditioning from which the delta will be computed.

- **conditioning_b** (CONDITIONING):  
  The conditioning to subtract from `conditioning_a`.

- **temporal_weight** (FLOAT, default: 1.0, min: 0.0, max: 10.0):  
  Applies a weighting factor to the temporal component beyond the first frame.

**Return:**  
- **CONDITIONING**:  
  The resulting conditioning delta computed by subtracting `conditioning_b` from `conditioning_a`.

**Notes:**  
- The node aligns temporal dimensions by padding or cropping as necessary.
- It handles pooled outputs appropriately to ensure consistency in the resulting delta.

---

### 5. Add LTX Video ConDelta
**Purpose:**  
Adds a conditioning delta to a base conditioning with adjustable strength and temporal/smoothing modifications. This node integrates a delta into the base conditioning while allowing precise control over its influence.

**Function:** `addDelta`

**Parameters:**
- **conditioning_base** (CONDITIONING):  
  The base conditioning data which will receive the delta addition.

- **conditioning_delta** (CONDITIONING):  
  The delta to be applied to the base conditioning.

- **conditioning_delta_strength** (FLOAT, default: 1.0, min: -100.0, max: 100.0):  
  Determines the strength (or weight) of the delta when added. Negative values can subtract influence.

- **temporal_scale** (FLOAT, default: 0.8, min: 0.0, max: 10.0):  
  Controls the scaling of temporal dimensions during application.

- **frame_window** (INT, default: 3, min: 1, max: 16):  
  Sets the window size for temporal smoothing operations.

- **scale_type** (Choice: "median", "mean", "max", default: "median"):  
  Selects the statistical method for calculating scaling ratios to modulate the delta.

**Return:**  
- **CONDITIONING**:  
  The modified conditioning after the delta has been incorporated.

**Notes:**  
- The node adjusts the delta by computing a base ratio and applying temporal factor adjustments.
- It ensures that dimensions between the base and the delta match before performing the addition.
- Pooled outputs are also modified accordingly to reflect changes in the conditions.

---

## Conclusion
The nodes within the **LTX ConDelta** module offer a comprehensive set of tools to manage video conditioning deltas in ComfyUI. Whether saving, loading, applying, subtracting, or adding deltas, these nodes provide flexibility and robustness for advanced video processing tasks.

Each node is designed with parameters that enable fine-tuning of the operation, ensuring that users can adapt these nodes to a variety of workflows and requirements in video conditioning and sonification.

Happy conditioning!
