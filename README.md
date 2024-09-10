# Simple Implementation of Time Series Foundation Model in PyTorch

This repository provides an implementation of the Time Series Foundation Model (TimesFM) using PyTorch, based on the original model available [here](https://github.com/google-research/timesfm).

The purpose of this repository is to offer a PyTorch variant that can load checkpoints from the JAX version and enable fine-tuning. Given the scope of my personal needs and available resources, this implementation includes only the essential components required to operate the model effectively.

## Update 09/10/24
Add dockerfiles in `timesfm_torch/docker` for building container that can run TimesFM JAX version. If you have trouble converting the JAX weights to PyTorch weights, try converting them in docker container. See [timesfm_torch/docker/README.md](https://github.com/tangh18/timesfm_torch/tree/main/timesfm_torch/docker/README.md) for detail.

## Features

### Provided
1. Capability to load JAX checkpoints into a PyTorch model with the same architecture.
2. Core components constituting the TimesFM model.
3. Output equivalence with the JAX version under specific conditions.(Numerical error exists)

### Not Provided
1. Padding handling (assumes no padding during inference).
2. Support for variable context and horizon lengths (easy to add).
3. Different frequency embeddings (same above).

### Differences from the Original Implementation
1. The mean and standard deviation are computed across the entire time series rather than just the first patch.

## Usage

### Installation
Install the package using pip:
```bash
pip install -e .
```

### Convert JAX Checkpoint to PyTorch Checkpoint
Navigate to the utility directory and run the conversion script:
```bash
cd timesfm_torch/timesfm_torch/utils
python convert_ckpt.py
```
This process will generate PyTorch checkpoints in `timesfm_torch/timesfm_torch/ckpt`.

### Run the PyTorch Model
By default, the model loads the checkpoint during initialization. The `forward()` method replicates the functionality of the `PatchedTimeSeriesDecoder.__call__()` method in the JAX version, maintaining the same input and output shapes. Note that the `forward()` method does not handle padding and only requires the input time series.

#### Example Usage
```python
from timesfm_torch.model.timesfm import TimesFm
input_ts = torch.rand((32, 512)).to('cuda') # Input shape: (batch_size, context_len)
timesfm = TimesFm(context_len=512)
timesfm.load_from_checkpoint(ckpt_dir=f"timesfm_torch/timesfm_torch/ckpt")
output_ts = timesfm(input_ts) # Output shape: (batch_size, patch_num, horizon_len, num_outputs)
```
