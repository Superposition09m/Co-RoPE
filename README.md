# Co-RoPE

Environment:
```
PyTorch: 2.9.1+cu128
CUDA Version: 12.8
GPU: 4x NVIDIA H200
Triton: 3.5.1
NumPy: 2.3.5
```
Install Env:
```bash
conda create -n corope python==3.12
conda activate corope
pip install torch triton
#use flash-attn as comparison
pip install packaging ninja psutil
pip install flash-attn --no-build-isolation
```
