# Co-RoPE
# Prerequisites
## Methodology

## Environment:
```
PyTorch: 2.9.1+cu128
CUDA Version: 12.8
GPU: 4x NVIDIA H200
Triton: 3.5.1
NumPy: 2.3.5
einops: 0.8.1
flash-attn: 2.8.3
transformers: 4.57.3
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

# Coding Plan

## Abandoned `co_rope_torch.py`

## Adopted `flash_attn_v2_triton.py` and `attn_pytorch.py`

## Next Step: Implement RoPE to both torch and triton

rope in torch:

seq_len=512 + theta=500000 ERR 4.5e-05

seems done.

now start to implement rope in triton.