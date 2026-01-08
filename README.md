# RoPE-CoRoPE

This repository presents a comprehensive exploration of **efficient positional embeddings** in modern attention mechanisms using **OpenAI Triton**. It features two core components: 
- A high-performance **Fused RoPE kernel** integrated into Flash Attention
- An experimental implementation of **Co-RoPE**, which is a context-aware improvement of RoPE.

## Code Structure

```
.
├── assets
├── corope-exp
│   ├── compare_corope.py
│   ├── corope_attn_gqa_pytorch.py
│   └── flash_attn_co_rope_gqa_triton.py
├── fused-rope
│   ├── baseline.py
│   ├── bench_compare.py
│   ├── flash_attn_v2_triton.py
│   ├── fused_rope_attn.py
│   ├── rope_attn_pytorch.py
│   └── verification.py
├── README.md
└── utils.py
```

## Environment

```
PyTorch: 2.9.1+cu128
CUDA Version: 12.8
GPU: NVIDIA H200
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
# Use flash-attn as comparison
pip install packaging ninja psutil
pip install flash-attn --no-build-isolation
pip install transformers
```

## Fused RoPE

### Algorithm

**Rotary Positional Embedding (RoPE)** encodes positional information by rotating the query and key vectors in a high-dimensional space. Given a position $m$ and a vector $\mathbf{x}(\mathbf{q} \text{ or } \mathbf{k})$, the rotation is defined as:

$$f(\mathbf{x}, m) = \begin{pmatrix} x_1, x_2, \cdots, x_d \end{pmatrix} \otimes \begin{pmatrix} \cos m\theta_1, \cos m\theta_1, \cdots, \cos m\theta_{d/2} \end{pmatrix} + \begin{pmatrix} -x_2, x_1, \cdots, -x_{d-1} \end{pmatrix} \otimes \begin{pmatrix} \sin m\theta_1, \sin m\theta_1, \cdots, \sin m\theta_{d/2} \end{pmatrix}$$

> [Note] In code implementation (like `transformers`), we often use half layout instead of interleaved layout, which is more efficient for GPU operations.

### Triton Kernel Optimization

**Core Insight**: Instead of applying RoPE in separate kernels before attention, we fuse the rotation directly into the Flash Attention loop body, eliminating intermediate memory traffic.

**Standard Pipeline (3 stages)**:

```python
# Kernel 1 & 2: Apply RoPE separately
q_rope = apply_rope(q, freqs_cos, freqs_sin)  # Write to HBM
k_rope = apply_rope(k, freqs_cos, freqs_sin)  # Write to HBM

# Kernel 3: Flash Attention
o = flash_attn(q_rope, k_rope, v)  # Read from HBM
```

**Our Fused Implementation (1 kernel)**:

```python
# In _attn_fwd (outer loop):
q1_rot = (q1 * cos_q - q2 * sin_q).to(q1.dtype)  # Compute once per query block
q2_rot = (q2 * cos_q + q1 * sin_q).to(q2.dtype)

# In _attn_fwd_inner (inner loop):
for start_n in tl.range(lo, hi, BLOCK_N):
    # Load K block
    k1 = tl.load(k1_ptrs, mask=mask_k, other=0.0)
    k2 = tl.load(k2_ptrs, mask=mask_k, other=0.0)
    
    # Load rotation frequencies
    cos_k = tl.load(freqs_cos_ptrs, mask=mask_k, other=1.0)
    sin_k = tl.load(freqs_sin_ptrs, mask=mask_k, other=0.0)
    
    # Rotate K in registers
    k1_rot = (k1 * cos_k - k2 * sin_k).to(q1_rot.dtype)
    k2_rot = (k2 * cos_k + k1 * sin_k).to(q2_rot.dtype)
    
    # Immediately compute QK^T
    qk = tl.dot(q1_rot, tl.trans(k1_rot))
    qk += tl.dot(q2_rot, tl.trans(k2_rot))
    # ... continue with softmax and attention ...
```

### Performance Benchmark

- Baseline 1: Transformers RoPE + PyTorch SDPA
- Baseline 2: Transformers RoPE + Flash Attention (Official CUDA)
- Baseline 3: Transformers RoPE + Flash Attention v2 (Triton)
- Ours: Fused RoPE (Triton)


| Configuration (B, H, N, D) | Pass | Baseline 1 (PyTorch SDPA) | Baseline 2 (Official)      | Baseline 3 (Triton v2)      | Fused RoPE (Ours)              | Speedup (vs B3) |
|---------------------------|------|---------------------------|----------------------------|-----------------------------|--------------------------------|-----------------|
| **Small-512** (4, 8, 512, 64) | FWD | 0.119ms (18.10 TFLOPS)   | 0.142ms (15.18 TFLOPS)     | 0.202ms (10.66 TFLOPS)      | **0.043ms (49.91 TFLOPS)**     | **4.68x ↑** |
|                           | BWD  | 0.524ms (10.24 TFLOPS)   | 0.692ms (7.76 TFLOPS)      | 0.757ms (7.09 TFLOPS)       | **0.322ms (16.66 TFLOPS)**     | **2.35x ↑** |
| **Small-1K** (4, 8, 1024, 64) | FWD | 0.168ms (51.19 TFLOPS)   | 0.167ms (51.46 TFLOPS)     | 0.207ms (41.43 TFLOPS)      | **0.050ms (171.74 TFLOPS)**    | **4.15x ↑** |
|                           | BWD  | 0.712ms (30.15 TFLOPS)   | 0.801ms (26.80 TFLOPS)     | 0.747ms (28.76 TFLOPS)      | **0.215ms (99.75 TFLOPS)**     | **3.47x ↑** |
| **Llama7B-2K** (2, 32, 2048, 128) | FWD | 0.807ms (170.38 TFLOPS)  | 0.789ms (174.21 TFLOPS)    | 0.709ms (193.82 TFLOPS)     | **0.381ms (360.47 TFLOPS)**    | **1.86x ↑** |
|                              | BWD | 1.556ms (220.89 TFLOPS)  | 1.423ms (241.44 TFLOPS)    | 2.785ms (123.38 TFLOPS)     | **2.880ms (119.30 TFLOPS)**    | **0.97x ↓** |
| **Llama7B-4K** (2, 32, 4096, 128) | FWD | 1.882ms (292.04 TFLOPS)  | 1.826ms (301.00 TFLOPS)    | 1.594ms (344.94 TFLOPS)     | **1.251ms (439.29 TFLOPS)**    | **1.27x ↑** |
|                              | BWD | 4.057ms (338.77 TFLOPS)  | 3.725ms (369.00 TFLOPS)    | 8.434ms (162.96 TFLOPS)     | **9.474ms (145.06 TFLOPS)**    | **0.89x ↓** |
| **Llama70B-1K** (2, 64, 1024, 128) | FWD | 0.695ms (98.93 TFLOPS)   | 0.683ms (100.55 TFLOPS)    | 0.643ms (106.91 TFLOPS)     | **0.220ms (312.36 TFLOPS)**    | **2.92x ↑** |
|                               | BWD | 1.251ms (137.35 TFLOPS)  | 1.137ms (151.13 TFLOPS)    | 1.958ms (87.73 TFLOPS)      | **1.834ms (93.65 TFLOPS)**     | **1.07x ↑** |
| **Long-64K** (1, 8, 65536, 128) | FWD | 27.062ms (650.07 TFLOPS) | 25.874ms (679.92 TFLOPS)   | 18.045ms (974.89 TFLOPS)    | **34.804ms (505.47 TFLOPS)**   | **0.52x ↓** |
|                               | BWD | 71.039ms (619.10 TFLOPS) | 66.682ms (659.56 TFLOPS)   | 204.990ms (214.55 TFLOPS)   | **254.385ms (172.89 TFLOPS)**  | **0.81x ↓** |

**Conclusion**
- **IO-Bound Regime (N ≤ 1024)**: Our Fused RoPE achieves a significant 2.9x – 4.6x speedup. In this regime, the kernel is limited by HBM bandwidth and kernel launch latency. By fusing the rotary transformation into the SRAM-resident tiles of Flash Attention, we eliminate the redundant R/W cycles of $Q_{rope}$ and $K_{rope}$ to global memory.

- **Compute-Bound Transition (N ≥ 4096)**: As sequence length increases, the attention mechanism transitions from being memory-bound to compute-bound. 

- **Register Pressure and Long-Context (64K)**: In ultra-long sequences ($N=64K$), our fused implementation exhibits a performance regression (0.52x vs. B3).
    - **Root Cause**: The addition of RoPE logic increases the Register Pressure per thread. To accommodate the rotary state, the Triton compiler may reduce the Occupancy or trigger Register Spilling, which is particularly costly in the massive loops of long-context attention.

- **Backward Pass Asymmetry**: The speedup in BWD is consistently lower than FWD (max 3.47x). This is expected as the BWD pass of Flash Attention is inherently more compute-intensive (calculating gradients for $Q, K, V$), making the relative savings from memory fusion less impactful.

## Co-RoPE (Experimental)
Co-RoPE is a context-aware improvement of RoPE.

### Preliminaries

- RoPE: https://arxiv.org/abs/2104.09864


- CoPE: https://arxiv.org/abs/2405.18719

![CoRoPE](./assets/corope.png)

### Our Methodology

Co-RoPE extends RoPE by introducing context-aware mileage computation. The key mathematical formulation is as follows:

**Co-RoPE** 

![CoRoPE1](./assets/corope-1.png)

For each query position $i$ and key position $j$, we compute the contextual mileage by summing up the sigmoid of the dot product between the query head and the key head:

$$z_{ij} = \sigma(\mathbf{q}_i \cdot \mathbf{k}_j \cdot s)$$

and the accumulated mileage is:

$$a_{ij} = \sum_{k=0}^{j} z_{ik}$$

where $\sigma$ is the sigmoid function, $s$ is the scaling factor, and $\mathbf{q}$ represents the query head. So the relative displacement between positions $i$ and $j$ is:

$$\Delta a_{ij} = a_{ii} - a_{ij}$$

This captures the contextual distance between query position $i$ and key position $j$.

The phase angle is computed as:

$$\phi_{ijd} = \Delta a_{ij} \cdot \omega_d$$

where $\omega_d = \frac{1}{\theta^{2d/D}}$ is the inverse frequency for dimension $d$, and $\theta$ is the RoPE base (typically 10000).


**Efficient Rotation**

To efficiently apply the rotation, we decompose the query and key vectors into two halves:

$$\mathbf{q} = [\mathbf{q}_1, \mathbf{q}_2], \quad \mathbf{k} = [\mathbf{k}_1, \mathbf{k}_2]$$

The energy fields are computed as:

$$E_A = \mathbf{q}_1 \mathbf{k}_1^T + \mathbf{q}_2 \mathbf{k}_2^T$$

$$E_B = \mathbf{q}_2 \mathbf{k}_1^T - \mathbf{q}_1 \mathbf{k}_2^T$$

The final attention score combines the energy fields with phase modulation:

$$\text{score}_{ij} = \sum_{d=0}^{D/2-1} \left[ E_A^{ijd} \cos(\phi_{ijd}) - E_B^{ijd} \sin(\phi_{ijd}) \right] \cdot s$$

**CoRoPE-GQA**

We use GQA to implement Co-RoPE to reduce the computational cost.

![Methodology](./assets/method.png)

We use a leader head to compute the contextual mileage and the accumulated mileage, and then broadcast the mileage to all the heads in the group.

### Bottleneck Analysis

This is a mathematically elegant but computationally expensive implementation. The context-aware mileage computation has $O(N^2)$ complexity, which becomes the primary bottleneck for long sequences. This is a runnable version but has not been fully optimized, and fully optimizing it also isn't worth the cost.

Even with an efficient rotation implementation, the 3D phase(because each D/2 feature is non-linear and has to be computed for each position) angle tensor introduces significant memory and computational overhead, posing challenges for Triton compiler optimization. 

The Trigonometric Disaster in Inner Loops is obvious. Pushing trigonometric computations (sin/cos) into the most nested loop of a Triton kernel is, frankly, a performance nightmare.

