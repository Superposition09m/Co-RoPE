# Co-RoPE: Contextual Rotary Positional Encoding

Co-RoPE is a context-aware improvement of RoPE. In this project, we implement the Co-RoPE in Triton and compare the performance with the PyTorch implementation.

## Preliminaries

- RoPE: https://arxiv.org/abs/2104.09864


- CoPE: https://arxiv.org/abs/2405.18719

![CoRoPE](./assets/CoRoPE.png)

### Our Methodology

![Methodology](./assets/method.png)

## Mathematical Formulation

### Core Formula

Co-RoPE extends RoPE by introducing context-aware mileage (里程) computation. The key mathematical formulation is as follows:

#### 1. Mileage Computation (里程计算)

For each query position $i$ and key position $j$, we compute the contextual mileage:

$$z_{ij} = \sigma(\mathbf{q}_i^{\text{leader}} \cdot \mathbf{k}_j \cdot s)$$

$$a_{ij} = \sum_{k=0}^{j} z_{ik}$$

where $\sigma$ is the sigmoid function, $s$ is the scaling factor, and $\mathbf{q}^{\text{leader}}$ represents the leader query head (used for GQA efficiency).

#### 2. Mileage Difference (里程差)

The relative displacement between positions $i$ and $j$ is:

$$\Delta a_{ij} = a_{ii} - a_{ij}$$

This captures the contextual distance between query position $i$ and key position $j$.

#### 3. Phase Modulation (相位调制)

The phase angle is computed as:

$$\phi_{ijd} = \Delta a_{ij} \cdot \omega_d$$

where $\omega_d = \frac{1}{\theta^{2d/D}}$ is the inverse frequency for dimension $d$, and $\theta$ is the RoPE base (typically 10000).

#### 4. Energy Field Decomposition (能量场分解)

To efficiently apply the rotation, we decompose the query and key vectors into two halves:

$$\mathbf{q} = [\mathbf{q}_1, \mathbf{q}_2], \quad \mathbf{k} = [\mathbf{k}_1, \mathbf{k}_2]$$

The energy fields are computed as:

$$E_A = \mathbf{q}_1 \mathbf{k}_1^T + \mathbf{q}_2 \mathbf{k}_2^T$$

$$E_B = \mathbf{q}_2 \mathbf{k}_1^T - \mathbf{q}_1 \mathbf{k}_2^T$$

#### 5. Attention Score (注意力分数)

The final attention score combines the energy fields with phase modulation:

$$\text{score}_{ij} = \sum_{d=0}^{D/2-1} \left[ E_A^{ijd} \cos(\phi_{ijd}) - E_B^{ijd} \sin(\phi_{ijd}) \right] \cdot s$$

### Energy Simplification

The key optimization in Co-RoPE is the **energy field decomposition**:

- **Traditional approach**: Directly compute rotated Q and K, then perform matrix multiplication
- **Co-RoPE approach**: Pre-compute energy fields $E_A$ and $E_B$ from unrotated Q and K, then apply rotation via simple trigonometric operations

This simplification:
1. **Reduces computation**: Energy fields are computed once and reused across different phase angles
2. **Improves numerical stability**: Separates the dot product computation from rotation
3. **Enables efficient implementation**: Allows for better memory access patterns in Triton kernels

The mathematical equivalence can be verified by expanding the rotated dot product:

$$\text{Re}(\mathbf{q}_\text{rot} \cdot \mathbf{k}_\text{rot}^*) = E_A \cos(\phi) - E_B \sin(\phi)$$

where $\mathbf{q}_\text{rot}$ and $\mathbf{k}_\text{rot}$ are the RoPE-rotated vectors, and $*$ denotes complex conjugation.

- we use GQA to implement the Co-RoPE.




## Environment:
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
#use flash-attn as comparison
pip install packaging ninja psutil
pip install flash-attn --no-build-isolation
```