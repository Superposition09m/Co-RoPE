# Co-RoPE: Contextual Rotary Positional Encoding

Co-RoPE is a context-aware improvement of RoPE. In this project, we implement the Co-RoPE in Triton and compare the performance with the PyTorch implementation.

## Preliminaries

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

To efficiently apply the rotation, we decompose the query and key vectors into two halves:

$$\mathbf{q} = [\mathbf{q}_1, \mathbf{q}_2], \quad \mathbf{k} = [\mathbf{k}_1, \mathbf{k}_2]$$

The energy fields are computed as:

$$E_A = \mathbf{q}_1 \mathbf{k}_1^T + \mathbf{q}_2 \mathbf{k}_2^T$$

$$E_B = \mathbf{q}_2 \mathbf{k}_1^T - \mathbf{q}_1 \mathbf{k}_2^T$$

The final attention score combines the energy fields with phase modulation:

$$\text{score}_{ij} = \sum_{d=0}^{D/2-1} \left[ E_A^{ijd} \cos(\phi_{ijd}) - E_B^{ijd} \sin(\phi_{ijd}) \right] \cdot s$$

This simplifies...

**CoRoPE-GQA**

we use GQA to implement the Co-RoPE, to reduce the computational cost of the Co-RoPE.

![Methodology](./assets/method.png)


We use a leader head to compute the contextual mileage and the accumulated mileage, and then broadcast the mileage to all the heads in the group.





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