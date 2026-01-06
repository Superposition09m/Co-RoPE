# Co-RoPE
## Methodology

Collaborative Rotary Positional Embedding (Co-RoPE) is a content-aware dynamic positional encoding mechanism. Unlike standard RoPE which uses fixed token indices to represent relative distances, Co-RoPE determines the distance between tokens based on their mutual affinity, allowing the model to dynamically "compress" or "expand" the positional space.

### 1. Leader-Driven GQA Mechanism
To maintain computational efficiency, especially in Grouped Query Attention (GQA) settings, Co-RoPE introduces a **Leader-Driven** architecture:
- **Leader Selection**: For each group of query heads (sharing the same KV heads), the first head is designated as the **Leader**.
- **Shared Odometry**: The Leader head computes a shared "mileage" (odometry) distribution for the entire group. This reduces the overhead of dynamic position calculation from $O(H_q)$ to $O(H_{kv})$.

### 2. Dynamic Mileage (Odometry) Calculation
The "mileage" $a_{i,j}$ represents the cumulative affinity a query at position $i$ has encountered when scanning keys from $0$ to $j$:
1. **Affinity Score**: $z_{i,j} = \sigma(\tau \cdot \langle q_i, k_j \rangle)$, where $\sigma$ is the sigmoid function and $\rho$ is the softmax scaling factor (usually $1/\sqrt{d}$).
2. **Cumulative Mileage**: $a_{i,j} = \sum_{k=0}^{j} z_{i,k}$.
3. **Self-Mileage**: The total "distance" traveled by query $i$ up to its own position is $a_{i,i}$.

### 3. Co-RoPE Phase Shift
The relative distance between query $i$ and key $j$ is defined as the mileage difference:
$$\Delta a_{i,j} = a_{i,i} - a_{i,j}$$
This dynamic distance replaces the static integer distance $(i - j)$ in the RoPE formulation. The attention score is computed as:
$$\text{Score}(i, j) = \langle R(\Delta a_{i,j}) q_i, k_j \rangle$$
where $R(\theta)$ is the rotary matrix. In practice, this is equivalent to:
$$\text{Score}(i, j) = \langle R(a_{i,i}) q_i, R(a_{i,j}) k_j \rangle$$
This literal interpretation allows each token to perceive its neighbors at distances proportional to their semantic relevance.

### 4. Energy Field Decomposition (EA-EB Optimization)
To avoid the $O(N^2 \cdot D)$ overhead of explicitly materializing rotated tensors or high-dimensional rotary matrices, we use the **Energy Field Decomposition**:
The rotary dot product can be expanded using trigonometric identities:
$$\text{Score}(i, j) = \sum_{d=0}^{D/2-1} \left[ (q_{2d}k_{2d} + q_{2d+1}k_{2d+1}) \cos(\Delta a_{i,j} \omega_d) - (q_{2d+1}k_{2d} - q_{2d}k_{2d+1}) \sin(\Delta a_{i,j} \omega_d) \right]$$

We define two "Energy Fields" $E_A$ and $E_B$:
- **Real Energy ($E_A$)**: $q_{2d}k_{2d} + q_{2d+1}k_{2d+1}$ (The standard dot product components)
- **Imaginary Energy ($E_B$)**: $q_{2d+1}k_{2d} - q_{2d}k_{2d+1}$ (The rotational interaction components)

Final computation:
$$\text{Score}(i, j) = \sum \left( E_A \circ \cos(\phi) - E_B \circ \sin(\phi) \right)$$
This decomposition is critical for the Triton implementation, as it allows us to compute the attention score in a single pass without redundant rotations, significantly improving memory throughput and compute utilization.

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