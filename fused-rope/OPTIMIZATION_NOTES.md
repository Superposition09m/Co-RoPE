# Fused RoPE 长序列性能优化分析

## 🔴 当前问题

根据 benchmark 结果，Fused RoPE 在长序列时性能严重退化：

| 序列长度 | Forward Speedup vs B3 | Backward Speedup vs B3 | 状态 |
|---------|----------------------|------------------------|------|
| 512-4K  | **1.2-4.9x ↑** | **1.1-2.1x ↑** | ✅ 优秀 |
| 8K-16K  | **0.8-0.9x ↓** | **0.7-0.8x ↓** | ⚠️ 退化 |
| 32K-128K | **0.47-0.57x ↓** | **0.8x ↓** | ❌ 严重退化 |

**结论**：在 ≥8K 序列长度时，Fused RoPE 反而比非融合版本慢。

---

## 🔍 根本原因

### 1. **Register Pressure（寄存器压力）**

Fused RoPE kernel 需要存储：
```python
# Baseline 3 (无RoPE融合):
q, k1_rot, k2_rot, v  # 加载后的数据

# Fused RoPE (有RoPE融合):
q1, q2, q1_rot, q2_rot  # Q 的原始和旋转版本
k1, k2, k1_rot, k2_rot  # K 的原始和旋转版本
cos_k, sin_k            # 频率数据
v                       # V
```

**增加的寄存器使用：**
- 额外的 `q1, q2, k1, k2, cos, sin` 变量
- 估计增加 **20-40% 寄存器**

**后果：**
- SM 能同时运行的 warp 数量减少 → **Occupancy 下降**
- 无法隐藏内存延迟 → **性能下降**

### 2. **L2 Cache Miss**

长序列的数据量：
```
32K seq: 4 * 1 * 32 * 32768 * 128 * 2B = 1024 MB > 60 MB L2
64K seq: 2048 MB >> 60 MB L2
```

当数据无法放入 L2 cache 时，频繁访问 HBM 导致 bandwidth 成为瓶颈。

### 3. **Tile 配置不适合长序列**

当前 autotune 配置：
```python
for BM in [64, 128]
for BN in [32, 64, 128]
```

长序列时，应该：
- **减小 BLOCK_M/BLOCK_N** 以降低寄存器使用
- **增加 pipeline stages** 以更好地 overlap 计算和内存访问

---

## 💡 优化方案

### 方案 1：动态 Kernel Selection（最简单）

根据序列长度选择不同实现：

```python
def attention_adaptive(q, k, v, causal, sm_scale, freqs_cos, freqs_sin):
    seq_len = q.shape[2]
    
    if seq_len <= 4096:
        # 短序列：使用 Fused RoPE（性能最佳）
        return fused_rope_attn(q, k, v, causal, sm_scale, freqs_cos, freqs_sin, False)
    else:
        # 长序列：使用非融合版本（避免 register pressure）
        return baseline3_rope_flashattn_triton(q, k, v, causal, sm_scale)
```

**优点**：
- ✅ 简单直接，立即生效
- ✅ 在所有序列长度都获得最优性能

**缺点**：
- ❌ 不是"真正"的优化，只是 workaround
- ❌ 需要维护两套代码

### 方案 2：优化 Tile 配置（推荐）

为长序列添加专门的 autotune 配置：

```python
# 在 fused_rope_attn.py 中修改 configs
configs = [
    # 原有配置（适合短序列）
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w)
    for BM in [64, 128]
    for BN in [32, 64, 128]
    for s in [2, 3, 4]
    for w in [4, 8]
]

# 为长序列添加小 tile 配置
if N_CTX >= 8192:
    configs += [
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
    ]
```

**关键思路**：
- 小 tile → 更少寄存器 → 更高 occupancy
- 更多 stages → 更好的 memory latency hiding

### 方案 3：减少中间变量（侵入式）

在 `_attn_fwd_inner` kernel 中：

```python
# 当前实现（保留原始 q1, q2, k1, k2）:
q1 = tl.load(...)
q2 = tl.load(...)
q1_rot = (q1 * cos - q2 * sin).to(dtype)
q2_rot = (q2 * cos + q1 * sin).to(dtype)

# 优化：立即 overwrite，不保留原始
q1 = tl.load(...)
q2 = tl.load(...)
q1_temp = q1  # 临时保存用于计算 q2_rot
q1 = (q1 * cos - q2 * sin).to(dtype)  # q1 → q1_rot
q2 = (q2 * cos + q1_temp * sin).to(dtype)  # q2 → q2_rot
```

**优点**：减少活跃寄存器数量
**缺点**：代码可读性下降

### 方案 4：使用 Shared Memory（高级）

将 `freqs_cos`, `freqs_sin` 放入 shared memory 而非寄存器：

```python
# 在 kernel 开始时加载到 SMEM
freqs_cos_smem = tl.shared_memory(...)
freqs_sin_smem = tl.shared_memory(...)
```

**优点**：大幅减少寄存器压力
**缺点**：增加 SMEM 使用，可能影响 occupancy（不同的瓶颈）

---

## 🎯 立即可行的优化（方案 2 简化版）

修改 `fused_rope_attn.py` 的 `prune_invalid_configs` 函数：

```python
def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]
    STAGE = kwargs["STAGE"]
    
    filtered = [
        conf for conf in configs 
        if conf.kwargs.get("BLOCK_M", 0) <= N_CTX 
        and (conf.kwargs.get("BLOCK_M", 0) >= conf.kwargs.get("BLOCK_N", 0) or STAGE == 1)
    ]
    
    # 长序列时，优先使用小 tile 以降低寄存器压力
    if N_CTX >= 8192:
        # 过滤掉大 tile (BLOCK_M >= 128 or BLOCK_N >= 128)
        filtered = [
            conf for conf in filtered
            if conf.kwargs.get("BLOCK_M", 0) <= 64 
            and conf.kwargs.get("BLOCK_N", 0) <= 64
        ]
    
    return filtered
```

---

## 📊 预期效果

修改后，长序列性能应提升到：
- 8K-16K: **0.9-1.0x**（接近 Baseline 3）
- 32K-128K: **0.8-0.9x**（显著改善，虽然可能仍略慢）

如果仍不理想，建议采用**方案 1**（动态选择）作为 production 方案。

---

## 🛠️ 下一步行动

1. **立即尝试**：修改 `prune_invalid_configs`（10分钟）
2. **验证效果**：重新运行 `bench_compare.py`（5分钟）
3. **如果仍不理想**：实现方案 1 的 adaptive kernel selection
4. **高级优化**：使用 `ncu` profile 定位具体瓶颈

需要我现在就实现方案 2 吗？

