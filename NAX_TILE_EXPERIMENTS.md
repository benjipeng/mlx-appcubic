# NAX Tile Experiments on M5 Max 40-core (applegpu_g17s)

Results from systematic tile-size experiments on the MLX NAX matmul kernels.
All measurements on Apple M5 Max 128GB, macOS 26.3.1, MLX 0.31.2-dev.

## Dense fp16 matmul (matmul.cpp)

### Problem

The NAX dispatch in `matmul.cpp:207-214` unconditionally downgrades tile sizes
for Max/Ultra chips (architecture suffix `s`, `c`, `d`):

```cpp
// Temp routing for larger devices
if (devc == 's' || devc == 'c' || devc == 'd') {
    bk = (K >= 8192 && K > (M + N)) ? 64 : 256;
    bm = 64;
    wm = 2;
}
```

Default config: `bm=128, bn=128, bk=512, wm=4, wn=4`
Forced config:  `bm=64,  bn=128, bk=256, wm=2, wn=4`

### Fix (on this branch)

Gate the downscale on `M < 128` so only decode-shaped ops get small tiles:

```cpp
if (devc == 's' || devc == 'c' || devc == 'd') {
    if (M < 128) {
        bk = (K >= 8192 && K > (M + N)) ? 64 : 256;
        bm = 64;
        wm = 2;
    }
}
```

### Results — transformer projections (Llama 3.1 8B shapes)

| Case     | DType | Before (TOPS) | After (TOPS) | Speedup |
|----------|-------|---------------|--------------|---------|
| qkv      | fp16  | 60.9          | 124.1        | 2.0x    |
| attn_out | fp16  | 57.4          | 292.7*       | 5.1x    |
| mlp_up   | fp16  | 60.1          | 124.7        | 2.1x    |
| mlp_down | fp16  | 53.4          | 109.9        | 2.1x    |

*attn_out inflated by auto inner_loops pipelining — real single-op is ~60 TOPS.

Honest single-op peak after fix: **~62 TOPS fp16**, matching the M5 Max hardware
ceiling (40 cores × 1024 FP16 ops/core/cycle × ~1.55 GHz ≈ 63 TFLOPS).

### Regression on MLX's large_gemm_bench.py

The fix **regresses** on wider/deeper GEMMs:

| Shape (bf16)        | Before (ms) | After (ms) | Change     |
|---------------------|-------------|------------|------------|
| 4096×4096×12288     | 7.78        | 7.58       | 3% faster  |
| 4096×4096×21504     | 12.92       | 16.55      | 28% slower |
| 4096×6144×21504     | 20.10       | 42.64      | 112% slower|

**Not suitable for upstream PR.** Shape-dependent tuning needed.

### Correctness

- 677/677 MLX tests pass (9,471 subtests)
- Frobenius norm errors identical to baseline across all shapes and dtypes

---

## Quantized int8 matmul (quantized.cpp + quantized_nax.metal)

### Problem

The NAX quantized kernel is hardcoded to `bm=64, bn=64, bk=64, wm=2, wn=2`.
Only one tile variant is compiled. The `QuantizedBlockLoader` in `quantized_nax.h`
dequantizes int8 weights to fp16 in threadgroup memory (`Ws`), then the NAX
consumes the dense fp16 tiles. The native int8 NAX datapath is never used.

Measured peak: **~55 TOPS** (vs ~62 TOPS fp16, vs ~126 TOPS theoretical int8).

### Experiments

**Experiment 1: BK=128 (bigger K chunk)**

- Added `group_size=64` specialization to `QuantizedBlockLoader` in
  `quantized_nax.h` (following the existing `group_size=32` pattern at line 703)
- Instantiated `64×64×128` kernel variants in `quantized_nax.metal`
- Dispatch: `bk = (M >= 128 && group_size == 64) ? 128 : 64`

Result: **slower** (0.68-0.85x of fp16, was 0.90x). Doubling `Ws` from
64×64 to 64×128 killed GPU occupancy — fewer threadgroups per core.

**Experiment 2: BM=128 (taller output tile)**

- Instantiated `128×64×64` kernel variants in `quantized_nax.metal`
- Dispatch: `bm = (M >= 128) ? 128 : 64`
- No `QuantizedBlockLoader` changes needed (BM is activation-side only)

Result: **slower** (0.50-0.98x of fp16). Fewer threadgroups in grid,
possible register pressure from larger accumulator tile.

**Experiment 3: BN=128 (wider weight tile)**

Not attempted — blocked by `QuantizedBlockLoader` static_assert:
`BCOLS <= group_size` (BCOLS maps to BK in the transpose kernel's loader,
but BN maps to BROWS which... actually isn't constrained. However, BN=128
would double `Ws` size same as BK=128, so occupancy impact would be similar.)

### Summary

| Config          | Ws size  | TOPS  | vs fp16 |
|-----------------|----------|-------|---------|
| 64×64×64 (stock)| 64×72=4.6K| ~55  | 0.90x   |
| 64×64×128       | 64×136=8.7K| ~46 | 0.68x   |
| 128×64×64       | 64×72=4.6K| ~39  | 0.50x   |

The `64×64×64` tile is actually the sweet spot for the current kernel
architecture. The bottleneck is not tile sizing — it's the
**dequant-to-threadgroup-memory architecture itself**. Every int8 element
gets unpacked to fp16 in shared memory before the NAX matmul sees it.

### What would actually help

To reach the ~126 TOPS int8 hardware ceiling, the kernel would need to:
1. Feed packed int8 directly into `mpp::tensor_ops::matmul2d` with int8
   cooperative tensors, bypassing `QuantizedBlockLoader` entirely
2. Only dequantize the accumulator output (int32 → fp16), not the inputs
3. This is a ground-up kernel rewrite, not a tile tuning exercise

---

## Key files

- `mlx/backend/metal/matmul.cpp:204-214` — dense NAX tile dispatch
- `mlx/backend/metal/quantized.cpp:489-493` — quantized NAX tile dispatch
- `mlx/backend/metal/kernels/quantized_nax.h:566-693` — QuantizedBlockLoader (generic)
- `mlx/backend/metal/kernels/quantized_nax.h:695-833` — QuantizedBlockLoader (group_size=32 specialization)
- `mlx/backend/metal/kernels/quantized_nax.h:938-1068` — qmm_t_nax_tgp_impl (transpose kernel)
- `mlx/backend/metal/kernels/quantized_nax.metal:60-112` — kernel instantiations
- `mlx/backend/metal/kernels/steel/gemm/nax.h:716-778` — subtile_matmad_nax (mpp::tensor_ops::matmul2d)
