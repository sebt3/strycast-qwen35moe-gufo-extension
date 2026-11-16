# Block-sparse prefill attention (opt-in)

`STRYCAST_SPARSE_ATTN=1` makes the full-attention layers of Qwen3.6-35B-A3B
(and the 27B / 122B head shapes) read only a selection of the cached keys during
prefill: each 32-query tile attends to the first `STRYCAST_SPARSE_SINK` blocks
(default 2), the last `STRYCAST_SPARSE_LOCAL` blocks (32) and the
`STRYCAST_SPARSE_TOPK` (256) older 64-key blocks whose per-channel key min/max
bound the tile's mean query best (Quest). Prefill cost then stops growing with
depth. It is training-free and approximate: greedy output is no longer
byte-identical to the dense kernel, which is why it is off by default.

- Applies to prefill chunks whose visible context exceeds
  `STRYCAST_SPARSE_MIN_CONTEXT` (8192 keys). Decode and the last layer's
  trimmed output tile keep the dense kernels.
- FP16 KV cache only (disabled with `--kv-cache q8_0`), context <= 262144.
- The per-layer key bounds are rebuilt lazily from the cache; a snapshot
  restore or `Reset()` invalidates them (`InvalidateSparseAttention`).
- Kernel and selection are ported from mikealanni/gufo (MIT); see NOTICE.

## Measured (Strix Halo gfx1151, Q4_K_XL, DFlash2, 1 request at a time; same build, env on/off)

| Prompt tokens | Dense prefill | Sparse prefill |
|---|---|---|
| ~10k | 2564 / 2301 tok/s | 2509 / 2159 |
| ~44k | 1738 | 1718 |
| ~109k | 1179-1182 | 1556-1602 |
| ~164k | 910 | 1557 |
| ~229k | 716 | 1545 |

Quality checks at ~109k, dense and sparse alike: 6/6 unique facts planted at
10-90 % depth, 15/15 `FLINT_*` environment variables listed from real source
code, and 4/4 facts from the restored prefix after a cache hit
(108544 cached tokens). These tests are easy; confirm on real workloads.
