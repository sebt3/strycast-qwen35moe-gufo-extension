# Block-sparse attention (opt-in)

`STRYCAST_SPARSE_ATTN=1` makes the full-attention layers of Qwen3.6-35B-A3B
read only a selection of the cached keys, in prefill and in decode (including
the rows the drafter's verification passes through). Each query tile attends to
the first `STRYCAST_SPARSE_SINK` blocks of 64 keys (default 2), the last
`STRYCAST_SPARSE_LOCAL` blocks (32) and the older blocks whose per-channel key
min/max bound the tile's mean query best (Quest): `STRYCAST_SPARSE_TOPK` (256)
in prefill, `STRYCAST_SPARSE_DECODE_TOPK` (512) in decode. Cost then stops
growing with depth. It is training-free and approximate: greedy output is no
longer byte-identical to the dense kernels, which is why it is off by default.

## Where it applies

- **Prefill** chunks whose visible context exceeds `STRYCAST_SPARSE_MIN_CONTEXT`
  (8192 keys). The last `STRYCAST_SPARSE_DENSE_TAIL` rows (1024) of a prompt run
  apart and dense: the question that follows a long document has to find
  things anywhere in it, and a selection made from a tile's mean query does not
  reliably find them (see below). A prompt shorter than that tail is dense.
- **Decode and verification rows** (35B shape only: 16 query heads over 2 KV
  heads of 256), from `STRYCAST_SPARSE_DECODE_MIN_CONTEXT` (32768) keys,
  switchable on its own with `STRYCAST_SPARSE_DECODE=0`. Up to 32 rows share one
  selection chosen from their mean query, then run the matrix-core decode
  kernel over the selected blocks. Rows are no longer independent of their
  neighbours: greedy output with and without the drafter may differ.
- FP16 KV cache only (disabled with `--kv-cache q8_0`), context <= 262144. Other
  head shapes keep the dense route in decode.
- The per-layer key bounds are rebuilt lazily from the cache; a snapshot
  restore or `Reset()` invalidates them (`InvalidateSparseAttention`).
- Kernels and selection are ported from mikealanni/gufo (MIT); see NOTICE. The
  decode kernel is `attention_wmma_decode.hip` driven by the block list.

## Measured (Strix Halo gfx1151, Q4_K_XL, DFlash2, one request at a time)

Real source code as prompt, 350-token answer, tok/s:

| Prompt tokens | Dense prefill / decode | Sparse prefill / decode |
|---|---|---|
| ~17k | 2394 / 66.5 | 2122 / 64.1 |
| ~128k | 1005 / 50.0 | 1453 / 52-55 |
| ~211k | 585 / 42.4 | 1385 / 55-57 |

Quality, ~110k tokens of code: 5 listing questions with a checkable answer
(environment variables, enumerators, files, constants): dense 35/37, sparse
36/37. 24 numeric constants asked one by one: dense 24/24, sparse with the
dense tail 24/24. Needles planted at 10-90 % depth, 6/6, and 4/4 facts from a
restored 108544-token prefix.

What went wrong without the dense tail: the same 24 numeric lookups scored
12-16/24 with sparse prefill (wider selections hardly helped: top 512 gave
16/24), and 24/24 with sparse decode alone. The rows that carry the question
are the ones that need the whole context, so they are the ones kept dense.
These tests are still small: confirm on real workloads, and set
`STRYCAST_SPARSE_ATTN=0` (or unset it) if you see degradation.
