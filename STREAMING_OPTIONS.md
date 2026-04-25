# Streaming KWS — Options & Open Issues

## Model architecture (strided-s16-nodil)

```
sinc_baked  kernel=129, stride=16, valid  → 8000 samples → 492 frames  (~2M MACs)
dw2/pw2     kernel=3,   stride=1,  same   → 492 → 492    (×48 ch)
pool2       pool=4, stride=4              → 492 → 123
dw3/pw3     kernel=3,   stride=1,  same   → 123 → 30
dw4/pw4     kernel=3,   stride=1,  same   → 30  → 15
dw5/pw5     kernel=3,   stride=1,  same   → 15  → 15
GAP → Dense(32) → Dense(11)
```

Current sliding window: HOP=1600 → full 8000-sample inference every time → 80% overlap recomputed wastefully.

---

## Option A — Retrain as RNN (GRU)

Replace temporal backbone with GRU operating on 160-sample (20 ms) frames.
NNoM has GRU/LSTM support; `model_run()` works normally, called every 160 samples.

- **Pro**: cleanest architecture, inherently streaming, no graph surgery
- **Con**: different architecture, requires retraining from scratch; GRU heavier than depthwise CNN on small MCU
- **Status**: not started

---

## Option B — Bypass model_run(), drive math primitives directly

Keep weights header. Write `kws_stream_conv.c` that:
1. Maintains per-layer state ring buffers (total ~1.2 KB)
2. Calls `local_conv1d_HWC_q7_fast` / `local_depthwise_conv1d_HWC_q7` for only the new HOP outputs
3. Maintains a running GAP sum (incremental update)
4. Calls NNoM Dense layers via `layer_run(layer[15])` / `layer_run(layer[17])` for the FC tail

### Memory impact

```
                     Current NNoM        Option B
nnom_static_buf      256 KB (declared)   still needed for Dense tail (~50 KB actual)
nnom_input_data      8 KB (full window)  can be dropped / repurposed as scratch
Streaming state      —                   ~1.2 KB (sinc boundary + dw states + GAP sum)
Per-HOP scratch      —                   ~9.6 KB (reusable across layers)
```

Net active RAM is similar; main gain is ~5× MAC reduction in the conv backbone.
If Dense is also reimplemented manually (~60 lines), NNoM can be dropped entirely → ~12 KB total RAM.

### Compute savings per HOP

```
Layer       Full inference    Streaming (HOP=1600)    Reduction
sinc        492 × 129 × 32   100 × 129 × 32           5×
dw2/pw2     492 × 3 × 48     100 × 3 × 48             5×
dw3/pw3     123 × 3 × 48     25 × 3 × 48              ~5×
dw4/pw4     30 × 3 × 48      6 × 3 × 48               5×
dw5/pw5     15 × 3 × 48      3 × 3 × 48               5×
Dense       unchanged         unchanged                1×
```

### Padding problem
`dw2–dw5` use `padding=same` (symmetric). Causal streaming requires left-only padding.
Causes ~1 frame temporal shift per block = ~64 sample / 8 ms shift total.
- No retraining: ~1–2% accuracy drop
- Fine-tune with causal padding: accuracy recovered, ~30 min training

- **Pro**: reuses existing weights, largest compute gain, no new architecture
- **Con**: bypasses NNoM graph runner; must drive CMSIS-NN primitives manually; fragile if weights format changes
- **Status**: not started

---

## Option C — Google kws_streaming re-export → TFLite Micro

1. `pip install kws-streaming`
2. `kws_streaming.utils.to_streaming(model)` — converts each layer to a causal streaming cell with explicit state tensors
3. Export to TFLite int8
4. Run on MCU via **TFLite Micro** (not NNoM — streaming state tensors use `ASSIGN_VARIABLE` ops that NNoM cannot convert)

### NNoM vs TFLite Micro

| | NNoM | TFLite Micro |
|---|---|---|
| Streaming | No | Yes (native) |
| Quantization | Int8 per-tensor | Int8 per-channel (more accurate) |
| Code size | ~20 KB C | ~50–100 KB C++ |
| CMSIS-NN | Yes | Yes |
| kws_streaming compat | No (ASSIGN_VARIABLE unsupported) | Yes |

- **Pro**: automated causal conversion, best accuracy, maintained runtime
- **Con**: adds C++ and TFLM dependency; larger code footprint; kws_streaming not installed yet
- **Status**: not started; kws_streaming needs `pip install`

---

## NNoM memory problems (general)

1. **`NNOM_STATIC_BUF_KB=256` is a guess** — actual usage for this model is ~50–60 KB.
   Fix: call `nnom_mem_stat()` after `model_compile()`, then tighten the define.

2. **`nnom_input_data[8000]` is always in BSS** regardless of whether you use the full window.
   In streaming, 7 of those 8 KB are wasted.

3. **Block-sharing only covers activations** — graph structs, layer configs, weight tensor metadata
   all sit in `nnom_mem()` and are never freed. True total RAM > activation peak.

4. **No partial execution API** — `model_run_to(m, end_layer)` exists but still runs each
   layer over its full input tensor; cannot express "run layer N over positions 392–492 only."

---

## Decision needed before implementing

- How tight is the SRAM budget on the target MCU?
- Acceptable to add C++ (TFLite Micro) dependency?
- Worth retraining for causal padding, or accept ~1–2% drop?
