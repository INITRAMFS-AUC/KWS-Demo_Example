# How to Convert an NNoM Model to a Streaming Inference Engine

This guide documents the exact process used to build `streaming_b/` from `strided_s16_nodil`.
The method applies to any model that is:
- A purely feedforward 1D temporal CNN (Conv1D, Pooling, GAP, Dense)
- Quantised to int8 and exported via NNoM's `nnom_model_create()`
- Trained on fixed-length clips (e.g. 1-second audio at 8 kHz)

The result is a C engine that calls NNoM's math primitives directly — no `model_run()`,
no NNoM static buffer — and produces a new prediction every HOP samples by recomputing
only the frames that changed.

---

## 1. Understand what "streaming" means here

The batch model does this every second:

```
8000 samples → model_run() → 1 prediction
```

The streaming model does this every HOP (e.g. 1600 samples):

```
1600 new samples
  + per-layer boundary state from the previous hop
  → compute only the ~100 new output frames per layer
  → get 1 prediction every 200 ms instead of 1 per second
```

The key insight: most of the computation in the batch model is redundant because
consecutive calls share ~80% of their audio. The streaming engine eliminates that
redundancy by keeping only the boundary context each layer needs.

---

## 2. Read off the architecture from the weights header

Open `<model>_weights.h` and find `nnom_model_create()`. Read the layer list from top
to bottom. For each layer note:

| What to record | Where to find it |
|---|---|
| Layer type | function name: `conv2d_s`, `maxpool_s`, `dense_s`, `global_avgpool_s` |
| Kernel size | `config.kernel_size = {H, W}` — for 1D audio W is the temporal dimension |
| Stride | `config.stride_size = {H, W}` |
| Padding | `config.padding_type = PADDING_VALID / PADDING_SAME` |
| Input channels | inferred from previous layer's output or from weight tensor dim |
| Output channels / filters | `config.filter_size` or last dim of `tensor_*_kernel_0_dim[]` |

For `strided_s16_nodil` this gives:

```
Layer       Type          kernel  stride  pad    in_ch  out_ch
sinc        conv2d_s      1×129   1×16    VALID  1      32
conv1d_1    conv2d_s      1×3     1×1     SAME   32     48
pool1       maxpool_s     1×4     1×4     VALID  48     48
conv1d_2    conv2d_s      1×3     1×1     SAME   48     48
pool2       maxpool_s     1×4     1×4     VALID  48     48
conv1d_3    conv2d_s      1×3     1×1     SAME   48     48
pool3       maxpool_s     1×2     1×2     VALID  48     48
conv1d_4    conv2d_s      1×3     1×1     SAME   48     48
GAP         global_avgpool_s      —       —      48     48
dense       dense_s       —       —       —      48     32
dense_1     dense_s       —       —       —      32     11
```

Note: NNoM exports Keras `Conv1D` as `conv2d_s` with `kernel_size = {1, W}`.
Activations (ReLU) are stored as `actail` on their parent layer — they are not
separate nodes in the shortcut list.

---

## 3. Compute the temporal stride pyramid

Starting from the raw audio, multiply all temporal strides together cumulatively.
This tells you the frame rate at each layer and how many audio samples each frame
represents.

```
Stage             Stride  Cumulative stride  Hz at 8 kHz input
raw audio         —       1                  8000 samples/sec
after sinc        ×16     16                 500 frames/sec
after pool1       ×4      64                 125 frames/sec
after pool2       ×4      256                31.25 frames/sec
after pool3       ×2      512                15.6 frames/sec   ← GAP input
```

This tells you:
- **How to choose HOP**: pick a HOP divisible by the sinc stride (16). HOP=1600 gives
  exactly 100 new sinc frames per hop and cascades cleanly through the pools.
- **Why state is small**: a single GAP-input frame covers 512 audio samples. Keeping
  2 left-context frames there costs 2 × 48 bytes = 96 bytes, not 2 × 512 × 48.

---

## 4. Compute state sizes for each layer

### Conv layers — left-context ring

Every conv layer with temporal kernel `k > 1` needs `k - 1` left-context frames
from its INPUT (not output). These are the frames from the previous hop that sit
at the boundary of the new computation window.

```
State size = (kernel_w - 1) × in_channels  bytes
```

For kernel_w = 3: need 2 left-context frames.
For kernel_w = 129 (sinc): need 128 left-context SAMPLES (ch=1), stored in `audio_ring`.

| Layer | Left-context frames | ch_in | State |
|-------|--------------------:|------:|------:|
| audio_ring (sinc input) | 128 samples | 1 | 128 B |
| conv1_ctx (conv1d_1 input) | 2 | 32 | 64 B |
| conv2_ctx (conv1d_2 input) | 2 | 48 | 96 B |
| conv3_ctx (conv1d_3 input) | 2 | 48 | 96 B |
| conv4_ctx (conv1d_4 input) | 2 | 48 | 96 B |

**General rule**: `ctx_bytes = (kernel_w - 1) × ch_in`

### Pool layers — partial-frame buffer

Non-overlapping pooling (stride = kernel) is stateless IF the number of input
frames per hop is divisible by the pool kernel. When it is not, the leftover
frames must be held for the next hop.

```
Max partial frames = kernel - 1
State size = (kernel - 1) × channels  bytes
```

Check divisibility hop-by-hop:

```
Pool1: 100 conv1 frames per hop, kernel=4 → 100 % 4 = 0  → NO partial buffer needed
Pool2: 25  pool1 frames per hop, kernel=4 → 25  % 4 = 1  → up to 3 pending (max partial)
Pool3: 6–7 pool2 frames per hop, kernel=2 → varies       → up to 1 pending
```

| Pool | kernel | Max partial | ch | State |
|------|-------:|------------:|---:|------:|
| pool1 | 4 | 0 | 48 | 0 B |
| pool2 | 4 | 3 | 48 | 144 B |
| pool3 | 2 | 1 | 48 | 48 B |

**General rule**: if `(frames_per_hop % kernel) == 0` always, no partial buffer.
Otherwise `partial_bytes = (kernel - 1) × channels`.

### GAP ring

The batch model's GAP averages over N frames where N = total_input_samples / total_stride.
For strided_s16_nodil: N = 8000 / 512 ≈ 15 frames (verified by counting conv4 output size).

The streaming GAP ring holds the last N conv4 output frames. New frames push in;
the oldest is evicted when the ring is full. At inference time, average all N slots.

```
State size = N × ch_out  bytes  =  15 × 48  =  720 B
```

To find N for your model: look at the output shape of the last conv layer before GAP
in the weights header. It's stored in `layer_out->tensor->dim[]` or derivable from the
stride pyramid: `N = clip_length_samples / total_cumulative_stride`.

### Dense / FC layers

Stateless — same computation every hop. No state needed.

### Summary for strided_s16_nodil

```
audio_ring      128 B
conv1_ctx        64 B
conv2_ctx        96 B
conv3_ctx        96 B
conv4_ctx        96 B
pool2_buf       144 B   (3 partial × 48 ch)
pool3_buf        48 B   (1 partial × 48 ch)
gap_ring        720 B   (15 × 48)
bookkeeping ints ~30 B
sample_buf     1600 B   (one HOP of raw audio, for sub-HOP push support)
─────────────────────
Total           ~3.0 KB
```

---

## 5. Choose causal vs symmetric padding

Your trained model uses `PADDING_SAME` (symmetric: `floor(kernel/2)` on each side).
Causal streaming requires left-only padding.

**Option A — Accept the shift (fastest, no retraining)**

Use the "prepend-context + valid conv" trick:
```
input = [ctx_frames (kernel_w-1) | new_frames (N_new)]   (contiguous buffer)
call local_convolve_HWC_q7_nonsquare with padding=0, dim_in_x=(kernel_w-1+N_new)
output = N_new frames (exactly, via valid convolution)
```

This gives causal output: frame k depends on inputs k-2, k-1, k (for kernel_w=3).
The symmetric model used: k-1, k, k+1. The output is shifted 1 frame to the left.
Accuracy impact: typically **0.5–2%** drop.

For `strided_s16_nodil` the measured drop was **0.9%** (84.6% vs 85.5%).

**Option B — Lookahead buffer (no accuracy drop, adds latency)**

Keep `floor(kernel/2)` future frames buffered before running each conv.
For kernel=3 this is 1 frame of lookahead per conv layer = 4 layers × 1 frame ×
(1/15.6 sec per frame at pool3 output) ≈ **250 ms extra latency**.

**Option C — Retrain with causal padding (~30 min)**

Fine-tune the existing model with causal padding. Recovers accuracy fully.
Use `tf.keras.layers.Conv1D(padding='causal')` and fine-tune for 5–10 epochs.

---

## 6. Implement the engine

### Header (`kws_stream_<model>.h`)

Define constants and the state struct. Mirror the architecture exactly:

```c
#define KWS_HOP         <hop_size>
#define KWS_N_CLASSES   <num_classes>
#define KWS_N_GAP       <gap_temporal_dim>

typedef struct {
    /* One audio_ring per conv layer with temporal kernel > 1.
     * Size = (kernel_w - 1) × ch_in */
    int8_t audio_ring[128];           /* sinc: (129-1) × 1 ch */
    int8_t conv1_ctx[2 * 32];         /* conv1: (3-1) × 32 ch */
    int8_t conv2_ctx[2 * 48];         /* conv2: (3-1) × 48 ch */
    /* ... */

    /* One partial_buf per pool where frames_per_hop % kernel != 0.
     * Size = (kernel - 1) × channels */
    int8_t pool2_buf[3 * 48];
    int    pool2_n;
    /* ... */

    /* GAP ring */
    int8_t gap_ring[KWS_N_GAP * <ch>];
    int    gap_head, gap_count;

    /* Sub-HOP sample accumulator */
    int    samples_pending;
    int8_t sample_buf[KWS_HOP];
} kws_stream_t;

void kws_stream_reset(kws_stream_t *s);
int  kws_stream_push(kws_stream_t *s, const int8_t *samples, int n,
                     int8_t logits[KWS_N_CLASSES]);
```

### Scratch buffers (`kws_stream_<model>.c`)

Declare static (non-reentrant) scratch buffers for all intermediate results.
The largest is always the first conv output (output_frames × output_channels).

```c
/* Sinc stage */
static int8_t scratch_sinc_in[128 + KWS_HOP];               /* 1728 B */
static int8_t scratch_sinc_out[100 * 32];                    /* 3200 B */

/* Conv1 stage: prepend 2 ctx frames to 100 sinc frames */
static int8_t scratch_conv1_in[(2 + 100) * 32];              /* 3264 B */
static int8_t scratch_conv1_out[100 * 48];                   /* 4800 B — largest */

/* Later stages are smaller because pool downsampling reduces frame count */
static int8_t scratch_pool1_out[25 * 48];
/* ... */
```

Size each buffer as `max_frames_this_stage × channels`. For stages after pools,
the max frames per hop is `ceil(frames_entering_pool / pool_kernel)` — small.

### Causal conv helper

```c
static void causal_conv1d(
    int8_t *scratch,           /* (n_ctx + n_new) × ch_in bytes */
    const int8_t *ctx,         /* n_ctx × ch_in: left-context from state */
    int n_ctx,
    const int8_t *new_frames,  /* n_new × ch_in: new input this hop */
    int n_new,
    int ch_in, int ch_out,
    const int8_t *weight,      /* [ch_out][1][kernel_w][ch_in] layout */
    const int8_t *bias,
    const nnom_qformat_param_t *bias_shift,
    const nnom_qformat_param_t *out_shift,
    nnom_qtype_t qtype,
    int8_t *out)               /* n_new × ch_out output */
{
    int total = n_ctx + n_new;
    memcpy(scratch,             ctx,        n_ctx  * ch_in);
    memcpy(scratch + n_ctx*ch_in, new_frames, n_new * ch_in);

    local_convolve_HWC_q7_nonsquare(
        scratch,
        total, 1, ch_in,           /* dim_in_x, dim_in_y, ch_in */
        weight, ch_out,
        3, 1,                      /* kernel_w, kernel_h */
        0, 0, 1, 1, 1, 1,          /* pad, stride, dilation (all 1D) */
        bias, bias_shift, out_shift, qtype,
        out, n_new, 1,             /* out_x = n_new, out_y = 1 */
        NULL, NULL);
}
```

After each causal_conv1d call, update the context with the last `n_ctx` frames of
that layer's INPUT (not output):

```c
/* Save last 2 input frames as left-context for next hop */
memcpy(ctx,       inputs + (n_new - 2) * ch_in, ch_in);
memcpy(ctx + ch_in, inputs + (n_new - 1) * ch_in, ch_in);
```

For sinc, the "input" is raw audio — save the last 128 samples:
```c
memcpy(s->audio_ring, new_audio + KWS_HOP - 128, 128);
```

### Pooling helper with partial state

```c
static int pool_with_partial(
    int8_t *partial_buf, int *partial_n,
    const int8_t *new_frames, int n_new,
    int ch, int k,
    int8_t *scratch,   /* (*partial_n + n_new) × ch bytes */
    int8_t *out)
{
    int total    = *partial_n + n_new;
    int n_out    = total / k;
    int leftover = total % k;

    memcpy(scratch,                    partial_buf,  *partial_n * ch);
    memcpy(scratch + *partial_n * ch,  new_frames,   n_new      * ch);

    if (n_out > 0)
        local_maxpool_q7_HWC(
            scratch,
            n_out * k, 1, ch,   /* process only the full groups */
            k, 1, 0, 0, k, 1,
            n_out, 1,
            NULL, out);

    memcpy(partial_buf, scratch + n_out * k * ch, leftover * ch);
    *partial_n = leftover;
    return n_out;
}
```

If `frames_per_hop % kernel == 0` always (like pool1 here), skip the partial buffer
entirely and call `local_maxpool_q7_HWC` directly on the full input.

### GAP ring and average

```c
/* Push new conv_final frames into the ring */
for (int f = 0; f < n_new; f++) {
    int dst = s->gap_head % N_GAP;
    memcpy(s->gap_ring + dst * CH, new_frames + f * CH, CH);
    s->gap_head++;
    if (s->gap_count < N_GAP) s->gap_count++;
}

/* Compute average at inference time */
int n = s->gap_count;
for (int c = 0; c < CH; c++) {
    int32_t sum = 0;
    for (int i = 0; i < n; i++)
        sum += s->gap_ring[i * CH + c];
    gap_out[c] = (int8_t)(sum / n);
}
```

When `gap_count == N_GAP` (ring full), all N slots are valid recent frames and can
be summed in any order. When `gap_count < N_GAP` (warmup), average over however
many frames exist.

### Dense tail

Pull constants directly from the weights header. No NNoM layer objects needed:

```c
/* DENSE_BIAS_LSHIFT = {0}, DENSE_OUTPUT_RSHIFT = {6} */
static const nnom_qformat_param_t _bs[] = DENSE_BIAS_LSHIFT;
static const nnom_qformat_param_t _os[] = DENSE_OUTPUT_RSHIFT;

local_fully_connected_q7_opt(
    gap_out,
    tensor_dense_kernel_0_data,
    48, 32,                       /* dim_vec, num_of_rows */
    (uint16_t)_bs[0], (uint16_t)_os[0],
    tensor_dense_bias_0_data,
    dense0_out,
    fc_scratch);                  /* q15_t[dim_vec] scratch */

local_relu_q7(dense0_out, 32);   /* ReLU actail */

/* Repeat for Dense_1 (32→11) */
```

For models with per-tensor quantisation (`NNOM_QTYPE_PER_TENSOR`): shift arrays are
size 1, access index [0]. For per-axis (`NNOM_QTYPE_PER_AXIS`): arrays are length
`ch_out`, pass directly to the conv primitive.

---

## 7. Wire the per-hop function

```c
static void run_hop(kws_stream_t *s, const int8_t *new_audio, int8_t *logits)
{
    /* 1. Sinc */
    memcpy(scratch_sinc_in,       s->audio_ring, 128);
    memcpy(scratch_sinc_in + 128, new_audio,     KWS_HOP);
    local_convolve_HWC_q7_nonsquare(/* sinc args */);
    local_relu_q7(scratch_sinc_out, N_SINC * CH_SINC);
    memcpy(s->audio_ring, new_audio + KWS_HOP - 128, 128);

    /* 2. Conv1 (causal) */
    causal_conv1d(scratch_conv1_in, s->conv1_ctx, 2,
                  scratch_sinc_out, N_SINC, CH_SINC, CH_CONV, /* weights */);
    local_relu_q7(scratch_conv1_out, N_SINC * CH_CONV);
    update_ctx2(s->conv1_ctx, scratch_sinc_out, N_SINC, CH_SINC);

    /* 3. Pool1 (exact: no partial) */
    int n_p1 = maxpool_valid(scratch_conv1_out, N_SINC, CH_CONV, 4, scratch_pool1_out);

    /* 4. Conv2 (causal) */
    causal_conv1d(scratch_conv2_in, s->conv2_ctx, 2,
                  scratch_pool1_out, n_p1, CH_CONV, CH_CONV, /* weights */);
    local_relu_q7(scratch_conv2_out, n_p1 * CH_CONV);
    update_ctx2(s->conv2_ctx, scratch_pool1_out, n_p1, CH_CONV);

    /* 5. Pool2 (partial accumulation) */
    int n_p2 = pool_with_partial(&s->pool2_buf, &s->pool2_n,
                                  scratch_conv2_out, n_p1, CH_CONV, 4, /* scratch, out */);

    /* 6–9. Conv3, Pool3, Conv4: same pattern */

    /* 10. GAP ring update */
    for (int f = 0; f < n_conv4; f++) { /* push frames into ring */ }

    /* 11. Compute GAP average */

    /* 12. Dense0 + ReLU + Dense1 → logits */
}
```

---

## 8. Build the Spike test harness

Mirror `strided_s16_nodil_main.c`:

```c
int main(void) {
    /* 1. open test_data.bin, read n_real_samples */
    /* 2. malloc full file */
    /* 3. init streaming engine: kws_stream_reset(&s) */

    while (/* batches */) {
        for (int i = 0; i < LABEL_BATCH; i++) {
            /* push HOPS_PER_CLIP × HOP samples */
            kws_stream_reset(&s);
            for (int h = 0; h < HOPS_PER_CLIP; h++)
                kws_stream_push(&s, audio + h*HOP, HOP, logits);

            /* argmax(logits) vs label[i] */
            printf("RESULT:%d,%d\n", label, pred);
        }
    }
    printf("ACCURACY:%.6f\n", (double)correct/total);
}
```

`HOPS_PER_CLIP = SAMPLES_PER_CLIP / HOP = 8000 / 1600 = 5`.

The test_data.bin format (same for all models):
```
int32 LE  n_real_samples          (4 bytes header)
repeat:
  int8[128]     labels            (LABEL_BATCH)
  int8[128×8000] audio            (Q7 signed, 8 kHz, 1 s clips)
```

Add to Makefile, mirroring the existing targets:
```makefile
$(BUILD)/<model>_stream: streaming_<model>/stream_main.c \
                         streaming_<model>/kws_stream.c \
                         <model>/<model>_weights.h \
                         $(NNOM_SRCS)
    $(CC) $(CFLAGS) -I<model> -Istreaming_<model> \
        $< streaming_<model>/kws_stream.c $(NNOM_SRCS) -o $@ $(LDFLAGS)
```

---

## 9. Expected accuracy and how to interpret results

After running on Spike:

| Result | Meaning |
|---|---|
| `ACCURACY` close to batch baseline | Engine is correct — minimal state corruption |
| Drop of 0–2% | Expected from causal padding shift — acceptable |
| Drop of 5–10% | Context update bug (wrong layer's frames being saved as ctx) |
| Drop of >10% | Q-format mismatch (wrong bias_shift/out_shift, wrong qtype) |
| Near-random (~9%) | Critical bug: wrong weight tensor, wrong layer order, or wrong tensor layout |

If accuracy is wrong, debug one layer at a time by comparing the streaming engine's
intermediate outputs to NNoM's intermediate outputs on the same input. Add a
`#ifdef DEBUG` path that:
1. Runs `model_run()` on the first hop's audio (with zero-padded batch input)
2. Compares sinc output, conv1 output, etc. against the streaming engine's scratch buffers
3. Finds the first layer where they diverge

---

## 10. Checklist

**Architecture analysis**
- [ ] Read every layer from `nnom_model_create()` in the weights header
- [ ] Note kernel_w, stride_w, padding_type, ch_in, ch_out for each layer
- [ ] Build the stride pyramid — compute cumulative stride at each stage
- [ ] Choose HOP: must be divisible by the first layer's temporal stride
- [ ] Verify `HOPS_PER_CLIP × HOP == SAMPLES_PER_CLIP`

**State sizing**
- [ ] For each conv with kernel_w > 1: `ctx_bytes = (kernel_w-1) × ch_in`
- [ ] For each pool: check if `(frames_entering_pool % kernel) == 0` always
- [ ] If not exact: `partial_bytes = (kernel-1) × channels`
- [ ] For GAP: `ring_bytes = N_GAP × ch_out` where `N_GAP = clip_samples / cumulative_stride`
- [ ] Add `HOP × 1` bytes for the sample accumulator buffer

**Q-format**
- [ ] Use `NNOM_QTYPE_PER_AXIS` for conv layers (per-channel shifts are arrays)
- [ ] Use `NNOM_QTYPE_PER_TENSOR` for dense layers (single scalar each)
- [ ] For dense: cast shift array index [0] to `uint16_t`
- [ ] Check which macros are the shifts: `<LAYER>_OUTPUT_RSHIFT`, `<LAYER>_BIAS_LSHIFT`

**Weight layout**
- [ ] Conv weights in `local_convolve_HWC_q7_nonsquare` are `[ch_out][ker_h][ker_w][ch_in]`
- [ ] Dense weights in `local_fully_connected_q7_opt` are `[out_row][in_col]` row-major
- [ ] Both match what NNoM's Python export script produces — no transposing needed

**Testing**
- [ ] Build and run on Spike against test_data.bin
- [ ] Streaming accuracy within 2.5% of batch baseline → pass
- [ ] If worse: check context update order, check qtype, check tensor names match

**Multiple definition guard**
- [ ] `<model>_weights.h` must be included in **exactly one** `.c` file
- [ ] `streaming_main.c` must NOT include the weights header — include only the engine header

---

## 11. Layer types not in strided_s16_nodil

### Depthwise Conv (Conv2D with `filter_mult`)

```c
/* Uses local_depthwise_separable_conv_HWC_q7_nonsquare */
/* State: same as regular conv — (kernel_w-1) × ch_in frames */
/* Weight layout: [ker_h][ker_w][ch_in × ch_mult] */
```

### BatchNorm (if not fused)

Stateless at inference — frozen mean/variance baked into a scale+bias per channel.
Apply as element-wise multiply+add on each output frame. No boundary state needed.

### Strided Conv (stride > 1, SAME padding)

Behaves like sinc: stride reduces frame rate. Left-context = `kernel_w - 1` frames
of input BUT only the frames that sit at the stride boundary:
```
effective_ctx = ceil((kernel_w - 1) / stride) × stride  samples of raw input
```

Simplest: convert to "audio_ring" style — keep last `kernel_w - 1` input samples.

### Global MaxPool instead of GAP

Replace the averaging loop with a running max-pool ring update:
```c
/* For each channel, ring holds the max of the last N frames */
for (int c = 0; c < CH; c++) {
    if (new_frame[c] > ring_max[c] || oldest_frame_evicted)
        ring_max[c] = recompute_max_over_ring();
}
```
Or simplest: just keep the ring and take max at inference time — same ring structure.

### LSTM / GRU

Inherently streaming — the hidden state IS the left-context. Call NNoM's
`local_lstm_q7` or `local_gru_q7` primitives per time step. State = hidden
vector size, no ring buffer needed.

---

## 12. Memory footprint summary

For a typical KWS model (3–5 conv layers, 2 pool layers, GAP, 2 dense layers):

```
State struct       = sum(ctx_bytes) + sum(partial_bytes) + gap_ring + sample_buf
                   ≈ 5×96 + 2×144 + 720 + 1600  ≈  3 KB

Per-hop scratch    = dominated by first conv output (N_sinc × ch_out)
                   ≈ 100 × 48  =  4800 B  × ~4 (different stages)  ≈  20 KB

NNoM static buf    = 0  (not needed — no model_run(), no graph)

Weights (flash)    = unchanged from batch model
```

Compare to the batch NNoM path:
```
nnom_static_buf    52 KB  (full graph activation scratch)
nnom_input_data     8 KB  (full clip buffer)
─────────────────────────
                   ~60 KB active RAM  →  streaming: ~23 KB active RAM
```
