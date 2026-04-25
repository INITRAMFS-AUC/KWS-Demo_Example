/* kws_stream_b.c — Option B streaming inference engine
 *
 * Calls NNoM math primitives (local_convolve_HWC_q7_nonsquare etc.) directly.
 * Weight constants imported from strided_s16_nodil_weights.h.
 * No model_run(); no NNoM static buffer required.
 *
 * Memory layout for all intermediate buffers: HWC (H=1 for 1D),
 * i.e. data[frame][channel] = data[frame * CH + channel].
 */
#include <string.h>
#include <stdint.h>
#include "kws_stream_b.h"

/* NNoM types and math primitives */
#include "nnom.h"
#include "nnom_local.h"

/* Model weights, biases, and Q-format constants */
#include "strided_s16_nodil_weights.h"

/* ── per-hop scratch (static, non-reentrant) ─────────────────────────────── */

/* The largest intermediate buffer is conv1_input = (2+100)×32 = 3264 B,
 * or conv1_output = 100×48 = 4800 B.  We allocate them separately and reuse
 * the earlier buffers for later (smaller) stages. */

#define N_SINC_PER_HOP   100   /* HOP/sinc_stride = 1600/16 */
#define N_POOL1_PER_HOP   25   /* N_SINC_PER_HOP / pool1_k = 100/4 */

static int8_t  scratch_sinc_in[128 + KWS_B_HOP];              /* 1728 B */
static int8_t  scratch_sinc_out[N_SINC_PER_HOP * KWS_B_SINC_CH];  /* 3200 B */
static int8_t  scratch_conv1_in[(2 + N_SINC_PER_HOP) * KWS_B_SINC_CH]; /* 3264 B */
static int8_t  scratch_conv1_out[N_SINC_PER_HOP * KWS_B_CONV_CH]; /* 4800 B */
static int8_t  scratch_pool1_out[N_POOL1_PER_HOP * KWS_B_CONV_CH]; /* 1200 B */

/* pool2 input needs at most 3 partial + 25 new = 28 frames × 48 ch = 1344 B */
static int8_t  scratch_pool2_in[28 * KWS_B_CONV_CH];
static int8_t  scratch_conv2_in[(2 + N_POOL1_PER_HOP) * KWS_B_CONV_CH]; /* 1296 B */
static int8_t  scratch_conv2_out[N_POOL1_PER_HOP * KWS_B_CONV_CH];  /* 1200 B */

/* pool2 output: up to 7 frames; pool3 input: up to 1 partial + 7 = 8 × 48 = 384 B */
static int8_t  scratch_pool2_out[7 * KWS_B_CONV_CH];
static int8_t  scratch_pool3_in[8 * KWS_B_CONV_CH];
static int8_t  scratch_conv3_in[(2 + 7) * KWS_B_CONV_CH];   /* 432 B */
static int8_t  scratch_conv3_out[7 * KWS_B_CONV_CH];        /* 336 B */
static int8_t  scratch_pool3_out[4 * KWS_B_CONV_CH];        /* 192 B */
static int8_t  scratch_conv4_in[(2 + 4) * KWS_B_CONV_CH];   /* 288 B */
static int8_t  scratch_conv4_out[4 * KWS_B_CONV_CH];        /* 192 B */

/* Dense intermediates */
static int8_t  scratch_dense0_out[32];
static q15_t   scratch_fc_buf[48];    /* vec_buffer for FC opt */
static int8_t  scratch_gap_out[KWS_B_CONV_CH]; /* 48 B */

/* ── helpers ─────────────────────────────────────────────────────────────── */

/* Run causal conv1d on (n_ctx + n_new) input frames using valid padding.
 * ctx[n_ctx * ch_in] holds the left-context frames.
 * new_frames[n_new * ch_in] holds the new frames.
 * out[n_new * ch_out] receives n_new output frames.
 * scratch_in must be at least (n_ctx + n_new)*ch_in bytes.
 * (n_ctx = kernel_w - 1 = 2 for kernel_w=3)
 */
static void causal_conv1d(
    int8_t *scratch_in,          /* caller-provided scratch for prepended input */
    const int8_t *ctx,           /* left-context frames (n_ctx × ch_in)         */
    int n_ctx,
    const int8_t *new_frames,    /* new input frames (n_new × ch_in)            */
    int n_new,
    int ch_in, int ch_out,
    const int8_t *weight,        /* [ch_out][1][kernel_w][ch_in]                */
    const int8_t *bias,
    const nnom_qformat_param_t *bias_shift,
    const nnom_qformat_param_t *out_shift,
    nnom_qtype_t qtype,
    int8_t *out)                 /* n_new × ch_out output                       */
{
    int total = n_ctx + n_new;
    memcpy(scratch_in,                  ctx,        (size_t)(n_ctx  * ch_in));
    memcpy(scratch_in + n_ctx * ch_in,  new_frames, (size_t)(n_new  * ch_in));

    local_convolve_HWC_q7_nonsquare(
        scratch_in,
        (uint16_t)total, 1, (uint16_t)ch_in,   /* dim_in_x, dim_in_y, ch_in */
        weight, (uint16_t)ch_out,
        3, 1,                                   /* kernel_x, kernel_y        */
        0, 0,                                   /* padding (valid)           */
        1, 1,                                   /* stride_x, stride_y        */
        1, 1,                                   /* dilation_x, dilation_y    */
        bias, bias_shift, out_shift, qtype,
        out,
        (uint16_t)n_new, 1,                     /* dim_out_x, dim_out_y      */
        NULL, NULL);
}

/* Apply relu in-place on n*ch bytes. */
static void relu_inplace(int8_t *buf, int n_frames, int ch)
{
    local_relu_q7(buf, (uint32_t)(n_frames * ch));
}

/* Run valid maxpool (kernel=k, stride=k) on a contiguous buffer of n_in frames.
 * Returns n_out = n_in / k.  Output written to out[n_out * ch]. */
static int maxpool_valid(const int8_t *in, int n_in, int ch, int k, int8_t *out)
{
    int n_out = n_in / k;
    if (n_out == 0) return 0;
    local_maxpool_q7_HWC(
        in,
        (uint16_t)n_in, 1, (uint16_t)ch,
        (uint16_t)k, 1,         /* kernel_x, kernel_y */
        0, 0,                   /* padding            */
        (uint16_t)k, 1,         /* stride_x, stride_y */
        (uint16_t)n_out, 1,     /* dim_out_x, dim_out_y */
        NULL, out);
    return n_out;
}

/* Run pool with partial-frame accumulation.
 * Prepends pending partial frames, runs maxpool, saves leftover back to buf.
 * Returns number of pool output frames produced. */
static int pool_with_partial(
    int8_t *partial_buf, int *partial_n,
    const int8_t *new_frames, int n_new,
    int ch, int k,
    int8_t *scratch,   /* at least (*partial_n + n_new) * ch bytes */
    int8_t *out)       /* at least ((n_new + *partial_n) / k) * ch bytes */
{
    int total = *partial_n + n_new;
    memcpy(scratch,                        partial_buf,  (size_t)(*partial_n * ch));
    memcpy(scratch + *partial_n * ch,      new_frames,   (size_t)(n_new       * ch));

    int n_out  = total / k;
    int leftover = total - n_out * k;

    if (n_out > 0)
        maxpool_valid(scratch, n_out * k, ch, k, out);

    /* Save leftover frames */
    memcpy(partial_buf, scratch + n_out * k * ch, (size_t)(leftover * ch));
    *partial_n = leftover;
    return n_out;
}

/* Update a 2-frame left-context array with the last 2 frames of a buffer. */
static void update_ctx2(int8_t *ctx, const int8_t *frames, int n_frames, int ch)
{
    if (n_frames >= 2) {
        memcpy(ctx,       frames + (n_frames - 2) * ch, (size_t)ch);
        memcpy(ctx + ch,  frames + (n_frames - 1) * ch, (size_t)ch);
    } else if (n_frames == 1) {
        memcpy(ctx,       ctx + ch, (size_t)ch);    /* shift ctx[1] → ctx[0] */
        memcpy(ctx + ch,  frames,   (size_t)ch);
    }
    /* n_frames == 0: ctx unchanged */
}

/* ── per-hop processing ──────────────────────────────────────────────────── */

static void run_hop(kws_stream_b_t *s, const int8_t *new_audio, int8_t *logits)
{
    /* ── sinc (kernel 1×129, stride 16, valid, 32ch output) ──────────────── */
    memcpy(scratch_sinc_in,         s->audio_ring, 128);
    memcpy(scratch_sinc_in + 128,   new_audio,     KWS_B_HOP);

    local_convolve_HWC_q7_nonsquare(
        scratch_sinc_in,
        (uint16_t)(128 + KWS_B_HOP), 1, 1,            /* in_x=1728, in_y=1, ch=1 */
        tensor_conv1d_kernel_0_data, KWS_B_SINC_CH,   /* weights, ch_out=32      */
        129, 1,                                         /* kernel_x, kernel_y      */
        0, 0, 16, 1,                                    /* pad, stride             */
        1, 1,                                           /* dilation                */
        tensor_conv1d_bias_0_data,
        conv1d_bias_shift, conv1d_output_shift, NNOM_QTYPE_PER_AXIS,
        scratch_sinc_out,
        N_SINC_PER_HOP, 1,                              /* out_x=100, out_y=1      */
        NULL, NULL);
    relu_inplace(scratch_sinc_out, N_SINC_PER_HOP, KWS_B_SINC_CH);

    /* update audio ring with last 128 samples */
    memcpy(s->audio_ring, new_audio + KWS_B_HOP - 128, 128);

    /* ── conv1 (kernel 3, 32→48ch, causal) ──────────────────────────────── */
    causal_conv1d(
        scratch_conv1_in,
        s->conv1_ctx, 2,
        scratch_sinc_out, N_SINC_PER_HOP,
        KWS_B_SINC_CH, KWS_B_CONV_CH,
        tensor_conv1d_1_kernel_0_data,
        tensor_conv1d_1_bias_0_data,
        conv1d_1_bias_shift, conv1d_1_output_shift, NNOM_QTYPE_PER_AXIS,
        scratch_conv1_out);
    relu_inplace(scratch_conv1_out, N_SINC_PER_HOP, KWS_B_CONV_CH);

    update_ctx2(s->conv1_ctx, scratch_sinc_out, N_SINC_PER_HOP, KWS_B_SINC_CH);

    /* ── pool1 (k=4, s=4): 100 → 25 frames, exact — no partial ─────────── */
    int n_pool1 = maxpool_valid(scratch_conv1_out, N_SINC_PER_HOP,
                                KWS_B_CONV_CH, 4, scratch_pool1_out);
    /* n_pool1 == 25 always */

    /* ── conv2 (kernel 3, 48→48ch, causal) ──────────────────────────────── */
    causal_conv1d(
        scratch_conv2_in,
        s->conv2_ctx, 2,
        scratch_pool1_out, n_pool1,
        KWS_B_CONV_CH, KWS_B_CONV_CH,
        tensor_conv1d_2_kernel_0_data,
        tensor_conv1d_2_bias_0_data,
        conv1d_2_bias_shift, conv1d_2_output_shift, NNOM_QTYPE_PER_AXIS,
        scratch_conv2_out);
    relu_inplace(scratch_conv2_out, n_pool1, KWS_B_CONV_CH);

    update_ctx2(s->conv2_ctx, scratch_pool1_out, n_pool1, KWS_B_CONV_CH);

    /* ── pool2 (k=4, s=4) with partial accumulation ─────────────────────── */
    int n_pool2 = pool_with_partial(
        s->pool2_buf, &s->pool2_n,
        scratch_conv2_out, n_pool1,
        KWS_B_CONV_CH, 4,
        scratch_pool2_in, scratch_pool2_out);

    if (n_pool2 == 0) return;   /* shouldn't happen; 25 frames always give ≥6 */

    /* ── conv3 (kernel 3, 48→48ch, causal) ──────────────────────────────── */
    causal_conv1d(
        scratch_conv3_in,
        s->conv3_ctx, 2,
        scratch_pool2_out, n_pool2,
        KWS_B_CONV_CH, KWS_B_CONV_CH,
        tensor_conv1d_3_kernel_0_data,
        tensor_conv1d_3_bias_0_data,
        conv1d_3_bias_shift, conv1d_3_output_shift, NNOM_QTYPE_PER_AXIS,
        scratch_conv3_out);
    relu_inplace(scratch_conv3_out, n_pool2, KWS_B_CONV_CH);

    update_ctx2(s->conv3_ctx, scratch_pool2_out, n_pool2, KWS_B_CONV_CH);

    /* ── pool3 (k=2, s=2) with partial accumulation ─────────────────────── */
    int n_pool3 = pool_with_partial(
        s->pool3_buf, &s->pool3_n,
        scratch_conv3_out, n_pool2,
        KWS_B_CONV_CH, 2,
        scratch_pool3_in, scratch_pool3_out);

    if (n_pool3 == 0) return;

    /* ── conv4 (kernel 3, 48→48ch, causal) ──────────────────────────────── */
    causal_conv1d(
        scratch_conv4_in,
        s->conv4_ctx, 2,
        scratch_pool3_out, n_pool3,
        KWS_B_CONV_CH, KWS_B_CONV_CH,
        tensor_conv1d_4_kernel_0_data,
        tensor_conv1d_4_bias_0_data,
        conv1d_4_bias_shift, conv1d_4_output_shift, NNOM_QTYPE_PER_AXIS,
        scratch_conv4_out);
    relu_inplace(scratch_conv4_out, n_pool3, KWS_B_CONV_CH);

    update_ctx2(s->conv4_ctx, scratch_pool3_out, n_pool3, KWS_B_CONV_CH);

    /* ── GAP ring: push new conv4 frames ────────────────────────────────── */
    for (int f = 0; f < n_pool3; f++) {
        const int8_t *frame = scratch_conv4_out + f * KWS_B_CONV_CH;
        int dst = s->gap_head % KWS_B_N_GAP;
        memcpy(s->gap_ring + dst * KWS_B_CONV_CH, frame, KWS_B_CONV_CH);
        s->gap_head++;
        if (s->gap_count < KWS_B_N_GAP)
            s->gap_count++;
    }

    /* ── GAP average ────────────────────────────────────────────────────── */
    int n = s->gap_count;
    if (n == 0) {
        /* can't produce logits yet — fill zeros and return */
        for (int c = 0; c < KWS_B_NUM_CLASSES; c++) logits[c] = -128;
        return;
    }

    /* Sum all valid slots in the ring (order doesn't matter for averaging) */
    for (int c = 0; c < KWS_B_CONV_CH; c++) {
        int32_t sum = 0;
        for (int i = 0; i < n; i++)
            sum += (int32_t)s->gap_ring[i * KWS_B_CONV_CH + c];
        scratch_gap_out[c] = (int8_t)(sum / n);
    }

    /* ── Dense 48→32 + ReLU ─────────────────────────────────────────────── */
    /* DENSE_BIAS_LSHIFT = {0}, DENSE_OUTPUT_RSHIFT = {6} (PER_TENSOR) */
    {
        static const nnom_qformat_param_t _d0_bs[] = DENSE_BIAS_LSHIFT;
        static const nnom_qformat_param_t _d0_os[] = DENSE_OUTPUT_RSHIFT;
        local_fully_connected_q7_opt(
            scratch_gap_out,
            tensor_dense_kernel_0_data,
            KWS_B_CONV_CH, 32,
            (uint16_t)_d0_bs[0], (uint16_t)_d0_os[0],
            tensor_dense_bias_0_data,
            scratch_dense0_out,
            scratch_fc_buf);
    }
    local_relu_q7(scratch_dense0_out, 32);

    /* ── Dense 32→11 ────────────────────────────────────────────────────── */
    /* DENSE_1_BIAS_LSHIFT = {3}, DENSE_1_OUTPUT_RSHIFT = {8} */
    {
        static const nnom_qformat_param_t _d1_bs[] = DENSE_1_BIAS_LSHIFT;
        static const nnom_qformat_param_t _d1_os[] = DENSE_1_OUTPUT_RSHIFT;
        local_fully_connected_q7_opt(
            scratch_dense0_out,
            tensor_dense_1_kernel_0_data,
            32, KWS_B_NUM_CLASSES,
            (uint16_t)_d1_bs[0], (uint16_t)_d1_os[0],
            tensor_dense_1_bias_0_data,
            logits,
            scratch_fc_buf);
    }
}

/* ── Public API ─────────────────────────────────────────────────────────── */

void kws_stream_b_reset(kws_stream_b_t *s)
{
    memset(s, 0, sizeof(*s));
}

int kws_stream_b_push(kws_stream_b_t *s, const int8_t *samples, int n,
                      int8_t logits[KWS_B_NUM_CLASSES])
{
    int produced = 0;
    int i = 0;

    while (i < n) {
        int space = KWS_B_HOP - s->samples_pending;
        int take  = (n - i < space) ? (n - i) : space;

        memcpy(s->sample_buf + s->samples_pending, samples + i, (size_t)take);
        s->samples_pending += take;
        i += take;

        if (s->samples_pending == KWS_B_HOP) {
            run_hop(s, s->sample_buf, logits);
            s->samples_pending = 0;
            produced = 1;   /* report only the last inference in this call */
        }
    }
    return produced;
}
