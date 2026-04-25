/* kws_stream_b.h — Option B streaming inference engine for strided_s16_nodil
 *
 * Maintains per-layer state rings so only new frames are computed each hop.
 * Calls NNoM math primitives directly; no model_run() call.
 *
 * Architecture:
 *   sinc    kernel 1×129 stride 1×16 valid 32ch → ~492 frames
 *   conv1   kernel 1×3  same  48ch  → 492
 *   pool1   kernel 1×4  s=4   valid → 123
 *   conv2   kernel 1×3  same  48ch  → 123
 *   pool2   kernel 1×4  s=4   valid → 30
 *   conv3   kernel 1×3  same  48ch  → 30
 *   pool3   kernel 1×2  s=2   valid → 15
 *   conv4   kernel 1×3  same  48ch  → 15
 *   GAP  →  Dense(32) ReLU  →  Dense(11)
 *
 * Padding: causal (left-only via context prepend).  Expects ≤2% accuracy drop
 * compared to same-padded batch model_run().
 *
 * Inference trigger: every KWS_B_HOP samples pushed.
 * State size: ~1.5 KB  (audio_ring 128 + conv ctxs 5×96 + pool partials + GAP ring 720)
 */
#pragma once
#include <stdint.h>

#define KWS_B_HOP           1600   /* samples per inference hop  */
#define KWS_B_SINC_CH       32     /* sinc output channels       */
#define KWS_B_CONV_CH       48     /* conv1-4 output channels    */
#define KWS_B_N_GAP         15     /* GAP temporal dimension     */
#define KWS_B_NUM_CLASSES   11

typedef struct {
    /* Audio left-context ring for sinc (kernel 129 → need 128 old samples) */
    int8_t  audio_ring[128];

    /* Conv left-context: 2 frames × channel_count bytes each.
     * conv1 input is sinc frames (32ch); conv2-4 inputs are pool outputs (48ch). */
    int8_t  conv1_ctx[2 * KWS_B_SINC_CH];   /* 64 B  */
    int8_t  conv2_ctx[2 * KWS_B_CONV_CH];   /* 96 B  */
    int8_t  conv3_ctx[2 * KWS_B_CONV_CH];   /* 96 B  */
    int8_t  conv4_ctx[2 * KWS_B_CONV_CH];   /* 96 B  */

    /* Partial pool buffers for non-divisible frame counts.
     * pool1 (k=4): 100 conv1 frames per hop — always divisible, no partial.
     * pool2 (k=4): 25 pool1 frames per hop — 25%4=1 leftover.
     * pool3 (k=2): pool2 output per hop — may have 1 leftover. */
    int8_t  pool2_buf[3 * KWS_B_CONV_CH];  /* up to 3 pending frames */
    int     pool2_n;
    int8_t  pool3_buf[1 * KWS_B_CONV_CH];  /* up to 1 pending frame  */
    int     pool3_n;

    /* GAP ring: last KWS_B_N_GAP conv4 output frames */
    int8_t  gap_ring[KWS_B_N_GAP * KWS_B_CONV_CH];  /* 720 B */
    int     gap_head;    /* next write position (0..N_GAP-1) */
    int     gap_count;   /* valid frames in ring (0..N_GAP)  */

    int     samples_pending;  /* samples buffered toward next hop */
    int8_t  sample_buf[KWS_B_HOP];
} kws_stream_b_t;

/* Zero all state — call once before first use (or between clips in test mode). */
void kws_stream_b_reset(kws_stream_b_t *s);

/* Push n Q7 int8 samples.
 * When KWS_B_HOP samples have been accumulated, runs inference and fills
 * scores[11] with Q7 softmax probabilities (value/128 = probability, 0–127).
 * Threshold example: scores[c] > 96 means class c has ≥75% confidence.
 * Returns 1 if scores were produced this call, 0 otherwise. */
int  kws_stream_b_push(kws_stream_b_t *s, const int8_t *samples, int n,
                       int8_t scores[KWS_B_NUM_CLASSES]);
