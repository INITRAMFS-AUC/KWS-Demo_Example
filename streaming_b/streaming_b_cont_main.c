/* streaming_b_cont_main.c — Continuous-stream test for Option B engine
 *
 * Unlike streaming_b_main.c (which resets the engine per clip), this harness
 * never resets between clips.  Each keyword is preceded by SILENCE_HOPS hops
 * of zeros to represent real-world silence between words, then 5 hops of the
 * keyword audio.  The engine state (conv contexts, pool partials, GAP ring)
 * carries over across the whole sequence.
 *
 * Why SILENCE_HOPS 5?
 *   Each hop produces ~3 new conv4 frames pushed into the 15-slot GAP ring.
 *   5 silence hops × 3 frames/hop = 15 frames → ring fully replaced by silence
 *   before the next keyword starts.  Conv context rings (2 frames each) and
 *   pool partial buffers drain in ≤1 hop.
 *
 * Scoring:
 *   PRIMARY   — prediction at the LAST hop of the keyword window (hop 4 of 5).
 *               This is what a real deployment sees: one decision per keyword
 *               window, taken when the GAP ring has the most complete picture.
 *   SECONDARY — best prediction across ALL 5 keyword hops (highest logit for
 *               any class wins).  Upper-bound accuracy estimate.
 *
 * Usage:
 *   make build_streaming_b_cont && make run_streaming_b_cont
 *
 * Expected output (vs 85.5% batch baseline):
 *   LAST_HOP_ACC  ≥ 0.83  (≤2.5% drop from causal padding)
 *   BEST_HOP_ACC  ≥ 0.84  (tighter bound)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "kws_stream_b.h"

#define NUM_CLASSES       11
#define SAMPLES_PER_CLIP  8000
#define LABEL_BATCH       128
#define HOPS_PER_CLIP     5
#define HOP               KWS_B_HOP   /* 1600 */

/* 5 hops × 1600 = 8000 silence samples — fully flushes the GAP ring */
#define SILENCE_HOPS      5

static const char * const class_names[NUM_CLASSES] = {
    "down","go","left","no","off","on","right","stop","up","yes","unknown"
};

static int argmax8(const int8_t *v, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

/* Q7 silence — 0 maps to 0.0 in the [-1,1] fixed-point range */
static const int8_t silence_buf[HOP] = {0};

int main(void)
{
    FILE   *fp;
    int8_t *buf;
    size_t  sz, seek;
    int     n_real = 0;

    fp = fopen("test_data.bin", "rb");
    if (!fp) {
        printf("ERROR: cannot open test_data.bin\n");
        printf("  Run: python3 generate_test_data.py\n");
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    sz = (size_t)ftell(fp);
    fseek(fp, 0, SEEK_SET);

    buf = (int8_t *)malloc(sz);
    if (!buf) { printf("ERROR: malloc failed\n"); fclose(fp); return 1; }
    fread(buf, 1, sz, fp);
    fclose(fp);

    { int32_t n; memcpy(&n, buf, 4); n_real = (int)n; }
    seek = 4;

    printf("Samples: %d\n", n_real);
    printf("Mode: continuous stream, %d silence hops between keywords\n",
           SILENCE_HOPS);

    /* Single engine instance — never reset between clips */
    static kws_stream_b_t stream;
    kws_stream_b_reset(&stream);

    int last_hop_correct = 0;
    int best_hop_correct = 0;
    int total = 0;

    int8_t scores[NUM_CLASSES];

    printf("RESULTS_START\n");

    while (seek < sz) {
        uint8_t labels[LABEL_BATCH];
        if (seek + LABEL_BATCH > sz) break;
        memcpy(labels, buf + seek, LABEL_BATCH);
        seek += LABEL_BATCH;

        for (int i = 0; i < LABEL_BATCH; i++) {
            if (total >= n_real || seek + SAMPLES_PER_CLIP > sz) goto done;

            const int8_t *audio = buf + seek;
            seek += SAMPLES_PER_CLIP;
            int label = (int)labels[i];

            /* 1. Push silence — flushes GAP ring and conv contexts from the
             *    previous keyword before this one starts. */
            for (int s = 0; s < SILENCE_HOPS; s++)
                kws_stream_b_push(&stream, silence_buf, HOP, scores);

            /* 2. Push keyword audio across 5 hops.
             *    Track best prediction across all hops (SECONDARY metric). */
            int best_pred = -1;
            int8_t best_logit = -128;

            for (int h = 0; h < HOPS_PER_CLIP; h++) {
                kws_stream_b_push(&stream, audio + h * HOP, HOP, scores);

                /* Find max logit and its class for best-hop scoring */
                int pred_h = argmax8(scores, NUM_CLASSES);
                if (scores[pred_h] > best_logit) {
                    best_logit = scores[pred_h];
                    best_pred  = pred_h;
                }
            }
            /* After the loop, scores[] holds the last-hop output (PRIMARY) */
            int last_pred = argmax8(scores, NUM_CLASSES);

            if (last_pred == label) last_hop_correct++;
            if (best_pred == label) best_hop_correct++;
            total++;

            printf("RESULT:%d,%d,%d\n", label, last_pred, best_pred);
        }
    }

done:
    printf("RESULTS_END\n");
    printf("TOTAL:%d\n",           total);
    printf("LAST_HOP_CORRECT:%d\n", last_hop_correct);
    printf("BEST_HOP_CORRECT:%d\n", best_hop_correct);
    if (total > 0) {
        printf("LAST_HOP_ACC:%.6f\n", (double)last_hop_correct / total);
        printf("BEST_HOP_ACC:%.6f\n", (double)best_hop_correct / total);
    }
    /* Batch baseline (no streaming, full reset per clip): 0.8554 */
    /* Isolated-clip streaming baseline (streaming_b):     0.8462 */

    free(buf);
    return 0;
}
