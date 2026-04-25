/* streaming_b_main.c — Spike test harness for Option B streaming engine
 *
 * Reads test_data.bin (same format as strided_s16_nodil_main.c).
 * For each 8000-sample clip, pushes 5 hops of 1600 samples and
 * uses the final-hop prediction.
 *
 * Batch baseline (85.5%) is from build/strided_s16_nodil — no rerun needed.
 *
 * Usage:
 *   make build_streaming_b && make run_streaming_b
 *
 * Expected output:
 *   STREAMING_ACC: ≥0.83  (≤2.5% drop from causal padding vs 85.5% batch)
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

    static kws_stream_b_t stream;

    int stream_correct = 0, total = 0;
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

            /* Push 5 hops of 1600 samples; use the final-hop scores. */
            kws_stream_b_reset(&stream);
            for (int h = 0; h < HOPS_PER_CLIP; h++)
                kws_stream_b_push(&stream, audio + h * HOP, HOP, scores);

            int pred = argmax8(scores, NUM_CLASSES);
            if (pred == label) stream_correct++;
            total++;

            printf("RESULT:%d,%d\n", label, pred);
        }
    }

done:
    printf("RESULTS_END\n");
    printf("TOTAL:%d\n",          total);
    printf("CORRECT:%d\n",        stream_correct);
    if (total > 0)
        printf("ACCURACY:%.6f\n", (double)stream_correct / total);
    /* Batch baseline for comparison: build/strided_s16_nodil → 0.855 */

    free(buf);
    return 0;
}
