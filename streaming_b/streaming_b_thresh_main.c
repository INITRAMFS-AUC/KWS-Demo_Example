/* streaming_b_thresh_main.c — Threshold sweep with real background filler
 *
 * Reads thresh_data.bin (produced by generate_thresh_data.py).
 * For each keyword entry the filler slot contains real audio drawn from
 * background noise recordings and unknown-class clips — NOT zeros.
 *
 * File format (thresh_data.bin):
 *   int32  n_keywords
 *   int32  n_filler_per_kw          (always 1 in current generator)
 *   per entry:
 *     int8  keyword_label           0–9
 *     int8  filler_label            10 (always)
 *     int8  filler_audio[8000]      background / unknown clip
 *     int8  keyword_audio[8000]     keyword clip
 *
 * Engine is reset once at start and never again.  Each entry pushes:
 *   1. filler_audio as 5 hops → scores during these hops tracked for FP
 *   2. keyword_audio as 5 hops → correct-class score tracked for TP
 *
 * Threshold sweep: for T in [-128..127], window-level:
 *   TP  keyword detected  — max scores[label] across 5 keyword hops  > T
 *   FN  keyword missed    — max scores[label] across 5 keyword hops  <= T
 *   FP  false alarm       — max non-unknown score across 5 filler hops > T
 *
 * Usage:
 *   python3 generate_thresh_data.py
 *   make build_streaming_b_thresh && make run_streaming_b_thresh
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "kws_stream_b.h"

#define NUM_CLASSES       11
#define UNKNOWN_CLASS     10
#define SAMPLES_PER_CLIP  8000
#define HOPS_PER_CLIP     5
#define HOP               KWS_B_HOP   /* 1600 */

#define MAX_CLIPS         5000

/* Per-keyword summary scores collected during Phase 1 */
static int8_t  keyword_score[MAX_CLIPS]; /* best scores[label] across keyword hops */
static int8_t  filler_max[MAX_CLIPS];    /* best non-unknown score across filler hops */

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
    size_t  sz;

    fp = fopen("thresh_data.bin", "rb");
    if (!fp) {
        printf("ERROR: cannot open thresh_data.bin\n");
        printf("  Run: python3 generate_thresh_data.py\n");
        return 1;
    }
    fseek(fp, 0, SEEK_END); sz = (size_t)ftell(fp); fseek(fp, 0, SEEK_SET);
    buf = (int8_t *)malloc(sz);
    if (!buf) { printf("ERROR: malloc failed\n"); fclose(fp); return 1; }
    fread(buf, 1, sz, fp);
    fclose(fp);

    int32_t n_keywords, n_filler;
    memcpy(&n_keywords, buf,     4);
    memcpy(&n_filler,   buf + 4, 4);

    printf("Keywords: %d  Filler per keyword: %d\n", n_keywords, n_filler);

    /* Entry layout: 1 (kw label) + n_filler (filler labels) +
     *               n_filler*8000 (filler audio) + 8000 (keyword audio) */
    size_t entry_size = (size_t)(1 + n_filler) + (size_t)(n_filler + 1) * SAMPLES_PER_CLIP;
    size_t seek       = 8;

    static kws_stream_b_t stream;
    kws_stream_b_reset(&stream);

    int total = 0;
    int8_t scores[NUM_CLASSES];

    /* ── Phase 1: collect per-keyword scores ─────────────────────────────── */
    while (total < n_keywords && total < MAX_CLIPS && seek + entry_size <= sz) {
        int8_t kw_label     = buf[seek];
        /* filler labels follow, then filler audio, then keyword audio */
        const int8_t *filler_audio  = buf + seek + 1 + n_filler;
        const int8_t *keyword_audio = filler_audio + (size_t)n_filler * SAMPLES_PER_CLIP;

        /* Filler hops — push all to flush the GAP ring, score ONLY the last hop.
         * Hops 1-4 still contain residue from the previous keyword (GAP ring
         * takes 5 hops to fully replace); only the 5th hop has clean state. */
        for (int f = 0; f < n_filler; f++) {
            const int8_t *clip = filler_audio + (size_t)f * SAMPLES_PER_CLIP;
            for (int h = 0; h < HOPS_PER_CLIP; h++)
                kws_stream_b_push(&stream, clip + h * HOP, HOP, scores);
        }
        /* scores[] now holds the last filler hop output — GAP ring fully flushed */
        int8_t fil_max = -128;
        for (int c = 0; c < NUM_CLASSES; c++) {
            if (c == UNKNOWN_CLASS) continue;
            if (scores[c] > fil_max) fil_max = scores[c];
        }
        filler_max[total] = fil_max;

        /* Keyword hops — push all, score ONLY the last hop.
         * Matches streaming_b_cont evaluation (last-hop decision). */
        for (int h = 0; h < HOPS_PER_CLIP - 1; h++)
            kws_stream_b_push(&stream, keyword_audio + h * HOP, HOP, scores);
        kws_stream_b_push(&stream, keyword_audio + (HOPS_PER_CLIP-1) * HOP, HOP, scores);
        keyword_score[total] = scores[(int)kw_label];

        seek += entry_size;
        total++;
    }

    printf("Clips processed: %d\n\n", total);

    /* ── Phase 2: threshold sweep ─────────────────────────────────────────── */
    printf("%-6s  %-6s  %-6s  %-6s  %-7s  %-7s  %-7s\n",
           "T", "TP", "FP", "FN", "P", "R", "F1");
    printf("--------------------------------------------------------------\n");

    int    best_T  = -128;
    double best_F1 = 0.0;
    double best_P  = 0.0, best_R = 0.0;

    for (int T = -128; T <= 127; T += 4) {
        int TP = 0, FP = 0, FN = 0;
        for (int i = 0; i < total; i++) {
            if (keyword_score[i] > T) TP++; else FN++;
            if (filler_max[i]    > T) FP++;
        }
        double P  = (TP + FP > 0) ? (double)TP / (TP + FP) : 1.0;
        double R  = (double)TP / (TP + FN);
        double F1 = (P + R > 0.0) ? 2.0 * P * R / (P + R) : 0.0;

        printf("T=%-4d  %-6d  %-6d  %-6d  %-7.4f  %-7.4f  %-7.4f\n",
               T, TP, FP, FN, P, R, F1);

        if (F1 > best_F1) { best_F1 = F1; best_T = T; best_P = P; best_R = R; }
    }

    /* Fine search ±3 around best T */
    for (int T = best_T - 3; T <= best_T + 3; T++) {
        if (T < -128 || T > 127) continue;
        int TP = 0, FP = 0, FN = 0;
        for (int i = 0; i < total; i++) {
            if (keyword_score[i] > T) TP++; else FN++;
            if (filler_max[i]    > T) FP++;
        }
        double P  = (TP + FP > 0) ? (double)TP / (TP + FP) : 1.0;
        double R  = (double)TP / (TP + FN);
        double F1 = (P + R > 0.0) ? 2.0 * P * R / (P + R) : 0.0;
        if (F1 > best_F1) { best_F1 = F1; best_T = T; best_P = P; best_R = R; }
    }

    printf("--------------------------------------------------------------\n\n");
    printf("BEST_T:%d\n",    best_T);
    printf("BEST_P:%.4f\n",  best_P);
    printf("BEST_R:%.4f\n",  best_R);
    printf("BEST_F1:%.4f\n", best_F1);
    printf("BEST_T_PROB:%.1f%%\n", (best_T + 128) / 255.0 * 100.0);

    free(buf);
    return 0;
}
