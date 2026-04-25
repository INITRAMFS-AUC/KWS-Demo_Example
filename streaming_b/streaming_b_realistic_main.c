/* streaming_b_realistic_main.c — Realistic streaming detector evaluation
 *
 * Simulates what the MCU detector actually does:
 *
 *   Every hop (200 ms):
 *     1. Receive 11 Q7 softmax scores from kws_stream_b_push()
 *     2. Smooth: average scores over a sliding window of SMOOTH_WIN hops
 *     3. Threshold: if max_smoothed > T AND argmax != unknown → fire
 *     4. Debounce: after any fire, suppress for DEBOUNCE_HOPS hops
 *
 * Evaluation strategy:
 *   Phase 1 — run the full stream once on Spike, store all per-hop scores.
 *   Phase 2 — replay scores in software at every threshold T, count
 *              TP / FP / FN per entry window, print P-R-F1 curve.
 *   Phase 3 — at the best T, sweep debounce values.
 *
 * Window labelling (per thresh_data.bin entry):
 *   hops 0 .. N_FILLER*5-1   → filler  (non-keyword background/unknown audio)
 *   hops N_FILLER*5 .. end   → keyword (the actual word being tested)
 *
 * Scoring:
 *   TP  — correct class fires during the keyword window
 *   FP  — any non-unknown class fires during a filler window
 *   FN  — no correct-class detection in the keyword window
 *
 * Debounce suppression that bridges from a keyword into the next filler window
 * is NOT counted as FP — the detector is correctly silent after a detection.
 *
 * Usage:
 *   python3 generate_thresh_data.py   (if thresh_data.bin not present)
 *   make build_streaming_b_realistic && make run_streaming_b_realistic
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "kws_stream_b.h"

#define NUM_CLASSES        11
#define UNKNOWN_CLASS      10
#define SAMPLES_PER_CLIP   8000
#define HOPS_PER_CLIP      5
#define HOP                KWS_B_HOP   /* 1600 samples */

#define SMOOTH_WIN         3    /* hops to average before threshold check    */
#define DEFAULT_DEBOUNCE   5    /* hops to suppress after a fire (1 second)  */
#define MAX_ENTRIES        5000
#define MAX_TOTAL_HOPS     (MAX_ENTRIES * 12)  /* generous upper bound       */

/* ── per-hop storage (Phase 1) ───────────────────────────────────────────── */
static int8_t  all_scores[MAX_TOTAL_HOPS][NUM_CLASSES]; /* Q7 softmax output */
static int8_t  hop_kw_label[MAX_TOTAL_HOPS]; /* -1 = filler hop, 0-9 = keyword hop */
static int     hop_entry[MAX_TOTAL_HOPS];    /* which entry this hop belongs to    */
static int8_t  entry_label[MAX_ENTRIES];

/* ── detector simulation ─────────────────────────────────────────────────── */
static void simulate(int n_hops, int n_entries, int T, int debounce,
                     int *out_TP, int *out_FP, int *out_FN)
{
    /* per-entry detection tracking */
    static int8_t entry_got_correct[MAX_ENTRIES];
    static int8_t entry_got_fp[MAX_ENTRIES];
    memset(entry_got_correct, 0, (size_t)n_entries);
    memset(entry_got_fp,      0, (size_t)n_entries);

    int debounce_ctr = 0;

    for (int h = 0; h < n_hops; h++) {
        /* --- smoothing: average last SMOOTH_WIN hops --- */
        int smoothed[NUM_CLASSES];
        for (int c = 0; c < NUM_CLASSES; c++) {
            int sum = 0, cnt = 0;
            for (int w = 0; w < SMOOTH_WIN && (h - w) >= 0; w++, cnt++)
                sum += (int)all_scores[h - w][c];
            smoothed[c] = sum / cnt;
        }

        /* --- threshold + debounce decision --- */
        if (debounce_ctr > 0) {
            debounce_ctr--;
            continue;   /* suppressed — no detection this hop */
        }

        int best_c = 0;
        for (int c = 1; c < NUM_CLASSES; c++)
            if (smoothed[c] > smoothed[best_c]) best_c = c;

        if (best_c == UNKNOWN_CLASS || smoothed[best_c] <= T)
            continue;   /* below threshold or unknown — no detection */

        /* --- fire --- */
        debounce_ctr = debounce;
        int ei       = hop_entry[h];
        int is_kw    = (hop_kw_label[h] >= 0);

        if (is_kw) {
            if (best_c == (int)entry_label[ei])
                entry_got_correct[ei] = 1;  /* TP: correct class in keyword window */
        } else {
            entry_got_fp[ei] = 1;           /* FP: any fire in filler window        */
        }
    }

    int TP = 0, FP = 0, FN = 0;
    for (int i = 0; i < n_entries; i++) {
        if (entry_got_correct[i]) TP++; else FN++;
        if (entry_got_fp[i])      FP++;
    }
    *out_TP = TP; *out_FP = FP; *out_FN = FN;
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
    if (!buf) { printf("ERROR: malloc\n"); fclose(fp); return 1; }
    fread(buf, 1, sz, fp);
    fclose(fp);

    int32_t n_entries_file, n_filler;
    memcpy(&n_entries_file, buf,     4);
    memcpy(&n_filler,       buf + 4, 4);
    int n_entries = (n_entries_file < MAX_ENTRIES) ? n_entries_file : MAX_ENTRIES;

    int hops_per_entry = (n_filler + 1) * HOPS_PER_CLIP;
    size_t entry_size  = (size_t)(1 + n_filler)
                       + (size_t)(n_filler + 1) * SAMPLES_PER_CLIP;
    size_t seek        = 8;

    printf("Entries: %d  Filler clips/entry: %d  Hops/entry: %d\n",
           n_entries, n_filler, hops_per_entry);
    printf("Smoothing: %d-hop sliding window\n", SMOOTH_WIN);

    /* ── Phase 1: run full stream, collect per-hop scores ────────────────── */
    static kws_stream_b_t stream;
    kws_stream_b_reset(&stream);

    int total_hops = 0;
    int8_t scores[NUM_CLASSES];

    for (int ei = 0; ei < n_entries && seek + entry_size <= sz; ei++) {
        int8_t kw_label               = buf[seek];
        const int8_t *filler_audio    = buf + seek + 1 + n_filler;
        const int8_t *keyword_audio   = filler_audio + (size_t)n_filler * SAMPLES_PER_CLIP;

        entry_label[ei] = kw_label;

        /* filler hops */
        for (int f = 0; f < n_filler; f++) {
            const int8_t *clip = filler_audio + (size_t)f * SAMPLES_PER_CLIP;
            for (int h = 0; h < HOPS_PER_CLIP; h++) {
                kws_stream_b_push(&stream, clip + h * HOP, HOP, scores);
                memcpy(all_scores[total_hops], scores, NUM_CLASSES);
                hop_kw_label[total_hops] = -1;   /* filler */
                hop_entry[total_hops]    = ei;
                total_hops++;
            }
        }

        /* keyword hops */
        for (int h = 0; h < HOPS_PER_CLIP; h++) {
            kws_stream_b_push(&stream, keyword_audio + h * HOP, HOP, scores);
            memcpy(all_scores[total_hops], scores, NUM_CLASSES);
            hop_kw_label[total_hops] = kw_label; /* keyword */
            hop_entry[total_hops]    = ei;
            total_hops++;
        }

        seek += entry_size;
    }

    printf("Total hops collected: %d\n\n", total_hops);

    /* ── Phase 2: threshold sweep at fixed debounce ──────────────────────── */
    int    best_T  = 0;
    double best_F1 = 0.0;
    double best_P  = 0.0, best_R = 0.0;

    printf("Debounce: %d hops (%d ms)\n\n", DEFAULT_DEBOUNCE,
           DEFAULT_DEBOUNCE * 200);
    printf("%-6s  %-6s  %-6s  %-6s  %-7s  %-7s  %-7s\n",
           "T", "TP", "FP", "FN", "P", "R", "F1");
    printf("--------------------------------------------------------------\n");

    for (int T = 0; T <= 127; T += 3) {
        int TP, FP, FN;
        simulate(total_hops, n_entries, T, DEFAULT_DEBOUNCE, &TP, &FP, &FN);
        double P  = (TP + FP > 0) ? (double)TP / (TP + FP) : 1.0;
        double R  = (double)TP / (TP + FN);
        double F1 = (P + R > 0.0) ? 2.0 * P * R / (P + R) : 0.0;
        printf("T=%-4d  %-6d  %-6d  %-6d  %-7.4f  %-7.4f  %-7.4f\n",
               T, TP, FP, FN, P, R, F1);
        if (F1 > best_F1) { best_F1 = F1; best_T = T; best_P = P; best_R = R; }
    }

    /* fine search */
    for (int T = best_T - 2; T <= best_T + 2; T++) {
        if (T < 0 || T > 127) continue;
        int TP, FP, FN;
        simulate(total_hops, n_entries, T, DEFAULT_DEBOUNCE, &TP, &FP, &FN);
        double P  = (TP + FP > 0) ? (double)TP / (TP + FP) : 1.0;
        double R  = (double)TP / (TP + FN);
        double F1 = (P + R > 0.0) ? 2.0 * P * R / (P + R) : 0.0;
        if (F1 > best_F1) { best_F1 = F1; best_T = T; best_P = P; best_R = R; }
    }

    printf("--------------------------------------------------------------\n\n");
    printf("BEST_T:%d\n",          best_T);
    printf("BEST_T_PROB:%.1f%%\n", best_T / 128.0 * 100.0);
    printf("BEST_P:%.4f\n",        best_P);
    printf("BEST_R:%.4f\n",        best_R);
    printf("BEST_F1:%.4f\n",       best_F1);

    /* ── Phase 3: debounce sensitivity at best T ─────────────────────────── */
    printf("\nDebounce sensitivity at T=%d (%.1f%% confidence):\n",
           best_T, best_T / 128.0 * 100.0);
    printf("%-6s  %-5s  %-6s  %-6s  %-6s  %-7s  %-7s  %-7s\n",
           "D", "ms", "TP", "FP", "FN", "P", "R", "F1");
    printf("--------------------------------------------------------------\n");

    int debounce_vals[] = {1, 2, 3, 5, 8, 10};
    int nd = (int)(sizeof(debounce_vals) / sizeof(debounce_vals[0]));
    for (int di = 0; di < nd; di++) {
        int D = debounce_vals[di];
        int TP, FP, FN;
        simulate(total_hops, n_entries, best_T, D, &TP, &FP, &FN);
        double P  = (TP + FP > 0) ? (double)TP / (TP + FP) : 1.0;
        double R  = (double)TP / (TP + FN);
        double F1 = (P + R > 0.0) ? 2.0 * P * R / (P + R) : 0.0;
        printf("D=%-4d  %-5d  %-6d  %-6d  %-6d  %-7.4f  %-7.4f  %-7.4f%s\n",
               D, D * 200, TP, FP, FN, P, R, F1,
               D == DEFAULT_DEBOUNCE ? "  ← default" : "");
    }

    free(buf);
    return 0;
}
