/*
 * kws_streaming.h — Sliding-window streaming KWS inference API
 *
 * Replaces the fill-and-reset ring buffer in kws_bare.c with a
 * sliding window: inference fires every HOP_SAMPLES, the window
 * content shifts by HOP_SAMPLES rather than resetting to zero.
 * Keywords that straddle a 1-second boundary are still detected.
 *
 * Usage:
 *   kws_stream_t s;
 *   kws_stream_init(&s, model);
 *
 *   // In ISR or main loop, push raw Q7 samples:
 *   int cls = kws_stream_push(&s, sample);
 *   if (cls >= 0)
 *       uart_puts(class_names[cls]);
 */

#ifndef KWS_STREAMING_H
#define KWS_STREAMING_H

#include <stdint.h>
#include "nnom.h"

/* ── Tunable parameters ──────────────────────────────────────────────────── */

#ifndef KWS_WINDOW_SAMPLES
#define KWS_WINDOW_SAMPLES  8000   /* 1 s @ 8 kHz — must match model input */
#endif

#ifndef KWS_HOP_SAMPLES
#define KWS_HOP_SAMPLES     1600   /* 200 ms — 5 inferences per second */
#endif

#ifndef KWS_NUM_CLASSES
#define KWS_NUM_CLASSES     11
#endif

/* Minimum output score (int8 softmax, 127 = 1.0) to count as a detection.
 * ~0.5 in Q7 = 64.  Tune against FAR in evaluate_streaming.py first. */
#ifndef KWS_THRESHOLD
#define KWS_THRESHOLD       64
#endif

/* Suppress repeated detections of the same class within this many hops.
 * At 5 hops/sec, 5 hops ≈ 1 second cooldown. */
#ifndef KWS_COOLDOWN_HOPS
#define KWS_COOLDOWN_HOPS   5
#endif

/* ── State struct ────────────────────────────────────────────────────────── */

typedef struct {
    int8_t          buf[KWS_WINDOW_SAMPLES]; /* circular audio window        */
    int             write_pos;               /* next write index (mod WINDOW) */
    int             samples_since_infer;     /* counts up to HOP_SAMPLES     */
    int             cooldown;                /* hops remaining before re-fire */
    int             last_class;             /* class detected in last fire   */
    nnom_model_t   *model;
    int8_t         *input_buf;              /* = nnom_input_data  (8000 B)  */
    int8_t         *output_buf;             /* = nnom_output_data (11 B)    */
} kws_stream_t;

/* ── API ─────────────────────────────────────────────────────────────────── */

/*
 * kws_stream_init — zero the state and bind the NNoM model.
 * Pass nnom_input_data and nnom_output_data from the weights header so this
 * module stays independent of the weights translation unit.
 * Call once after nnom_model_create().
 */
void kws_stream_init(kws_stream_t *s, nnom_model_t *model,
                     int8_t *input_buf, int8_t *output_buf);

/*
 * kws_stream_push — add one Q7 int8 sample to the window.
 *
 * Returns the detected class index (0–10) when an inference fires and
 * a score exceeds KWS_THRESHOLD, otherwise returns -1.
 *
 * Safe to call from an ISR — all state is in *s, no heap allocation.
 */
int kws_stream_push(kws_stream_t *s, int8_t sample);

/*
 * kws_stream_push_block — add n samples at once (e.g. from I2S FIFO).
 * Returns the detected class from the last inference that fired, or -1.
 */
int kws_stream_push_block(kws_stream_t *s, const int8_t *samples, int n);

#endif /* KWS_STREAMING_H */
