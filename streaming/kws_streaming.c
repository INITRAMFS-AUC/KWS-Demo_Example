/*
 * kws_streaming.c — Sliding-window streaming KWS inference
 *
 * See kws_streaming.h for API docs and parameter tuning.
 *
 * Window layout (circular buffer, write_pos advances modulo WINDOW):
 *
 *   oldest sample ←─────────────────────────── newest sample
 *   [ ... | write_pos | write_pos+1 | ... | write_pos-1 ]
 *
 * At inference time, samples are linearised into nnom_input_data[]
 * starting from write_pos (oldest), wrapping to write_pos-1 (newest).
 * This avoids a second ring buffer: the NNoM input tensor IS the
 * linearised window, written once per hop.
 */

#include <string.h>
#include "kws_streaming.h"

/* ── helpers ──────────────────────────────────────────────────────────────── */

/* Linearise the circular window into nnom_input_data[].
 * write_pos is the index of the OLDEST sample (next to be overwritten). */
static void _copy_window(const kws_stream_t *s)
{
    int wp  = s->write_pos;
    int rem = KWS_WINDOW_SAMPLES - wp;      /* samples from wp to end of buf */
    memcpy(s->input_buf,       s->buf + wp, (size_t)rem);
    memcpy(s->input_buf + rem, s->buf,      (size_t)wp);
}

/* Run one inference, return detected class or -1. */
static int _infer(kws_stream_t *s)
{
    _copy_window(s);
    model_run(s->model);

    /* Argmax */
    int    best_cls   = 0;
    int8_t best_score = s->output_buf[0];
    for (int i = 1; i < KWS_NUM_CLASSES; i++) {
        if (s->output_buf[i] > best_score) {
            best_score = s->output_buf[i];
            best_cls   = i;
        }
    }

    if (best_score < KWS_THRESHOLD)
        return -1;

    /* Cooldown: suppress re-fire of the same class within N hops */
    if (s->cooldown > 0 && best_cls == s->last_class) {
        s->cooldown--;
        return -1;
    }

    s->last_class = best_cls;
    s->cooldown   = KWS_COOLDOWN_HOPS;
    return best_cls;
}

/* ── public API ──────────────────────────────────────────────────────────── */

void kws_stream_init(kws_stream_t *s, nnom_model_t *model,
                     int8_t *input_buf, int8_t *output_buf)
{
    memset(s, 0, sizeof(*s));
    s->model      = model;
    s->input_buf  = input_buf;
    s->output_buf = output_buf;
}

int kws_stream_push(kws_stream_t *s, int8_t sample)
{
    s->buf[s->write_pos] = sample;
    s->write_pos = (s->write_pos + 1) % KWS_WINDOW_SAMPLES;
    s->samples_since_infer++;

    if (s->cooldown > 0)
        s->cooldown--;   /* count down even between hops */

    if (s->samples_since_infer >= KWS_HOP_SAMPLES) {
        s->samples_since_infer = 0;
        return _infer(s);
    }
    return -1;
}

int kws_stream_push_block(kws_stream_t *s, const int8_t *samples, int n)
{
    int result = -1;
    for (int i = 0; i < n; i++) {
        int r = kws_stream_push(s, samples[i]);
        if (r >= 0)
            result = r;
    }
    return result;
}
