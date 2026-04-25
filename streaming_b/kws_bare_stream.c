/**
 * kws_bare_stream.c — Bare-metal streaming KWS firmware for KWS-SoC
 *
 * Drop-in replacement for strided_s16_nodil/kws_bare.c that uses the Option B
 * streaming engine instead of NNoM model_run().  Key differences:
 *
 *   Old (kws_bare.c):
 *     - Accumulates 8000 samples (1 second), then calls model_run()
 *     - Needs nnom_static_buf (~48 KB) + nnom_input_data[8000] (8 KB)
 *     - 1-second latency before first result
 *
 *   This firmware:
 *     - Accumulates 1600 samples (200 ms hop), then calls kws_stream_b_push()
 *     - Needs only kws_stream_b_t state (~3 KB) + engine scratch (~19 KB)
 *     - 200 ms hop latency; GAP ring filled after 5 hops (1 second warmup)
 *     - Threshold + debounce detector — no output during silence/unknown
 *
 * DETECTOR LOGIC (per hop):
 *   1. kws_stream_b_push() → 11 Q7 softmax probabilities
 *   2. Smooth: average scores over last SMOOTH_WIN hops
 *   3. Threshold: if max_smoothed > KWS_DETECT_THRESHOLD AND class != unknown
 *   4. Debounce: suppress for KWS_DEBOUNCE_HOPS hops after any detection
 *
 * Tuned from Spike evaluation (streaming_b_realistic):
 *   KWS_DETECT_THRESHOLD = 60   (46.9% confidence → F1=0.60)
 *   KWS_DEBOUNCE_HOPS    = 5    (1000 ms)
 *   SMOOTH_WIN           = 3    (3-hop sliding window)
 *
 * REGISTER MAP, UART, I2S: identical to kws_bare.c.
 *
 * MEMORY (vs kws_bare.c on 128 KB SoC SRAM):
 *   kws_bare.c:        nnom_static_buf ~48 KB + ring_buf 8 KB  = ~56 KB
 *   kws_bare_stream.c: engine state    ~3 KB  + hop_buf 1.6 KB = ~22 KB total
 */

#include <stdint.h>
#include <string.h>
#include "kws_stream_b.h"

/* ── UART (identical to kws_bare.c) ─────────────────────────────────────── */

#define UART_BASE            0x40004000UL
#define UART_CSR_OFFS        0x00
#define UART_DIV_OFFS        0x04
#define UART_FSTAT_OFFS      0x08
#define UART_TX_OFFS         0x0C
#define UART_REG(off)        (*(volatile uint32_t *)(UART_BASE + (off)))
#define UART_CSR_EN_MASK     (1u << 0)
#define UART_FSTAT_TXFULL    (1u << 8)

#ifdef CLK_MHZ
#define SYS_CLK_HZ   ((uint32_t)((CLK_MHZ) * 1000000UL))
#define UART_DIV_VAL (((2u * SYS_CLK_HZ) + (UART_BAUD_RATE / 2u)) / UART_BAUD_RATE)
#endif

static void uart_init(void) {
#ifdef UART_DIV_VAL
    UART_REG(UART_DIV_OFFS) = UART_DIV_VAL;
#endif
    UART_REG(UART_CSR_OFFS) |= UART_CSR_EN_MASK;
}
static void uart_putc(char c) {
    while (UART_REG(UART_FSTAT_OFFS) & UART_FSTAT_TXFULL);
    UART_REG(UART_TX_OFFS) = (uint32_t)(uint8_t)c;
}
static void uart_puts(const char *s) { while (*s) uart_putc(*s++); }
static void uart_putdec(int v) {
    char buf[12]; int i = 0;
    if (v < 0) { uart_putc('-'); v = -v; }
    if (v == 0) { uart_putc('0'); return; }
    while (v > 0) { buf[i++] = (char)('0' + (v % 10)); v /= 10; }
    while (i > 0) uart_putc(buf[--i]);
}
static void uart_puthex(uint32_t v) {
    static const char h[] = "0123456789abcdef";
    uart_putc('0'); uart_putc('x');
    for (int s = 28; s >= 0; s -= 4) uart_putc(h[(v >> s) & 0xfu]);
}

/* ── I2S (identical register layout to kws_bare.c) ──────────────────────── */

typedef struct {
    volatile uint32_t id;
    volatile uint32_t conf;
    volatile uint32_t fifo;
} i2s_hw_t;

#define I2S_BASE_ADDR  0x40008000UL
#define I2S            ((i2s_hw_t *)I2S_BASE_ADDR)

/* Real SoC: 8 samples per ISR (hardware FIFO depth).
 * Spike simulation: override to 8000 — one ISR delivers a full 1-second clip,
 * matching the plugin's I2S_FIFO_DEPTH so the IRQ deasserts after the ISR
 * drains the FIFO completely (avoids livelock from premature reassert).
 * kws_stream_b_push handles n > KWS_B_HOP by running multiple hops. */
#ifndef I2S_FIFO_DEPTH
#define I2S_FIFO_DEPTH  8
#endif

/* HOP_BUF_SZ: size of the ISR accumulation buffer.
 * Spike  (I2S_FIFO_DEPTH=8000): collect a full 1-second clip in one ISR.
 * SoC    (I2S_FIFO_DEPTH=8):    accumulate 1600 samples over 200 ISRs. */
#define HOP_BUF_SZ  ((I2S_FIFO_DEPTH) >= (KWS_B_HOP) ? (I2S_FIFO_DEPTH) : (KWS_B_HOP))

#ifndef I2S_CLK_DIV
#define I2S_CLK_DIV  70
#endif

static void i2s_init(uint32_t clk_div) {
    I2S->conf = ((clk_div & 0xFFFFFFu) << 8) | (1u << 4);
}

/* ── RISC-V CSR helpers (identical to kws_bare.c) ───────────────────────── */

static inline void csr_set_mtvec(void (*handler)(void)) {
    asm volatile ("csrw mtvec, %0" :: "r"((uintptr_t)handler));
}
static inline void csr_enable_meie(void) {
    asm volatile ("csrs mie, %0" :: "r"(1u << 11));
}
static inline void csr_enable_mie(void) {
    asm volatile ("csrsi mstatus, 8");
}

/* ── PLIC (Spike simulation only) ────────────────────────────────────────── */
/* The real KWS-SoC uses Hazard3 with direct IRQ wiring (no PLIC needed).
 * Spike uses a SiFive-compatible PLIC at 0x0C000000 that must be configured
 * before machine external interrupts can fire.
 *
 * SiFive PLIC register layout:
 *   base + src*4          : interrupt source priority (0 = disabled)
 *   base + 0x2000 + c*0x80: interrupt enable for context c (one bit per src)
 *   base + 0x200000+c*0x1000 : priority threshold for context c
 *   base + 0x200004+c*0x1000 : claim / complete register for context c
 *
 * Spike context numbering (single hart): 0 = S-mode, 1 = M-mode. */
#ifdef SPIKE_PLIC
#define PLIC_BASE          0x0C000000UL
#define PLIC_REG(off)      (*(volatile uint32_t *)(PLIC_BASE + (off)))
#define I2S_PLIC_IRQ       2
#define PLIC_CTX_MMODE     1   /* M-mode context for hart 0 */

static void plic_init(void) {
    PLIC_REG(I2S_PLIC_IRQ * 4)                              = 1;  /* priority */
    PLIC_REG(0x2000 + PLIC_CTX_MMODE * 0x80)               |= (1u << I2S_PLIC_IRQ);
    PLIC_REG(0x200000 + (uint32_t)PLIC_CTX_MMODE * 0x1000)  = 0;  /* threshold=0 */
}
static uint32_t plic_claim(void) {
    return PLIC_REG(0x200004 + (uint32_t)PLIC_CTX_MMODE * 0x1000);
}
static void plic_complete(uint32_t irq_id) {
    PLIC_REG(0x200004 + (uint32_t)PLIC_CTX_MMODE * 0x1000) = irq_id;
}
#else
/* Real SoC: no PLIC — IRQ deasserts automatically when FIFO drains */
#define I2S_PLIC_IRQ       2   /* unused on real SoC; defined for plic_complete call */
static inline void plic_init(void)             {}
static inline uint32_t plic_claim(void)        { return I2S_PLIC_IRQ; }
static inline void plic_complete(uint32_t id)  { (void)id; }
#endif

/* ── Detector tuning ─────────────────────────────────────────────────────── */

/* Q7 confidence threshold: fire if scores[best_class] > this value.
 * 60/128 = 46.9% — from streaming_b_realistic Spike evaluation (F1=0.60).
 * Raise to reduce false alarms (costs recall); lower to catch more keywords. */
#ifndef KWS_DETECT_THRESHOLD
#define KWS_DETECT_THRESHOLD  60
#endif

/* Hops to suppress after a detection fires (1 hop = 200 ms).
 * 5 hops = 1000 ms — prevents repeated fires on a single keyword. */
#ifndef KWS_DEBOUNCE_HOPS
#define KWS_DEBOUNCE_HOPS  5
#endif

/* Hops to average scores over before threshold comparison.
 * Smoothing kills single-hop noise spikes. */
#ifndef SMOOTH_WIN
#define SMOOTH_WIN  3
#endif

/* ── Hop ring buffer (filled by ISR) ─────────────────────────────────────── */

static volatile int8_t  hop_buf[HOP_BUF_SZ];
static volatile int     hop_pos   = 0;
static volatile int     hop_ready = 0;

/* Safe copy — ISR writes hop_buf; main reads safe_hop after hop_ready set */
static int8_t safe_hop[HOP_BUF_SZ];

/* ── Streaming engine state ──────────────────────────────────────────────── */

static kws_stream_b_t stream;

/* ── Score smoothing ring ────────────────────────────────────────────────── */

static int8_t  score_history[SMOOTH_WIN][KWS_B_NUM_CLASSES];
static int     score_hist_head = 0;   /* next write slot */
static int     score_hist_cnt  = 0;   /* valid slots (0..SMOOTH_WIN)  */

static int8_t  smooth_scores[KWS_B_NUM_CLASSES];

static void update_smooth(const int8_t *new_scores)
{
    /* Write new scores into ring */
    memcpy(score_history[score_hist_head], new_scores, KWS_B_NUM_CLASSES);
    score_hist_head = (score_hist_head + 1) % SMOOTH_WIN;
    if (score_hist_cnt < SMOOTH_WIN) score_hist_cnt++;

    /* Average over valid history */
    for (int c = 0; c < KWS_B_NUM_CLASSES; c++) {
        int sum = 0;
        for (int w = 0; w < score_hist_cnt; w++)
            sum += (int)score_history[w][c];
        smooth_scores[c] = (int8_t)(sum / score_hist_cnt);
    }
}

/* ── Class names ─────────────────────────────────────────────────────────── */

#define NUM_CLASSES  KWS_B_NUM_CLASSES
static const char * const class_names[NUM_CLASSES] = {
    "down","go","left","no","off","on","right","stop","up","yes","unknown"
};

/* ── Trap handler ────────────────────────────────────────────────────────── */

void __attribute__((interrupt("machine"))) kws_trap_handler(void)
{
    uint32_t mcause;
    asm volatile ("csrr %0, mcause" : "=r"(mcause));

    if (!(mcause & 0x80000000u)) {
        uint32_t mepc, mtval;
        asm volatile ("csrr %0, mepc"  : "=r"(mepc));
        asm volatile ("csrr %0, mtval" : "=r"(mtval));
        uart_puts("EXCEPTION cause="); uart_putdec((int)(mcause & 0x7fffffffu));
        uart_puts(" mepc=");           uart_puthex(mepc);
        uart_puts(" mtval=");          uart_puthex(mtval);
        uart_puts("\r\n");
        while (1);
    }

    /* PLIC claim (Spike): acknowledge interrupt before draining FIFO */
    uint32_t irq_id = plic_claim();

    /* I2S external interrupt — drain FIFO into hop buffer */
    for (int i = 0; i < I2S_FIFO_DEPTH; i++) {
        int32_t raw = (int32_t)I2S->fifo;
        int8_t  q7  = (int8_t)(raw >> 16);
        if (hop_pos < HOP_BUF_SZ)
            hop_buf[hop_pos++] = q7;
    }

    if (hop_pos >= HOP_BUF_SZ)
        hop_ready = 1;

    /* PLIC complete (Spike): signal interrupt handling is done */
    plic_complete(irq_id);
}

/* ── main ────────────────────────────────────────────────────────────────── */

int main(void)
{
    uart_init();
    uart_puts("KWS streaming firmware (Option B, hop=200ms)\r\n");
    uart_puts("Threshold:");  uart_putdec(KWS_DETECT_THRESHOLD);
    uart_puts(" Debounce:");  uart_putdec(KWS_DEBOUNCE_HOPS);
    uart_puts(" hops\r\n");

    /* Engine and smoothing init */
    kws_stream_b_reset(&stream);
    memset(score_history, 0, sizeof(score_history));

    /* Interrupt setup */
    csr_set_mtvec(kws_trap_handler);
    csr_enable_meie();
    plic_init();           /* no-op on real SoC; required for Spike PLIC */
    i2s_init(I2S_CLK_DIV);
    uart_puts("I2S started\r\n");
    csr_enable_mie();

    int debounce_ctr = 0;

    while (1) {
        if (!hop_ready)
            asm volatile ("wfi");
        if (!hop_ready)
            continue;

        /* Snapshot hop buffer; let ISR start refilling immediately */
        memcpy(safe_hop, (const void *)hop_buf, HOP_BUF_SZ);
        hop_pos   = 0;
        hop_ready = 0;

        /* Run streaming inference — kws_stream_b_push handles HOP_BUF_SZ > KWS_B_HOP
         * by processing multiple internal hops; scores holds the last hop's output */
        int8_t scores[NUM_CLASSES];
        kws_stream_b_push(&stream, safe_hop, HOP_BUF_SZ, scores);

        /* Update smoothing ring */
        update_smooth(scores);

        /* Debounce countdown */
        if (debounce_ctr > 0) {
            debounce_ctr--;
            continue;
        }

        /* Find best class in smoothed scores */
        int    best_c     = 0;
        int8_t best_score = smooth_scores[0];
        for (int c = 1; c < NUM_CLASSES; c++) {
            if (smooth_scores[c] > best_score) {
                best_score = smooth_scores[c];
                best_c     = c;
            }
        }

        /* Fire if above threshold and not unknown */
        if (best_c != (NUM_CLASSES - 1) && best_score > KWS_DETECT_THRESHOLD) {
            uart_puts("DETECT:");
            uart_putdec(best_c);
            uart_putc(',');
            uart_puts(class_names[best_c]);
            uart_putc(',');
            uart_putdec((int)best_score);   /* Q7 confidence for debugging */
            uart_puts("\r\n");

            debounce_ctr = KWS_DEBOUNCE_HOPS;
        }
    }

    return 0;
}
