/**
 * kws_bare.c — Bare-metal KWS inference firmware for KWS-SoC
 *
 * Model:    mel_compact_4blk_ch36 (seed 789)
 * Accuracy: 88.7% NNoM int8
 * Params:   15,851  (15.0 KB weights, 28.7 KB total flash)
 *
 * Frontend: Conv2D(1×65, stride=16, 16 filters) initialised as mel bandpass
 * filters and trained end-to-end — takes raw Q7 audio, same as strided model.
 *
 * For full pipeline, register map, and build instructions see the parent
 * model's kws_bare.c (strided_s16_nodil/kws_bare.c). This file is identical
 * except for the weights include.
 *
 * CLASSES (index order matches mel_compact_4blk_ch36_weights.h)
 * ----------------------------------------
 *  0=down 1=go 2=left 3=no 4=off 5=on 6=right 7=stop 8=up 9=yes 10=unknown
 */

#include <stdint.h>
#include <string.h>
#include "mel_compact_4blk_ch36_weights.h"

/* ── UART (KWS-SoC uart_mini at 0x40004000) ─────────────────────────────── */

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
    UART_REG(UART_DIV_OFFS)  = UART_DIV_VAL;
#endif
    UART_REG(UART_CSR_OFFS) |= UART_CSR_EN_MASK;
}

static void uart_putc(char c) {
    while (UART_REG(UART_FSTAT_OFFS) & UART_FSTAT_TXFULL);
    UART_REG(UART_TX_OFFS) = (uint32_t)(uint8_t)c;
}

static void uart_puts(const char *s) {
    while (*s) uart_putc(*s++);
}

static void uart_putdec(int v) {
    char buf[12];
    int i = 0;
    if (v < 0) { uart_putc('-'); v = -v; }
    if (v == 0) { uart_putc('0'); return; }
    while (v > 0) { buf[i++] = (char)('0' + (v % 10)); v /= 10; }
    while (i > 0) uart_putc(buf[--i]);
}

static void uart_puthex(uint32_t v) {
    static const char h[] = "0123456789abcdef";
    uart_putc('0'); uart_putc('x');
    for (int s = 28; s >= 0; s -= 4)
        uart_putc(h[(v >> s) & 0xfu]);
}

/* ── I2S (KWS-SoC apb_i2s_receiver at 0x40008000) ──────────────────────── */

typedef struct {
    volatile uint32_t id;
    volatile uint32_t conf;
    volatile uint32_t fifo;
} i2s_hw_t;

#define I2S_BASE_ADDR    0x40008000UL
#define I2S              ((i2s_hw_t *)I2S_BASE_ADDR)
#ifndef I2S_FIFO_DEPTH
#define I2S_FIFO_DEPTH   8
#endif

#ifndef I2S_CLK_DIV
#define I2S_CLK_DIV  70
#endif

static void i2s_init(uint32_t clk_div) {
    I2S->conf = ((clk_div & 0xFFFFFFu) << 8) | (1u << 4);
}

/* ── RISC-V CSR helpers ──────────────────────────────────────────────────── */

static inline void csr_set_mtvec(void (*handler)(void)) {
    asm volatile ("csrw mtvec, %0" :: "r"((uintptr_t)handler));
}
static inline void csr_enable_meie(void) {
    asm volatile ("csrs mie, %0" :: "r"(1u << 11));
}
static inline void csr_enable_mie(void) {
    asm volatile ("csrsi mstatus, 8");
}

/* ── NNoM static memory buffer ───────────────────────────────────────────── */
/* Peak measured on Spike: 32,960 bytes. 36 KB gives ~3.8 KB headroom.
 * Compare to strided_s16_nodil which needs 52 KB — this model saves 16 KB. */
#ifndef NNOM_STATIC_BUF_KB
#define NNOM_STATIC_BUF_KB 36
#endif
#ifdef NNOM_USING_STATIC_MEMORY
static uint8_t nnom_static_buf[NNOM_STATIC_BUF_KB * 1024];
#endif

/* ── Audio ring buffer ───────────────────────────────────────────────────── */
#define SAMPLES_PER_CLIP  8000
#define NUM_CLASSES       11

static volatile int8_t ring_buf[SAMPLES_PER_CLIP];
static volatile int    ring_pos   = 0;
static volatile int    ring_ready = 0;

/* ── Class names ─────────────────────────────────────────────────────────── */
static const char * const class_names[NUM_CLASSES] = {
    "down", "go", "left", "no", "off", "on",
    "right", "stop", "up", "yes", "unknown"
};

/* ── I2S / Machine External Interrupt handler ────────────────────────────── */

void __attribute__((interrupt("machine"))) kws_trap_handler(void) {
    uint32_t mcause, mepc, mtval;
    asm volatile ("csrr %0, mcause" : "=r"(mcause));

    if (!(mcause & 0x80000000u)) {
        asm volatile ("csrr %0, mepc"  : "=r"(mepc));
        asm volatile ("csrr %0, mtval" : "=r"(mtval));
        uart_puts("EXCEPTION cause=");
        uart_putdec((int)(mcause & 0x7fffffffu));
        uart_puts(" mepc=");
        uart_puthex(mepc);
        uart_puts(" mtval=");
        uart_puthex(mtval);
        uart_puts("\r\n");
        while (1);
    }

    for (int i = 0; i < I2S_FIFO_DEPTH; i++) {
        int32_t raw = (int32_t)I2S->fifo;
        int8_t  q7  = (int8_t)(raw >> 16);
        if (ring_pos < SAMPLES_PER_CLIP)
            ring_buf[ring_pos++] = q7;
    }

    if (ring_pos >= SAMPLES_PER_CLIP)
        ring_ready = 1;
}

/* ── main ────────────────────────────────────────────────────────────────── */

int main(void) {
    uart_init();
    uart_puts("KWS bare-metal firmware (mel_compact_4blk_ch36, NNoM int8)\r\n");

#ifdef NNOM_USING_STATIC_MEMORY
    nnom_set_static_buf(nnom_static_buf, sizeof(nnom_static_buf));
#endif

    nnom_model_t *model = nnom_model_create();
    if (!model) {
        uart_puts("ERROR: nnom_model_create failed\r\n");
        while (1);
    }

    if (model->head == NULL || model->head->in->tensor->p_data == NULL) {
        uart_puts("ERROR: model_compile failed — nnom_static_buf too small\r\n");
        while (1);
    }
    uart_puts("Model loaded\r\n");

    csr_set_mtvec(kws_trap_handler);
    csr_enable_meie();
    i2s_init(I2S_CLK_DIV);
    uart_puts("I2S started\r\n");
    csr_enable_mie();

    while (1) {
        if (!ring_ready)
            asm volatile ("wfi");

        if (!ring_ready)
            continue;

        memcpy(nnom_input_data, (const void *)ring_buf,
               (size_t)SAMPLES_PER_CLIP);

        ring_pos   = 0;
        ring_ready = 0;

        model_run(model);

        int    pred      = 0;
        int8_t max_score = nnom_output_data[0];
        for (int j = 1; j < NUM_CLASSES; j++) {
            if (nnom_output_data[j] > max_score) {
                max_score = nnom_output_data[j];
                pred      = j;
            }
        }

        uart_puts("DETECT:");
        uart_putdec(pred);
        uart_putc(',');
        uart_puts(class_names[pred]);
        uart_puts("\r\n");
    }

    return 0;
}
