# kws-spike-validate

RISC-V validation and deployment firmware for the KWS NNoM int8 model, targeting
the **KWS-SoC** (Hazard3 RV32IMAC, 128 KB SRAM, I2S mic, UART).

---

## Two tracks

### Track 1 — Accuracy benchmark (pk-based, Spike only)

`strided_s16_nodil_main.c` + `strided_s16_nodil_weights.h`

Runs the full GSCD test split through the model on Spike using the proxy kernel
(`pk`) for file I/O.  No hardware simulation.  Use this to confirm that a new
set of weights produces the expected accuracy before spending time on firmware.

**Expected: 85.3% on the full GSCD test split.**

### Track 2 — Bare-metal SoC firmware (the real thing)

`kws_bare.c` + `strided_s16_nodil_weights.h`

Single-file bare-metal firmware that:
- Drives the real KWS-SoC register map (UART `0x40004000`, I2S `0x40008000`)
- Uses the Hazard3 interrupt system (no PLIC — CSR-based, `EXTENSION_XH3IRQ=0`)
- Collects audio via I2S ISR into a ring buffer
- Runs NNoM inference on each 1-second window
- Prints `DETECT:<index>,<word>` over UART

The **same source file** targets both Spike (with MMIO plugins) and the FPGA SoC.
Only the build flags differ.

---

## Repository layout

```
kws-spike-validate/
│
├── kws_bare.c                    Bare-metal KWS firmware (SoC deployment target)
├── strided_s16_nodil_weights.h   NNoM int8 model weights (85.3% accuracy)
├── strided_s16_nodil_main.c      pk-based accuracy benchmark harness
├── weights.h                     Alias weights header (same model)
│
├── soc/
│   ├── crt0.s                    Bare-metal startup (reg init, BSS zero, data copy)
│   ├── spike_soc.ld              Linker script for Spike (RAM at 0x80000000)
│   ├── link.ld                   SoC XIP linker script (flash 0x80000000, SRAM 0x0)
│   └── sram_link.ld              SoC SRAM-only linker script (GDB load workflow)
│
├── plugins/
│   ├── spike_uart.cpp            Spike MMIO plugin: uart_mini at 0x40004000
│   └── spike_i2s.cpp             Spike MMIO plugin: apb_i2s_receiver at 0x40008000
│
├── scripts/
│   ├── gen_spike_audio.py        Convert GSCD clips → raw int8 Q7 for I2S plugin
│   └── run_spike_batch.sh        Batch accuracy test over all test_clips/
│
├── extern/nnom/                  NNoM library (git submodule)
├── generate_test_data.py         Generate test_data.bin from GSCD
└── Makefile
```

---

## Prerequisites

```bash
make setup    # checks all tools
```

| Tool | Path | Purpose |
|---|---|---|
| `riscv32-unknown-elf-gcc` | `/opt/riscv/gcc15/bin/` | Cross-compiler |
| `spike` | `/opt/riscv/bin/` | RISC-V ISA simulator |
| `pk` | `/opt/riscv/riscv32-unknown-elf/bin/` | Proxy kernel (Track 1 only) |
| `g++` | system | Build Spike MMIO plugins |
| `python3` | system | Generate test data / clips |

NNoM is a git submodule at `extern/nnom/`.  After cloning:
```bash
git submodule update --init --recursive
```

---

## Track 1: Accuracy benchmark

### Generate test data

```bash
make test_data       # generates build/test_data.bin from GSCD (~35 MB)
```

Options:
```bash
python3 generate_test_data.py \
    --dataset /workspace/Desktop/Models/data/dataset \
    --output test_data.bin \
    --max-samples 50    # optional: quick smoke test
```

**Audio format (must match training):**
- 16 kHz WAV → downsample to 8 kHz by taking every other sample (`[::2]`).
  Naive decimation, no anti-alias filter.  This is intentional — the model
  was trained on data processed identically.  Using a proper filter reduces accuracy.
- Scale to int8 Q7: `clip(round(sample * 128), -128, 127)`
- Pad/crop to exactly 8000 samples per clip

### Build and run

```bash
make build_strided_s16_nodil
make run_strided_s16_nodil
```

Expected output:
```
ACCURACY:0.853000
TOTAL:4928
CORRECT:4203
```

---

## Track 2: Bare-metal firmware

### Architecture overview

```
I2S FIFO (8 samples @ 8 kHz)
    │  IRQ every 1 ms
    ▼
kws_trap_handler() — ISR
    │  fills ring_buf[8000]
    ▼
ring_ready = 1
    │
    ▼
main() inference loop (wfi → wake)
    │  memcpy ring_buf → nnom_input_data
    │  model_run()
    │  argmax(nnom_output_data[11])
    ▼
UART: "DETECT:9,yes\r\n"
```

### Build

```bash
# Spike simulation build (MMIO plugins, I2S_FIFO_DEPTH=8000 for speed)
make build/kws_bare

# SoC deployment build (XIP: flash+SRAM, I2S_FIFO_DEPTH=8, NNOM_STATIC_BUF_KB=52)
make build/kws_soc

# SoC SRAM-only build (GDB load workflow — only if simulation SRAM > 128 KB)
make build/kws_soc_sram
```

Size summary:

| Build | text (code+weights) | bss (SRAM) | Notes |
|---|---|---|---|
| `kws_bare` (Spike) | ~85 KB | ~278 KB | Spike has 16 MB RAM |
| `kws_soc` (XIP) | ~85 KB → flash | ~68 KB → SRAM | Fits 128 KB SRAM |
| `kws_soc_sram` | ~154 KB → SRAM | — | Exceeds 128 KB — see note |

> **Note on SRAM build:** The NNoM weights (~55 KB of `.rodata`) plus activation
> buffer (52 KB) plus code exceed the SoC's 128 KB SRAM.  Use the XIP target
> (`kws_soc`) for normal deployment.  The SRAM target is only usable if the
> simulation is configured with more than 128 KB.

### Run on Spike (bare-metal, single clip)

```bash
# Build Spike MMIO plugins first (one-time)
make plugins

# Generate test clips
make test_clips      # extracts individual .bin files from test_data.bin

# Run one clip
make run_kws_bare AUDIO_FILE=test_clips/yes_0000.bin
```

Expected output:
```
[I2S] Loaded 8000 samples from test_clips/yes_0000.bin
KWS bare-metal firmware (NNoM int8)
Model loaded
[I2S] Configured: clk_div=70 irq_en=1
I2S started
DETECT:9,yes
```

### Batch accuracy test on Spike

```bash
make batch_kws_bare
```

Runs all clips in `test_clips/` and reports accuracy.

### Deploy on KWS-SoC FPGA

The `build/kws_soc` ELF uses:
- **XIP linker script** (`soc/link.ld`): code and weights execute from flash at
  `0x80000000`; BSS lives in SRAM at `0x00000000`
- **I2S_FIFO_DEPTH=8**: real hardware FIFO depth
- **NNOM_STATIC_BUF_KB=52**: 52 KB activation buffer (peak usage measured at ~48 KB)

Loading workflow depends on your simulation setup:
- **GDB + OpenOCD (XIP):** flash the binary to the SPI flash model, then reset
- **GDB load (SRAM only):** use `kws_soc_sram` if your simulation supports it

Expected boot sequence over UART:
```
KWS bare-metal firmware (NNoM int8)
Model loaded
I2S started
DETECT:<index>,<word>
DETECT:<index>,<word>
...    (continuous, one line per 1-second audio window)
```

---

## Firmware internals

### Startup: `soc/crt0.s`

1. **Zero all 31 GPRs** — Hazard3 `RESET_REGFILE=0`; registers hold garbage
   on reset and across GDB `monitor reset`
2. **Set `mtvec`** to a safe infinite-loop trap handler — catches any early
   exception before the real handler is installed in `main()`
3. **Set `gp`** using `lui+addi` (absolute, not PC-relative `auipc`) — required
   for XIP builds where code runs from flash but SRAM is at a different address
4. **Set `sp`** same way, aligned to 16 bytes (RISC-V ABI)
5. **Copy `.data`** from LMA to VMA — no-op for pure-SRAM builds; essential
   for XIP where `.data` LMA is in flash and VMA is in SRAM
6. **Zero `.bss`** — C standard guarantees for uninitialized globals
7. **`call main`**

### Interrupt system: Hazard3 without XH3IRQ

The KWS-SoC sets `EXTENSION_XH3IRQ=0` in `hazard3_config.vh`.  This means:

- **No PLIC**, no memory-mapped interrupt controller
- **No `meiea`/`meipa`/`meicontext` CSRs** — calling `h3irq_enable()` would
  cause an illegal instruction exception
- External interrupts are a single OR of all IRQ lines gated by `mie.MEIE`
- The Verilog wires `.irq({uart_irq, i2s_irq})` — I2S is bit 0, UART is bit 1

**Setup in `main()` (4 instructions):**
```c
csr_set_mtvec(kws_trap_handler);  // csrw mtvec, handler
csr_enable_meie();                 // csrs mie, (1<<11)
i2s_init(I2S_CLK_DIV);            // write I2S CONF: start FIFO + assert IRQ_EN
csr_enable_mie();                  // csrsi mstatus, 8  — IRQs now live
```

**ISR — no claim/complete:**

The I2S `irq` line deasserts automatically when the firmware drains the FIFO
to empty.  Reading `I2S_FIFO_DEPTH` samples inside the ISR is sufficient.
On MRET, `mstatus.MIE` is restored from `mstatus.MPIE` and the CPU returns.

```c
void __attribute__((interrupt("machine"))) kws_trap_handler(void) {
    uint32_t mcause;
    asm volatile ("csrr %0, mcause" : "=r"(mcause));

    if (!(mcause & 0x80000000u)) {
        // exception — print diagnostics and spin
    }

    // external interrupt (only source: I2S)
    for (int i = 0; i < I2S_FIFO_DEPTH; i++) {
        int8_t q7 = (int8_t)((int32_t)I2S->fifo >> 16);
        if (ring_pos < SAMPLES_PER_CLIP)
            ring_buf[ring_pos++] = q7;
    }
    if (ring_pos >= SAMPLES_PER_CLIP)
        ring_ready = 1;
    // MRET: Hazard3 auto-deasserts when FIFO empty; no claim/complete needed
}
```

### I2S audio format

INMP441 MEMS mic outputs 24-bit signed audio left-aligned in a 32-bit I2S word:
`raw_32bit = (audio_24bit << 8)`.  The SoC's `apb_i2s_receiver` preserves this
format in the FIFO.

To extract the int8 Q7 value for NNoM:
```c
int8_t q7 = (int8_t)((int32_t)I2S->fifo >> 16);
```
This takes bits `[23:16]` of the 32-bit word — the top 8 bits of the 24-bit sample.

The Spike I2S plugin (`plugins/spike_i2s.cpp`) applies the inverse:
```cpp
val = (uint32_t)((int32_t)q7 << 16);
```
so the firmware's `>> 16` recovers the original Q7 byte exactly.

### NNoM static memory

Defined with `-DNNOM_USING_STATIC_MEMORY`.  NNoM uses a bump allocator over a
single statically-declared buffer.

```c
static uint8_t nnom_static_buf[NNOM_STATIC_BUF_KB * 1024];   // in .bss
// ...
nnom_set_static_buf(nnom_static_buf, sizeof(nnom_static_buf));
nnom_model_t *model = nnom_model_create();   // all allocs from bump allocator
```

`nnom_free()` is a no-op.  The allocator never reclaims memory — this is correct
because the model is created once and lives for the entire program lifetime.

**Measured peak usage (this model):** `nnom_memory_taken = 47968` bytes.
`NNOM_STATIC_BUF_KB=52` gives ~4 KB headroom.

### NNoM kernel_size overflow bug (fixed)

**Symptom:** `nnom_model_create()` reported 5.7 MB activation memory instead of
~48 KB; the first Conv1D output tensor was `{1, 61948, 32}` instead of `{1, 492, 32}`.

**Root cause:** `nnom_conv2d_config_t` in the upstream NNoM source declared:
```c
int8_t kernel_size[2];   // max value = 127
```
This model uses `kernel_size = {1, 129}`.  The value 129 overflows `int8_t` to
`-127`, which sign-extends to `size_t` `4294967169`, which truncates to `uint16_t`
`65409` when computing the output length.  The inflated tensor caused allocation
failure (silent NULL return from bump allocator) and then a store fault in
`model_run()`.

**Fix applied** in `extern/nnom/inc/layers/nnom_conv2d.h`:
```c
// Before (broken for kernel > 127):
int8_t kernel_size[2];
int8_t stride_size[2];
int8_t padding_size[2];
int8_t dilation_size[2];

// After (fixed):
uint16_t kernel_size[2];
uint16_t stride_size[2];
uint16_t padding_size[2];
uint16_t dilation_size[2];
```
`nnom_shape_data_t` is already `uint16_t`; the fix aligns the config struct to
match the internal representation.

---

## Spike MMIO plugins

The plugins simulate KWS-SoC peripherals at their real register addresses.
Built against Spike's C++ headers (`/opt/riscv/include`).

### `spike_uart.cpp` — `0x40004000`

| Offset | Register | Behaviour |
|---|---|---|
| `0x00` | CSR | enable bit; ignored by plugin |
| `0x04` | DIV | baud divisor; ignored by plugin |
| `0x08` | FSTAT | always returns 0 (TX never full) |
| `0x0C` | TX | write prints character to host stdout |

### `spike_i2s.cpp` — `0x40008000`

| Offset | Register | Behaviour |
|---|---|---|
| `0x00` | ID | returns `0xDEADCAFE` |
| `0x04` | CONF | write: records config; on IRQ_EN 0→1 asserts PLIC IRQ 2 |
| `0x08` | FIFO | read: returns next sample as `(int8_t q7) << 16` |

After every `I2S_FIFO_DEPTH` reads the plugin deasserts then reasserts the IRQ
to simulate the hardware FIFO refill.  When all audio data is consumed the IRQ
stays deasserted and the firmware exits via HTIF.

**`I2S_FIFO_DEPTH=8000` for Spike** (vs 8 on real hardware): reduces ISR count
from 1000 to 1, cutting simulation time by ~1000×.  The entire ISR/PLIC claim
/complete path is still exercised once.

> **Spike uses its built-in PLIC at `0x0C000000`** for interrupt delivery.
> The Spike firmware build (`kws_bare`) therefore still configures the PLIC.
> The SoC firmware build (`kws_soc`) removes all PLIC code and uses only
> standard CSR instructions, matching the Hazard3 hardware.

---

## Memory map

| Address | SoC peripheral | Spike |
|---|---|---|
| `0x00000000` | SRAM (128 KB) | not mapped (all in 0x80000000 region) |
| `0x40004000` | UART (`uart_mini`) | `spike_uart.so` plugin |
| `0x40008000` | I2S (`apb_i2s_receiver`) | `spike_i2s.so` plugin |
| `0x80000000` | Flash (XIP) | Spike DRAM (16 MB) |

---

## Build flags reference

| Flag | Spike | SoC |
|---|---|---|
| `I2S_FIFO_DEPTH` | `8000` | `8` |
| `NNOM_STATIC_BUF_KB` | `256` | `52` |
| `CLK_MHZ` | `36` | `36` |
| `UART_BAUD_RATE` | `115200` | `115200` |
| Linker script | `soc/spike_soc.ld` | `soc/link.ld` |
| SRAM_SIZE | N/A | `131072` (128 KB) |

---

## Troubleshooting

**Firmware prints `ERROR: model_compile failed — nnom_static_buf too small`**
→ Increase `NNOM_STATIC_BUF_KB`.  Measured peak for this model is ~48 KB.

**Firmware hangs after `Model loaded`, never prints `I2S started`**
→ The `I2S->conf` write is hanging.  On the SoC, verify the I2S peripheral is
  clocked and the address decode is correct in the simulation.

**Firmware prints `I2S started` but never prints `DETECT:`**
→ The ISR is not firing.  Check:
  - `i2s_irq` is wired in the testbench (`.irq({uart_irq, i2s_irq})` in `kws_soc.v`)
  - `mie.MEIE` was set before `mstatus.MIE` (correct order in firmware)
  - The I2S peripheral is actually generating IRQ pulses (check simulation waveform)

**`EXCEPTION cause=5` (load fault) or `cause=7` (store fault) in ISR**
→ A NULL pointer dereference.  Almost always means `nnom_model_create()` returned
  a model with unallocated tensors due to OOM.  The firmware checks for this and
  prints an error — if you see a fault instead, the check is being bypassed.
  Increase `NNOM_STATIC_BUF_KB`.

**`EXCEPTION cause=2` (illegal instruction)**
→ A CSR instruction the CPU doesn't support.  On the SoC this would happen if
  code accidentally calls `h3irq_enable()` or another XH3IRQ function
  (`EXTENSION_XH3IRQ=0` — these CSRs don't exist).  Verify no `h3irq_*` calls
  are in the firmware.

**Linker fails with `undefined reference to 'expf'` (or `__extenddftf2`, `__addsf3`)**
→ NNoM's softmax layer calls `expf()` from libm, and the compiler runtime supplies
  soft-float helpers (`__addsf3`, `__mulsf3`, `__extenddftf2`).  Two fixes:

  *Option A (quick):* add `-lm -lgcc` to the link line.  The Makefile already does
  this for the `kws_soc` target; if building manually make sure both are present and
  appear **after** all object files on the command line:
  ```
  riscv32-unknown-elf-gcc ... kws_bare.c nnom/src/... -lm -lgcc
  ```
  Use `gcc` as the driver (not `ld` directly) — `gcc` automatically resolves the
  multilib path for the target `-march`; `ld` does not and will miss libgcc.

  *Option B (eliminate float entirely):* stop model compilation before the Softmax
  layer so inference produces raw int8 logits.  Argmax over logits gives the same
  predicted class as argmax over softmax probabilities:
  ```c
  // In nnom_model_create() (weights.h), change final compile call from:
  //   model_compile(&model, input_layer, output_layer_softmax);
  // to:
  //   model_compile(&model, input_layer, output_layer_dense11);
  ```
  This eliminates every floating-point operation from the inference path.  The
  `DETECT:` output is unchanged; only softmax scores (unused for detection) are lost.

**Accuracy on Spike batch test lower than 85.3%**
→ Regenerate test clips with `make test_clips`.  The clips must be generated from
  the same `test_data.bin` that was validated with the pk-based benchmark.

**Spike runs but no UART output appears**
→ Check that `plugins/spike_uart.so` was built (`make plugins`) and that
  `LD_LIBRARY_PATH=/opt/riscv/lib` is set when invoking Spike.

---

## Class index mapping

| Index | Word    |
|---|---|
| 0 | down |
| 1 | go |
| 2 | left |
| 3 | no |
| 4 | off |
| 5 | on |
| 6 | right |
| 7 | stop |
| 8 | up |
| 9 | yes |
| 10 | unknown |

Fixed alphabetically in both the training script and the weights header.

---

## Model: strided-s16-nodil

| Property | Value |
|---|---|
| Float accuracy (GSCD) | 91.6% |
| NNoM int8 accuracy (Spike) | 85.3% |
| Parameters | 11,867 |
| MACs per inference | ~5.46M |
| Inference time @ 36 MHz | ~150 ms (fits in 1-second audio window) |
| Input | `{1, 8000, 1}` int8 Q7 |
| Output | `{11}` int8 softmax scores |
| QAT | Yes (per-channel KLD) |

Architecture:
```
Input  {1, 8000, 1}
  Conv1D k=129, s=16, f=32, VALID  →  {1, 492, 32}   learned filterbank
  ReLU
  Conv1D k=3, s=1, f=48, VALID     →  {1, 490, 48}
  ReLU → MaxPool k=4, s=4          →  {1, 122, 48}
  Conv1D k=3, s=1, f=48, VALID     →  {1, 120, 48}
  ReLU → MaxPool k=2, s=2          →  {1,  60, 48}
  Conv1D k=3, s=1, f=48, VALID     →  {1,  58, 48}
  ReLU → MaxPool k=2, s=2          →  {1,  29, 48}
  Conv1D k=3, s=1, f=48, VALID     →  {1,  27, 48}
  ReLU → GlobalAvgPool             →  {48}
  Dense 32 → ReLU                  →  {32}
  Dense 11 → Softmax               →  {11}
Output {11}
```

Peak activation memory (measured): **47,968 bytes** across 3 live tensors:
- largest tensor: `{1, 492, 48}` = 23,616 bytes
