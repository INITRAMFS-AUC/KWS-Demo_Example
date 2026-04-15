# kws-spike-validate

Minimal RISC-V Spike validation for KWS NNoM models.

**Purpose:** Confirm that a trained model produces correct output when compiled
for the target ISA (`rv32imc`) before migrating to real hardware.  This is the
go/no-go check before connecting any microphone, DMA, or I2S peripheral.

---

## What this repository does

For each model there are exactly two files:

```
<model_name>_weights.h   — NNoM int8 weights (auto-generated C header)
<model_name>_main.c      — Minimal inference harness
```

The harness:
1. Loads `test_data.bin` into heap (one `malloc`)
2. Copies each 8000-byte audio clip directly into NNoM's input buffer
3. Calls `model_run()`
4. Takes `argmax` of the 11 output scores
5. Prints `RESULT:<true_label>,<pred_label>` and a final `ACCURACY:` line

No ring buffer.  No streaming.  No VAD.  No I2S.  Just the model weights and
the NNoM inference library on a RISC-V core.

---

## Repository layout

```
kws-spike-validate/
│
├── strided_s16_nodil_weights.h   NNoM int8 weights — best validated model
│                                  (nnom-qat-strided-s16-nodil, 85.3% int8)
├── strided_s16_nodil_main.c      Harness for the above model
│
├── generate_test_data.py         Generates test_data.bin from GSCD audio
├── Makefile                      Build and run targets
├── .gitmodules                   NNoM as git submodule at nnom/
└── nnom/                         NNoM library (git submodule)
```

**Adding a new model:** drop `<name>_weights.h` and `<name>_main.c` into this
directory and add the corresponding target to the Makefile.  See the commented
template in the Makefile.

---

## Prerequisites

### 1. RISC-V toolchain

| Tool | Default path | What it does |
|---|---|---|
| `riscv32-unknown-elf-gcc` | `/opt/riscv/gcc15/bin/` | Cross-compiles C to RV32IMC |
| `spike` | `/opt/riscv/bin/` | RISC-V ISA simulator |
| `pk` | `/opt/riscv/riscv32-unknown-elf/bin/` | Proxy kernel — forwards `fopen`, `malloc`, `printf` to Linux host |

Check that all tools are present:
```bash
make setup
```

If any tool is missing, see the installation notes in [kws-nnom/README.md](../kws-nnom/README.md)
(companion repo) or the [RISC-V GNU Toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain)
documentation.

To use different paths, edit the three lines at the top of the Makefile:
```makefile
CC    = /opt/riscv/gcc15/bin/riscv32-unknown-elf-gcc
SPIKE = /opt/riscv/bin/spike
PK    = /opt/riscv/riscv32-unknown-elf/bin/pk
```

### 2. NNoM

NNoM is **pre-installed in the container** at `/opt/nnom`.  The Dockerfile
clones it there directly:
```
git clone https://github.com/majianjia/nnom.git   →   /opt/nnom
```
No submodule initialisation or extra installation is required.
If for any reason it is missing: `git clone https://github.com/majianjia/nnom /opt/nnom`

### 3. Python and GSCD (for test data only)

```bash
pip install numpy soundfile scipy
```

Google Speech Commands v2 dataset:
```bash
wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar xf speech_commands_v0.02.tar.gz -C /path/to/gscd_v2
```

Python and GSCD are only needed to generate `test_data.bin`.  If you already
have the file, you can skip this step.

---

## Quickstart

```bash
# 1. Generate test_data.bin  (~35 MB, full GSCD test split)
python3 generate_test_data.py

# 2. Build and run on Spike
make run_strided_s16_nodil
```

Expected output (tail):
```
...
RESULT:9,9
RESULT:8,8
RESULTS_END
ACCURACY:0.853000
TOTAL:4928
CORRECT:4203
```

---

## Step-by-step

### Step 1 — Generate test_data.bin

```bash
python3 generate_test_data.py [OPTIONS]

  --dataset PATH        GSCD v2 root directory
                        (default: /workspace/Desktop/Models/data/dataset)
  --output  PATH        Output file (default: test_data.bin)
  --max-samples N       Limit to N samples per class — useful for a quick
                        sanity check (e.g. --max-samples 50 takes ~2 min
                        on Spike instead of ~20 min for the full set)
```

The file format is NNoM's standard evaluation batch format:
```
Repeat until all samples consumed:
  int8  true_labels[128]         — ground-truth class index (0–10) for each clip
  int8  audio[128 × 8000]        — 8000 int8 Q7 bytes per clip
```

**Audio processing (critical — must match training):**

- 16 kHz WAV files are downsampled to 8 kHz by taking every other sample
  (`audio[::2]`).  This is naive decimation without an anti-alias filter — it
  is intentional because the model was trained on data downsampled the same way.
  Using a proper filter would reduce accuracy.
- Audio is converted to int8 Q7 by multiplying by 128 and clipping to [-128, 127].
  This matches `AUDIO_INPUT_OUTPUT_DEC = 7` in the weights header.
- Clips are centre-padded/cropped to exactly 8000 samples.

### Step 2 — Build

```bash
make                             # builds all models
make build_strided_s16_nodil     # builds only this model
```

Compiler flags used:
```
-march=rv32imc_zicsr_zifencei  — RV32IMC + CSR + fence.i
-mabi=ilp32                    — 32-bit pointers, no hardware FPU
-O2 -std=c99
-DNNOM_USING_STATIC_MEMORY     — disable NNoM's malloc; use static buffer
```

The binary includes: `strided_s16_nodil_main.c` + all NNoM C sources from
`nnom/src/core/`, `nnom/src/layers/`, `nnom/src/backends/`.

### Step 3 — Run on Spike

```bash
make run_strided_s16_nodil
```

Spike runs the ELF on a virtual RV32IMC core.  The proxy kernel (`pk`) handles
`fopen`, `malloc`, and `printf` so the program can read files and write output
as if running on Linux.  The simulation is instruction-accurate but slow —
expect 1–2 minutes per inference on a typical workstation.

For a quick smoke test (fewer samples):
```bash
python3 generate_test_data.py --max-samples 50 --output test_data_small.bin
# temporarily rename for the make target:
cp test_data_small.bin test_data.bin
make run_strided_s16_nodil
```

---

## What the output means

```
RESULTS_START
RESULT:<true_label>,<pred_label>
...
RESULTS_END
ACCURACY:<float>
TOTAL:<N>
CORRECT:<N>
```

`ACCURACY` is the fraction of clips where `true_label == pred_label`.

**Expected result: ~85.3%** on the full GSCD test split.

### Class label index mapping

| Index | Word    |
|-------|---------|
| 0     | down    |
| 1     | go      |
| 2     | left    |
| 3     | no      |
| 4     | off     |
| 5     | on      |
| 6     | right   |
| 7     | stop    |
| 8     | up      |
| 9     | yes     |
| 10    | unknown |

This mapping is alphabetical and is fixed in both the training script and this
test harness.  Do not reorder it.

---

## Troubleshooting

**`ERROR: cannot open test_data.bin`**
→ Run `python3 generate_test_data.py` first.

**Compile error: `nnom.h: No such file or directory`**
→ NNoM should be at `/opt/nnom` (pre-installed by Dockerfile). Check with
  `ls /opt/nnom/inc/nnom.h`. If missing: `git clone https://github.com/majianjia/nnom /opt/nnom`

**`ACCURACY:0.000000` or no `RESULT:` lines printed**
→ The model produced no output.  Check that `nnom/inc/nnom.h` exists and that
  the NNoM sources compiled without warnings.  Also check that `test_data.bin`
  is not empty (`ls -lh test_data.bin`).

**Accuracy is ~9% (≈ random for 11 classes)**
→ The model's int8 weights are loading but computing random output.  Likely cause:
  `test_data.bin` was generated with a different class order.  Regenerate it.

**Accuracy is ~50–70% instead of 85%**
→ Possible causes:
  - `test_data.bin` was generated using a proper anti-alias filter for
    downsampling — regenerate with the naive `[::2]` method (default).
  - The wrong `weights.h` file is being compiled.  Verify that
    `AUDIO_INPUT_OUTPUT_DEC` in the weights header is `7`.

**Spike runs but produces no output at all**
→ `pk` may not be found.  Check `make setup` and verify the PK path.

**Spike killed / out of memory**
→ Reduce `-m256` in `SPIKE_FLAGS` (Makefile line `SPIKE_FLAGS = --isa=$(ARCH) -m256`)
  if your host has less than 4 GB of free RAM.

---

## How to add a new model

1. Get the new model's `weights.h` from the ML team (generated by `export_model.py`
   in the `kws-nnom` companion repo).

2. Copy it here with a descriptive name:
   ```bash
   cp /path/to/weights.h da4a_weights.h
   ```

3. Copy `strided_s16_nodil_main.c` as a template:
   ```bash
   cp strided_s16_nodil_main.c da4a_main.c
   ```

4. In `da4a_main.c`, change the one include line:
   ```c
   // Before:
   #include "strided_s16_nodil_weights.h"
   // After:
   #include "da4a_weights.h"
   ```
   If the new model has a different number of output classes or input size,
   update `NUM_CLASSES` and `SAMPLES_PER_CLIP` in the `#define` section as well.

5. Add the build and run targets to the Makefile (copy the commented template
   at the bottom of the model targets section).

6. Run:
   ```bash
   make run_da4a
   ```

---

## What this validates — and what it does not

| This repo validates | Not validated here |
|---|---|
| Model weights are correctly quantised | I2S input processing |
| NNoM runs correctly on RV32IMC | DMA buffer sizing |
| `model_run()` produces correct output | Ring buffer logic |
| Int8 MAC arithmetic is correct | VAD / RMS normalisation |
| Spike proxy kernel works | SoC linker / memory map |

The `kws-nnom` companion repo adds the full firmware pipeline (ring buffer,
VAD, RMS norm, debounce) as the next validation step.

---

## Model details: strided_s16_nodil

| Property | Value |
|---|---|
| Full name | nnom-qat-strided-s16-nodil |
| Float accuracy (GSCD) | 91.6% |
| NNoM int8 accuracy (Spike) | 85.3% |
| Parameters | 11,867 |
| MACs per inference | ~5.46M |
| Input | 8000 × int8 Q7 (1 second at 8 kHz) |
| Output | 11 × int8 (softmax scores, non-negative) |
| `AUDIO_INPUT_OUTPUT_DEC` | 7 (Q7 format, scale = 128) |
| Training data | Google Speech Commands v2 |
| QAT | Yes (NNoM per-channel KLD quantisation) |

Architecture:
```
Input (1, 8000, 1) int8
  ↓  Conv2D [1,129,1,32] stride=16   SincConv baked + BN fused
  ↓  ReLU
  ↓  Conv2D [1,3,32,32]              DS block 2 (DW+PW fused) + BN fused
  ↓  ReLU → MaxPool(1,2)
  ↓  Conv2D [1,3,32,32]              DS block 3
  ↓  ReLU → MaxPool(1,2)
  ↓  Conv2D [1,3,32,32]              DS block 4
  ↓  ReLU → MaxPool(1,2)
  ↓  Conv2D [1,3,32,32]              DS block 5
  ↓  ReLU → MaxPool(1,2)
  ↓  GlobalAvgPool
  ↓  Dense(32) → ReLU
  ↓  Dense(11) → Softmax
Output (11,) int8
```
