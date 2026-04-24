# Makefile — KWS Spike Validation
#
# Two firmware tracks:
#
#  1. BARE-METAL (kws_bare) — SoC-compatible firmware, no pk.
#     Uses real KWS-SoC register addresses (UART 0x40004000, I2S 0x40008000).
#     Tested on Spike via MMIO plugins (plugins/spike_uart.so, spike_i2s.so).
#     The SAME binary can be flashed directly onto the KWS-SoC FPGA.
#
#       make plugins              — build Spike MMIO plugins
#       make build/kws_bare       — cross-compile bare-metal firmware
#       make run_kws_bare AUDIO_FILE=test_clips/yes_0000.bin
#                                 — run one clip on Spike
#       make test_clips           — extract test clips from test_data.bin
#       make batch_kws_bare       — batch accuracy over all test_clips/
#
#  2. PK-BASED (strided_s16_nodil) — original accuracy benchmark via pk.
#     Uses fopen/printf/malloc (proxied through Spike proxy kernel).
#     Useful for quick model accuracy checks.
#
#       make build_strided_s16_nodil
#       make run_strided_s16_nodil
#
# GENERATE TEST DATA
# ------------------
#   make test_data                      # generate test_data.bin from GSCD
#   make test_clips                     # extract individual clips for bare-metal
#
# PREREQUISITES
# -------------
#   make setup    — check that all tools are found

# ── Toolchain ─────────────────────────────────────────────────────────────────

CC    = /opt/riscv/gcc15/bin/riscv32-unknown-elf-gcc
SPIKE = /opt/riscv/bin/spike
PK    = /opt/riscv/riscv32-unknown-elf/bin/pk

# ISA matches the hardware Hazard3 configuration:
#   I   — base integer
#   M   — hardware multiply/divide   (EXTENSION_M = 1)
#   A   — atomics                    (EXTENSION_A = 1)
#   C   — compressed instructions    (EXTENSION_C = 1)
#   Zicsr   — CSR instructions       (implied by CSR_M_MANDATORY/CSR_M_TRAP = 1)
#   Zifencei — included because this exact string is in the toolchain multilib
#              generator, so GCC can find its runtime libraries.  NNoM never
#              emits fence.i, so this extension is never executed on hardware
#              (EXTENSION_ZIFENCEI = 0 in Hazard3 config is therefore safe).
ARCH  = rv32imac_zicsr_zifencei
ABI   = ilp32

# ── NNoM ──────────────────────────────────────────────────────────────────────

# NNoM is tracked as a git submodule at extern/nnom (fork: INITRAMFS-AUC/nnom).
# After cloning this repo, initialise it with:
#   git submodule update --init --recursive
NNOM_DIR  = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))extern/nnom
NNOM_SRCS = $(wildcard $(NNOM_DIR)/src/core/*.c) \
            $(wildcard $(NNOM_DIR)/src/layers/*.c) \
            $(wildcard $(NNOM_DIR)/src/backends/*.c)
NNOM_INC  = -I$(NNOM_DIR)/inc -I$(NNOM_DIR)/port

# ── Compiler flags ────────────────────────────────────────────────────────────

CFLAGS = -march=$(ARCH) -mabi=$(ABI) \
         -O2 -std=c99 \
         -DNNOM_USING_STATIC_MEMORY \
         $(NNOM_INC) \
         -I. -Istrided_s16_nodil

LDFLAGS = -lm

# ── Spike flags ───────────────────────────────────────────────────────────────

# -m256: 256 MB of guest memory.
# NNoM static buffer = 1 MB; test file (full GSCD test split) = ~35 MB.
# 256 MB is more than enough.
# --isa must match -march so Spike can execute the binary.
SPIKE_FLAGS = --isa=$(ARCH) -m256

# ── Build directory ───────────────────────────────────────────────────────────

BUILD = build

# ── Spike MMIO plugins ────────────────────────────────────────────────────────
# These shared libraries simulate the KWS-SoC peripherals on Spike.
# Built with the host g++ against Spike's installed headers+library.

PLUGIN_CXX   = g++
# -DI2S_FIFO_DEPTH=8000: match firmware's batch size for Spike simulation.
# Reduces ISR invocations from 1000 (×8 samples each) to 1 (×8000 samples),
# cutting PLIC overhead by ~1000x. For real SoC use -DI2S_FIFO_DEPTH=8.
PLUGIN_FLAGS = -std=c++17 -shared -fPIC \
               -DI2S_FIFO_DEPTH=8000 \
               -I/opt/riscv/include \
               -Wl,-rpath,/opt/riscv/lib -L/opt/riscv/lib -lriscv

PLUGINS = plugins/spike_uart.so plugins/spike_i2s.so

plugins/spike_uart.so: plugins/spike_uart.cpp
	@mkdir -p plugins
	$(PLUGIN_CXX) $(PLUGIN_FLAGS) -o $@ $<
	@echo "Built: $@"

plugins/spike_i2s.so: plugins/spike_i2s.cpp
	@mkdir -p plugins
	$(PLUGIN_CXX) $(PLUGIN_FLAGS) -o $@ $<
	@echo "Built: $@"

.PHONY: plugins
plugins: $(PLUGINS)

# ── Bare-metal KWS firmware ───────────────────────────────────────────────────
# SoC-compatible firmware using real register addresses.
# -DNNOM_LOG(...)= suppresses NNoM's printf-based debug output (no printf on MCU).
# -DCLK_MHZ and -DUART_BAUD_RATE configure the UART baud divisor for the SoC.
# For Spike, the UART plugin ignores the divisor, so these values don't matter.

BARE_CFLAGS  = -march=$(ARCH) -mabi=$(ABI) \
               -O2 -std=gnu99 \
               -DNNOM_USING_STATIC_MEMORY \
               -DNNOM_BARE_METAL \
               -DCLK_MHZ=36 -DUART_BAUD_RATE=115200 \
               -DI2S_FIFO_DEPTH=8000 \
               -DNNOM_STATIC_BUF_KB=256 \
               $(NNOM_INC) \
               -I.

BARE_LDFLAGS = -nostartfiles \
               -T soc/spike_soc.ld \
               -lm -lgcc

$(BUILD)/kws_bare: strided_s16_nodil/kws_bare.c strided_s16_nodil/strided_s16_nodil_weights.h soc/crt0.s $(NNOM_SRCS) | $(BUILD)
	@echo "Compiling kws_bare (bare-metal) ..."
	$(CC) $(BARE_CFLAGS) -Istrided_s16_nodil soc/crt0.s strided_s16_nodil/kws_bare.c $(NNOM_SRCS) \
	    -o $@ $(BARE_LDFLAGS)
	@echo "Built: $@"
	$(CC:%gcc=%size) $@

.PHONY: build_kws_bare
build_kws_bare: $(BUILD)/kws_bare

# ── KWS-SoC firmware (XIP: code+weights in flash, bss in SRAM) ───────────────
#
# This is the deployment target for the KWS-SoC FPGA.
#
# Memory layout (XIP via soc/link.ld):
#   Flash  0x80000000  code + rodata (weights) — ~86 KB
#   SRAM   0x00000000  data + bss              — ~60 KB (fits in 128 KB SRAM)
#
# SRAM breakdown:
#   nnom_static_buf  52 KB   activation scratch (model needs ~48 KB)
#   ring_buf          8 KB   audio samples
#   stack             4 KB   (grows down from top of SRAM)
#   other statics    ~1 KB
#
# Loading on the FPGA simulation:
#   Use the XIP workflow: pre-load the flash model with the firmware binary,
#   or use GDB `load` if the testbench supports SRAM-only loading (use
#   build/kws_soc_sram instead, see below).
#
# I2S_FIFO_DEPTH=8: real hardware FIFO depth (INMP441).
# NNOM_STATIC_BUF_KB=52: measured peak usage is 47968 B; 52 KB gives headroom.

SOC_CFLAGS = -march=$(ARCH) -mabi=$(ABI) \
             -O2 -std=gnu99 \
             -DNNOM_USING_STATIC_MEMORY \
             -DNNOM_BARE_METAL \
             -DCLK_MHZ=36 -DUART_BAUD_RATE=115200 \
             -DI2S_FIFO_DEPTH=8 \
             -DNNOM_STATIC_BUF_KB=52 \
             $(NNOM_INC) \
             -I.

SRAM_SIZE_BYTES = 131072   # 128 KB: SRAM_DEPTH=32768 words × 4

SOC_XIP_LDFLAGS  = -nostartfiles \
                   -T soc/link.ld \
                   -Wl,--defsym=SRAM_SIZE=$(SRAM_SIZE_BYTES) \
                   -lm -lgcc

SOC_SRAM_LDFLAGS = -nostartfiles \
                   -T soc/sram_link.ld \
                   -Wl,--defsym=SRAM_SIZE=$(SRAM_SIZE_BYTES) \
                   -lm -lgcc

$(BUILD)/kws_soc: strided_s16_nodil/kws_bare.c strided_s16_nodil/strided_s16_nodil_weights.h soc/crt0.s $(NNOM_SRCS) | $(BUILD)
	@echo "Compiling kws_soc (XIP: flash+SRAM) ..."
	$(CC) $(SOC_CFLAGS) -Istrided_s16_nodil soc/crt0.s strided_s16_nodil/kws_bare.c $(NNOM_SRCS) \
	    -o $@ $(SOC_XIP_LDFLAGS)
	@echo "Built: $@"
	$(CC:%gcc=%size) $@

# SRAM-only variant for GDB `load` workflow (everything in SRAM — requires
# testbench to support loading ~146 KB into the 128 KB SRAM, which will NOT
# fit at default SRAM size; only use if the simulation uses a larger SRAM).
$(BUILD)/kws_soc_sram: strided_s16_nodil/kws_bare.c strided_s16_nodil/strided_s16_nodil_weights.h soc/crt0.s $(NNOM_SRCS) | $(BUILD)
	@echo "Compiling kws_soc_sram (SRAM-only) ..."
	$(CC) $(SOC_CFLAGS) -Istrided_s16_nodil soc/crt0.s strided_s16_nodil/kws_bare.c $(NNOM_SRCS) \
	    -o $@ $(SOC_SRAM_LDFLAGS)
	@echo "Built: $@"
	$(CC:%gcc=%size) $@

.PHONY: build_kws_soc build_kws_soc_sram
build_kws_soc: $(BUILD)/kws_soc
build_kws_soc_sram: $(BUILD)/kws_soc_sram

# ── Run bare-metal firmware on Spike (single clip) ────────────────────────────
# Usage: make run_kws_bare AUDIO_FILE=test_clips/yes_0000.bin

AUDIO_FILE ?= test_clips/yes_0000.bin

SPIKE_BARE_FLAGS = --isa=$(ARCH)
SPIKE_PLUGINS    = --extlib=plugins/spike_uart.so --device=spike_uart \
                   --extlib=plugins/spike_i2s.so  --device=spike_i2s,$(AUDIO_FILE)

.PHONY: run_kws_bare
run_kws_bare: $(BUILD)/kws_bare $(PLUGINS)
	@echo ""
	@echo "Running kws_bare on Spike (bare-metal, no pk) ..."
	@echo "Audio: $(AUDIO_FILE)"
	@echo ""
	LD_LIBRARY_PATH=/opt/riscv/lib $(SPIKE) $(SPIKE_BARE_FLAGS) \
	    $(SPIKE_PLUGINS) \
	    $(BUILD)/kws_bare
	@echo ""

# ── Generate test clips for bare-metal testing ────────────────────────────────

.PHONY: test_clips
test_clips: build/test_data.bin
	@echo "Extracting test clips from test_data.bin ..."
	python3 scripts/gen_spike_audio.py \
	    --from-test-data build/test_data.bin \
	    --output-dir test_clips/
	@echo "Clips written to test_clips/"

# ── Batch accuracy test (bare-metal, all clips) ───────────────────────────────

.PHONY: batch_kws_bare
batch_kws_bare: $(BUILD)/kws_bare $(PLUGINS)
	@echo "Running batch accuracy test ..."
	bash scripts/run_spike_batch.sh test_clips/

# ── All models ────────────────────────────────────────────────────────────────

ALL_TARGETS = $(BUILD)/strided_s16_nodil

# ── Default target ────────────────────────────────────────────────────────────

.PHONY: all clean test_data setup help
all: $(ALL_TARGETS) $(BUILD)/kws_bare
	@echo ""
	@echo "All targets built."
	@echo "  Bare-metal:  make run_kws_bare AUDIO_FILE=test_clips/yes_0000.bin"
	@echo "  pk-based:    make run_strided_s16_nodil"

# ── strided_s16_nodil ─────────────────────────────────────────────────────────
#
# Model:    nnom-qat-strided-s16-nodil
# Accuracy: 85.3% NNoM int8 (validated on Spike)
# MACs:     ~5.46M per inference
# Weights:  strided_s16_nodil_weights.h

$(BUILD)/strided_s16_nodil: strided_s16_nodil/strided_s16_nodil_main.c \
                             strided_s16_nodil/strided_s16_nodil_weights.h \
                             $(NNOM_SRCS)
	@mkdir -p $(BUILD)
	@echo "Compiling strided_s16_nodil ..."
	$(CC) $(CFLAGS) -Istrided_s16_nodil \
	    strided_s16_nodil/strided_s16_nodil_main.c \
	    $(NNOM_SRCS) \
	    -o $@ $(LDFLAGS)
	@echo "Built: $@"

.PHONY: build_strided_s16_nodil run_strided_s16_nodil

build_strided_s16_nodil: $(BUILD)/strided_s16_nodil

run_strided_s16_nodil: $(BUILD)/strided_s16_nodil test_data.bin
	@echo ""
	@echo "Running strided_s16_nodil on Spike ..."
	@echo "Expected accuracy: ~85.3%"
	@echo ""
	cd $(BUILD) && $(SPIKE) $(SPIKE_FLAGS) $(PK) strided_s16_nodil \
	    | tee strided_s16_nodil.log
	@echo ""
	@grep "ACCURACY:" $(BUILD)/strided_s16_nodil.log || true
	@echo "Full log: $(BUILD)/strided_s16_nodil.log"

# ── Template for adding a new model ──────────────────────────────────────────
# To add a new model (e.g. da4a), uncomment and edit:
#
# $(BUILD)/da4a: da4a_main.c da4a_weights.h $(NNOM_SRCS)
# 	@mkdir -p $(BUILD)
# 	$(CC) $(CFLAGS) da4a_main.c $(NNOM_SRCS) -o $@ $(LDFLAGS)
#
# .PHONY: build_da4a run_da4a
# build_da4a: $(BUILD)/da4a
#
# run_da4a: $(BUILD)/da4a test_data.bin
# 	cd $(BUILD) && $(SPIKE) $(SPIKE_FLAGS) $(PK) da4a | tee da4a.log
#
# Also add $(BUILD)/da4a to ALL_TARGETS above.

# ── Test data ─────────────────────────────────────────────────────────────────

test_data: test_data.bin

test_data.bin:
	@echo "Generating test_data.bin ..."
	python3 generate_test_data.py --output test_data.bin
	@cp test_data.bin $(BUILD)/test_data.bin 2>/dev/null || true

# Copy test_data.bin into build/ when running a benchmark
$(BUILD)/test_data.bin: test_data.bin
	@mkdir -p $(BUILD)
	@cp test_data.bin $(BUILD)/test_data.bin

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	rm -rf $(BUILD) plugins/*.so
	@echo "Cleaned build/ and plugins/*.so"

# ── Setup check ───────────────────────────────────────────────────────────────

setup:
	@echo "=== KWS Spike validation toolchain check ==="
	@echo ""
	@printf "%-35s" "riscv32-unknown-elf-gcc ..."
	@$(CC) --version 2>/dev/null | head -1 || echo "NOT FOUND (see README.md)"
	@printf "%-35s" "spike ..."
	@$(SPIKE) --version 2>/dev/null | head -1 || echo "NOT FOUND (see README.md)"
	@printf "%-35s" "pk (proxy kernel) ..."
	@test -f $(PK) && echo "found: $(PK)" || echo "NOT FOUND (see README.md)"
	@printf "%-35s" "python3 ..."
	@python3 --version 2>/dev/null || echo "NOT FOUND"
	@echo ""
	@printf "%-35s" "NNoM submodule ..."
	@test -f $(NNOM_DIR)/inc/nnom.h && echo "OK ($(NNOM_DIR))" || \
	    echo "NOT FOUND — run: git submodule update --init --recursive"
	@echo ""

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo "KWS Spike Validation Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all                          Build all model binaries"
	@echo "  build_strided_s16_nodil      Compile strided_s16_nodil ELF"
	@echo "  run_strided_s16_nodil        Run strided_s16_nodil on Spike"
	@echo "  test_data                    Generate test_data.bin from GSCD"
	@echo "  setup                        Check toolchain prerequisites"
	@echo "  clean                        Remove build/ directory"
	@echo ""
	@echo "See README.md for full documentation."
