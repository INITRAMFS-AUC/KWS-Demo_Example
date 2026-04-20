#!/usr/bin/env bash
# run_spike_batch.sh — Batch accuracy test for bare-metal KWS on Spike
#
# Runs one Spike instance per audio clip, collects DETECT output, computes
# accuracy vs ground truth (embedded in clip filename: <class>_NNNN.bin).
#
# Usage:
#   bash scripts/run_spike_batch.sh [TEST_CLIPS_DIR]
#
# Defaults:
#   TEST_CLIPS_DIR = test_clips/
#   SPIKE_ELF      = build/kws_bare
#   UART_PLUGIN    = plugins/spike_uart.so
#   I2S_PLUGIN     = plugins/spike_i2s.so
#
# Output:
#   Per-class accuracy + overall accuracy printed to stdout
#   Detailed results saved to build/kws_bare_results.txt

set -euo pipefail

CLIPS_DIR="${1:-test_clips}"
ELF="build/kws_bare"
UART_SO="plugins/spike_uart.so"
I2S_SO="plugins/spike_i2s.so"
SPIKE="/opt/riscv/bin/spike"
ARCH="rv32imac_zicsr_zifencei"
RESULTS="build/kws_bare_results.txt"
TIMEOUT=120  # seconds per inference (generous; actual < 5s per clip)

CLASSES=(down go left no off on right stop up yes unknown)

if [ ! -f "$ELF" ]; then
    echo "ERROR: $ELF not found. Run: make build/kws_bare" >&2
    exit 1
fi
if [ ! -f "$UART_SO" ] || [ ! -f "$I2S_SO" ]; then
    echo "ERROR: plugins not built. Run: make plugins" >&2
    exit 1
fi
if [ ! -d "$CLIPS_DIR" ]; then
    echo "ERROR: $CLIPS_DIR not found. Run: python3 scripts/gen_spike_audio.py" >&2
    exit 1
fi

mkdir -p build
> "$RESULTS"

declare -A total_per_class
declare -A correct_per_class
for cls in "${CLASSES[@]}"; do
    total_per_class[$cls]=0
    correct_per_class[$cls]=0
done

total=0
correct=0
failed=0

echo "Running batch inference on $(find "$CLIPS_DIR" -name '*.bin' | wc -l) clips..."
echo ""

for clip in "$CLIPS_DIR"/*.bin; do
    [ -f "$clip" ] || continue
    basename="$(basename "$clip" .bin)"

    # Extract true class from filename: <class>_NNNN.bin
    true_class="${basename%%_*}"

    if [[ ! " ${CLASSES[*]} " =~ " ${true_class} " ]]; then
        echo "WARNING: unrecognised class in filename: $clip" >&2
        continue
    fi

    # Run Spike (capture stdout, discard stderr from plugins)
    output=$(LD_LIBRARY_PATH=/opt/riscv/lib timeout "$TIMEOUT" \
        "$SPIKE" --isa="$ARCH" \
        --extlib="$UART_SO" --device=spike_uart \
        --extlib="$I2S_SO"  --device="spike_i2s,$clip" \
        "$ELF" 2>/dev/null) || {
        echo "WARNING: Spike failed/timed out for $clip" >&2
        ((failed++)) || true
        continue
    }

    # Parse DETECT line: "DETECT:<index>,<name>"
    detect_line=$(echo "$output" | grep -m1 '^DETECT:' || true)
    if [ -z "$detect_line" ]; then
        echo "WARNING: no DETECT output for $clip" >&2
        ((failed++)) || true
        continue
    fi

    pred_class="${detect_line#DETECT:*,}"
    pred_class="${pred_class%%$'\r'}"  # strip CR if present

    # Score
    is_correct=0
    if [ "$pred_class" = "$true_class" ]; then
        is_correct=1
        ((correct++)) || true
        ((correct_per_class[$true_class]++)) || true
    fi
    ((total++)) || true
    ((total_per_class[$true_class]++)) || true

    echo "$true_class -> $pred_class $([ $is_correct -eq 1 ] && echo OK || echo WRONG)" \
        >> "$RESULTS"
done

# Print summary
echo ""
echo "========================================"
echo "  Bare-metal KWS on Spike — Results"
echo "========================================"
echo ""
echo "  Per-class accuracy:"
for cls in "${CLASSES[@]}"; do
    n=${total_per_class[$cls]}
    c=${correct_per_class[$cls]}
    if [ "$n" -gt 0 ]; then
        pct=$(awk "BEGIN { printf \"%.1f\", 100*$c/$n }")
        printf "    %-10s: %s%%  (%d/%d)\n" "$cls" "$pct" "$c" "$n"
    fi
done
echo ""
if [ "$total" -gt 0 ]; then
    acc=$(awk "BEGIN { printf \"%.2f\", 100*$correct/$total }")
    echo "  Overall accuracy: ${acc}%  ($correct/$total)"
else
    echo "  No clips processed."
fi
if [ "$failed" -gt 0 ]; then
    echo "  WARNING: $failed clips failed/timed out"
fi
echo ""
echo "  Full results: $RESULTS"
