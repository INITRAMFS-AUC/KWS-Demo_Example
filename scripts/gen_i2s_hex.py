#!/usr/bin/env python3
"""
gen_i2s_hex.py — Convert a raw int8 Q7 audio clip to an I2S hex file
for the KWS-SoC simulation testbench.

The KWS-SoC's apb_i2s_receiver presents audio as 24-bit samples
left-aligned in a 32-bit I2S word:
    word = (int32_t)q7 << 16   (bits [23:16] carry the Q7 value)

The firmware extracts: q7 = (int8_t)((int32_t)fifo >> 16)

The simulation's I2SMicSim (sim/i2s_mic_sim.cpp) reads one hex word
per line and clocks it out bit-serially over the I2S interface.

Usage:
    # Single clip
    python3 scripts/gen_i2s_hex.py test_clips/yes_0000.bin -o test_audio/yes_0000.hex

    # All _0000 clips at once (generates test_audio/<word>_0000.hex)
    python3 scripts/gen_i2s_hex.py --all-keywords test_clips/ -o test_audio/

    # With silence padding between words (for continuous detection testing)
    python3 scripts/gen_i2s_hex.py test_clips/yes_0000.bin --pad-silence 4000 -o out.hex
"""

import argparse
import struct
import os
import sys

SAMPLES_PER_CLIP = 8000
SILENCE_WORD     = 0x00000000   # 0 in I2S format = silence

KEYWORDS = ["down", "go", "left", "no", "off", "on",
            "right", "stop", "up", "yes", "unknown"]


def clip_to_words(path: str) -> list[int]:
    """Read a raw int8 Q7 clip and return list of 32-bit I2S words."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) != SAMPLES_PER_CLIP:
        raise ValueError(
            f"{path}: expected {SAMPLES_PER_CLIP} bytes, got {len(data)}")
    words = []
    for b in data:
        q7 = struct.unpack("b", bytes([b]))[0]   # signed int8
        word = (q7 << 16) & 0xFFFFFFFF           # left-align in 32-bit word
        words.append(word)
    return words


def write_hex(words: list[int], path: str, label: str = "") -> None:
    """Write list of 32-bit words as hex file, one word per line."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        if label:
            f.write(f"# KWS-SoC I2S audio: {label}\n")
            f.write(f"# {len(words)} samples, int8 Q7 left-aligned in 32-bit I2S word\n")
            f.write(f"# word = (int32_t)q7 << 16  →  firmware reads (int8_t)(fifo >> 16)\n")
        for w in words:
            f.write(f"{w:08X}\n")
    print(f"Wrote {len(words)} samples → {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?",
                    help="Input .bin clip (raw int8 Q7, 8000 bytes)")
    ap.add_argument("-o", "--output", required=True,
                    help="Output .hex file (or directory when --all-keywords)")
    ap.add_argument("--all-keywords", action="store_true",
                    help="Convert <input_dir>/<kw>_0000.bin for all 11 keywords")
    ap.add_argument("--pad-silence", type=int, default=0, metavar="N",
                    help="Append N silent samples after each clip (default: 0)")
    ap.add_argument("--clip-index", type=int, default=0, metavar="N",
                    help="Which clip index to use with --all-keywords (default: 0)")
    args = ap.parse_args()

    silence = [SILENCE_WORD] * args.pad_silence

    if args.all_keywords:
        if not args.input:
            ap.error("--all-keywords requires an input directory")
        clip_dir = args.input.rstrip("/")
        out_dir  = args.output
        for kw in KEYWORDS:
            fname = f"{kw}_{args.clip_index:04d}.bin"
            clip_path = os.path.join(clip_dir, fname)
            if not os.path.exists(clip_path):
                print(f"WARNING: {clip_path} not found, skipping", file=sys.stderr)
                continue
            out_path = os.path.join(out_dir, fname.replace(".bin", ".hex"))
            words = clip_to_words(clip_path) + silence
            write_hex(words, out_path, label=fname)
    else:
        if not args.input:
            ap.error("Provide an input .bin file or use --all-keywords")
        words = clip_to_words(args.input) + silence
        label = os.path.basename(args.input)
        write_hex(words, args.output, label=label)


if __name__ == "__main__":
    main()
