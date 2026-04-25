#!/usr/bin/env python3
"""
generate_thresh_data.py — Generate thresh_data.bin for threshold sweep testing.

For each keyword clip in the test split, prepends one real filler clip drawn
from background noise recordings and unknown-class audio.  This gives the
threshold sweep harness realistic non-keyword audio to measure false-positive
rates against, instead of all-zero silence.

FILLER sources (50/50 split):
  - Background noise: random 1-second windows from _background_noise_/*.wav
    (doing_the_dishes, dude_miaowing, exercise_bike, pink_noise, running_tap,
     white_noise) — downsampled 16kHz→8kHz the same naive way as training.
  - Unknown-class clips: test-split WAVs from non-keyword word folders.

OUTPUT: thresh_data.bin
-----------------------
  int32  n_keywords          total keyword entries
  int32  n_filler_per_kw     always 1 (one 8000-sample clip before each keyword)

  For each of n_keywords entries:
    int8  keyword_label          0–9  (same mapping as test_data.bin)
    int8  filler_label           10   (always — filler is non-keyword audio)
    int8  filler_audio[8000]     real background/unknown audio, Q7 int8
    int8  keyword_audio[8000]    keyword clip, Q7 int8

Usage:
  python3 generate_thresh_data.py
  python3 generate_thresh_data.py --dataset /path/to/gscd_v2 --out thresh_data.bin
"""

import argparse
import os
import sys
import numpy as np

try:
    import soundfile as sf
except ImportError:
    sys.exit("ERROR: soundfile not installed.  Run: pip install soundfile")

TARGET_SR      = 8000
WINDOW_SAMPLES = 8000
Q7_SCALE       = 128
LABEL_UNKNOWN  = 10
CORE_CLASSES   = ['down','go','left','no','off','on','right','stop','up','yes']
DEFAULT_DATASET = '/workspace/Desktop/Models/data/dataset'
DEFAULT_OUTPUT  = 'thresh_data.bin'


def load_as_int8(path):
    """Load a WAV, downsample 16kHz→8kHz if needed, pad/trim to 8000, return int8 Q7."""
    audio, sr = sf.read(path, dtype='float32', always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr == 16000:
        audio = audio[::2]          # naive decimation — matches training
    elif sr != TARGET_SR:
        sys.exit(f"Unexpected sample rate {sr} in {path}")
    if len(audio) < WINDOW_SAMPLES:
        pad = WINDOW_SAMPLES - len(audio)
        audio = np.pad(audio, (pad // 2, pad - pad // 2))
    else:
        audio = audio[:WINDOW_SAMPLES]
    audio = np.clip(audio * Q7_SCALE, -128, 127).astype(np.int8)
    return audio


def load_background_windows(dataset_root, n_windows, rng):
    """Cut n_windows random 1-second (8000-sample) windows from background noise files."""
    noise_dir = os.path.join(dataset_root, '_background_noise_')
    wavs = [os.path.join(noise_dir, f)
            for f in os.listdir(noise_dir) if f.endswith('.wav')]
    if not wavs:
        sys.exit(f"No wav files found in {noise_dir}")

    windows = []
    while len(windows) < n_windows:
        path = wavs[rng.integers(len(wavs))]
        audio, sr = sf.read(path, dtype='float32', always_2d=False)
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr == 16000:
            audio = audio[::2]
        elif sr != TARGET_SR:
            continue
        if len(audio) < WINDOW_SAMPLES:
            continue
        start = rng.integers(0, len(audio) - WINDOW_SAMPLES + 1)
        window = audio[start:start + WINDOW_SAMPLES]
        window = np.clip(window * Q7_SCALE, -128, 127).astype(np.int8)
        windows.append(window)
    return windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=DEFAULT_DATASET)
    ap.add_argument('--out',     default=DEFAULT_OUTPUT)
    ap.add_argument('--seed',    type=int, default=42)
    args = ap.parse_args()

    dataset_root   = args.dataset
    output_path    = args.out
    rng            = np.random.default_rng(args.seed)

    test_list_path = os.path.join(dataset_root, 'testing_list.txt')
    if not os.path.exists(test_list_path):
        sys.exit(f"ERROR: testing_list.txt not found in {dataset_root}")

    with open(test_list_path) as f:
        test_files = set(line.strip() for line in f if line.strip())
    print(f"Test split: {len(test_files)} files")

    label_map = {cls: idx for idx, cls in enumerate(CORE_CLASSES)}

    # Collect keyword test clips (classes 0–9 only)
    keyword_clips = []   # list of (label, path)
    unknown_paths = []   # for filler

    all_dirs = sorted(d for d in os.listdir(dataset_root)
                      if os.path.isdir(os.path.join(dataset_root, d))
                      and not d.startswith('_'))

    for word_dir in all_dirs:
        class_idx = label_map.get(word_dir, LABEL_UNKNOWN)
        full_dir  = os.path.join(dataset_root, word_dir)
        for fname in sorted(os.listdir(full_dir)):
            if not fname.endswith('.wav'):
                continue
            rel = f"{word_dir}/{fname}"
            if rel not in test_files:
                continue
            path = os.path.join(full_dir, fname)
            if class_idx < LABEL_UNKNOWN:
                keyword_clips.append((class_idx, path))
            else:
                unknown_paths.append(path)

    rng.shuffle(keyword_clips)
    n_kw = len(keyword_clips)
    print(f"Keyword clips: {n_kw}")
    print(f"Unknown clips available: {len(unknown_paths)}")

    # Build filler pool: 50% background noise windows, 50% unknown clips
    n_bg      = n_kw // 2
    n_unk_use = n_kw - n_bg

    print(f"Generating {n_bg} background noise windows ...")
    bg_windows = load_background_windows(dataset_root, n_bg, rng)

    print(f"Loading {n_unk_use} unknown clips ...")
    unk_idx = rng.choice(len(unknown_paths), size=n_unk_use, replace=len(unknown_paths) < n_unk_use)
    unk_windows = [load_as_int8(unknown_paths[i]) for i in unk_idx]

    filler_pool = bg_windows + unk_windows
    rng.shuffle(filler_pool)
    print(f"Filler pool: {len(filler_pool)} clips")

    # Write thresh_data.bin
    import struct
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<i', n_kw))   # int32 n_keywords
        f.write(struct.pack('<i', 1))      # int32 n_filler_per_kw = 1

        for i, (label, kw_path) in enumerate(keyword_clips):
            kw_audio     = load_as_int8(kw_path)
            filler_audio = filler_pool[i % len(filler_pool)]

            f.write(struct.pack('b', label))         # keyword label (int8)
            f.write(struct.pack('b', LABEL_UNKNOWN)) # filler label (int8, always 10)
            f.write(filler_audio.tobytes())          # filler audio [8000]
            f.write(kw_audio.tobytes())              # keyword audio [8000]

            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{n_kw}")

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\nWrote {output_path}  ({n_kw} entries, {size_kb:.0f} KB)")

    # Print class distribution
    from collections import Counter
    counts = Counter(label for label, _ in keyword_clips)
    print("\nClass distribution:")
    for cls, idx in label_map.items():
        print(f"  {idx:2d} {cls:8s}: {counts[idx]}")


if __name__ == '__main__':
    main()
