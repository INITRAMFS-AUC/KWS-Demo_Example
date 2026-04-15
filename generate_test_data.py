#!/usr/bin/env python3
"""
generate_test_data.py — Generate test_data.bin for the Spike accuracy benchmark.

This script reads the Google Speech Commands v2 (GSCD) dataset and produces
test_data.bin — the input file consumed by *_main.c harnesses.

USAGE
-----
  python3 generate_test_data.py
  python3 generate_test_data.py --dataset /path/to/gscd_v2
  python3 generate_test_data.py --max-samples 200   # quick test (200 per class)

REQUIREMENTS
------------
  pip install numpy soundfile scipy

  Google Speech Commands v2 must be extracted so each word has its own folder:
    <dataset_root>/yes/<uuid>.wav
    <dataset_root>/no/<uuid>.wav
    ...
  Download:
    wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
    tar xf speech_commands_v0.02.tar.gz -C /path/to/gscd_v2

OUTPUT: test_data.bin
---------------------
  Batches of 128 samples (repeating until all samples consumed):

    int8  true_labels[128]          — ground-truth class indices 0–10
    int8  audio[128 × 8000]         — one int8 Q7 audio clip per sample
                                      8000 bytes = 1 second at 8 kHz

  This format is read by the *_main.c harnesses — do not change it.

CLASS INDEX → LABEL MAPPING
----------------------------
  The order below is fixed (alphabetical).  It matches the training script.
  Changing the order would break the model.

    0=down  1=go  2=left  3=no  4=off  5=on  6=right  7=stop  8=up  9=yes  10=unknown

AUDIO PROCESSING (must match training)
---------------------------------------
  1. Load WAV as float32 in [-1, 1].
  2. If 16 kHz: take every other sample (i2s[::2]).  Do NOT use a proper
     decimation filter — the model was trained on naively downsampled data;
     using a filter is a domain shift that reduces accuracy.
  3. Centre/zero-pad to exactly 8000 samples.
  4. Scale float32 to int8 Q7: multiply by 128, clip to [-128, 127].
"""

import argparse
import os
import sys
import numpy as np

try:
    import soundfile as sf
except ImportError:
    sys.exit("ERROR: soundfile not installed.  Run: pip install soundfile")

# ── Constants — must match the model and firmware config ─────────────────────

TARGET_SR      = 8000    # Model sample rate
WINDOW_SAMPLES = 8000    # 1 second of audio = one inference window
INPUT_DEC_BITS = 7       # Q7: float 1.0 → int8 128
LABEL_BATCH    = 128     # NNoM evaluation batch size

# Core keyword classes in alphabetical order.
# This is the fixed class index mapping — do not change.
CORE_CLASSES = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']

DEFAULT_DATASET = '/workspace/Desktop/Models/data/dataset'
DEFAULT_OUTPUT  = 'test_data.bin'


# ── Audio loading ─────────────────────────────────────────────────────────────

def load_as_int8(path):
    """
    Load a WAV file and return 8000 int8 Q7 samples.
    Returns None if the file cannot be loaded.
    """
    try:
        audio, sr = sf.read(path, dtype='float32')
    except Exception as e:
        print(f"  WARNING: {os.path.basename(path)}: {e}")
        return None

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Downsample to 8 kHz
    if sr == 16000:
        # Naive every-other-sample decimation — intentional, must match training.
        # Do NOT replace with resample_poly or any anti-alias filter.
        audio = audio[::2]
    elif sr != TARGET_SR:
        # Fallback for other sample rates
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(TARGET_SR, sr)
        audio = resample_poly(audio, TARGET_SR // g, sr // g).astype(np.float32)

    # Centre / zero-pad to exactly WINDOW_SAMPLES
    n = len(audio)
    if n >= WINDOW_SAMPLES:
        start = (n - WINDOW_SAMPLES) // 2
        audio = audio[start:start + WINDOW_SAMPLES]
    else:
        pad = WINDOW_SAMPLES - n
        audio = np.pad(audio, (pad // 2, pad - pad // 2))

    # Convert float [-1, 1] to int8 Q7 [-128, 127]
    scale   = float(1 << INPUT_DEC_BITS)   # 128.0
    samples = np.clip(np.round(audio * scale), -128, 127).astype(np.int8)
    return samples


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_test_samples(dataset_root, max_per_class=None):
    """
    Collect all samples in the official GSCD test split.

    Returns a list of (class_idx, int8_array) tuples, shuffled.
    """
    test_list_path = os.path.join(dataset_root, 'testing_list.txt')
    if not os.path.exists(test_list_path):
        sys.exit(f"ERROR: testing_list.txt not found in {dataset_root}\n"
                 f"  Is this the GSCD v2 root directory?")

    with open(test_list_path) as f:
        test_files = set(line.strip() for line in f if line.strip())
    print(f"  Test split: {len(test_files)} files listed in testing_list.txt")

    label_map  = {cls: idx for idx, cls in enumerate(CORE_CLASSES)}
    per_class  = {i: [] for i in range(11)}

    all_dirs = sorted(d for d in os.listdir(dataset_root)
                      if os.path.isdir(os.path.join(dataset_root, d))
                      and not d.startswith('_'))

    for word_dir in all_dirs:
        class_idx = label_map.get(word_dir, 10)    # 10 = unknown
        full_dir  = os.path.join(dataset_root, word_dir)
        count     = 0
        for fname in os.listdir(full_dir):
            if not fname.endswith('.wav'):
                continue
            rel = f"{word_dir}/{fname}"
            if rel not in test_files:
                continue
            per_class[class_idx].append(os.path.join(full_dir, fname))
            count += 1

    # Subsample unknown to match mean core-class count
    core_counts = [len(per_class[i]) for i in range(10)]
    mean_core   = int(np.mean(core_counts)) if core_counts else 490

    # Cap unknown
    rng = np.random.default_rng(42)
    if len(per_class[10]) > mean_core:
        idx = rng.choice(len(per_class[10]), size=mean_core, replace=False)
        per_class[10] = [per_class[10][i] for i in sorted(idx)]

    # Apply per-class cap if requested
    if max_per_class:
        for c in range(11):
            if len(per_class[c]) > max_per_class:
                idx = rng.choice(len(per_class[c]), size=max_per_class, replace=False)
                per_class[c] = [per_class[c][i] for i in sorted(idx)]

    # Print class summary
    print(f"  {'Class':<12} {'Count':>6}")
    for i, name in enumerate(CORE_CLASSES + ['unknown']):
        print(f"  {name:<12} {len(per_class[i]):>6}")

    # Load audio
    print(f"\n  Loading audio ...")
    samples = []
    total_files = sum(len(v) for v in per_class.values())
    loaded = 0
    for class_idx in range(11):
        for path in per_class[class_idx]:
            audio = load_as_int8(path)
            if audio is not None:
                samples.append((class_idx, audio))
            loaded += 1
            if loaded % 200 == 0:
                print(f"  {loaded}/{total_files} ...", end='\r')

    print(f"  Loaded {len(samples)} / {total_files} samples          ")
    rng.shuffle(samples)
    return samples


# ── Write test_data.bin ───────────────────────────────────────────────────────

def write_test_bin(samples, output_path):
    """
    Write samples to test_data.bin in NNoM batch format.

    Format:
      Repeat until all samples consumed:
        int8  labels[LABEL_BATCH]
        int8  audio[LABEL_BATCH × WINDOW_SAMPLES]
    """
    # Pad to a multiple of LABEL_BATCH
    while len(samples) % LABEL_BATCH != 0:
        samples.append((0, np.zeros(WINDOW_SAMPLES, dtype=np.int8)))

    print(f"\nWriting {len(samples)} samples to {output_path} ...")
    with open(output_path, 'wb') as f:
        for batch_start in range(0, len(samples), LABEL_BATCH):
            batch = samples[batch_start:batch_start + LABEL_BATCH]

            # Labels
            labels = np.array([c for c, _ in batch], dtype=np.int8)
            f.write(labels.tobytes())

            # Audio clips
            for _, audio in batch:
                f.write(audio.tobytes())

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Written: {output_path}  ({size_mb:.1f} MB, {len(samples)} samples)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate test_data.bin for Spike KWS benchmarks")
    parser.add_argument('--dataset', default=DEFAULT_DATASET,
                        help=f"GSCD v2 root directory (default: {DEFAULT_DATASET})")
    parser.add_argument('--output',  default=DEFAULT_OUTPUT,
                        help=f"Output file path (default: {DEFAULT_OUTPUT})")
    parser.add_argument('--max-samples', type=int, default=None, metavar='N',
                        help="Max samples per class — useful for quick tests")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset):
        sys.exit(f"ERROR: Dataset not found: {args.dataset}\n"
                 f"  Pass the correct path with --dataset /path/to/gscd_v2")

    print(f"Dataset: {args.dataset}")
    print(f"Output:  {args.output}")
    if args.max_samples:
        print(f"Limit:   {args.max_samples} samples per class")
    print()

    samples = load_test_samples(args.dataset, args.max_samples)
    write_test_bin(samples, args.output)

    print()
    print("Done.  Now run the benchmark:")
    print("  make run_strided_s16_nodil")


if __name__ == '__main__':
    main()
