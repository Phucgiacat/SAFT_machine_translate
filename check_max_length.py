"""
Quick analysis: How many samples exceed max_seq_length?
Checks both SAFT and Baseline tracks without loading the full model.
"""
import os, sys, pickle
import numpy as np
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Config values (match saft_train.py)
SAFT_MAX_SEQ = 1280
BASELINE_MAX_SEQ = 768

def read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [l.strip() for l in f]

def analyze_linear_amr(linear_amr_lines):
    """Analyze AMR label counts and token estimates."""
    label_counts = []
    stop_counts = []
    for line in linear_amr_lines:
        labels = line.strip().split()
        label_counts.append(len(labels))
        stop_counts.append(sum(1 for l in labels if l == '<stop>'))
    return np.array(label_counts), np.array(stop_counts)

def estimate_saft_tokens(vi_lines, en_lines, amr_label_counts):
    """Estimate total token count per sample for SAFT track.
    Rough: prefix~50 + AMR~2*labels + suffix~30 + response~1.3*en_words"""
    estimates = []
    for i in range(len(vi_lines)):
        prefix = 50  # system + user opening
        amr = amr_label_counts[i] * 2  # ~2 tokens per label avg
        vi_tokens = len(vi_lines[i].split()) * 1.5  # ~1.5 tokens per word
        suffix = 30  # Vietnamese: ... English: + closing
        response = len(en_lines[i].split()) * 1.3  # ~1.3 tokens per word
        total = prefix + amr + vi_tokens + suffix + response
        estimates.append(total)
    return np.array(estimates)

def estimate_baseline_tokens(vi_lines, en_lines):
    """Estimate total token count for Baseline track."""
    estimates = []
    for i in range(len(vi_lines)):
        prefix = 60  # system + user opening + "Translate..." + "Vietnamese: "
        vi_tokens = len(vi_lines[i].split()) * 1.5
        suffix = 20  # English: + closing
        response = len(en_lines[i].split()) * 1.3
        total = prefix + vi_tokens + suffix + response
        estimates.append(total)
    return np.array(estimates)

def print_stats(name, estimates, max_seq, label_counts=None):
    """Print overflow statistics."""
    n = len(estimates)
    over = estimates > max_seq
    n_over = over.sum()
    pct = 100.0 * n_over / n

    print(f"\n{'='*60}")
    print(f"  {name} (max_seq={max_seq})")
    print(f"{'='*60}")
    print(f"  Total samples:    {n:,}")
    print(f"  Overflow:         {n_over:,} ({pct:.2f}%)")
    print(f"  Token estimates:  min={estimates.min():.0f}, "
          f"median={np.median(estimates):.0f}, "
          f"mean={estimates.mean():.0f}, "
          f"p95={np.percentile(estimates, 95):.0f}, "
          f"p99={np.percentile(estimates, 99):.0f}, "
          f"max={estimates.max():.0f}")

    if n_over > 0:
        overflow_amounts = estimates[over] - max_seq
        print(f"  Overflow amount:  min={overflow_amounts.min():.0f}, "
              f"median={np.median(overflow_amounts):.0f}, "
              f"max={overflow_amounts.max():.0f}")

    if label_counts is not None:
        print(f"\n  AMR labels/sample: min={label_counts.min()}, "
              f"median={np.median(label_counts):.0f}, "
              f"mean={label_counts.mean():.0f}, "
              f"p95={np.percentile(label_counts, 95):.0f}, "
              f"p99={np.percentile(label_counts, 99):.0f}, "
              f"max={label_counts.max()}")

    # Histogram buckets
    buckets = [0, 256, 512, 768, 1024, 1280, 1536, 2048, 3000, 99999]
    print(f"\n  Token distribution:")
    for j in range(len(buckets) - 1):
        lo, hi = buckets[j], buckets[j+1]
        count = ((estimates >= lo) & (estimates < hi)).sum()
        bar = '█' * (count * 40 // n) if n > 0 else ''
        label = f"  {lo:>5}-{hi:>5}" if hi < 99999 else f"  {lo:>5}+     "
        print(f"  {label}: {count:>6,} ({100*count/n:>5.1f}%) {bar}")

def main():
    for split_name, vi_name, en_name, amr_name in [
        ("TRAIN", "train.vi", "train.en", "train.linear.amr"),
        ("VAL (tst2012)", "tst2012.vi", "tst2012.en", "tst2012.linear.amr"),
        ("TEST (tst2013)", "tst2013.vi", "tst2013.en", "tst2013.linear.amr"),
    ]:
        vi_path = os.path.join(DATA_DIR, vi_name)
        en_path = os.path.join(DATA_DIR, en_name)
        amr_path = os.path.join(DATA_DIR, amr_name)

        if not all(os.path.exists(p) for p in [vi_path, en_path, amr_path]):
            print(f"[SKIP] Missing files for {split_name}")
            continue

        vi = read_lines(vi_path)
        en = read_lines(en_path)
        amr = read_lines(amr_path)
        n = min(len(vi), len(en), len(amr))
        vi, en, amr = vi[:n], en[:n], amr[:n]

        label_counts, stop_counts = analyze_linear_amr(amr)

        # SAFT track
        saft_est = estimate_saft_tokens(vi, en, label_counts)
        print_stats(f"{split_name} — SAFT Track", saft_est, SAFT_MAX_SEQ, label_counts)

        # Baseline track
        base_est = estimate_baseline_tokens(vi, en)
        print_stats(f"{split_name} — Baseline Track", base_est, BASELINE_MAX_SEQ)

        # Show worst offenders
        if (saft_est > SAFT_MAX_SEQ).any():
            worst_idx = np.argsort(saft_est)[-5:][::-1]
            print(f"\n  Top 5 longest SAFT samples:")
            for rank, idx in enumerate(worst_idx, 1):
                print(f"    #{rank} idx={idx}: ~{saft_est[idx]:.0f} tokens, "
                      f"{label_counts[idx]} AMR labels, "
                      f"{stop_counts[idx]} segments, "
                      f"vi={len(vi[idx].split())}w, en={len(en[idx].split())}w")
                amr_preview = amr[idx][:100] + "..." if len(amr[idx]) > 100 else amr[idx]
                print(f"         AMR: {amr_preview}")

if __name__ == '__main__':
    main()
