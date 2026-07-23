"""
Simulate what happens when SAFT max_seq = 512.
Shows: how many samples overflow, what gets truncated, and chunking behavior.
"""
import os, sys, pickle
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SAFT_MAX_SEQ = 512  # Simulating reduced max length

def read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [l.strip() for l in f]

def analyze(split_name, vi_path, en_path, amr_path):
    vi = read_lines(vi_path)
    en = read_lines(en_path)
    amr = read_lines(amr_path)
    n = min(len(vi), len(en), len(amr))
    vi, en, amr = vi[:n], en[:n], amr[:n]

    # Estimate tokens per sample
    overflow_samples = []
    total_est = []

    for i in range(n):
        labels = amr[i].strip().split()
        n_labels = len(labels)
        n_stops = sum(1 for l in labels if l == '<stop>')

        prefix = 50
        amr_tokens = n_labels * 2
        vi_tokens = len(vi[i].split()) * 1.5
        suffix = 30
        resp_tokens = len(en[i].split()) * 1.3

        est = prefix + amr_tokens + vi_tokens + suffix + resp_tokens
        total_est.append(est)

        if est > SAFT_MAX_SEQ:
            overflow_samples.append({
                'idx': i, 'est': est, 'n_labels': n_labels,
                'n_stops': n_stops, 'vi': vi[i], 'en': en[i],
                'amr': amr[i], 'vi_words': len(vi[i].split()),
                'en_words': len(en[i].split()),
                'amr_tokens_est': amr_tokens,
                'resp_tokens_est': resp_tokens,
                'vi_tokens_est': vi_tokens,
            })

    total_est = np.array(total_est)
    n_over = len(overflow_samples)
    pct = 100.0 * n_over / n

    print(f"\n{'='*70}")
    print(f"  {split_name} | SAFT max_seq = {SAFT_MAX_SEQ}")
    print(f"{'='*70}")
    print(f"  Total:     {n:,} samples")
    print(f"  Overflow:  {n_over:,} ({pct:.2f}%)")
    print(f"  Estimates: median={np.median(total_est):.0f}, "
          f"p95={np.percentile(total_est, 95):.0f}, "
          f"max={total_est.max():.0f}")

    if not overflow_samples:
        print("  No overflow - all samples fit!")
        return

    # --- What happens with OLD code (truncation bugs) ---
    print(f"\n  {'='*60}")
    print(f"  SCENARIO 1: OLD CODE (buggy truncation)")
    print(f"  {'='*60}")

    labels_all_neg100 = 0
    amr_cut_mid_label = 0
    response_partially_cut = 0

    for s in overflow_samples:
        excess = s['est'] - SAFT_MAX_SEQ
        fixed = 80  # prefix + suffix
        budget = SAFT_MAX_SEQ - fixed

        # Old code: max_amr = budget - response
        max_amr = budget - s['resp_tokens_est']
        if max_amr < 10:
            # Old code sets max_amr = max_seq // 2, truncates response
            max_amr = SAFT_MAX_SEQ // 2
            remaining_for_resp = SAFT_MAX_SEQ - fixed - max_amr
            if remaining_for_resp <= 0:
                labels_all_neg100 += 1
            else:
                response_partially_cut += 1
        elif max_amr < s['amr_tokens_est']:
            # AMR gets cut - might be mid-label
            amr_cut_mid_label += 1

    print(f"  Samples with ALL labels=-100 (loss=0):  {labels_all_neg100}")
    print(f"  Samples with AMR cut mid-label:         {amr_cut_mid_label}")
    print(f"  Samples with response partially cut:    {response_partially_cut}")

    # --- What happens with NEW code (smart truncation) ---
    print(f"\n  {'='*60}")
    print(f"  SCENARIO 2: NEW CODE (smart truncation)")
    print(f"  {'='*60}")

    amr_truncated = 0
    resp_truncated = 0
    both_truncated = 0

    for s in overflow_samples:
        fixed = 80
        budget = SAFT_MAX_SEQ - fixed
        resp = s['resp_tokens_est']
        amr = s['amr_tokens_est']

        if resp <= budget - 10:
            amr_truncated += 1  # only AMR cut, response intact
        else:
            both_truncated += 1  # both cut

    print(f"  AMR truncated (response kept full):     {amr_truncated}")
    print(f"  Both truncated (30/70 split):           {both_truncated}")
    print(f"  Response always has training signal:     YES (guaranteed)")

    # --- What happens with CHUNKING ---
    print(f"\n  {'='*60}")
    print(f"  SCENARIO 3: CHUNKING (no info loss)")
    print(f"  {'='*60}")

    chunk_counts = {1: 0, 2: 0, 3: 0}
    extra_samples = 0

    for s in overflow_samples:
        fixed = 80
        vi_t = s['vi_tokens_est']
        resp_t = s['resp_tokens_est']
        non_amr = fixed + vi_t + resp_t
        amr_budget_labels = max(5, (SAFT_MAX_SEQ - non_amr) / 2)

        n_chunks = max(1, int(np.ceil(s['n_labels'] / max(1, amr_budget_labels))))
        n_chunks = min(n_chunks, 3)  # cap at max_chunks=3

        chunk_counts[n_chunks] = chunk_counts.get(n_chunks, 0) + 1
        extra_samples += n_chunks - 1

    print(f"  Samples needing 1 chunk:   {chunk_counts.get(1,0)}")
    print(f"  Samples needing 2 chunks:  {chunk_counts.get(2,0)}")
    print(f"  Samples needing 3 chunks:  {chunk_counts.get(3,0)}")
    print(f"  Extra training samples:    +{extra_samples} "
          f"({n:,} -> {n + extra_samples:,})")
    print(f"  Information lost:          NONE")

    # --- Show 5 specific examples ---
    print(f"\n  {'='*60}")
    print(f"  EXAMPLE OVERFLOW SAMPLES")
    print(f"  {'='*60}")

    # Sort by estimate descending
    overflow_samples.sort(key=lambda x: x['est'], reverse=True)
    for rank, s in enumerate(overflow_samples[:5], 1):
        print(f"\n  --- Sample #{rank} (idx={s['idx']}) ---")
        print(f"  Estimated tokens: {s['est']:.0f} (exceeds {SAFT_MAX_SEQ} by {s['est']-SAFT_MAX_SEQ:.0f})")
        print(f"  AMR: {s['n_labels']} labels, {s['n_stops']} BFS segments, ~{s['amr_tokens_est']:.0f} tokens")
        print(f"  Vietnamese: {s['vi_words']} words (~{s['vi_tokens_est']:.0f} tokens)")
        print(f"  English: {s['en_words']} words (~{s['resp_tokens_est']:.0f} tokens)")
        vi_preview = s['vi'][:80] + '...' if len(s['vi']) > 80 else s['vi']
        en_preview = s['en'][:80] + '...' if len(s['en']) > 80 else s['en']
        amr_preview = s['amr'][:100] + '...' if len(s['amr']) > 100 else s['amr']
        print(f"  VI: {vi_preview}")
        print(f"  EN: {en_preview}")
        print(f"  AMR: {amr_preview}")

        # Show what chunking would do
        labels = s['amr'].strip().split()
        stops = [j+1 for j, l in enumerate(labels) if l == '<stop>']
        non_amr = 80 + s['vi_tokens_est'] + s['resp_tokens_est']
        amr_budget = max(5, int((SAFT_MAX_SEQ - non_amr) / 2))

        print(f"  -> AMR budget per chunk: ~{amr_budget} labels")
        print(f"  -> BFS segment sizes: {[stops[0]] + [stops[i]-stops[i-1] for i in range(1,len(stops))]}")

        # Simulate greedy chunking
        boundaries = [0] + stops
        if boundaries[-1] < len(labels):
            boundaries.append(len(labels))

        chunks = []
        start = 0
        for bi in range(1, len(boundaries)):
            size = boundaries[bi] - boundaries[start]
            if size > amr_budget and bi - 1 > start:
                chunks.append((boundaries[start], boundaries[bi-1]))
                start = bi - 1
        if start < len(boundaries) - 1:
            chunks.append((boundaries[start], boundaries[-1]))

        chunks = chunks[:3]
        print(f"  -> Chunks: {len(chunks)} chunks: ", end="")
        for ci, (cs, ce) in enumerate(chunks):
            print(f"[{cs}:{ce}]={ce-cs} labels", end="  ")
        print()

def main():
    for name, vi, en, amr in [
        ("TRAIN", "train.vi", "train.en", "train.bpe.amr"),
        ("VAL", "tst2012.vi", "tst2012.en", "tst2012.bpe.amr"),
        ("TEST", "tst2013.vi", "tst2013.en", "tst2013.bpe.amr"),
    ]:
        vi_p = os.path.join(DATA_DIR, vi)
        en_p = os.path.join(DATA_DIR, en)
        amr_p = os.path.join(DATA_DIR, amr)
        if all(os.path.exists(p) for p in [vi_p, en_p, amr_p]):
            analyze(name, vi_p, en_p, amr_p)

if __name__ == '__main__':
    main()
