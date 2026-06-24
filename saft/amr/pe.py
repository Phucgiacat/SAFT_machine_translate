"""
SAFT PE Precomputation (BFS Linearization)
═════════════════════════════════════════════════════════
Precompute node-level Magnetic Laplacian PEs for all AMR graphs.
Uses BFS-linearized AMR (*.linear.amr) and builds SPG from
the linearization directly (following SAFT paper Section 3.1/B.1).

Key difference from old version:
  - SPG built from BFS linearization (not Penman parse)
  - Every label gets a PE (bijective alignment: node i ↔ label i)
  - No heuristic matching needed

Output: {split}_pes.pkl containing per-sample label PEs.

Usage:
    python saft_pe_precompute.py --data-dir data
"""

import os
import pickle
import argparse
import time
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amr_graph_parser import (
    build_spg_from_bfs_linear,
    compute_magnetic_laplacian,
    extract_spectral_features,
)


# ─────────────────────────────────────────────────────────
# 1. Compute PEs from BFS Linearization (Paper-compliant)
# ─────────────────────────────────────────────────────────

def compute_pes_from_linear(
    linear_amr: str,
    k: int = 20,
    q: float = 0.25,
) -> Optional[Dict]:
    """
    For a single sample, build SPG from BFS linearization and compute PEs.

    Following SAFT paper:
      - SPG built from BFS linearization (bijective: label i ↔ node i)
      - Magnetic Laplacian PE computed on SPG
      - Every label gets its own PE vector (concept, role, <stop>, <P>)

    Returns:
        {
            'labels': ['want-01', ':arg0', 'child', '<P1>', '<stop>', ...],
            'label_pes': np.array (n_labels, 2k),
        }
    or None if parsing fails.
    """
    if not linear_amr or not linear_amr.strip():
        return None

    labels = linear_amr.strip().split()
    if len(labels) == 0:
        return None

    # Step 1: Build SPG from BFS linearization
    SPG = build_spg_from_bfs_linear(linear_amr)
    if SPG is None or len(SPG.nodes) == 0:
        return None

    # Step 2: Compute spectral features on SPG
    spectral = extract_spectral_features(SPG, k=k, q=q)
    node_pe = spectral['node_pe']  # Dict[node_id -> np.array(2k,)]

    # Step 3: Bijective alignment — node i ↔ label i
    pe_dim = 2 * k
    n_labels = len(labels)
    label_pes = np.zeros((n_labels, pe_dim), dtype=np.float32)

    for i in range(n_labels):
        if i in node_pe:
            pe_vec = node_pe[i]
            # Ensure correct dimension
            if len(pe_vec) >= pe_dim:
                label_pes[i] = pe_vec[:pe_dim]
            else:
                label_pes[i, :len(pe_vec)] = pe_vec

    return {
        'labels': labels,
        'label_pes': label_pes,
    }


# ─────────────────────────────────────────────────────────
# 2. Batch precomputation
# ─────────────────────────────────────────────────────────

def precompute_pes(
    linear_file: str,
    output_file: str,
    k: int = 20,
    q: float = 0.25,
):
    """Precompute PEs for all samples in a split."""
    print(f"\n{'='*60}")
    print(f"  Precomputing PEs (BFS): {os.path.basename(linear_file)}")
    print(f"  k={k}, q={q}")
    print(f"{'='*60}")

    # Read BFS-linearized AMR
    print("Reading BFS-linearized AMR file...")
    with open(linear_file, 'r', encoding='utf-8') as f:
        linear_lines = [line.strip() for line in f]
    print(f"  {len(linear_lines)} entries")

    n = len(linear_lines)
    results = []
    stats = {'success': 0, 'fail': 0, 'total': n}
    start = time.time()

    for i in tqdm(range(n), desc="Computing PEs"):
        linear_amr = linear_lines[i]

        try:
            result = compute_pes_from_linear(linear_amr, k=k, q=q)
            if result is not None:
                results.append(result)
                stats['success'] += 1
            else:
                # Fallback: all zeros
                labels = linear_amr.strip().split()
                results.append({
                    'labels': labels,
                    'label_pes': np.zeros((len(labels), 2 * k), dtype=np.float32),
                })
                stats['fail'] += 1
        except Exception as e:
            labels = linear_amr.strip().split()
            results.append({
                'labels': labels,
                'label_pes': np.zeros((len(labels), 2 * k), dtype=np.float32),
            })
            stats['fail'] += 1

    elapsed = time.time() - start

    # Save
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n  Success: {stats['success']}/{stats['total']}")
    print(f"  Failed:  {stats['fail']}/{stats['total']}")
    print(f"  Time:    {elapsed:.1f}s")
    print(f"  Saved:   {output_file} ({os.path.getsize(output_file)/1e6:.1f} MB)")

    return stats


# ─────────────────────────────────────────────────────────
# 3. Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Precompute SAFT PEs (BFS)')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--k', type=int, default=20, help='Number of eigenvectors')
    parser.add_argument('--q', type=float, default=0.25, help='Magnetic parameter')
    parser.add_argument('--test-only', action='store_true', help='Only process test sets')
    args = parser.parse_args()

    splits = [
        ('tst2012.linear.amr', 'tst2012_pes.pkl'),
        ('tst2013.linear.amr', 'tst2013_pes.pkl'),
    ]
    if not args.test_only:
        splits.insert(0, ('train.linear.amr', 'train_pes.pkl'))

    for linear_name, out_name in splits:
        linear_path = os.path.join(args.data_dir, linear_name)
        out_path = os.path.join(args.data_dir, out_name)

        if not os.path.exists(linear_path):
            print(f"[SKIP] Missing: {linear_path}")
            print(f"  Run: python saft_bfs_linearize.py --data-dir {args.data_dir}")
            continue

        precompute_pes(linear_path, out_path, k=args.k, q=args.q)

    print("\nAll done!")


if __name__ == '__main__':
    main()
