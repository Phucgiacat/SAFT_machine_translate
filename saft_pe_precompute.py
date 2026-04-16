"""
SAFT PE Precomputation
═════════════════════════════════════════════════════════
Precompute node-level Magnetic Laplacian PEs for all AMR graphs.
Aligns SPG node PEs with BPE AMR labels for downstream use.

Output: {split}_pes.pkl containing per-sample label PEs.

Usage:
    python saft_pe_precompute.py --data-dir data
"""

import os
import re
import pickle
import argparse
import time
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amr_graph_parser import (
    parse_penman_to_graph,
    transform_to_spg,
    compute_magnetic_laplacian,
    extract_spectral_features,
    read_amr_file,
)


# ─────────────────────────────────────────────────────────
# 1. Extract DFS concept traversal from Penman (incl. re-entrancies)
# ─────────────────────────────────────────────────────────

def extract_dfs_concept_sequence(penman_str: str) -> List[Tuple[str, str, bool]]:
    """
    Walk Penman notation in DFS order and extract concept references
    in the same order they appear in the BPE AMR linearization.

    Returns: [(variable, concept_name, is_reentrant), ...]
    
    - Concept definitions:  (z0 / muốn)    → ('z0', 'muốn', False)
    - Re-entrant references: z0 (bare var)  → ('z0', 'muốn', True)
    """
    if not penman_str or not penman_str.strip():
        return []

    token_pattern = re.compile(
        r'(\()'
        r'|(\))'
        r'|(:[a-zA-Z0-9_-]+)'
        r'|("(?:[^"\\]|\\.)*")'
        r'|(/)'
        r'|([\w.+-]+)'
    )

    tokens = []
    for m in token_pattern.finditer(penman_str):
        tok = m.group(0)
        if tok.strip():
            tokens.append(tok)

    var_to_concept = {}
    concept_sequence = []
    idx = 0
    current_role = None

    while idx < len(tokens):
        tok = tokens[idx]

        if tok == '(':
            idx += 1
            if idx >= len(tokens):
                break
            var_name = tokens[idx]

            concept = var_name
            if idx + 2 < len(tokens) and tokens[idx + 1] == '/':
                concept = tokens[idx + 2]
                idx += 2

            var_to_concept[var_name] = concept
            concept_sequence.append((var_name, concept, False))
            current_role = None
            idx += 1

        elif tok == ')':
            current_role = None
            idx += 1

        elif tok.startswith(':'):
            current_role = tok
            idx += 1

        elif tok.startswith('"'):
            if current_role:
                str_val = tok.strip('"')
                concept_sequence.append((f'_str_{len(concept_sequence)}', str_val, False))
                current_role = None
            idx += 1

        elif tok == '/':
            idx += 1

        else:
            if current_role:
                if tok in var_to_concept:
                    # Re-entrant reference
                    concept_sequence.append((tok, var_to_concept[tok], True))
                else:
                    # Atomic value (number, constant)
                    concept_sequence.append((f'_val_{len(concept_sequence)}', tok, False))
                current_role = None
            idx += 1

    return concept_sequence


# ─────────────────────────────────────────────────────────
# 2. Align BPE AMR labels with Penman concept PEs
# ─────────────────────────────────────────────────────────

def align_bpe_labels_with_pes(
    bpe_amr: str,
    penman_str: str,
    k: int = 20,
    q: float = 0.25,
) -> Optional[Dict]:
    """
    For a single sample, compute SPG PEs and align them to BPE AMR labels.

    Returns:
        {
            'labels': ['muốn', ':arg0', 'tôi', ...],
            'label_pes': np.array (n_labels, 2k),
            'label_is_concept': [True, False, True, ...],
        }
    or None if parsing fails.
    """
    if not bpe_amr or not penman_str:
        return None

    # Step 1: Parse Penman → graph → SPG → PEs
    G = parse_penman_to_graph(penman_str)
    if G is None or len(G.nodes) == 0:
        return None

    SPG = transform_to_spg(G)
    spectral = extract_spectral_features(SPG, k=k, q=q)
    node_pe = spectral['node_pe']

    # Step 2: Get DFS concept sequence from Penman
    concept_seq = extract_dfs_concept_sequence(penman_str)

    # Build var → PE mapping (using SPG PE for the concept node)
    var_pe = {}
    for var, concept, is_reentrant in concept_seq:
        if var in node_pe and var not in var_pe:
            var_pe[var] = node_pe[var]

    # Step 3: Parse BPE AMR into labels
    labels = bpe_amr.strip().split()

    # Step 4: Match BPE concept labels to Penman concepts
    pe_dim = 2 * k
    label_pes = np.zeros((len(labels), pe_dim), dtype=np.float32)
    label_is_concept = [False] * len(labels)

    concept_idx = 0
    for i, label in enumerate(labels):
        # Skip roles, parens, structural tokens
        if label.startswith(':') or label in ('(', ')'):
            continue

        # Try to match with next concept in DFS sequence
        if concept_idx < len(concept_seq):
            var, concept, is_reentrant = concept_seq[concept_idx]

            if _is_match(label, concept):
                # Assign PE for this concept node
                if var in var_pe:
                    label_pes[i] = var_pe[var]
                elif is_reentrant and var in node_pe:
                    label_pes[i] = node_pe[var]
                label_is_concept[i] = True
                concept_idx += 1
            else:
                # Mismatch — try to skip ahead in concept sequence
                found = False
                for j in range(concept_idx, min(concept_idx + 3, len(concept_seq))):
                    v, c, r = concept_seq[j]
                    if _is_match(label, c):
                        if v in var_pe:
                            label_pes[i] = var_pe[v]
                        label_is_concept[i] = True
                        concept_idx = j + 1
                        found = True
                        break
                if not found:
                    # Still a concept token, just couldn't find matching PE
                    label_is_concept[i] = True

    return {
        'labels': labels,
        'label_pes': label_pes,
        'label_is_concept': label_is_concept,
    }


def _is_match(bpe_token: str, penman_concept: str) -> bool:
    """Check if a BPE token matches a Penman concept."""
    if bpe_token == penman_concept:
        return True
    if bpe_token.lower() == penman_concept.lower():
        return True
    if bpe_token.replace('_', '-') == penman_concept.replace('_', '-'):
        return True
    clean = penman_concept.strip('"')
    if bpe_token == clean or bpe_token.lower() == clean.lower():
        return True
    # Handle num_0 → number mapping
    if bpe_token.startswith('num_') or bpe_token.startswith('temporal-quantity'):
        return True
    return False


# ─────────────────────────────────────────────────────────
# 3. Batch precomputation
# ─────────────────────────────────────────────────────────

def precompute_pes(
    penman_file: str,
    bpe_file: str,
    output_file: str,
    k: int = 20,
    q: float = 0.25,
):
    """Precompute PEs for all samples in a split."""
    print(f"\n{'='*60}")
    print(f"  Precomputing PEs: {os.path.basename(bpe_file)}")
    print(f"  k={k}, q={q}")
    print(f"{'='*60}")

    # Read data
    print("Reading Penman AMR file...")
    penman_entries = read_amr_file(penman_file)
    print(f"  {len(penman_entries)} Penman entries")

    print("Reading BPE AMR file...")
    with open(bpe_file, 'r', encoding='utf-8') as f:
        bpe_lines = [line.strip() for line in f]
    print(f"  {len(bpe_lines)} BPE lines")

    n = min(len(penman_entries), len(bpe_lines))
    if len(penman_entries) != len(bpe_lines):
        print(f"  [WARN] Count mismatch: Penman={len(penman_entries)}, BPE={len(bpe_lines)}")

    results = []
    stats = {'success': 0, 'fail': 0, 'total': n}
    start = time.time()

    for i in tqdm(range(n), desc="Computing PEs"):
        _, _, penman_str = penman_entries[i]
        bpe_amr = bpe_lines[i]

        try:
            result = align_bpe_labels_with_pes(bpe_amr, penman_str, k=k, q=q)
            if result is not None:
                results.append(result)
                stats['success'] += 1
            else:
                # Fallback: all zeros
                labels = bpe_amr.strip().split()
                results.append({
                    'labels': labels,
                    'label_pes': np.zeros((len(labels), 2 * k), dtype=np.float32),
                    'label_is_concept': [not l.startswith(':') and l not in ('(', ')') for l in labels],
                })
                stats['fail'] += 1
        except Exception as e:
            labels = bpe_amr.strip().split()
            results.append({
                'labels': labels,
                'label_pes': np.zeros((len(labels), 2 * k), dtype=np.float32),
                'label_is_concept': [not l.startswith(':') and l not in ('(', ')') for l in labels],
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
# 4. Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Precompute SAFT PEs')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--k', type=int, default=20, help='Number of eigenvectors')
    parser.add_argument('--q', type=float, default=0.25, help='Magnetic parameter')
    parser.add_argument('--test-only', action='store_true', help='Only process test sets')
    args = parser.parse_args()

    splits = [
        ('tst2012-amr.txt', 'tst2012.bpe.amr', 'tst2012_pes.pkl'),
        ('tst2013-amr.txt', 'tst2013.bpe.amr', 'tst2013_pes.pkl'),
    ]
    if not args.test_only:
        splits.insert(0, ('train-amr.txt', 'train.bpe.amr', 'train_pes.pkl'))

    for penman_name, bpe_name, out_name in splits:
        penman_path = os.path.join(args.data_dir, penman_name)
        bpe_path = os.path.join(args.data_dir, bpe_name)
        out_path = os.path.join(args.data_dir, out_name)

        if not os.path.exists(penman_path) or not os.path.exists(bpe_path):
            print(f"[SKIP] Missing files for {bpe_name}")
            continue

        precompute_pes(penman_path, bpe_path, out_path, k=args.k, q=args.q)

    print("\nAll done!")


if __name__ == '__main__':
    main()
