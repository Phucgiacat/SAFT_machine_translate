"""
SAFT BFS Linearization
═════════════════════════════════════════════════════════
Convert Penman AMR graphs to BFS linearization following
the SAFT paper (arXiv:2507.13381) Section 3.1.

Output format:
    concept :role1 child1 :role2 child2 <stop> child1 <stop> child2 :role3 grandchild <stop> ...

Re-entrant references use pointer tokens: <P1>, <P2>, ...

Usage:
    python saft_bfs_linearize.py --data-dir data
═════════════════════════════════════════════════════════
"""

import os
import re
import argparse
from collections import deque
from typing import List, Tuple, Dict, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from saft.amr.parser import parse_penman_to_graph, read_amr_file


# ─────────────────────────────────────────────────────────
# 1. BFS Linearization of AMR DiGraph
# ─────────────────────────────────────────────────────────

def bfs_linearize(penman_str: str) -> Optional[str]:
    """
    Convert a Penman AMR string to BFS linearization following the SAFT paper.

    Paper Section 3.1 — BFS Linearization:
      - BFS traversal from root node
      - Each node expanded once as a segment: concept :role1 target1 :role2 target2 ... <stop>
      - Re-entrant references → pointer tokens <P1>, <P2>, ...
      - Role labels preserved as labels in the linearization

    Returns:
        A single-line BFS linearized string, or None if parsing fails.
    """
    G = parse_penman_to_graph(penman_str)
    if G is None or len(G.nodes) == 0:
        return None

    # Find root node
    root = None
    for node, data in G.nodes(data=True):
        if data.get('is_root', False):
            root = node
            break
    if root is None:
        return None

    # Assign pointer IDs to nodes that are referenced more than once
    # (re-entrant nodes = nodes with in-degree > 1)
    in_degree = dict(G.in_degree())
    # Also count nodes that appear as targets of multiple edges
    ref_count = {}
    for u, v, data in G.edges(data=True):
        ref_count[v] = ref_count.get(v, 0) + 1

    # Nodes that need pointers: referenced more than once (re-entrant)
    pointer_map = {}  # node_id -> pointer_id
    pointer_counter = 1
    for node in G.nodes():
        if node == root:
            # Root can also be re-entrant
            total_refs = ref_count.get(node, 0)
        else:
            total_refs = ref_count.get(node, 0)
        if total_refs > 1:
            pointer_map[node] = f"<P{pointer_counter}>"
            pointer_counter += 1

    # BFS traversal
    visited_expanded = set()  # nodes that have been expanded (their segment written)
    queue = deque([root])
    segments = []

    while queue:
        node = queue.popleft()

        if node in visited_expanded:
            continue
        visited_expanded.add(node)

        concept = G.nodes[node].get('concept', node)

        # Build segment: concept :role1 target1 :role2 target2 ... <stop>
        segment_labels = [concept]

        # Get outgoing edges sorted by role name for deterministic order
        out_edges = list(G.out_edges(node, data=True))
        # Sort: :ARG0 < :ARG1 < ... < :mod < :name < ...
        out_edges.sort(key=lambda e: e[2].get('role', ':UNK'))

        for u, v, data in out_edges:
            role = data.get('role', ':UNK')
            target_concept = G.nodes[v].get('concept', v)

            # Add role label
            segment_labels.append(role.lower())

            if v in visited_expanded:
                # Re-entrant reference — use pointer
                if v in pointer_map:
                    segment_labels.append(pointer_map[v])
                else:
                    # Should not happen, but fallback
                    segment_labels.append(target_concept)
            else:
                # First reference — emit concept (and pointer if needed)
                if v in pointer_map:
                    segment_labels.append(f"{target_concept} {pointer_map[v]}")
                else:
                    segment_labels.append(target_concept)

                # Enqueue for future expansion
                if v not in visited_expanded and v not in queue:
                    queue.append(v)

        segment_labels.append("<stop>")
        segments.append(segment_labels)

    # Flatten all segments into a single line
    all_labels = []
    for seg in segments:
        all_labels.extend(seg)

    return " ".join(all_labels)


# ─────────────────────────────────────────────────────────
# 2. Batch Processing
# ─────────────────────────────────────────────────────────

def linearize_split(
    penman_file: str,
    output_file: str,
):
    """Linearize all AMR entries in a split file."""
    print(f"\n{'='*60}")
    print(f"  BFS Linearizing: {os.path.basename(penman_file)}")
    print(f"{'='*60}")

    entries = read_amr_file(penman_file)
    print(f"  {len(entries)} Penman entries")

    results = []
    stats = {'success': 0, 'fail': 0}

    # Load dictionary once
    from saft.amr.dictionary import SAFTDictionary
    saft_dict = SAFTDictionary()
    saft_dict.load_dictionary()

    for idx, (entry_id, snt, penman_str) in enumerate(entries):
        linear = bfs_linearize(penman_str)
        if linear is not None:
            # Apply Dictionary Enrichment
            enriched_linear = saft_dict.enrich_linear_amr(linear)
            results.append(enriched_linear)
            stats['success'] += 1
        else:
            # Fallback: use concept from root if possible
            results.append("<unk> <stop>")
            stats['fail'] += 1

    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')

    print(f"  Success: {stats['success']}/{len(entries)}")
    print(f"  Failed:  {stats['fail']}/{len(entries)}")
    print(f"  Saved:   {output_file}")

    # Show first 3 examples
    print(f"\n  Sample outputs:")
    for i in range(min(3, len(results))):
        text = results[i]
        if len(text) > 120:
            text = text[:120] + "..."
        print(f"    [{i}] {text}")

    return stats


# ─────────────────────────────────────────────────────────
# 3. Main
# ─────────────────────────────────────────────────────────

def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(description='BFS Linearize AMR graphs (SAFT paper)')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    args = parser.parse_args()

    splits = [
        ('train-amr.txt', 'train.linear.amr'),
        ('tst2012-amr.txt', 'tst2012.linear.amr'),
        ('tst2013-amr.txt', 'tst2013.linear.amr'),
    ]

    for penman_name, out_name in splits:
        penman_path = os.path.join(args.data_dir, penman_name)
        out_path = os.path.join(args.data_dir, out_name)

        if not os.path.exists(penman_path):
            print(f"[SKIP] Missing: {penman_path}")
            continue

        linearize_split(penman_path, out_path)

    print("\nAll done!")


if __name__ == '__main__':
    main()
