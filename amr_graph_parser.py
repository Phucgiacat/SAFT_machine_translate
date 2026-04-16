"""
═══════════════════════════════════════════════════════════
  AMR Graph Parser & Magnetic Laplacian Feature Extractor
  ─────────────────────────────────────────────────────────
  Input:  Penman notation AMR graphs (from -amr.txt files)
  Output: NetworkX DiGraph + spectral features per node

  Follows SAFT paper (KDD 2025 Workshop):
  - Semantically-Preserving Graph transformation
  - Magnetic Laplacian positional encodings
  - Spectral clustering for structural tags
═══════════════════════════════════════════════════════════
"""

import re
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import List, Tuple, Dict, Optional


# ─────────────────────────────────────────────────────────
# 1. Penman Notation Parser
# ─────────────────────────────────────────────────────────

def parse_penman_to_graph(penman_str: str) -> Optional[nx.DiGraph]:
    """
    Parse a Penman notation AMR string into a NetworkX DiGraph.

    Each node has attributes:
      - 'concept': the concept label (e.g., 'muốn', 'tôi', 'name')
      - 'is_root': True if root node

    Each edge has attributes:
      - 'role': the relation label (e.g., ':ARG0', ':mod', ':location')

    Handles:
      - Nested structures with parentheses
      - Re-entrancies (back-references like z0, z1)
      - Named entities (:op1 "value")
      - Multi-sentence AMRs

    Returns None if parsing fails.
    """
    if not penman_str or not penman_str.strip():
        return None

    try:
        G = nx.DiGraph()
        # Token patterns
        # Match: opening paren, closing paren, variable/concept, roles, strings, numbers
        token_pattern = re.compile(
            r'(\()'           # opening paren
            r'|(\))'          # closing paren
            r'|(:[a-zA-Z0-9_-]+)'  # role label like :ARG0, :mod, :op1
            r'|("(?:[^"\\]|\\.)*")'  # quoted string
            r'|(/)'           # concept separator
            r'|([\w.+-]+)'    # variable or concept or number
        )

        tokens = []
        for m in token_pattern.finditer(penman_str):
            tok = m.group(0)
            if tok.strip():
                tokens.append(tok)

        if not tokens:
            return None

        # Parse state
        var_to_concept = {}  # z0 -> 'muốn'
        node_stack = []      # stack of current parent variable
        root_var = None
        idx = 0
        current_role = None

        while idx < len(tokens):
            tok = tokens[idx]

            if tok == '(':
                # Next token should be variable
                idx += 1
                if idx >= len(tokens):
                    break
                var_name = tokens[idx]

                # Next should be '/' followed by concept
                concept = var_name  # fallback
                if idx + 2 < len(tokens) and tokens[idx + 1] == '/':
                    concept = tokens[idx + 2]
                    idx += 2

                var_to_concept[var_name] = concept
                G.add_node(var_name, concept=concept, is_root=(root_var is None))

                if root_var is None:
                    root_var = var_name

                # If there's a parent and a role, add edge
                if node_stack and current_role:
                    parent_var = node_stack[-1]
                    # Direction: parent --role--> child
                    if current_role.startswith(':ARG') and current_role.endswith('-of'):
                        # Inverse role: child -> parent
                        actual_role = current_role[:-3]  # Remove '-of'
                        G.add_edge(var_name, parent_var, role=actual_role)
                    else:
                        G.add_edge(parent_var, var_name, role=current_role)
                    current_role = None

                node_stack.append(var_name)
                idx += 1

            elif tok == ')':
                if node_stack:
                    node_stack.pop()
                current_role = None
                idx += 1

            elif tok.startswith(':'):
                current_role = tok
                idx += 1

            elif tok.startswith('"'):
                # String value - treat as a leaf node
                if node_stack and current_role:
                    parent_var = node_stack[-1]
                    # Create a unique node for the string value
                    str_val = tok.strip('"')
                    str_node_id = f"_str_{len(G.nodes)}_{str_val}"
                    G.add_node(str_node_id, concept=str_val, is_root=False)
                    G.add_edge(parent_var, str_node_id, role=current_role)
                    current_role = None
                idx += 1

            elif tok == '/':
                idx += 1

            else:
                # Could be: re-entrant variable, number, or concept
                if node_stack and current_role:
                    parent_var = node_stack[-1]
                    if tok in var_to_concept:
                        # Re-entrant reference
                        if current_role.startswith(':ARG') and current_role.endswith('-of'):
                            actual_role = current_role[:-3]
                            G.add_edge(tok, parent_var, role=actual_role)
                        else:
                            G.add_edge(parent_var, tok, role=current_role)
                    else:
                        # Atomic value (number, constant, etc.)
                        val_node_id = f"_val_{len(G.nodes)}_{tok}"
                        G.add_node(val_node_id, concept=tok, is_root=False)
                        G.add_edge(parent_var, val_node_id, role=current_role)
                    current_role = None
                idx += 1

        if len(G.nodes) == 0:
            return None

        return G

    except Exception:
        return None


# ─────────────────────────────────────────────────────────
# 2. Semantically-Preserving Graph (SPG) Transformation
#    Following SAFT paper Section 3.1
# ─────────────────────────────────────────────────────────

def transform_to_spg(G: nx.DiGraph) -> nx.DiGraph:
    """
    Transform AMR DiGraph into a Semantically-Preserving Graph (SPG).

    Following SAFT paper:
    - Edge labels → role nodes (edge-to-node conversion)
    - Result: unlabeled directed graph suitable for Laplacian computation

    Each labeled edge (u --:ARG0--> v) becomes:
        u --> role_node_ARG0 --> v
    """
    SPG = nx.DiGraph()

    # Copy all original nodes
    for node, data in G.nodes(data=True):
        SPG.add_node(node, **data, node_type='concept')

    # Convert each labeled edge into role node
    role_counter = 0
    for u, v, data in G.edges(data=True):
        role = data.get('role', ':UNK')
        role_node_id = f"_role_{role_counter}_{role}"
        role_counter += 1

        # Add role node
        SPG.add_node(role_node_id, concept=role, is_root=False, node_type='role')

        # Add unlabeled edges: u -> role_node -> v
        SPG.add_edge(u, role_node_id)
        SPG.add_edge(role_node_id, v)

    return SPG


# ─────────────────────────────────────────────────────────
# 3. Magnetic Laplacian Computation
#    Following SAFT paper Section 2.1 & 3.2
# ─────────────────────────────────────────────────────────

def compute_magnetic_laplacian(G: nx.DiGraph, q: float = 0.25) -> np.ndarray:
    """
    Compute the magnetic Laplacian L^(q) of a directed graph.

    L^(q) = D_S - D_S^{-1/2} * (A_S ⊙ exp(iΘ^(q))) * D_S^{-1/2}

    where:
    - A_S = A ∨ A^T (symmetrized adjacency)
    - Θ^(q)_{u,v} = 2πq(A_{u,v} - A_{v,u})
    - D_S is the degree matrix of A_S

    Args:
        G: NetworkX DiGraph
        q: magnetic parameter (default 0.25 from paper)

    Returns:
        L_q: complex-valued magnetic Laplacian matrix (n x n)
    """
    nodes = list(G.nodes())
    n = len(nodes)

    if n == 0:
        return np.array([[]], dtype=complex)

    if n == 1:
        return np.array([[0.0]], dtype=complex)

    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Adjacency matrix
    A = np.zeros((n, n), dtype=float)
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        A[i, j] = 1.0

    # Symmetrized adjacency (element-wise OR)
    A_S = np.maximum(A, A.T)

    # Degree matrix of A_S
    d_S = A_S.sum(axis=1)

    # Handle isolated nodes
    d_S_inv_sqrt = np.zeros(n, dtype=float)
    nonzero = d_S > 0
    d_S_inv_sqrt[nonzero] = 1.0 / np.sqrt(d_S[nonzero])
    D_S_inv_sqrt = np.diag(d_S_inv_sqrt)

    # Phase matrix Θ^(q)
    Theta = 2 * np.pi * q * (A - A.T)

    # Magnetic adjacency: A_S ⊙ exp(iΘ)
    A_mag = A_S * np.exp(1j * Theta)

    # Magnetic Laplacian: L^(q) = I - D_S^{-1/2} A_mag D_S^{-1/2}
    L_q = np.eye(n, dtype=complex) - D_S_inv_sqrt @ A_mag @ D_S_inv_sqrt

    return L_q


def extract_spectral_features(G: nx.DiGraph, k: int = 30, q: float = 0.25) -> Dict:
    """
    Extract spectral features from the magnetic Laplacian.

    Returns dict with:
    - 'node_pe': Dict[node_id -> np.array of shape (2k,)] real-valued PE
    - 'node_list': ordered list of nodes
    - 'eigenvalues': k smallest eigenvalues
    """
    nodes = list(G.nodes())
    n = len(nodes)

    if n <= 1:
        # Trivial case
        pe = {node: np.zeros(2 * k, dtype=float) for node in nodes}
        return {
            'node_pe': pe,
            'node_list': nodes,
            'eigenvalues': np.zeros(min(k, n)),
        }

    L_q = compute_magnetic_laplacian(G, q=q)

    # Eigendecomposition of Hermitian matrix
    actual_k = min(k, n)
    eigenvalues, eigenvectors = np.linalg.eigh(L_q)

    # Take k smallest eigenvectors
    idx_sorted = np.argsort(eigenvalues.real)[:actual_k]
    selected_eigenvalues = eigenvalues[idx_sorted].real
    selected_eigenvectors = eigenvectors[:, idx_sorted]  # shape: (n, actual_k)

    # Convert complex eigenvectors to real: concat real + imaginary parts
    # PE(v_i) = [Re(Γ_i), Im(Γ_i)] ∈ R^{2k}
    node_pe = {}
    for i, node in enumerate(nodes):
        ev = selected_eigenvectors[i, :]  # shape: (actual_k,)
        pe_real = ev.real
        pe_imag = ev.imag
        pe = np.concatenate([pe_real, pe_imag])

        # Pad if actual_k < k
        if actual_k < k:
            pe = np.pad(pe, (0, 2 * k - 2 * actual_k), mode='constant')

        node_pe[node] = pe

    return {
        'node_pe': node_pe,
        'node_list': nodes,
        'eigenvalues': selected_eigenvalues,
    }


# ─────────────────────────────────────────────────────────
# 4. Node Feature Extraction (Depth + Cluster)
# ─────────────────────────────────────────────────────────

def compute_node_depth(G: nx.DiGraph) -> Dict[str, int]:
    """
    Compute BFS depth from root for each node.
    Root = node with is_root=True, or node with no incoming edges.

    For undirected BFS (to handle all nodes), we use the undirected version.
    """
    # Find root
    root = None
    for node, data in G.nodes(data=True):
        if data.get('is_root', False):
            root = node
            break

    if root is None:
        # Fallback: node with minimum in-degree
        in_degrees = dict(G.in_degree())
        if in_degrees:
            root = min(in_degrees, key=in_degrees.get)
        else:
            return {}

    # BFS on undirected version to reach all nodes
    G_undirected = G.to_undirected()
    depth = {root: 0}
    queue = [root]
    visited = {root}

    while queue:
        current = queue.pop(0)
        for neighbor in G_undirected.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                depth[neighbor] = depth[current] + 1
                queue.append(neighbor)

    # Assign max_depth + 1 for unreachable nodes
    max_depth = max(depth.values()) if depth else 0
    for node in G.nodes():
        if node not in depth:
            depth[node] = max_depth + 1

    return depth


def spectral_clustering(node_pe: Dict, n_clusters: int = 5) -> Dict[str, int]:
    """
    Simple k-means clustering on node positional encodings.
    Returns: Dict[node_id -> cluster_id]
    """
    if not node_pe:
        return {}

    nodes = list(node_pe.keys())
    if len(nodes) <= n_clusters:
        return {node: i for i, node in enumerate(nodes)}

    # Stack PEs into matrix
    pe_matrix = np.array([node_pe[n] for n in nodes])

    # Simple k-means (avoid sklearn dependency)
    cluster_ids = _simple_kmeans(pe_matrix, n_clusters)

    return {nodes[i]: cluster_ids[i] for i in range(len(nodes))}


def _simple_kmeans(X: np.ndarray, k: int, max_iter: int = 20) -> np.ndarray:
    """Minimal k-means without sklearn dependency."""
    n, d = X.shape
    if n <= k:
        return np.arange(n)

    # Initialize centroids randomly (deterministic seed)
    rng = np.random.RandomState(42)
    centroid_idx = rng.choice(n, k, replace=False)
    centroids = X[centroid_idx].copy()

    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        # Assign clusters
        for i in range(n):
            dists = np.sum((centroids - X[i]) ** 2, axis=1)
            labels[i] = np.argmin(dists)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            mask = labels == c
            if mask.any():
                new_centroids[c] = X[mask].mean(axis=0)
            else:
                new_centroids[c] = centroids[c]

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels


# ─────────────────────────────────────────────────────────
# 5. File Reader: Parse -amr.txt files
# ─────────────────────────────────────────────────────────

def read_amr_file(filepath: str) -> List[Tuple[int, str, str]]:
    """
    Read an -amr.txt file and extract individual AMR entries.

    Returns list of tuples: (id, sentence, penman_string)
    """
    entries = []
    current_id = None
    current_snt = None
    current_amr_lines = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\r\n')

            if line.startswith('# ::id '):
                # Save previous entry
                if current_amr_lines and current_id is not None:
                    amr_str = '\n'.join(current_amr_lines).strip()
                    if amr_str:
                        entries.append((current_id, current_snt or '', amr_str))

                current_id = int(line.split('# ::id ')[1].strip())
                current_snt = None
                current_amr_lines = []

            elif line.startswith('# ::snt '):
                current_snt = line.split('# ::snt ')[1].strip()

            elif line.startswith('# ::'):
                # Other metadata, skip
                continue

            elif line.strip():
                current_amr_lines.append(line)

        # Save last entry
        if current_amr_lines and current_id is not None:
            amr_str = '\n'.join(current_amr_lines).strip()
            if amr_str:
                entries.append((current_id, current_snt or '', amr_str))

    return entries


# ─────────────────────────────────────────────────────────
# 6. Full Pipeline: AMR → Graph → Spectral Features
# ─────────────────────────────────────────────────────────

def extract_structural_features(penman_str: str,
                                 k_eigenvectors: int = 30,
                                 n_clusters: int = 5,
                                 q: float = 0.25) -> Optional[Dict]:
    """
    Full pipeline: Penman AMR → Graph → SPG → Magnetic Laplacian → Features.

    Returns dict with:
    - 'graph': original AMR DiGraph
    - 'spg': Semantically-Preserving Graph
    - 'node_depth': Dict[node -> depth]
    - 'node_cluster': Dict[node -> cluster_id]
    - 'node_pe': Dict[node -> PE vector]
    - 'concept_nodes': Dict[node -> concept] (only concept nodes, not role nodes)
    """
    # Step 1: Parse Penman → DiGraph
    G = parse_penman_to_graph(penman_str)
    if G is None or len(G.nodes) == 0:
        return None

    # Step 2: Transform to SPG (edge labels → role nodes)
    SPG = transform_to_spg(G)

    # Step 3: Compute spectral features on SPG
    spectral = extract_spectral_features(SPG, k=k_eigenvectors, q=q)

    # Step 4: Node depth on original graph
    node_depth = compute_node_depth(G)

    # Step 5: Spectral clustering on SPG node PEs
    # But we only care about concept nodes (not role nodes)
    concept_pe = {
        n: spectral['node_pe'][n]
        for n in G.nodes()
        if n in spectral['node_pe']
    }
    node_cluster = spectral_clustering(concept_pe, n_clusters=n_clusters)

    # Concept node info
    concept_nodes = {
        n: G.nodes[n].get('concept', n) for n in G.nodes()
    }

    return {
        'graph': G,
        'spg': SPG,
        'node_depth': node_depth,
        'node_cluster': node_cluster,
        'node_pe': concept_pe,
        'concept_nodes': concept_nodes,
    }


# ─────────────────────────────────────────────────────────
# 7. Quick Test
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    # Test with a simple AMR
    test_amr = """(z0 / muốn
    :ARG0 (z1 / tôi)
    :ARG1 (z2 / cho
              :ARG0 z1
              :ARG1 (z3 / biết
                        :ARG0 (z4 / bạn)
                        :ARG1 (z5 / to_lớn
                                  :ARG1 (z6 / nỗ_lực
                                            :mod (z7 / khoa_học))))
              :ARG2 z4))"""

    print("=" * 60)
    print("Testing AMR Parser + Magnetic Laplacian Features")
    print("=" * 60)

    features = extract_structural_features(test_amr, k_eigenvectors=10, n_clusters=3)

    if features:
        G = features['graph']
        print(f"\n[STATS] Original Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        print("\n[NODES] Concept Nodes:")
        for node, concept in features['concept_nodes'].items():
            depth = features['node_depth'].get(node, -1)
            cluster = features['node_cluster'].get(node, -1)
            print(f"  {node}: concept={concept}, depth={depth}, cluster={cluster}")

        SPG = features['spg']
        print(f"\n[SPG] SPG: {SPG.number_of_nodes()} nodes, {SPG.number_of_edges()} edges")

        print("\n[OK] All tests passed!")
    else:
        print("[FAIL] Parsing failed!")
