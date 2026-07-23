"""
SAFT Dataset — Node-to-Token Alignment (BFS Linearization)
═════════════════════════════════════════════════════════
Custom PyTorch Dataset that:
1. Loads precomputed node-level PEs from BFS-linearized AMR
2. Builds prompts with AMR + Vietnamese + English
3. Aligns AMR labels to token positions (bijective: label i ↔ node i)
4. Returns per-token PE vectors for embedding injection
5. ALL labels get PE (concept + role + <stop> + <P>)

Supports both SAFT (with PEs) and Baseline (without AMR) modes.
═════════════════════════════════════════════════════════
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────
# 1. Prompt Templates
# ─────────────────────────────────────────────────────────

SYSTEM_MSG_SAFT = (
    "You are an expert Vietnamese-to-English translation assistant. "
    "You are given an Abstract Meaning Representation (AMR) graph of the source sentence. "
    "Use the AMR as a semantic blueprint to produce an accurate, fluent English translation."
)

SYSTEM_MSG_BASELINE = "You are a helpful translation assistant."


def build_saft_prompt_parts(vi_text: str, amr_text: str, en_text: str = None):
    """Build prompt parts for SAFT mode. Returns (system, user_before_amr, amr, user_after_amr, assistant)."""
    user_before = "AMR Graph:\n"
    user_after = f"\n\nVietnamese: {vi_text}\nEnglish:"
    return SYSTEM_MSG_SAFT, user_before, amr_text, user_after, en_text


def build_baseline_prompt_parts(vi_text: str, amr_text: str, en_text: str = None):
    """Build prompt parts for Baseline mode. Returns (system, user_content, assistant)."""
    user_content = f"Translate the source text from Vietnamese to English. Using AMR: {amr_text}\nsource: {vi_text}\nEnglish:"
    return SYSTEM_MSG_BASELINE, user_content, en_text


# ─────────────────────────────────────────────────────────
# 2. Tokenization with AMR alignment
# ─────────────────────────────────────────────────────────

def tokenize_with_amr_alignment(
    tokenizer,
    system_msg: str,
    user_before_amr: str,
    amr_labels: List[str],
    label_pes: np.ndarray,       # (n_labels, 2k)
    user_after_amr: str,
    en_text: str,
    max_seq_length: int = 1280,
    pe_dim: int = 40,
) -> Dict:
    """
    Tokenize the full prompt with per-label AMR alignment.
    All AMR labels get PE (bijective alignment from paper).

    Returns dict with:
        input_ids, attention_mask, labels,
        amr_node_pe, amr_intra_pos, amr_mask
    """
    # Tokenize non-AMR parts
    system_prefix = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_before_amr}"
    prefix_ids = tokenizer.encode(system_prefix, add_special_tokens=False)

    suffix_text = f"{user_after_amr}<|im_end|>\n<|im_start|>assistant\n"
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)

    response_ids = tokenizer.encode(f"{en_text}<|im_end|>", add_special_tokens=False)

    # Tokenize each AMR label individually to ensure perfect alignment
    amr_token_ids = []
    amr_token_pe = []       # per-token PE vectors
    amr_token_intra = []    # per-token intra-node positions
    amr_token_mask = []     # per-token AMR mask (1.0 for ALL labels)

    label_boundaries = []  # Track where each label's tokens start (for clean truncation)

    for label_idx, label in enumerate(amr_labels):
        label_boundaries.append(len(amr_token_ids))
        # Add space separator between labels (except first)
        if label_idx > 0:
            sep_ids = tokenizer.encode(" " + label, add_special_tokens=False)
        else:
            sep_ids = tokenizer.encode(label, add_special_tokens=False)

        # All labels get PE (paper: bijective alignment)
        pe = label_pes[label_idx]

        for j, tid in enumerate(sep_ids):
            amr_token_ids.append(tid)
            amr_token_pe.append(pe)
            amr_token_intra.append(j)  # intra-node position
            amr_token_mask.append(1.0)  # ALL AMR labels get mask=1.0

    # Combine all parts
    all_ids = prefix_ids + amr_token_ids + suffix_ids + response_ids

    # ── Smart truncation: priority = prefix+suffix (fixed) > response > AMR ──
    if len(all_ids) > max_seq_length:
        fixed_overhead = len(prefix_ids) + len(suffix_ids)
        budget = max_seq_length - fixed_overhead

        if budget < 20:
            # Edge case: structural tokens alone nearly fill the limit
            all_ids = all_ids[:max_seq_length]
        else:
            amr_len = len(amr_token_ids)
            resp_len = len(response_ids)

            if resp_len <= budget - 10:
                # Common case: keep full response, truncate AMR only
                amr_budget = budget - resp_len
            else:
                # Both need cutting: 30% to response (min 20), rest to AMR
                resp_budget = max(budget * 3 // 10, min(20, budget))
                amr_budget = budget - resp_budget
                response_ids = response_ids[:resp_budget]

            # Truncate AMR at label boundary (avoid cutting mid-label)
            if amr_budget <= 0:
                amr_token_ids, amr_token_pe = [], []
                amr_token_intra, amr_token_mask = [], []
            elif amr_budget < amr_len:
                cut_at = 0
                for lb in label_boundaries:
                    if lb <= amr_budget:
                        cut_at = lb
                    else:
                        break
                if cut_at == 0:
                    cut_at = amr_budget  # fallback: hard cut
                amr_token_ids = amr_token_ids[:cut_at]
                amr_token_pe = amr_token_pe[:cut_at]
                amr_token_intra = amr_token_intra[:cut_at]
                amr_token_mask = amr_token_mask[:cut_at]

            all_ids = prefix_ids + amr_token_ids + suffix_ids + response_ids

    # Final safety truncation (should not trigger with correct logic above)
    if len(all_ids) > max_seq_length:
        all_ids = all_ids[:max_seq_length]

    seq_len = len(all_ids)
    amr_start = len(prefix_ids)
    amr_end = amr_start + len(amr_token_ids)

    # Build per-token PE arrays for full sequence
    full_pe = np.zeros((seq_len, pe_dim), dtype=np.float32)
    full_intra = np.zeros(seq_len, dtype=np.int64)
    full_mask = np.zeros(seq_len, dtype=np.float32)

    for j in range(len(amr_token_ids)):
        pos = amr_start + j
        if pos < seq_len:
            full_pe[pos] = amr_token_pe[j]
            full_intra[pos] = amr_token_intra[j]
            full_mask[pos] = amr_token_mask[j]

    # Build labels: -100 for all non-response tokens
    response_start = len(prefix_ids) + len(amr_token_ids) + len(suffix_ids)
    labels = [-100] * seq_len
    for j in range(response_start, seq_len):
        labels[j] = all_ids[j]

    return {
        'input_ids': torch.tensor(all_ids, dtype=torch.long),
        'attention_mask': torch.ones(seq_len, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
        'amr_node_pe': torch.tensor(full_pe, dtype=torch.float32),
        'amr_intra_pos': torch.tensor(full_intra, dtype=torch.long),
        'amr_mask': torch.tensor(full_mask, dtype=torch.float32),
    }


def tokenize_baseline(
    tokenizer,
    vi_text: str,
    amr_text: str,
    en_text: str,
    max_seq_length: int = 1280,
) -> Dict:
    """Tokenize baseline prompt with AMR text prompt (no PE)."""
    system, user_content, assistant = build_baseline_prompt_parts(vi_text, amr_text, en_text)

    struct_prefix = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    prefix_ids = tokenizer.encode(struct_prefix, add_special_tokens=False)
    response_ids = tokenizer.encode(f"{en_text}<|im_end|>", add_special_tokens=False)

    all_ids = prefix_ids + response_ids
    if len(all_ids) > max_seq_length:
        all_ids = all_ids[:max_seq_length]

    seq_len = len(all_ids)
    labels = [-100] * seq_len
    for j in range(len(prefix_ids), seq_len):
        labels[j] = all_ids[j]

    return {
        'input_ids': torch.tensor(all_ids, dtype=torch.long),
        'attention_mask': torch.ones(seq_len, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
    }


# ─────────────────────────────────────────────────────────
# 3. AMR Chunking (split at <stop> boundaries, no info loss)
# ─────────────────────────────────────────────────────────

def chunk_amr_at_stop_boundaries(
    labels: List[str],
    label_pes: np.ndarray,
    max_labels_per_chunk: int,
    max_chunks: int = 3,
) -> List[Tuple[List[str], np.ndarray]]:
    """
    Split AMR labels into chunks at <stop> boundaries (BFS segment boundaries).
    Preserves complete BFS segments within each chunk — no information loss.

    Each BFS segment ends with <stop>, representing one node's expansion.
    Segments are greedily grouped to fit within max_labels_per_chunk.

    Args:
        labels: BFS-linearized AMR labels
        label_pes: Per-label PE vectors (n_labels, pe_dim)
        max_labels_per_chunk: Max labels per chunk (estimated from token budget)
        max_chunks: Cap on number of chunks (excess merges into last chunk)

    Returns:
        List of (labels_chunk, pes_chunk) tuples
    """
    if len(labels) <= max_labels_per_chunk:
        return [(labels, label_pes)]

    # Find segment boundaries (position after each <stop>)
    boundaries = [0]
    for i, label in enumerate(labels):
        if label == '<stop>':
            boundaries.append(i + 1)
    if boundaries[-1] < len(labels):
        boundaries.append(len(labels))

    # Greedily merge consecutive segments into chunks that fit
    chunks = []
    chunk_start_idx = 0  # index into boundaries list

    for i in range(1, len(boundaries)):
        current_size = boundaries[i] - boundaries[chunk_start_idx]

        if current_size > max_labels_per_chunk and i - 1 > chunk_start_idx:
            # Adding this segment would exceed limit → flush accumulated segments
            start = boundaries[chunk_start_idx]
            end = boundaries[i - 1]
            chunks.append((labels[start:end], label_pes[start:end]))
            chunk_start_idx = i - 1

    # Add remaining labels as last chunk
    if chunk_start_idx < len(boundaries) - 1:
        start = boundaries[chunk_start_idx]
        end = boundaries[-1]
        chunks.append((labels[start:end], label_pes[start:end]))

    if not chunks:
        return [(labels, label_pes)]

    # Cap at max_chunks (merge excess into last chunk)
    if len(chunks) > max_chunks:
        merged_labels = []
        merged_pes_list = []
        for c_labels, c_pes in chunks[max_chunks - 1:]:
            merged_labels.extend(c_labels)
            merged_pes_list.append(c_pes)
        chunks = chunks[:max_chunks - 1]
        chunks.append((merged_labels, np.concatenate(merged_pes_list, axis=0)))

    return chunks


# ─────────────────────────────────────────────────────────
# 4. Datasets
# ─────────────────────────────────────────────────────────

class SAFTDataset(Dataset):
    """
    Dataset for SAFT training with AMR PE injection.
    Uses BFS-linearized AMR with bijective PE alignment.

    Supports chunking: when AMR is too long to fit in max_seq_length,
    it is split at <stop> boundaries into multiple training samples.
    Each chunk gets partial AMR (with correct PEs) + full Vietnamese + full English.
    No information is lost — all AMR content is covered across chunks.
    """

    def __init__(
        self,
        vi_file: str,
        en_file: str,
        linear_amr_file: str,
        pe_file: str,
        tokenizer,
        max_seq_length: int = 1280,
        k_eigenvectors: int = 20,
        max_chunks: int = 3,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pe_dim = 2 * k_eigenvectors
        self.max_chunks = max_chunks

        # Load text data
        with open(vi_file, 'r', encoding='utf-8') as f:
            self.vi_texts = [l.strip() for l in f]
        with open(en_file, 'r', encoding='utf-8') as f:
            self.en_texts = [l.strip() for l in f]
        with open(linear_amr_file, 'r', encoding='utf-8') as f:
            self.amr_texts = [l.strip() for l in f]

        # Load precomputed PEs
        with open(pe_file, 'rb') as f:
            self.pe_data = pickle.load(f)

        n = min(len(self.vi_texts), len(self.en_texts),
                len(self.amr_texts), len(self.pe_data))
        self.vi_texts = self.vi_texts[:n]
        self.en_texts = self.en_texts[:n]
        self.amr_texts = self.amr_texts[:n]
        self.pe_data = self.pe_data[:n]

        # Pre-compute response token lengths for chunking decisions
        self._response_lens = [
            len(tokenizer.encode(en, add_special_tokens=False))
            for en in self.en_texts
        ]

        # Estimate fixed overhead (system + user structure + suffix + closing)
        overhead_text = (
            f"<|im_start|>system\n{SYSTEM_MSG_SAFT}<|im_end|>\n"
            f"<|im_start|>user\nAMR Graph:\n\n\n"
            f"Vietnamese: \nEnglish:<|im_end|>\n"
            f"<|im_start|>assistant\n<|im_end|>"
        )
        self._fixed_overhead = len(tokenizer.encode(overhead_text, add_special_tokens=False))

        print(f"  SAFTDataset (BFS): {n} samples, pe_dim={self.pe_dim}, max_chunks={max_chunks}")

    def __len__(self):
        return len(self.vi_texts)

    def __getitem__(self, idx):
        vi = self.vi_texts[idx]
        en = self.en_texts[idx]
        pe_info = self.pe_data[idx]

        labels_list = pe_info['labels']
        label_pes = pe_info['label_pes']

        # Estimate total tokens to decide if chunking is needed
        vi_token_len = len(self.tokenizer.encode(vi, add_special_tokens=False))
        resp_len = self._response_lens[idx]
        estimated_amr = len(labels_list) * 2  # rough: ~2 tokens per label
        estimated_total = self._fixed_overhead + vi_token_len + resp_len + estimated_amr

        if estimated_total <= self.max_seq_length:
            # Fits in one sample — no chunking needed
            result = tokenize_with_amr_alignment(
                tokenizer=self.tokenizer,
                system_msg=SYSTEM_MSG_SAFT,
                user_before_amr="AMR Graph:\n",
                amr_labels=labels_list,
                label_pes=label_pes,
                user_after_amr=f"\n\nVietnamese: {vi}\nEnglish:",
                en_text=en,
                max_seq_length=self.max_seq_length,
                pe_dim=self.pe_dim,
            )
            return [result]

        # ── Chunking: split AMR at <stop> boundaries ──
        non_amr_tokens = self._fixed_overhead + vi_token_len + resp_len
        amr_label_budget = max(5, (self.max_seq_length - non_amr_tokens) // 2)

        chunks = chunk_amr_at_stop_boundaries(
            labels_list, label_pes,
            max_labels_per_chunk=amr_label_budget,
            max_chunks=self.max_chunks,
        )

        results = []
        for chunk_labels, chunk_pes in chunks:
            result = tokenize_with_amr_alignment(
                tokenizer=self.tokenizer,
                system_msg=SYSTEM_MSG_SAFT,
                user_before_amr="AMR Graph:\n",
                amr_labels=list(chunk_labels),
                label_pes=chunk_pes,
                user_after_amr=f"\n\nVietnamese: {vi}\nEnglish:",
                en_text=en,
                max_seq_length=self.max_seq_length,
                pe_dim=self.pe_dim,
            )
            results.append(result)

        return results


class BaselineDataset(Dataset):
    """Dataset for Baseline training (with AMR text)."""

    def __init__(
        self,
        vi_file: str,
        amr_file: str,
        en_file: str,
        tokenizer,
        max_seq_length: int = 768,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        with open(vi_file, 'r', encoding='utf-8') as f:
            self.vi_texts = [l.strip() for l in f]
        with open(amr_file, 'r', encoding='utf-8') as f:
            self.amr_texts = [l.strip() for l in f]
        with open(en_file, 'r', encoding='utf-8') as f:
            self.en_texts = [l.strip() for l in f]

        n = min(len(self.vi_texts), len(self.amr_texts), len(self.en_texts))
        self.vi_texts = self.vi_texts[:n]
        self.amr_texts = self.amr_texts[:n]
        self.en_texts = self.en_texts[:n]
        print(f"  BaselineDataset: {n} samples")
        
        # Print a sample prompt for debugging
        if n > 0:
            sys_msg, user_msg, _ = build_baseline_prompt_parts(self.vi_texts[0], self.amr_texts[0])
            sample_prompt = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{self.en_texts[0]}<|im_end|>"
            print("\n" + "═" * 60)
            print("  SAMPLE PROMPT FORMAT (BaselineDataset):")
            print("═" * 60)
            print(sample_prompt)
            print("═" * 60 + "\n")

    def __len__(self):
        return len(self.vi_texts)

    def __getitem__(self, idx):
        result = tokenize_baseline(
            self.tokenizer, self.vi_texts[idx], self.amr_texts[idx], self.en_texts[idx],
            self.max_seq_length,
        )
        return [result]


# ─────────────────────────────────────────────────────────
# 5. Collate functions (support chunked samples)
# ─────────────────────────────────────────────────────────

def _flatten_batch(batch: List) -> List[Dict]:
    """Flatten a batch of lists-of-dicts into a single list of dicts.
    Handles both chunked samples (list of dicts) and single dicts."""
    flat = []
    for item in batch:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def saft_collate_fn(batch: List, pad_token_id: int = 0, pe_dim: int = 40) -> Dict:
    """Collate SAFT samples with padding. Handles chunked samples (list of lists)."""
    flat_batch = _flatten_batch(batch)

    max_len = max(b['input_ids'].size(0) for b in flat_batch)
    bs = len(flat_batch)

    input_ids = torch.full((bs, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(bs, max_len, dtype=torch.long)
    labels = torch.full((bs, max_len), -100, dtype=torch.long)
    amr_node_pe = torch.zeros(bs, max_len, pe_dim, dtype=torch.float32)
    amr_intra_pos = torch.zeros(bs, max_len, dtype=torch.long)
    amr_mask = torch.zeros(bs, max_len, dtype=torch.float32)

    for i, b in enumerate(flat_batch):
        seq_len = b['input_ids'].size(0)
        input_ids[i, :seq_len] = b['input_ids']
        attention_mask[i, :seq_len] = b['attention_mask']
        labels[i, :seq_len] = b['labels']
        amr_node_pe[i, :seq_len] = b['amr_node_pe']
        amr_intra_pos[i, :seq_len] = b['amr_intra_pos']
        amr_mask[i, :seq_len] = b['amr_mask']

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'amr_node_pe': amr_node_pe,
        'amr_intra_pos': amr_intra_pos,
        'amr_mask': amr_mask,
    }


def baseline_collate_fn(batch: List, pad_token_id: int = 0) -> Dict:
    """Collate Baseline samples with padding. Handles list format."""
    flat_batch = _flatten_batch(batch)

    max_len = max(b['input_ids'].size(0) for b in flat_batch)
    bs = len(flat_batch)

    input_ids = torch.full((bs, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(bs, max_len, dtype=torch.long)
    labels = torch.full((bs, max_len), -100, dtype=torch.long)

    for i, b in enumerate(flat_batch):
        seq_len = b['input_ids'].size(0)
        input_ids[i, :seq_len] = b['input_ids']
        attention_mask[i, :seq_len] = b['attention_mask']
        labels[i, :seq_len] = b['labels']

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }
