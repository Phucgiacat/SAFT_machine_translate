"""
SAFT Dataset — Node-to-Token Alignment
═════════════════════════════════════════════════════════
Custom PyTorch Dataset that:
1. Loads precomputed node-level PEs
2. Builds prompts with AMR + Vietnamese + English
3. Aligns AMR labels to token positions
4. Returns per-token PE vectors for embedding injection

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


def build_baseline_prompt_parts(vi_text: str, en_text: str = None):
    """Build prompt parts for Baseline mode. Returns (system, user_content, assistant)."""
    user_content = f"Translate the source text from Vietnamese to English.\nVietnamese: {vi_text}\nEnglish:"
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
    label_is_concept: List[bool],
    user_after_amr: str,
    en_text: str,
    max_seq_length: int = 1280,
    pe_dim: int = 40,
) -> Dict:
    """
    Tokenize the full prompt with per-label AMR alignment.

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
    amr_token_mask = []     # per-token concept mask

    for label_idx, label in enumerate(amr_labels):
        # Add space separator between labels (except first)
        if label_idx > 0:
            sep_ids = tokenizer.encode(" " + label, add_special_tokens=False)
        else:
            sep_ids = tokenizer.encode(label, add_special_tokens=False)
        n_tokens = len(sep_ids)

        is_concept = label_is_concept[label_idx]
        pe = label_pes[label_idx] if is_concept else np.zeros(pe_dim, dtype=np.float32)

        for j, tid in enumerate(sep_ids):
            amr_token_ids.append(tid)
            amr_token_pe.append(pe)
            amr_token_intra.append(j)  # intra-node position
            amr_token_mask.append(1.0 if is_concept else 0.0)

    # Combine all parts
    all_ids = prefix_ids + amr_token_ids + suffix_ids + response_ids

    # Truncate if needed
    if len(all_ids) > max_seq_length:
        # Truncate AMR tokens (keep prefix, suffix, response)
        max_amr_tokens = max_seq_length - len(prefix_ids) - len(suffix_ids) - len(response_ids)
        if max_amr_tokens < 10:
            # Can't fit, truncate response
            max_amr_tokens = max_seq_length // 2
            response_ids = response_ids[:max_seq_length - len(prefix_ids) - max_amr_tokens - len(suffix_ids)]

        amr_token_ids = amr_token_ids[:max_amr_tokens]
        amr_token_pe = amr_token_pe[:max_amr_tokens]
        amr_token_intra = amr_token_intra[:max_amr_tokens]
        amr_token_mask = amr_token_mask[:max_amr_tokens]
        all_ids = prefix_ids + amr_token_ids + suffix_ids + response_ids

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
    en_text: str,
    max_seq_length: int = 1280,
) -> Dict:
    """Tokenize baseline prompt (no AMR, no PE)."""
    system, user_content, assistant = build_baseline_prompt_parts(vi_text, en_text)

    prompt = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(f"{en_text}<|im_end|>", add_special_tokens=False)

    all_ids = prompt_ids + response_ids
    if len(all_ids) > max_seq_length:
        all_ids = all_ids[:max_seq_length]

    seq_len = len(all_ids)
    labels = [-100] * seq_len
    for j in range(len(prompt_ids), seq_len):
        labels[j] = all_ids[j]

    return {
        'input_ids': torch.tensor(all_ids, dtype=torch.long),
        'attention_mask': torch.ones(seq_len, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
    }


# ─────────────────────────────────────────────────────────
# 3. Datasets
# ─────────────────────────────────────────────────────────

class SAFTDataset(Dataset):
    """
    Dataset for SAFT training with AMR PE injection.
    Loads precomputed PEs and builds aligned prompts.
    """

    def __init__(
        self,
        vi_file: str,
        en_file: str,
        bpe_amr_file: str,
        pe_file: str,
        tokenizer,
        max_seq_length: int = 1280,
        k_eigenvectors: int = 20,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pe_dim = 2 * k_eigenvectors

        # Load text data
        with open(vi_file, 'r', encoding='utf-8') as f:
            self.vi_texts = [l.strip() for l in f]
        with open(en_file, 'r', encoding='utf-8') as f:
            self.en_texts = [l.strip() for l in f]
        with open(bpe_amr_file, 'r', encoding='utf-8') as f:
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

        print(f"  SAFTDataset: {n} samples, pe_dim={self.pe_dim}")

    def __len__(self):
        return len(self.vi_texts)

    def __getitem__(self, idx):
        vi = self.vi_texts[idx]
        en = self.en_texts[idx]
        amr = self.amr_texts[idx]
        pe_info = self.pe_data[idx]

        labels_list = pe_info['labels']
        label_pes = pe_info['label_pes']
        label_is_concept = pe_info['label_is_concept']

        result = tokenize_with_amr_alignment(
            tokenizer=self.tokenizer,
            system_msg=SYSTEM_MSG_SAFT,
            user_before_amr="AMR Graph:\n",
            amr_labels=labels_list,
            label_pes=label_pes,
            label_is_concept=label_is_concept,
            user_after_amr=f"\n\nVietnamese: {vi}\nEnglish:",
            en_text=en,
            max_seq_length=self.max_seq_length,
            pe_dim=self.pe_dim,
        )
        return result


class BaselineDataset(Dataset):
    """Dataset for Baseline training (no AMR)."""

    def __init__(
        self,
        vi_file: str,
        en_file: str,
        tokenizer,
        max_seq_length: int = 768,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        with open(vi_file, 'r', encoding='utf-8') as f:
            self.vi_texts = [l.strip() for l in f]
        with open(en_file, 'r', encoding='utf-8') as f:
            self.en_texts = [l.strip() for l in f]

        n = min(len(self.vi_texts), len(self.en_texts))
        self.vi_texts = self.vi_texts[:n]
        self.en_texts = self.en_texts[:n]

        print(f"  BaselineDataset: {n} samples")

    def __len__(self):
        return len(self.vi_texts)

    def __getitem__(self, idx):
        return tokenize_baseline(
            self.tokenizer, self.vi_texts[idx], self.en_texts[idx],
            self.max_seq_length,
        )


# ─────────────────────────────────────────────────────────
# 4. Collate functions
# ─────────────────────────────────────────────────────────

def saft_collate_fn(batch: List[Dict], pad_token_id: int = 0, pe_dim: int = 40) -> Dict:
    """Collate SAFT samples with padding."""
    max_len = max(b['input_ids'].size(0) for b in batch)
    bs = len(batch)

    input_ids = torch.full((bs, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(bs, max_len, dtype=torch.long)
    labels = torch.full((bs, max_len), -100, dtype=torch.long)
    amr_node_pe = torch.zeros(bs, max_len, pe_dim, dtype=torch.float32)
    amr_intra_pos = torch.zeros(bs, max_len, dtype=torch.long)
    amr_mask = torch.zeros(bs, max_len, dtype=torch.float32)

    for i, b in enumerate(batch):
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


def baseline_collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict:
    """Collate Baseline samples with padding."""
    max_len = max(b['input_ids'].size(0) for b in batch)
    bs = len(batch)

    input_ids = torch.full((bs, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(bs, max_len, dtype=torch.long)
    labels = torch.full((bs, max_len), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b['input_ids'].size(0)
        input_ids[i, :seq_len] = b['input_ids']
        attention_mask[i, :seq_len] = b['attention_mask']
        labels[i, :seq_len] = b['labels']

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }
