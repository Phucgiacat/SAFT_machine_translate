"""
End-to-End SAFT Translation Pipeline
═════════════════════════════════════════════════════════
This script combines all the steps of the SAFT machine translation pipeline:
1. English Sentence -> AMR parsing (using AMRBART)
2. AMR Graph -> BFS Linearization
3. Linearized AMR -> PE precomputation
4. Sentence + PE -> Qwen Translation Model -> Vietnamese Translation

Usage:
    python e2e_translation.py --model-path <path_to_qwen_model> --brand qwen2.5 --translate "Your English sentence."
"""

import os
import sys
import argparse
import torch
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Import SAFT modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saft_config import get_config
from saft_dataset import set_chat_format
from saft_eval import load_model
from saft_bfs_linearize import bfs_linearize
from saft_pe_precompute import compute_pes_from_linear
from transformers import BartForConditionalGeneration

SYSTEM_MSG_SAFT_VI2EN = (
    "You are an expert Vietnamese-to-English translation assistant. "
    "You are given an Abstract Meaning Representation (AMR) graph of the source sentence. "
    "Use the AMR as a semantic blueprint to produce an accurate, fluent English translation."
)

def _build_eval_prompt_with_pe_vi2en(tokenizer, vi_text, pe_info, max_length, k_eigenvectors=20):
    import numpy as np
    from saft_dataset import fmt
    pe_dim = 2 * k_eigenvectors
    labels_list = pe_info['labels']
    label_pes = pe_info['label_pes']

    f = fmt()
    prefix_text = (
        f"{f['sys_start']}{SYSTEM_MSG_SAFT_VI2EN}{f['sys_end']}"
        f"{f['user_start']}AMR Graph:\n"
    )
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

    suffix_text = (
        f"\n\nVietnamese: {vi_text}\nEnglish:{f['user_end']}"
        f"{f['asst_start']}"
    )
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)

    amr_token_ids = []
    amr_token_pe = []
    amr_token_intra = []
    label_boundaries = []

    for label_idx, label in enumerate(labels_list):
        label_boundaries.append(len(amr_token_ids))
        tids = tokenizer.encode((" " if label_idx > 0 else "") + label, add_special_tokens=False)
        pe = label_pes[label_idx]
        for j, tid in enumerate(tids):
            amr_token_ids.append(tid)
            amr_token_pe.append(pe)
            amr_token_intra.append(j)

    fixed_len = len(prefix_ids) + len(suffix_ids)
    amr_budget = max_length - fixed_len - 10

    if amr_budget <= 0:
        amr_token_ids, amr_token_pe, amr_token_intra = [], [], []
    elif len(amr_token_ids) > amr_budget:
        cut_at = next((lb for lb in reversed(label_boundaries) if lb <= amr_budget), amr_budget)
        amr_token_ids = amr_token_ids[:cut_at]
        amr_token_pe = amr_token_pe[:cut_at]
        amr_token_intra = amr_token_intra[:cut_at]

    all_ids = prefix_ids + amr_token_ids + suffix_ids
    if len(all_ids) > max_length:
        all_ids = all_ids[:max_length]

    seq_len = len(all_ids)
    amr_start = len(prefix_ids)

    full_pe = np.zeros((seq_len, pe_dim), dtype=np.float32)
    full_intra = np.zeros(seq_len, dtype=np.int64)
    full_mask = np.zeros(seq_len, dtype=np.float32)

    for j in range(len(amr_token_ids)):
        pos = amr_start + j
        if pos < seq_len:
            full_pe[pos] = amr_token_pe[j]
            full_intra[pos] = amr_token_intra[j]
            full_mask[pos] = 1.0

    return (
        torch.tensor(all_ids, dtype=torch.long),
        torch.tensor(full_pe, dtype=torch.float32),
        torch.tensor(full_intra, dtype=torch.long),
        torch.tensor(full_mask, dtype=torch.float32),
    )


def parse_vi_to_amr(text, amr_model, amr_tokenizer, device):
    """Parse a Vietnamese sentence into a Penman AMR string using AMRBART."""
    import penman
    input_ids = amr_tokenizer.encode(text, return_tensors="pt").to(device)
    # The AMRBART model generates the AMR graph
    outputs = amr_model.generate(input_ids, max_length=1024, num_beams=5)
    
    ith_pred = outputs[0].cpu().tolist()
    ith_pred[0] = amr_tokenizer.bos_token_id
    ith_pred = [
        amr_tokenizer.eos_token_id if itm == amr_tokenizer.amr_eos_token_id else itm
        for itm in ith_pred if itm != amr_tokenizer.pad_token_id
    ]
    
    graph, status, (lin, backr) = amr_tokenizer.decode_amr(ith_pred, restore_name_ops=False)
    graph.status = status
    graph.nodes = lin
    graph.backreferences = backr
    graph.tokens = ith_pred
    
    return penman.encode(graph).strip()

@torch.no_grad()
def translate_sentence_with_pe(model, tokenizer, vi_text, pe_info, config, max_seq=1280):
    """Generate translation using SAFT model with PE injection."""
    model.eval()
    device = next(model.parameters()).device
    
    # Build the prompt with PE alignment for Vietnamese-to-English
    ids, pe, intra, mask = _build_eval_prompt_with_pe_vi2en(
        tokenizer, vi_text, pe_info, max_seq, config.k_eigenvectors
    )
    
    # Add batch dimension
    input_ids = ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    amr_node_pe = pe.unsqueeze(0).to(device)
    amr_intra_pos = intra.unsqueeze(0).to(device)
    amr_mask = mask.unsqueeze(0).to(device)
    
    # Generate translation
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        amr_node_pe=amr_node_pe,
        amr_intra_pos=amr_intra_pos,
        amr_mask=amr_mask,
        max_new_tokens=config.max_new_tokens,
        num_beams=config.num_beams,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    input_len = input_ids.shape[-1]
    # Handle if generate returns full sequence or just new tokens
    if outputs[0].shape[0] > input_len:
        gen_ids = outputs[0][input_len:]
    else:
        gen_ids = outputs[0]
        
    translation = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return translation

def main():
    parser = argparse.ArgumentParser(description='E2E SAFT Translation Pipeline')
    parser.add_argument('--model-path', required=True, help='Path to saved translation best_model')
    parser.add_argument('--brand', default='qwen2.5', help='Model brand preset (e.g., qwen2.5)')
    parser.add_argument('--amrbart-path', default='../AMRBART', help='Path to AMRBART repository for tokenizer')
    parser.add_argument('--translate', type=str, default=None, help='Vietnamese sentence to translate')
    parser.add_argument('--interactive', action='store_true', help='Start interactive mode')
    args = parser.parse_args()
    
    if not args.translate and not args.interactive:
        print("Please provide --translate or use --interactive.")
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Setup config
    config = get_config(args.brand)
    set_chat_format(config.chat_format)
    
    # 2. Load AMRBART parser
    print("\nLoading AMRBART Parser...")
    
    # Import AMRBartTokenizer from AMRBART repo
    amrbart_repo = os.path.abspath(args.amrbart_path)
    amrbart_finetune = os.path.join(amrbart_repo, "fine-tune")
    
    if not os.path.exists(amrbart_repo) or not os.path.exists(amrbart_finetune):
        print(f"Error: AMRBART fine-tune directory not found at {amrbart_finetune}")
        print("Please clone https://github.com/goodbai-nlp/AMRBART.git and point --amrbart-path to it.")
        return
        
    if amrbart_finetune not in sys.path:
        sys.path.insert(0, amrbart_finetune)
    
    try:
        from model_interface.tokenization_bart import AMRBartTokenizer
    except ImportError as e:
        print(f"Failed to import AMRBartTokenizer from {amrbart_finetune}: {e}")
        return

    amr_model_name = "phucgiacat/AMRBART-parser-grpo"
    amr_tokenizer = AMRBartTokenizer.from_pretrained(amr_model_name)
    amr_model = BartForConditionalGeneration.from_pretrained(amr_model_name).to(device)
    amr_model.eval()
    
    # 3. Load SAFT Translation model
    print("\nLoading Translation Model...")
    model, tokenizer = load_model(args.model_path, config=config, mode="saft")
    model.eval()
    
    print("\n" + "="*50)
    print(" Pipeline Ready!")
    print("="*50 + "\n")
    
    def process_sentence(vi_text):
        print(f"\n[1] Parsing to AMR...")
        penman_amr = parse_vi_to_amr(vi_text, amr_model, amr_tokenizer, device)
        print(f"    Raw AMR: {penman_amr}")
        
        print(f"[2] Linearizing AMR...")
        linear_amr = bfs_linearize(penman_amr)
        if not linear_amr:
            print("    Failed to linearize AMR. Falling back to plain text.")
            return None
        print(f"    Linear AMR: {linear_amr[:100]}...")
        
        print(f"[3] Computing Positional Encodings (PE)...")
        pe_info = compute_pes_from_linear(linear_amr, k=config.k_eigenvectors, q=0.25)
        if not pe_info:
            print("    Failed to compute PE.")
            return None
        print(f"    Extracted {len(pe_info['labels'])} label PEs.")
        
        print(f"[4] Generating Translation...")
        translation = translate_sentence_with_pe(
            model, tokenizer, vi_text, pe_info, config, max_seq=config.saft_max_seq
        )
        return translation
        
    if args.translate:
        translation = process_sentence(args.translate)
        print(f"\n🇻🇳 Vietnamese: {args.translate}")
        print(f"🇬🇧 English:    {translation}")
        
    if args.interactive:
        while True:
            try:
                vi_text = input("\n🇻🇳 Enter Vietnamese text (or 'quit'): ").strip()
                if not vi_text or vi_text.lower() in ('quit', 'q', 'exit'):
                    break
                translation = process_sentence(vi_text)
                if translation:
                    print(f"🇬🇧 English: {translation}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
