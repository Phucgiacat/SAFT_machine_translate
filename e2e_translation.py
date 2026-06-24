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

Optimizations (--fast flag):
    - AMRBART loaded in float16 for faster AMR parsing
    - torch.backends.cudnn.benchmark enabled
    - AMR parse cache (avoids re-parsing identical sentences)
    - Reduced AMRBART beam search (5 -> 3)
    - torch.compile on AMRBART (optional, via --compile)
"""

import os
import sys
import argparse
import time
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


# ─────────────────────────────────────────────────────────
# AMR Parse Cache — avoid re-parsing identical sentences
# ─────────────────────────────────────────────────────────
class AMRCache:
    """Simple in-memory cache for AMR parse results."""
    def __init__(self, maxsize=256):
        self._cache = {}
        self._maxsize = maxsize

    def get(self, text):
        return self._cache.get(text)

    def put(self, text, amr):
        if len(self._cache) >= self._maxsize:
            # Evict oldest entry (FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[text] = amr

    def __contains__(self, text):
        return text in self._cache


_amr_cache = AMRCache()

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


def parse_vi_to_amr(text, amr_model, amr_tokenizer, device, num_beams=5, use_cache=True):
    """Parse a Vietnamese sentence into a Penman AMR string using AMRBART.
    
    Args:
        text: Input Vietnamese text
        amr_model: AMRBART model
        amr_tokenizer: AMRBart tokenizer
        device: torch device
        num_beams: beam search width (reduce for speed, e.g. 3)
        use_cache: if True, cache results to avoid re-parsing identical sentences
    """
    global _amr_cache
    
    # Check cache first
    if use_cache and text in _amr_cache:
        return _amr_cache.get(text)
    
    import penman
    input_ids = amr_tokenizer.encode(text, return_tensors="pt").to(device)
    
    # Use autocast for mixed precision inference
    with torch.cuda.amp.autocast(enabled=device != "cpu"):
        outputs = amr_model.generate(input_ids, max_length=1024, num_beams=num_beams)
    
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
    
    result = penman.encode(graph).strip()
    
    # Store in cache
    if use_cache:
        _amr_cache.put(text, result)
    
    return result

@torch.inference_mode()
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
    parser.add_argument('--base-model', type=str, default=None, help='Override base model name from config (e.g., Qwen/Qwen2.5-7B-Instruct)')
    parser.add_argument('--amrbart-path', default='../AMRBART', help='Path to AMRBART repository for tokenizer')
    parser.add_argument('--amr-checkpoint', type=str, default=None, help='Path to custom AMR parser checkpoint (default: phucgiacat/AMRBART-parser-grpo from HuggingFace)')
    parser.add_argument('--translate', type=str, default=None, help='Vietnamese sentence to translate')
    parser.add_argument('--interactive', action='store_true', help='Start interactive mode')
    parser.add_argument('--fast', action='store_true', help='Enable inference optimizations (FP16 AMRBART, reduced beams, cache)')
    parser.add_argument('--compile', action='store_true', help='Apply torch.compile to AMRBART (requires PyTorch 2.0+, slow first run)')
    args = parser.parse_args()
    
    if not args.translate and not args.interactive:
        print("Please provide --translate or use --interactive.")
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Apply CUDA optimizations
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        if args.fast:
            print("⚡ Fast mode enabled: FP16 AMRBART, reduced beams, AMR cache")
    
    pipeline_start = time.time()
    
    # 1. Setup config
    config = get_config(args.brand)
    if args.base_model:
        config.model_name = args.base_model
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
        
    # Monkey-patch transformers to bypass AdamW/Adafactor import errors in AMRBART's constant.py
    import transformers
    if not hasattr(transformers, 'AdamW'):
        transformers.AdamW = None
    if not hasattr(transformers, 'Adafactor'):
        transformers.Adafactor = None
    
    try:
        from model_interface.tokenization_bart import AMRBartTokenizer
        
        # Patch AMRBartTokenizer.__init__ to avoid "multiple values for argument 'vocab'" in Colab's transformers
        def patched_init(self, *args, **kwargs):
            from transformers import BartTokenizer
            import regex as re
            from common.constant import recategorizations
            
            # Avoid multiple values error if both positional and kwarg are passed
            if len(args) > 0:
                kwargs.pop('vocab', None)
                kwargs.pop('vocab_file', None)
                
            BartTokenizer.__init__(self, *args, **kwargs)
            
            # In some transformers versions, self.encoder is not set or replaced by self.vocab
            if not hasattr(self, 'encoder'):
                self.encoder = self.get_vocab().copy()
            if not hasattr(self, 'decoder'):
                self.decoder = {v: k for k, v in self.encoder.items()}
                
            self.modified = 0
            self.recategorizations = set(recategorizations)
            self.patterns = re.compile(r""" ?<[a-z]+:?\d*>| ?:[^\s]+|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
            self.remove_pars = False
            
        AMRBartTokenizer.__init__ = patched_init
        
        # Patch AMRBartTokenizer.init_amr_vocabulary to avoid "AttributeError: property 'decoder' has no setter"
        def patched_init_amr_vocabulary(self):
            from common.constant import raw_special_tokens
            self.old_enc_size = old_enc_size = len(self.encoder)
            tokens = [t for t in raw_special_tokens if t not in self.encoder]

            for i, t in enumerate(tokens, start=old_enc_size):
                self.encoder[t] = i

            self.encoder = {k: i for i, (k,v) in enumerate(sorted(self.encoder.items(), key=lambda x: x[1]))}
            my_decoder = {v: k for k, v in sorted(self.encoder.items(), key=lambda x: x[1])}
            
            try:
                self.decoder = my_decoder
            except AttributeError:
                # If decoder is a read-only property in this transformers version, override it on the class
                self.__class__.decoder = property(lambda self: self._amr_decoder)
                self._amr_decoder = my_decoder

            self.modified = len(tokens)
            self.amr_bos_token = "<AMR>"
            self.amr_bos_token_id = self.encoder[self.amr_bos_token]
            self.amr_eos_token = "</AMR>"
            self.amr_eos_token_id = self.encoder[self.amr_eos_token]
            # print(f"Added {self.modified} AMR tokens")
            
        AMRBartTokenizer.init_amr_vocabulary = patched_init_amr_vocabulary
        
    except ImportError as e:
        print(f"Failed to import AMRBartTokenizer from {amrbart_finetune}: {e}")
        return

    amr_model_name = args.amr_checkpoint if args.amr_checkpoint else "phucgiacat/AMRBART-parser-grpo"
    print(f"    Loading AMR model from: {amr_model_name}")
    amr_tokenizer = AMRBartTokenizer.from_pretrained(amr_model_name)
    
    # Load AMRBART — use float16 in fast mode for ~2x speedup
    if args.fast and device == "cuda":
        amr_model = BartForConditionalGeneration.from_pretrained(
            amr_model_name, torch_dtype=torch.float16
        ).to(device)
        print("    ✓ AMRBART loaded in float16")
    else:
        amr_model = BartForConditionalGeneration.from_pretrained(amr_model_name).to(device)
    amr_model.eval()
    
    # Optional: torch.compile for further speedup (slow first run, fast subsequent)
    if args.compile:
        try:
            amr_model = torch.compile(amr_model, mode="reduce-overhead")
            print("    ✓ AMRBART compiled with torch.compile")
        except Exception as e:
            print(f"    [WARN] torch.compile failed: {e}")
    
    # Set AMR beam count — reduced in fast mode
    amr_num_beams = 3 if args.fast else 5
    print(f"    AMR beams: {amr_num_beams}")
    
    # 3. Load SAFT Translation model
    print("\nLoading Translation Model...")
    model, tokenizer = load_model(args.model_path, config=config, mode="saft")
    model.eval()
    
    print("\n" + "="*50)
    print(" Pipeline Ready!")
    print("="*50 + "\n")
    
    def process_sentence(vi_text):
        total_start = time.time()
        
        t0 = time.time()
        print(f"\n[0] Word Segmentation...")
        try:
            from pyvi import ViTokenizer
            segmented_text = ViTokenizer.tokenize(vi_text)
            print(f"    Segmented: {segmented_text}")
        except ImportError:
            print("    [WARN] pyvi not installed. Using raw text. Run '!pip install pyvi' on Colab.")
            segmented_text = vi_text
        print(f"    ⏱ {time.time() - t0:.2f}s")
            
        t1 = time.time()
        print(f"\n[1] Parsing to AMR...")
        cached = segmented_text in _amr_cache
        penman_amr = parse_vi_to_amr(
            segmented_text, amr_model, amr_tokenizer, device,
            num_beams=amr_num_beams, use_cache=True
        )
        print(f"    Raw AMR: {penman_amr}")
        print(f"    ⏱ {time.time() - t1:.2f}s" + (" (cached)" if cached else ""))
        
        t2 = time.time()
        print(f"[2] Linearizing AMR...")
        linear_amr = bfs_linearize(penman_amr)
        if not linear_amr:
            print("    Failed to linearize AMR. Falling back to plain text.")
            return None
        print(f"    Linear AMR: {linear_amr[:100]}...")
        print(f"    ⏱ {time.time() - t2:.2f}s")
        
        t3 = time.time()
        print(f"[3] Computing Positional Encodings (PE)...")
        pe_info = compute_pes_from_linear(linear_amr, k=config.k_eigenvectors, q=0.25)
        if not pe_info:
            print("    Failed to compute PE.")
            return None
        print(f"    Extracted {len(pe_info['labels'])} label PEs.")
        print(f"    ⏱ {time.time() - t3:.2f}s")
        
        t4 = time.time()
        print(f"[4] Generating Translation...")
        translation = translate_sentence_with_pe(
            model, tokenizer, vi_text, pe_info, config, max_seq=config.saft_max_seq
        )
        print(f"    ⏱ {time.time() - t4:.2f}s")
        
        print(f"\n    ⏱ Total pipeline: {time.time() - total_start:.2f}s")
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
