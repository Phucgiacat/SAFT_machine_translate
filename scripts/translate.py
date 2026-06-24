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
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from saft.config import get_config
from saft.dataset import set_chat_format
from evaluate import load_model
from saft.amr.bfs import bfs_linearize
from saft.amr.pe import compute_pes_from_linear
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

def _build_eval_prompt_with_pe(tokenizer, src_text, pe_info, max_length, k_eigenvectors, src_lang="vi", tgt_lang="en"):
    import numpy as np
    from saft.dataset import fmt
    pe_dim = 2 * k_eigenvectors
    labels_list = pe_info['labels']
    label_pes = pe_info['label_pes']

    lang_map = {'en': 'English', 'vi': 'Vietnamese'}
    src = lang_map.get(src_lang, src_lang)
    tgt = lang_map.get(tgt_lang, tgt_lang)
    
    system_msg = (
        f"You are an expert {src}-to-{tgt} translation assistant. "
        "You are given an Abstract Meaning Representation (AMR) graph of the source sentence. "
        f"Use the AMR as a semantic blueprint to produce an accurate, fluent {tgt} translation."
    )

    f = fmt()
    prefix_text = (
        f"{f['sys_start']}{system_msg}{f['sys_end']}"
        f"{f['user_start']}AMR Graph:\n"
    )
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

    suffix_text = (
        f"\n\n{src}: {src_text}\n{tgt}:{f['user_end']}"
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


def get_dependency_mask(segmented_snt, tokenizer, phonlp_model, max_src_length=1024):
    words = segmented_snt.split()
    n = len(words)
    word_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        word_matrix[i][i] = 1
        
    if phonlp_model and n > 0:
        try:
            annotations = phonlp_model.annotate(segmented_snt)
            if len(annotations[0]) > 0:
                deps = annotations[3][0]
                if len(deps) == n:
                    for i, (head_idx, rel) in enumerate(deps):
                        head_idx = int(head_idx)
                        if head_idx > 0:
                            word_matrix[i][head_idx - 1] = 1
                            word_matrix[head_idx - 1][i] = 1
        except Exception:
            pass

    subword_to_word = []
    for w_idx, word in enumerate(words):
        prefix = " " if w_idx > 0 else ""
        tokens = tokenizer.tokenize(prefix + word)
        subword_to_word.extend([w_idx] * len(tokens))
        
    n_subwords = len(subword_to_word)
    sub_matrix = [[0] * n_subwords for _ in range(n_subwords)]
    if len(word_matrix) == n:
        for sw_i in range(n_subwords):
            for sw_j in range(n_subwords):
                w_i = subword_to_word[sw_i]
                w_j = subword_to_word[sw_j]
                if w_i < n and w_j < n:
                    sub_matrix[sw_i][sw_j] = word_matrix[w_i][w_j]
                    
    tokenized_txt = tokenizer(segmented_snt, max_length=max_src_length, padding=False, truncation=True)
    r_ids = tokenized_txt["input_ids"]
    
    final_len = len(r_ids)
    final_matrix = [[0] * final_len for _ in range(final_len)]
    
    for sw_i in range(min(n_subwords, final_len - 2)):
        for sw_j in range(min(n_subwords, final_len - 2)):
            final_matrix[sw_i + 1][sw_j + 1] = sub_matrix[sw_i][sw_j]
            
    import torch
    return torch.tensor([final_matrix], dtype=torch.float32)


def parse_vi_to_amr(text, amr_model, amr_tokenizer, device, num_beams=5, use_cache=True, phonlp_model=None):
    """Parse a Vietnamese sentence into a Penman AMR string using AMRBART.
    Includes Beam-search fallback and Semantic Concept Injection.
    """
    global _amr_cache
    if use_cache and text in _amr_cache:
        return _amr_cache.get(text)
    
    import penman
    input_ids = amr_tokenizer.encode(text, return_tensors="pt").to(device)
    
    dependency_mask = None
    if phonlp_model is not None:
        dependency_mask = get_dependency_mask(text, amr_tokenizer, phonlp_model).to(device)
    
    with torch.cuda.amp.autocast(enabled=device != "cpu"):
        # Bypass transformers strict kwargs validation for dependency_mask
        if hasattr(amr_model, "_validate_model_kwargs"):
            amr_model._validate_model_kwargs = lambda kwargs: None
            
        outputs = amr_model.generate(
            input_ids, 
            max_length=1024, 
            num_beams=num_beams,
            num_return_sequences=num_beams, # Enable returning all beams for fallback
            dependency_mask=dependency_mask
        )
    
    valid_graph = None
    best_ith_pred = None
    
    # Lớp 1: Beam-search Fallback
    for i in range(outputs.shape[0]):
        ith_pred = outputs[i].cpu().tolist()
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
        
        try:
            # Check valid penman encoding
            penman_str = penman.encode(graph).strip()
            valid_graph = graph
            best_ith_pred = ith_pred
            break
        except Exception:
            continue
            
    if not valid_graph:
        # Fallback to beam 0
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
        valid_graph = graph
        
    try:
        result = penman.encode(valid_graph).strip()
    except Exception:
        result = "(z1 / amr-empty)"
        
    # Lớp 2: Semantic Concept Injection
    # Lọc danh từ/động từ quan trọng (giả lập bằng từ > 2 ký tự)
    words = [w for w in text.split() if w.isalpha() and len(w) > 2]
    existing_concepts_str = result.lower()
    injections = []
    var_idx = 100
    for w in words:
        if w.lower() not in existing_concepts_str:
            injections.append(f":topic (z{var_idx} / {w})")
            var_idx += 1
            
    if injections and result != "(z1 / amr-empty)":
        parts = result.rsplit(')', 1)
        if len(parts) == 2:
            result = parts[0] + "\n      " + "\n      ".join(injections) + ")" + parts[1]
    
    if use_cache:
        _amr_cache.put(text, result)
    
    return result

@torch.inference_mode()
def translate_sentence_with_pe(model, tokenizer, src_text, pe_info, config, src_lang="vi", tgt_lang="en", max_seq=1280):
    """Generate translation using SAFT model with PE injection."""
    model.eval()
    device = next(model.parameters()).device
    
    # Build the prompt with PE alignment
    ids, pe, intra, mask = _build_eval_prompt_with_pe(
        tokenizer, src_text, pe_info, max_seq, config.k_eigenvectors, src_lang, tgt_lang
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
    parser.add_argument('--amr-checkpoint', type=str, default=None, help='Path to custom AMR parser checkpoint')
    parser.add_argument('--translate', type=str, default=None, help='Sentence to translate')
    parser.add_argument('--interactive', action='store_true', help='Start interactive mode')
    parser.add_argument('--fast', action='store_true', help='Enable inference optimizations')
    parser.add_argument('--compile', action='store_true', help='Apply torch.compile to AMRBART')
    parser.add_argument('--in', dest='src_lang', default='vi', help='Source language code (default: vi)')
    parser.add_argument('--out', dest='tgt_lang', default='en', help='Target language code (default: en)')
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
    
    # 3.5 Load PhoNLP for dependency matrix if vi
    phonlp_model = None
    if args.src_lang == 'vi':
        print("\nLoading PhoNLP for Dependency-aware AMR Parsing...")
        try:
            import phonlp
            if not os.path.exists('./phonlp'):
                phonlp.download(save_dir='./phonlp')
            phonlp_model = phonlp.load(save_dir='./phonlp')
            print("    ✓ PhoNLP loaded")
        except ImportError:
            print("    [WARN] phonlp not installed. AMR parsing might degrade.")
    
    # 4. Load Dictionary for Enrichment
    from saft.amr.dictionary import SAFTDictionary
    saft_dict = SAFTDictionary()
    saft_dict.load_dictionary()
    
    print("\n" + "="*50)
    print(" Pipeline Ready!")
    print(f" Mode: {args.src_lang.upper()} -> {args.tgt_lang.upper()}")
    print("="*50 + "\n")
    
    def process_sentence(src_text):
        total_start = time.time()
        
        t0 = time.time()
        print(f"\n[0] Word Segmentation...")
        if args.src_lang == 'vi':
            try:
                from pyvi import ViTokenizer
                segmented_text = ViTokenizer.tokenize(src_text)
                print(f"    Segmented: {segmented_text}")
            except ImportError:
                print("    [WARN] pyvi not installed. Using raw text.")
                segmented_text = src_text
        else:
            segmented_text = src_text
        print(f"    ⏱ {time.time() - t0:.2f}s")
            
        t1 = time.time()
        print(f"\n[1] Parsing to AMR...")
        cached = segmented_text in _amr_cache
        penman_amr = parse_vi_to_amr(
            segmented_text, amr_model, amr_tokenizer, device,
            num_beams=amr_num_beams, use_cache=True, phonlp_model=phonlp_model
        )
        print(f"    Raw AMR: {penman_amr}")
        print(f"    ⏱ {time.time() - t1:.2f}s" + (" (cached)" if cached else ""))
        
        t2 = time.time()
        print(f"[2] Linearizing and Enriching AMR...")
        linear_amr = bfs_linearize(penman_amr)
        if not linear_amr:
            print("    Failed to linearize AMR. Falling back to plain text.")
            return None
            
        # Dictionary Enrichment
        linear_amr = saft_dict.enrich_linear_amr(linear_amr)
        
        print(f"    Enriched Linear AMR: {linear_amr[:100]}...")
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
            model, tokenizer, src_text, pe_info, config, 
            src_lang=args.src_lang, tgt_lang=args.tgt_lang, max_seq=config.saft_max_seq
        )
        print(f"    ⏱ {time.time() - t4:.2f}s")
        
        print(f"\n    ⏱ Total pipeline: {time.time() - total_start:.2f}s")
        return translation
        
    if args.translate:
        translation = process_sentence(args.translate)
        print(f"\n{args.src_lang.upper()} Source: {args.translate}")
        print(f"{args.tgt_lang.upper()} Translation:    {translation}")
        
    if args.interactive:
        while True:
            try:
                src_text = input(f"\n{args.src_lang.upper()} Enter text (or 'quit'): ").strip()
                if not src_text or src_text.lower() in ('quit', 'q', 'exit'):
                    break
                translation = process_sentence(src_text)
                if translation:
                    print(f"{args.tgt_lang.upper()} Translation: {translation}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
