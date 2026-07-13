"""
End-to-End SAFT Translation Pipeline (Bidirectional)
═════════════════════════════════════════════════════════
This script combines all the steps of the SAFT machine translation pipeline:
1. Source Sentence -> AMR parsing
   - Vietnamese source: AMRBART (phucgiacat/AMRBART-parser-grpo)
   - English source:    IBM transition-amr-parser (AMR3-structbart-L)
2. AMR Graph -> BFS Linearization
3. Linearized AMR -> PE precomputation
4. Sentence + PE -> Qwen Translation Model -> Target Translation

Supports both directions:
  - Vietnamese -> English (default):  --in vi --out en  (uses AMRBART)
  - English -> Vietnamese:            --in en --out vi  (uses IBM transition-amr-parser)

Usage:
    # Vi -> En (default, uses AMRBART)
    python e2e_translation.py --model-path <path> --brand qwen2.5 \
        --translate "Tôn_Ngộ_Không phá khóa bay vào"

    # En -> Vi (uses IBM transition-amr-parser)
    python e2e_translation.py --model-path <path> --brand qwen2.5 \
        --in en --out vi --translate "The monkey king broke the lock and flew in"

    # En -> Vi with custom IBM parser model
    python e2e_translation.py --model-path <path> --brand qwen2.5 \
        --in en --out vi --en-amr-model AMR3-structbart-L-smpl \
        --translate "The monkey king broke the lock and flew in"

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
# Language Utilities
# ─────────────────────────────────────────────────────────

LANG_MAP = {"vi": "Vietnamese", "en": "English"}
LANG_EMOJI = {"vi": "🇻🇳", "en": "🇬🇧"}


def get_system_msg(src_lang, tgt_lang):
    """Generate system message matching the training prompt format.

    en2vi branch:   'expert English-to-Vietnamese translation assistant'
    qwen2.5-brand:  'expert Vietnamese-to-English translation assistant'
    """
    src = LANG_MAP.get(src_lang, src_lang)
    tgt = LANG_MAP.get(tgt_lang, tgt_lang)
    return (
        f"You are an expert {src}-to-{tgt} translation assistant. "
        f"You are given an Abstract Meaning Representation (AMR) graph of the source sentence. "
        f"Use the AMR as a semantic blueprint to produce an accurate, fluent {tgt} translation."
    )


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


# ─────────────────────────────────────────────────────────
# PE-aligned Prompt Builder (bidirectional)
# ─────────────────────────────────────────────────────────

def _build_eval_prompt_with_pe(tokenizer, src_text, pe_info, max_length,
                               k_eigenvectors=20, src_lang="vi", tgt_lang="en"):
    """Build eval prompt with aligned PE tensors for a single sample.

    Prompt format matches training exactly:
      en2vi:        AMR Graph:\n{amr}\n\nEnglish: {src}\nVietnamese:
      qwen2.5-brand: AMR Graph:\n{amr}\n\nVietnamese: {src}\nEnglish:
    """
    import numpy as np
    from saft_dataset import fmt

    pe_dim = 2 * k_eigenvectors
    labels_list = pe_info['labels']
    label_pes = pe_info['label_pes']

    src_label = LANG_MAP.get(src_lang, src_lang)
    tgt_label = LANG_MAP.get(tgt_lang, tgt_lang)
    system_msg = get_system_msg(src_lang, tgt_lang)

    f = fmt()
    prefix_text = (
        f"{f['sys_start']}{system_msg}{f['sys_end']}"
        f"{f['user_start']}AMR Graph:\n"
    )
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

    suffix_text = (
        f"\n\n{src_label}: {src_text}\n{tgt_label}:{f['user_end']}"
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


# ─────────────────────────────────────────────────────────
# AMR Parsing
# ─────────────────────────────────────────────────────────

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


def parse_en_to_amr(text, en_amr_venv, en_amr_model, en_amr_repo, use_cache=True):
    """Parse an English sentence into a Penman AMR string using IBM transition-amr-parser.

    Calls the parser via subprocess using the Python 3.8 venv created by setup_phuc.sh,
    since IBM transition-amr-parser requires Python 3.8 + fairseq==0.10.2 + torch 1.13.

    Args:
        text: Input English text
        en_amr_venv: Path to Python 3.8 venv (e.g. /content/transition-amr-parser/.venv)
        en_amr_model: Model name (e.g. AMR3-structbart-L)
        en_amr_repo: Path to the cloned transition-amr-parser repo
        use_cache: if True, cache results to avoid re-parsing identical sentences
    """
    import subprocess
    global _amr_cache

    # Check cache first
    if use_cache and text in _amr_cache:
        return _amr_cache.get(text)

    python_bin = os.path.join(en_amr_venv, 'bin', 'python')
    helper_script = os.path.join(en_amr_repo, '_amr_parse_helper.py')

    result = subprocess.run(
        [python_bin, helper_script, '--model', en_amr_model, '--text', text],
        capture_output=True, text=True, cwd=en_amr_repo,
        timeout=120
    )

    if result.returncode != 0:
        raise RuntimeError(f"AMR parsing failed: {result.stderr.strip()}")

    amr_output = result.stdout.strip()

    # Store in cache
    if use_cache:
        _amr_cache.put(text, amr_output)

    return amr_output


# ─────────────────────────────────────────────────────────
# Translation with PE injection
# ─────────────────────────────────────────────────────────

@torch.inference_mode()
def translate_sentence_with_pe(model, tokenizer, src_text, pe_info, config,
                               src_lang="vi", tgt_lang="en", max_seq=1280):
    """Generate translation using SAFT model with PE injection."""
    model.eval()
    device = next(model.parameters()).device

    # Build the prompt with PE alignment (direction-aware)
    ids, pe, intra, mask = _build_eval_prompt_with_pe(
        tokenizer, src_text, pe_info, max_seq, config.k_eigenvectors,
        src_lang, tgt_lang
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
    parser = argparse.ArgumentParser(description='E2E SAFT Translation Pipeline (Bidirectional)')
    parser.add_argument('--model-path', required=True, help='Path to saved translation best_model')
    parser.add_argument('--brand', default='qwen2.5', help='Model brand preset (e.g., qwen2.5)')
    parser.add_argument('--base-model', type=str, default=None, help='Override base model name from config (e.g., Qwen/Qwen2.5-7B-Instruct)')
    parser.add_argument('--amrbart-path', default='../AMRBART', help='Path to AMRBART repository for tokenizer (Vi->En)')
    parser.add_argument('--amr-checkpoint', type=str, default=None, help='Path to custom AMRBART checkpoint (Vi->En, default: phucgiacat/AMRBART-parser-grpo)')
    parser.add_argument('--en-amr-model', type=str, default='AMR3-structbart-L',
                        help='IBM transition-amr-parser model name for English AMR parsing (En->Vi, default: AMR3-structbart-L)')
    parser.add_argument('--en-amr-repo', type=str, default='/content/transition-amr-parser',
                        help='Path to cloned IBM transition-amr-parser repo (En->Vi, default: /content/transition-amr-parser)')
    parser.add_argument('--en-amr-venv', type=str, default=None,
                        help='Path to Python 3.8 venv for IBM parser (default: <en-amr-repo>/.venv)')
    parser.add_argument('--translate', type=str, default=None, help='Sentence to translate')
    parser.add_argument('--interactive', action='store_true', help='Start interactive mode')
    parser.add_argument('--fast', action='store_true', help='Enable inference optimizations (FP16 AMRBART, reduced beams, cache)')
    parser.add_argument('--compile', action='store_true', help='Apply torch.compile to AMRBART (requires PyTorch 2.0+, slow first run)')
    parser.add_argument('--in', dest='src_lang', default='vi',
                        help='Source language: vi or en (default: vi)')
    parser.add_argument('--out', dest='tgt_lang', default='en',
                        help='Target language: vi or en (default: en)')
    args = parser.parse_args()

    if not args.translate and not args.interactive:
        print("Please provide --translate or use --interactive.")
        return

    # Validate language pair
    valid_langs = {'vi', 'en'}
    if args.src_lang not in valid_langs or args.tgt_lang not in valid_langs:
        print(f"Error: --in and --out must be 'vi' or 'en'. Got: --in {args.src_lang} --out {args.tgt_lang}")
        return
    if args.src_lang == args.tgt_lang:
        print(f"Error: source and target language must differ. Got: --in {args.src_lang} --out {args.tgt_lang}")
        return

    src_label = LANG_MAP[args.src_lang]
    tgt_label = LANG_MAP[args.tgt_lang]
    src_emoji = LANG_EMOJI[args.src_lang]
    tgt_emoji = LANG_EMOJI[args.tgt_lang]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Direction: {src_label} → {tgt_label}")

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

    # 2. Load AMR parser (direction-specific)
    amr_model = None
    amr_tokenizer = None
    en_amr_parser = None
    amr_num_beams = 5

    if args.src_lang == 'vi':
        # ── Vi→En: Load AMRBART for Vietnamese AMR parsing ──
        print("\nLoading AMRBART Parser (Vietnamese)...")

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

                if len(args) > 0:
                    kwargs.pop('vocab', None)
                    kwargs.pop('vocab_file', None)

                BartTokenizer.__init__(self, *args, **kwargs)

                if not hasattr(self, 'encoder'):
                    self.encoder = self.get_vocab().copy()
                if not hasattr(self, 'decoder'):
                    self.decoder = {v: k for k, v in self.encoder.items()}

                self.modified = 0
                self.recategorizations = set(recategorizations)
                self.patterns = re.compile(r""" ?<[a-z]+:?\d*>| ?:[^\s]+|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
                self.remove_pars = False

            AMRBartTokenizer.__init__ = patched_init

            # Patch init_amr_vocabulary to avoid "property 'decoder' has no setter"
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
                    self.__class__.decoder = property(lambda self: self._amr_decoder)
                    self._amr_decoder = my_decoder

                self.modified = len(tokens)
                self.amr_bos_token = "<AMR>"
                self.amr_bos_token_id = self.encoder[self.amr_bos_token]
                self.amr_eos_token = "</AMR>"
                self.amr_eos_token_id = self.encoder[self.amr_eos_token]

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

        # Optional: torch.compile for further speedup
        if args.compile:
            try:
                amr_model = torch.compile(amr_model, mode="reduce-overhead")
                print("    ✓ AMRBART compiled with torch.compile")
            except Exception as e:
                print(f"    [WARN] torch.compile failed: {e}")

        # Set AMR beam count — reduced in fast mode
        amr_num_beams = 3 if args.fast else 5
        print(f"    AMR beams: {amr_num_beams}")

    else:
        # ── En→Vi: Setup IBM transition-amr-parser via subprocess (Python 3.8 venv) ──
        en_amr_repo = os.path.abspath(args.en_amr_repo)
        en_amr_venv = args.en_amr_venv if args.en_amr_venv else os.path.join(en_amr_repo, '.venv')
        en_amr_model = args.en_amr_model

        print(f"\nSetting up IBM transition-amr-parser (English, subprocess)...")
        print(f"    Repo:  {en_amr_repo}")
        print(f"    Venv:  {en_amr_venv}")
        print(f"    Model: {en_amr_model}")

        # Verify venv exists
        python_bin = os.path.join(en_amr_venv, 'bin', 'python')
        if not os.path.exists(python_bin):
            print(f"    [ERROR] Python 3.8 venv not found at {python_bin}")
            print(f"    Please run setup first (see amr_parsing.ipynb):")
            print(f"    1. git clone https://github.com/IBM/transition-amr-parser.git")
            print(f"    2. curl -fsSL https://raw.githubusercontent.com/Phucgiacat/transition-amr/main/setup.sh -o setup_phuc.sh")
            print(f"    3. bash setup_phuc.sh")
            return

        # Create helper script for subprocess AMR parsing
        helper_script = os.path.join(en_amr_repo, '_amr_parse_helper.py')
        with open(helper_script, 'w') as f:
            f.write('''#!/usr/bin/env python
"""Helper script for AMR parsing via subprocess. Called by e2e_translation.py."""
import argparse
import sys
import os

# Add src to path for fairseq_ext
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--text', required=True)
    args = parser.parse_args()

    from transition_amr_parser.parse import AMRParser
    parser_amr = AMRParser.from_pretrained(args.model)
    tokens, positions = parser_amr.tokenize(args.text)
    annotations, machines = parser_amr.parse_sentence(tokens)
    amr = machines.get_amr()
    penman_str = amr.to_penman(jamr=False, isi=True)
    print(penman_str)

if __name__ == "__main__":
    main()
''')
        print(f"    ✓ Helper script created: {helper_script}")

        # Quick verify: check if venv Python can import the parser
        import subprocess as sp
        verify = sp.run(
            [python_bin, '-c', 'from transition_amr_parser.parse import AMRParser; print("OK")'],
            capture_output=True, text=True, cwd=en_amr_repo
        )
        if verify.returncode == 0 and 'OK' in verify.stdout:
            print(f"    ✓ IBM AMR parser verified in venv")
        else:
            print(f"    [WARN] Could not verify IBM parser in venv:")
            print(f"    {verify.stderr.strip()[:200]}")
            print(f"    Parsing may fail at runtime.")

    # 3. Load SAFT Translation model
    print("\nLoading Translation Model...")
    model, tokenizer = load_model(args.model_path, config=config, mode="saft")
    model.eval()

    print(f"\n{'='*50}")
    print(f" Pipeline Ready!")
    print(f" Mode: {src_label} → {tgt_label}")
    print(f"{'='*50}\n")

    def process_sentence(src_text):
        total_start = time.time()

        # Step 0: Word Segmentation (only for Vietnamese source)
        t0 = time.time()
        if args.src_lang == 'vi':
            print(f"\n[0] Word Segmentation (pyvi)...")
            try:
                from pyvi import ViTokenizer
                segmented_text = ViTokenizer.tokenize(src_text)
                print(f"    Segmented: {segmented_text}")
            except ImportError:
                print("    [WARN] pyvi not installed. Using raw text. Run '!pip install pyvi' on Colab.")
                segmented_text = src_text
        else:
            print(f"\n[0] Preprocessing...")
            segmented_text = src_text
            print(f"    Text: {segmented_text}")
        print(f"    ⏱ {time.time() - t0:.2f}s")

        # Step 1: Parse to AMR (direction-specific parser)
        t1 = time.time()
        cached = segmented_text in _amr_cache
        if args.src_lang == 'vi':
            print(f"\n[1] Parsing to AMR (AMRBART)...")
            penman_amr = parse_vi_to_amr(
                segmented_text, amr_model, amr_tokenizer, device,
                num_beams=amr_num_beams, use_cache=True
            )
        else:
            print(f"\n[1] Parsing to AMR (IBM transition-amr-parser, subprocess)...")
            penman_amr = parse_en_to_amr(
                segmented_text, en_amr_venv, en_amr_model, en_amr_repo,
                use_cache=True
            )
        print(f"    Raw AMR: {penman_amr}")
        print(f"    ⏱ {time.time() - t1:.2f}s" + (" (cached)" if cached else ""))

        # Step 2: BFS Linearization
        t2 = time.time()
        print(f"[2] Linearizing AMR...")
        linear_amr = bfs_linearize(penman_amr)
        if not linear_amr:
            print("    Failed to linearize AMR. Falling back to plain text.")
            return None
        print(f"    Linear AMR: {linear_amr[:100]}...")
        print(f"    ⏱ {time.time() - t2:.2f}s")

        # Step 3: Compute PE
        t3 = time.time()
        print(f"[3] Computing Positional Encodings (PE)...")
        pe_info = compute_pes_from_linear(linear_amr, k=config.k_eigenvectors, q=0.25)
        if not pe_info:
            print("    Failed to compute PE.")
            return None
        print(f"    Extracted {len(pe_info['labels'])} label PEs.")
        print(f"    ⏱ {time.time() - t3:.2f}s")

        # Step 4: Generate Translation
        t4 = time.time()
        print(f"[4] Generating Translation ({src_label} → {tgt_label})...")
        translation = translate_sentence_with_pe(
            model, tokenizer, src_text, pe_info, config,
            src_lang=args.src_lang, tgt_lang=args.tgt_lang,
            max_seq=config.saft_max_seq
        )
        print(f"    ⏱ {time.time() - t4:.2f}s")

        print(f"\n    ⏱ Total pipeline: {time.time() - total_start:.2f}s")
        return translation

    if args.translate:
        translation = process_sentence(args.translate)
        print(f"\n{src_emoji} {src_label}: {args.translate}")
        print(f"{tgt_emoji} {tgt_label}:    {translation}")

    if args.interactive:
        while True:
            try:
                src_text = input(f"\n{src_emoji} Enter {src_label} text (or 'quit'): ").strip()
                if not src_text or src_text.lower() in ('quit', 'q', 'exit'):
                    break
                translation = process_sentence(src_text)
                if translation:
                    print(f"{tgt_emoji} {tgt_label}: {translation}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
