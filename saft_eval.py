"""
Evaluate a saved model on test set, or translate single sentences.
Now supports PE injection for SAFT mode via SAFTModel wrapper.

Usage:
    # Eval on test set (baseline — no PE)
    python saft_eval.py --model-path outputs/baseline/best_model --data-dir ../data --mode baseline

    # Eval on test set (SAFT — with PE injection)
    python saft_eval.py --model-path outputs/saft/best_model --data-dir ../data --mode saft

    # Translate a single sentence
    python saft_eval.py --model-path outputs/baseline/best_model --mode baseline \
        --translate "Tôi muốn cho bạn biết về nỗ lực khoa học to lớn."

    # Interactive mode (type sentences, Ctrl+C to exit)
    python saft_eval.py --model-path outputs/saft/best_model --mode saft --interactive
"""

import os
import sys
import json
import pickle
import argparse
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import torch
from tqdm.auto import tqdm
import sacrebleu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from saft_model import SAFTModel
from saft_config import get_config, BRAND_CONFIGS

# ── Prompt templates ──
SYSTEM_MSG_SAFT = (
    "You are an expert Vietnamese-to-English translation assistant. "
    "You are given an Abstract Meaning Representation (AMR) graph of the source sentence. "
    "Use the AMR as a semantic blueprint to produce an accurate, fluent English translation."
)
SYSTEM_MSG_BASELINE = "You are a helpful translation assistant."


def read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [l.strip() for l in f]


# ─────────────────────────────────────────────────────────
# PE-aligned prompt builder (shared logic with saft_train.py)
# ─────────────────────────────────────────────────────────

def _build_eval_prompt_with_pe(tokenizer, vi_text, pe_info, max_length, k_eigenvectors=20):
    """
    Build an eval prompt with aligned PE tensors for a single sample.
    Tokenizes parts separately so AMR labels are bijectively aligned to PE vectors.

    Returns: (input_ids, node_pe, intra_pos, amr_mask) — all 1D/2D tensors
    """
    pe_dim = 2 * k_eigenvectors
    labels_list = pe_info['labels']
    label_pes = pe_info['label_pes']

    # Tokenize structural parts
    prefix_text = (
        f"<|im_start|>system\n{SYSTEM_MSG_SAFT}<|im_end|>\n"
        f"<|im_start|>user\nAMR Graph:\n"
    )
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

    suffix_text = (
        f"\n\nVietnamese: {vi_text}\nEnglish:<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)

    # Tokenize each AMR label individually (same as training dataset)
    amr_token_ids = []
    amr_token_pe = []
    amr_token_intra = []
    label_boundaries = []

    for label_idx, label in enumerate(labels_list):
        label_boundaries.append(len(amr_token_ids))
        if label_idx > 0:
            tids = tokenizer.encode(" " + label, add_special_tokens=False)
        else:
            tids = tokenizer.encode(label, add_special_tokens=False)

        pe = label_pes[label_idx]
        for j, tid in enumerate(tids):
            amr_token_ids.append(tid)
            amr_token_pe.append(pe)
            amr_token_intra.append(j)

    # Truncate AMR at label boundary if needed
    fixed_len = len(prefix_ids) + len(suffix_ids)
    amr_budget = max_length - fixed_len - 10

    if amr_budget <= 0:
        amr_token_ids, amr_token_pe, amr_token_intra = [], [], []
    elif len(amr_token_ids) > amr_budget:
        cut_at = 0
        for lb in label_boundaries:
            if lb <= amr_budget:
                cut_at = lb
            else:
                break
        if cut_at == 0:
            cut_at = amr_budget
        amr_token_ids = amr_token_ids[:cut_at]
        amr_token_pe = amr_token_pe[:cut_at]
        amr_token_intra = amr_token_intra[:cut_at]

    # Assemble full sequence
    all_ids = prefix_ids + amr_token_ids + suffix_ids
    if len(all_ids) > max_length:
        all_ids = all_ids[:max_length]

    seq_len = len(all_ids)
    amr_start = len(prefix_ids)

    # Build per-token PE arrays
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
# Generation functions
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def generate_translations(model, tokenizer, vi_texts, amr_texts=None,
                           pe_data=None, mode="baseline", batch_size=16,
                           max_new_tokens=256, num_beams=4, max_seq=1280,
                           k_eigenvectors=20):
    """Generate translations for all samples. Supports PE injection for SAFT mode."""
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    is_saft_model = isinstance(model, SAFTModel)

    for bs_start in tqdm(range(0, len(vi_texts), batch_size), desc="Generating"):
        bs_end = min(bs_start + batch_size, len(vi_texts))

        if mode == "saft" and pe_data is not None and is_saft_model:
            # ── SAFT path: build prompts with PE alignment ──
            batch_ids_list = []
            batch_pe_list = []
            batch_intra_list = []
            batch_mask_list = []

            for j in range(bs_start, bs_end):
                pe_info = pe_data[j] if j < len(pe_data) else None
                if pe_info is not None:
                    ids, pe, intra, mask = _build_eval_prompt_with_pe(
                        tokenizer, vi_texts[j], pe_info, max_seq, k_eigenvectors
                    )
                else:
                    # Fallback: no PE for this sample
                    prompt = (
                        f"<|im_start|>system\n{SYSTEM_MSG_SAFT}<|im_end|>\n"
                        f"<|im_start|>user\nAMR Graph:\n{amr_texts[j] if amr_texts else ''}\n\n"
                        f"Vietnamese: {vi_texts[j]}\nEnglish:<|im_end|>\n"
                        f"<|im_start|>assistant\n"
                    )
                    tok_ids = tokenizer.encode(prompt, add_special_tokens=False)[:max_seq]
                    seq_len = len(tok_ids)
                    pe_dim = 2 * k_eigenvectors
                    ids = torch.tensor(tok_ids, dtype=torch.long)
                    pe = torch.zeros(seq_len, pe_dim, dtype=torch.float32)
                    intra = torch.zeros(seq_len, dtype=torch.long)
                    mask = torch.zeros(seq_len, dtype=torch.float32)

                batch_ids_list.append(ids)
                batch_pe_list.append(pe)
                batch_intra_list.append(intra)
                batch_mask_list.append(mask)

            # Left-pad the batch
            n_batch = len(batch_ids_list)
            max_len = max(ids.size(0) for ids in batch_ids_list)
            pe_dim = 2 * k_eigenvectors

            padded_ids = torch.full((n_batch, max_len), tokenizer.pad_token_id, dtype=torch.long)
            padded_attn = torch.zeros(n_batch, max_len, dtype=torch.long)
            padded_pe = torch.zeros(n_batch, max_len, pe_dim, dtype=torch.float32)
            padded_intra = torch.zeros(n_batch, max_len, dtype=torch.long)
            padded_mask = torch.zeros(n_batch, max_len, dtype=torch.float32)

            for i in range(n_batch):
                seq_len = batch_ids_list[i].size(0)
                offset = max_len - seq_len
                padded_ids[i, offset:] = batch_ids_list[i]
                padded_attn[i, offset:] = 1
                padded_pe[i, offset:] = batch_pe_list[i]
                padded_intra[i, offset:] = batch_intra_list[i]
                padded_mask[i, offset:] = batch_mask_list[i]

            outputs = model.generate(
                input_ids=padded_ids.to(device),
                attention_mask=padded_attn.to(device),
                amr_node_pe=padded_pe.to(device),
                amr_intra_pos=padded_intra.to(device),
                amr_mask=padded_mask.to(device),
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            for k in range(n_batch):
                gen_ids = outputs[k][max_len:]
                pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                predictions.append(pred)

        else:
            # ── Baseline / no-PE path ──
            prompts = []
            for j in range(bs_start, bs_end):
                if mode == "saft" and amr_texts:
                    p = (f"<|im_start|>system\n{SYSTEM_MSG_SAFT}<|im_end|>\n"
                         f"<|im_start|>user\nAMR Graph:\n{amr_texts[j]}\n\n"
                         f"Vietnamese: {vi_texts[j]}\nEnglish:<|im_end|>\n"
                         f"<|im_start|>assistant\n")
                else:
                    p = (f"<|im_start|>system\n{SYSTEM_MSG_BASELINE}<|im_end|>\n"
                         f"<|im_start|>user\n"
                         f"Translate the source text from Vietnamese to English.\n"
                         f"Vietnamese: {vi_texts[j]}\nEnglish:<|im_end|>\n"
                         f"<|im_start|>assistant\n")
                prompts.append(p)

            tokenizer.padding_side = "left"
            inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                              truncation=True, max_length=max_seq).to(device)

            gen_model = model.base_model if is_saft_model else model
            outputs = gen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            for k in range(len(prompts)):
                input_len = inputs.input_ids[k].shape[-1]
                pred = tokenizer.decode(outputs[k][input_len:], skip_special_tokens=True).strip()
                predictions.append(pred)

    tokenizer.padding_side = "right"
    return predictions


@torch.no_grad()
def translate_single(model, tokenizer, vi_text, amr_text=None, mode="baseline",
                     num_beams=4, max_new_tokens=256):
    """Translate a single Vietnamese sentence to English."""
    model.eval()
    device = next(model.parameters()).device

    if mode == "saft" and amr_text:
        prompt = (f"<|im_start|>system\n{SYSTEM_MSG_SAFT}<|im_end|>\n"
                  f"<|im_start|>user\nAMR Graph:\n{amr_text}\n\n"
                  f"Vietnamese: {vi_text}\nEnglish:<|im_end|>\n"
                  f"<|im_start|>assistant\n")
    else:
        prompt = (f"<|im_start|>system\n{SYSTEM_MSG_BASELINE}<|im_end|>\n"
                  f"<|im_start|>user\n"
                  f"Translate the source text from Vietnamese to English.\n"
                  f"Vietnamese: {vi_text}\nEnglish:<|im_end|>\n"
                  f"<|im_start|>assistant\n")

    is_saft_model = isinstance(model, SAFTModel)
    gen_model = model.base_model if is_saft_model else model

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = gen_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    input_len = inputs.input_ids.shape[-1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def interactive_loop(model, tokenizer, mode, num_beams, max_new_tokens):
    """Interactive translation loop."""
    print(f"\n{'='*50}")
    print(f"  Interactive Translation (mode={mode})")
    print(f"  Type Vietnamese text, press Enter to translate.")
    print(f"  Type 'quit' or Ctrl+C to exit.")
    print(f"{'='*50}\n")

    while True:
        try:
            vi_text = input("🇻🇳 Vietnamese: ").strip()
            if not vi_text or vi_text.lower() in ('quit', 'exit', 'q'):
                print("Bye!")
                break

            amr_text = None
            if mode == "saft":
                amr_input = input("📊 AMR (paste or press Enter to skip): ").strip()
                if amr_input:
                    amr_text = amr_input

            translation = translate_single(model, tokenizer, vi_text, amr_text,
                                           mode, num_beams, max_new_tokens)
            print(f"🇬🇧 English:    {translation}\n")

        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def load_model(model_path, mode="baseline", k_eigenvectors=20, sin_dim=8):
    """
    Load model and tokenizer from saved path.
    For SAFT mode: wraps with SAFTModel and loads PE projection weights.
    """
    print(f"\nLoading model from: {model_path}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Loaded: {model.config.hidden_size}d, {dtype}")

    if mode == "saft":
        # Wrap with SAFTModel for PE injection
        saft_model = SAFTModel(
            model, k_eigenvectors=k_eigenvectors, sin_dim=sin_dim
        ).to(model.device)

        # Load PE projection weights if available
        pe_proj_path = os.path.join(model_path, "pe_projection.pt")
        if os.path.exists(pe_proj_path):
            saft_model.load_pe_projection(pe_proj_path)
            print(f"  PE projection loaded: {pe_proj_path}")
        else:
            print(f"  [WARN] PE projection not found: {pe_proj_path}")
            print(f"  Continuing without learned PE projection weights.")

        return saft_model, tokenizer

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description='Evaluate or translate with saved model')
    parser.add_argument('--model-path', required=True, help='Path to saved best_model')
    parser.add_argument('--brand', default=None,
                        choices=sorted(BRAND_CONFIGS.keys()),
                        help='Model brand preset (for k_eigenvectors, sin_dim)')
    parser.add_argument('--data-dir', default='../data', help='Data directory')
    parser.add_argument('--mode', choices=['baseline', 'saft'], default='baseline')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-beams', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--max-seq', type=int, default=1280, help='Max sequence length')
    parser.add_argument('--split', default='tst2013', help='Test split name')
    parser.add_argument('--translate', type=str, default=None,
                        help='Translate a single Vietnamese sentence')
    parser.add_argument('--amr', type=str, default=None,
                        help='AMR text for --translate (SAFT mode)')
    parser.add_argument('--interactive', action='store_true',
                        help='Start interactive translation mode')
    args = parser.parse_args()

    # Get config for PE dimensions
    if args.brand:
        config = get_config(args.brand)
        k_eigenvectors = config.k_eigenvectors
        sin_dim = config.sin_dim
    else:
        k_eigenvectors = 20
        sin_dim = 8

    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load model ──
    model, tokenizer = load_model(
        args.model_path, mode=args.mode,
        k_eigenvectors=k_eigenvectors, sin_dim=sin_dim
    )

    # ── Single sentence translation ──
    if args.translate:
        result = translate_single(model, tokenizer, args.translate, args.amr,
                                   args.mode, args.num_beams, args.max_new_tokens)
        print(f"\n🇻🇳 Vietnamese: {args.translate}")
        if args.amr: print(f"📊 AMR:        {args.amr}")
        print(f"🇬🇧 English:    {result}")
        return

    # ── Interactive mode ──
    if args.interactive:
        interactive_loop(model, tokenizer, args.mode, args.num_beams, args.max_new_tokens)
        return

    # ── Test set evaluation ──
    test_vi = read_lines(os.path.join(args.data_dir, f"{args.split}.vi"))
    test_en = read_lines(os.path.join(args.data_dir, f"{args.split}.en"))
    test_amr = None
    amr_path = os.path.join(args.data_dir, f"{args.split}.linear.amr")
    if not os.path.exists(amr_path):
        # Fallback to old format
        amr_path = os.path.join(args.data_dir, f"{args.split}.bpe.amr")
    if os.path.exists(amr_path):
        test_amr = read_lines(amr_path)
    print(f"  Test samples: {len(test_vi)} ({args.split})")

    # Load PE data for SAFT mode
    test_pe_data = None
    if args.mode == "saft":
        pe_path = os.path.join(args.data_dir, f"{args.split}_pes.pkl")
        if os.path.exists(pe_path):
            with open(pe_path, 'rb') as f:
                test_pe_data = pickle.load(f)
            print(f"  PE data loaded: {len(test_pe_data)} samples")
        else:
            print(f"  [WARN] PE data not found: {pe_path}")
            print(f"  SAFT mode will run WITHOUT PE injection (text-only AMR).")

    # ── Generate ──
    pe_status = "with PE injection" if test_pe_data else "text-only"
    print(f"\nGenerating translations (mode={args.mode}, {pe_status}, beams={args.num_beams})...")
    predictions = generate_translations(
        model, tokenizer, test_vi,
        amr_texts=test_amr if args.mode == "saft" else None,
        pe_data=test_pe_data,
        mode=args.mode,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        max_seq=args.max_seq,
        k_eigenvectors=k_eigenvectors,
    )

    # ── BLEU ──
    bleu = sacrebleu.corpus_bleu(predictions, [test_en])
    print(f"\n{'='*50}")
    print(f"  BLEU = {bleu.score:.2f}")
    print(f"  {bleu}")

    # ── COMET ──
    comet_score = None
    try:
        from comet import download_model, load_from_checkpoint
        print("\nComputing COMET...")
        comet_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(comet_path)
        data = [{"src": s, "mt": m, "ref": r}
                for s, m, r in zip(test_vi, predictions, test_en)]
        output = comet_model.predict(data, batch_size=64, gpus=1)
        comet_score = output.system_score
        print(f"  COMET = {comet_score:.4f}")
    except ImportError:
        print("\n  COMET: skipped (unbabel-comet not installed)")
    except Exception as e:
        print(f"\n  COMET failed: {e}")

    # ── Summary ──
    print(f"\n{'='*50}")
    print(f"  Mode:  {args.mode}")
    print(f"  Split: {args.split} ({len(test_vi)} samples)")
    print(f"  PE:    {'injected' if test_pe_data else 'none'}")
    print(f"  BLEU:  {bleu.score:.2f}")
    if comet_score: print(f"  COMET: {comet_score:.4f}")
    print(f"{'='*50}")

    # ── Save ──
    output_dir = os.path.dirname(args.model_path)
    pred_file = os.path.join(output_dir, f"{args.split}_predictions.txt")
    with open(pred_file, 'w', encoding='utf-8') as f:
        f.writelines(p + '\n' for p in predictions)
    print(f"\nPredictions saved: {pred_file}")

    results = {
        'mode': args.mode, 'split': args.split,
        'pe_injection': test_pe_data is not None,
        'bleu': bleu.score, 'comet': comet_score,
        'num_samples': len(test_vi),
    }
    result_file = os.path.join(output_dir, f"{args.split}_results.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {result_file}")

    # Show some samples
    print(f"\n{'='*50}")
    print("  Sample Predictions:")
    print(f"{'='*50}")
    for i in range(min(5, len(predictions))):
        print(f"\n  [SRC] {test_vi[i]}")
        print(f"  [REF] {test_en[i]}")
        print(f"  [GEN] {predictions[i]}")


if __name__ == '__main__':
    main()
