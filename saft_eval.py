"""
SAFT Evaluation — Paper-Compliant Inference with PE Injection
═════════════════════════════════════════════════════════════
Evaluate a saved model on test/val set, or translate single sentences.

For SAFT mode: loads SAFTModel wrapper + MLP projection weights,
injects Magnetic Laplacian PEs into prompt embeddings during generation.
  H = Embed(tokens) + f_θ( PE(v_i) ‖ SinPE(j) )

Usage:
    # Eval baseline on test set
    python saft_eval.py --model-path outputs/baseline/best_model --data-dir data --mode baseline

    # Eval SAFT on test set (with PE injection)
    python saft_eval.py --model-path outputs/saft/best_model --data-dir data --mode saft

    # Eval on validation set
    python saft_eval.py --model-path outputs/saft/best_model --data-dir data --mode saft --split tst2012

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
from saft_dataset import tokenize_with_amr_alignment

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


# ═══════════════════════════════════════════════════════════
# Generation: Baseline (no PE)
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def generate_baseline(model, tokenizer, vi_texts, batch_size=16,
                      max_new_tokens=256, num_beams=4, max_seq=768):
    """Generate translations without PE injection (baseline mode)."""
    model.eval()
    device = next(model.parameters()).device
    predictions = []

    for bs in tqdm(range(0, len(vi_texts), batch_size), desc="Generating"):
        be = min(bs + batch_size, len(vi_texts))
        prompts = []
        for j in range(bs, be):
            p = (f"<|im_start|>system\n{SYSTEM_MSG_BASELINE}<|im_end|>\n"
                 f"<|im_start|>user\n"
                 f"Translate the source text from Vietnamese to English.\n"
                 f"Vietnamese: {vi_texts[j]}\nEnglish:<|im_end|>\n"
                 f"<|im_start|>assistant\n")
            prompts.append(p)

        tokenizer.padding_side = "left"
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_seq).to(device)

        outputs = model.generate(
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


# ═══════════════════════════════════════════════════════════
# Generation: SAFT (with PE injection)
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def generate_saft(saft_model, tokenizer, vi_texts, pe_data,
                  batch_size=8, max_new_tokens=256, num_beams=4,
                  max_seq=1280, k_eigenvectors=20):
    """
    Generate translations with PE injection (SAFT mode).
    Uses tokenize_with_amr_alignment (same as training) to build
    per-token PE tensors, then calls SAFTModel.generate() for
    proper embedding injection: H = Embed(tokens) + AmrPE.
    """
    saft_model.eval()
    device = next(saft_model.parameters()).device
    predictions = []
    pe_dim = 2 * k_eigenvectors

    for bs in tqdm(range(0, len(vi_texts), batch_size), desc="Generating (SAFT+PE)"):
        be = min(bs + batch_size, len(vi_texts))

        batch_input_ids = []
        batch_attn_mask = []
        batch_amr_pe = []
        batch_amr_intra = []
        batch_amr_mask = []

        for j in range(bs, be):
            pe_info = pe_data[j]
            labels_list = pe_info['labels']
            label_pes = pe_info['label_pes']

            # Tokenize with PE alignment (en_text="" for inference prompt)
            result = tokenize_with_amr_alignment(
                tokenizer=tokenizer,
                system_msg=SYSTEM_MSG_SAFT,
                user_before_amr="AMR Graph:\n",
                amr_labels=labels_list,
                label_pes=label_pes,
                user_after_amr=f"\n\nVietnamese: {vi_texts[j]}\nEnglish:",
                en_text="",  # no response for inference
                max_seq_length=max_seq,
                pe_dim=pe_dim,
            )
            batch_input_ids.append(result['input_ids'])
            batch_attn_mask.append(result['attention_mask'])
            batch_amr_pe.append(result['amr_node_pe'])
            batch_amr_intra.append(result['amr_intra_pos'])
            batch_amr_mask.append(result['amr_mask'])

        # Left-pad batch for generation
        max_len = max(ids.size(0) for ids in batch_input_ids)
        bsz = len(batch_input_ids)

        padded_ids = torch.full((bsz, max_len), tokenizer.pad_token_id, dtype=torch.long)
        padded_attn = torch.zeros(bsz, max_len, dtype=torch.long)
        padded_pe = torch.zeros(bsz, max_len, pe_dim, dtype=torch.float32)
        padded_intra = torch.zeros(bsz, max_len, dtype=torch.long)
        padded_mask = torch.zeros(bsz, max_len, dtype=torch.float32)

        for i in range(bsz):
            seq_len = batch_input_ids[i].size(0)
            offset = max_len - seq_len  # left-pad
            padded_ids[i, offset:] = batch_input_ids[i]
            padded_attn[i, offset:] = batch_attn_mask[i]
            padded_pe[i, offset:] = batch_amr_pe[i]
            padded_intra[i, offset:] = batch_amr_intra[i]
            padded_mask[i, offset:] = batch_amr_mask[i]

        padded_ids = padded_ids.to(device)
        padded_attn = padded_attn.to(device)
        padded_pe = padded_pe.to(device)
        padded_intra = padded_intra.to(device)
        padded_mask = padded_mask.to(device)

        # Generate with PE injection via SAFTModel.generate()
        outputs = saft_model.generate(
            input_ids=padded_ids,
            attention_mask=padded_attn,
            amr_node_pe=padded_pe,
            amr_intra_pos=padded_intra,
            amr_mask=padded_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode
        for k in range(bsz):
            input_len = padded_ids[k].shape[-1]
            pred = tokenizer.decode(outputs[k][input_len:], skip_special_tokens=True).strip()
            predictions.append(pred)

    return predictions


# ═══════════════════════════════════════════════════════════
# Single-sentence translation
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def translate_single(model_or_saft, tokenizer, vi_text, amr_text=None,
                     mode="baseline", num_beams=4, max_new_tokens=256,
                     k_eigenvectors=20, max_seq=1280):
    """
    Translate a single Vietnamese sentence to English.
    
    For SAFT mode with AMR linearization:
      - Computes PEs on-the-fly (build SPG → eigendecomposition)
      - Injects PEs into prompt embeddings via SAFTModel
    
    Args:
        model_or_saft: base model (baseline) or SAFTModel (saft)
        amr_text: BFS-linearized AMR string (e.g. "want-01 :arg0 child <stop> ...")
    """
    from saft_pe_precompute import compute_pes_from_linear

    if isinstance(model_or_saft, SAFTModel):
        saft_model = model_or_saft
        device = next(saft_model.parameters()).device
    else:
        saft_model = None
        device = next(model_or_saft.parameters()).device

    # ── SAFT with PE injection ──
    if mode == "saft" and amr_text and saft_model is not None:
        saft_model.eval()
        pe_dim = 2 * k_eigenvectors

        # Step 1: Compute PEs on-the-fly
        pe_info = compute_pes_from_linear(amr_text, k=k_eigenvectors)
        if pe_info is None:
            # Fallback: all-zero PEs
            labels = amr_text.strip().split()
            pe_info = {
                'labels': labels,
                'label_pes': np.zeros((len(labels), pe_dim), dtype=np.float32),
            }

        # Step 2: Tokenize with PE alignment
        result = tokenize_with_amr_alignment(
            tokenizer=tokenizer,
            system_msg=SYSTEM_MSG_SAFT,
            user_before_amr="AMR Graph:\n",
            amr_labels=pe_info['labels'],
            label_pes=pe_info['label_pes'],
            user_after_amr=f"\n\nVietnamese: {vi_text}\nEnglish:",
            en_text="",  # no response for inference
            max_seq_length=max_seq,
            pe_dim=pe_dim,
        )

        # Step 3: Generate with PE injection
        input_ids = result['input_ids'].unsqueeze(0).to(device)
        attn_mask = result['attention_mask'].unsqueeze(0).to(device)
        amr_pe = result['amr_node_pe'].unsqueeze(0).to(device)
        amr_intra = result['amr_intra_pos'].unsqueeze(0).to(device)
        amr_mask = result['amr_mask'].unsqueeze(0).to(device)

        outputs = saft_model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            amr_node_pe=amr_pe,
            amr_intra_pos=amr_intra,
            amr_mask=amr_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        input_len = input_ids.shape[-1]
        return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    # ── Baseline (no PE) ──
    model = model_or_saft.base_model if saft_model else model_or_saft
    if hasattr(model, 'eval'):
        model.eval()

    prompt = (f"<|im_start|>system\n{SYSTEM_MSG_BASELINE}<|im_end|>\n"
              f"<|im_start|>user\n"
              f"Translate the source text from Vietnamese to English.\n"
              f"Vietnamese: {vi_text}\nEnglish:<|im_end|>\n"
              f"<|im_start|>assistant\n")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    input_len = inputs.input_ids.shape[-1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def interactive_loop(model_or_saft, tokenizer, mode, num_beams, max_new_tokens,
                     k_eigenvectors=20):
    """Interactive translation loop with on-the-fly PE computation for SAFT."""
    print(f"\n{'='*50}")
    print(f"  Interactive Translation (mode={mode})")
    print(f"  Type Vietnamese text, press Enter to translate.")
    if mode == "saft":
        print(f"  Paste BFS-linearized AMR for PE injection.")
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
                amr_input = input("📊 AMR (BFS-linearized, or Enter to skip): ").strip()
                if amr_input:
                    amr_text = amr_input

            translation = translate_single(
                model_or_saft, tokenizer, vi_text, amr_text,
                mode, num_beams, max_new_tokens,
                k_eigenvectors=k_eigenvectors,
            )
            print(f"🇬🇧 English:    {translation}\n")

        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


# ═══════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════

def load_model(model_path, base_model_name=None):
    """Load model and tokenizer from saved path.

    Handles both:
      - Full model saves (config.json + model weights)
      - LoRA adapter saves (adapter_config.json + adapter weights)
    """
    print(f"\nLoading model from: {model_path}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    adapter_config_path = os.path.join(model_path, "adapter_config.json")

    if os.path.exists(adapter_config_path):
        # ── LoRA adapter: load base model first, then apply adapter ──
        if base_model_name is None:
            with open(adapter_config_path, 'r') as f:
                adapter_cfg = json.load(f)
            base_model_name = adapter_cfg.get("base_model_name_or_path")
            if not base_model_name:
                raise ValueError(
                    f"Cannot determine base model from {adapter_config_path}. "
                    "Please specify --base-model explicitly."
                )
        print(f"  Detected LoRA adapter. Base model: {base_model_name}")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=dtype, device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        print(f"  LoRA adapter loaded.")

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True)
    else:
        # ── Full model save ──
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Loaded: {model.config.hidden_size}d, {dtype}")
    return model, tokenizer


def load_saft_model(model_path, base_model_name=None,
                    k_eigenvectors=20, sin_dim=8):
    """Load model + wrap with SAFTModel + load MLP projection weights.

    For SAFT mode evaluation, we need the SAFTModel wrapper to inject
    PEs into the embedding layer during generation.
    """
    model, tokenizer = load_model(model_path, base_model_name)

    # Wrap with SAFTModel
    saft_model = SAFTModel(
        model, k_eigenvectors=k_eigenvectors, sin_dim=sin_dim
    ).to(next(model.parameters()).device)

    # Load MLP projection weights
    pe_proj_path = os.path.join(model_path, "pe_projection.pt")
    if os.path.exists(pe_proj_path):
        saft_model.load_pe_projection(pe_proj_path)
        print(f"  PE projection loaded: {pe_proj_path}")
        mlp_params = sum(p.numel() for p in saft_model.pe_projection.parameters())
        print(f"  MLP params: {mlp_params:,}")
    else:
        print(f"  [WARN] PE projection not found: {pe_proj_path}")
        print(f"         SAFT eval will use untrained MLP (results may be poor)")

    return saft_model, tokenizer


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Evaluate or translate with saved model')
    parser.add_argument('--model-path', required=True, help='Path to saved best_model')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--mode', choices=['baseline', 'saft'], default='baseline')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-beams', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--max-seq', type=int, default=None,
                        help='Max sequence length (default: 768 baseline, 1280 saft)')
    parser.add_argument('--split', default='tst2013', help='Test split name')
    parser.add_argument('--translate', type=str, default=None,
                        help='Translate a single Vietnamese sentence')
    parser.add_argument('--amr', type=str, default=None,
                        help='AMR text for --translate (SAFT mode)')
    parser.add_argument('--interactive', action='store_true',
                        help='Start interactive translation mode')
    parser.add_argument('--base-model', type=str, default=None,
                        help='Base model name (auto-detected from adapter_config.json if not set)')
    parser.add_argument('--k', type=int, default=20, help='Number of eigenvectors (SAFT)')
    args = parser.parse_args()

    max_seq = args.max_seq or (1280 if args.mode == "saft" else 768)

    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load model ──
    if args.mode == "saft":
        saft_model, tokenizer = load_saft_model(
            args.model_path, args.base_model, k_eigenvectors=args.k)
        model_or_saft = saft_model  # pass SAFTModel to all functions
    else:
        model, tokenizer = load_model(args.model_path, args.base_model)
        # Merge LoRA for faster baseline inference
        if hasattr(model, 'merge_and_unload'):
            model = model.merge_and_unload()
            print("  LoRA merged for baseline inference.")
        model_or_saft = model
        saft_model = None

    # ── Single sentence translation ──
    if args.translate:
        result = translate_single(
            model_or_saft, tokenizer, args.translate, args.amr,
            args.mode, args.num_beams, args.max_new_tokens,
            k_eigenvectors=args.k,
        )
        print(f"\n🇻🇳 Vietnamese: {args.translate}")
        if args.amr: print(f"📊 AMR:        {args.amr}")
        print(f"🇬🇧 English:    {result}")
        return

    # ── Interactive mode ──
    if args.interactive:
        interactive_loop(model_or_saft, tokenizer, args.mode,
                        args.num_beams, args.max_new_tokens,
                        k_eigenvectors=args.k)
        return

    # ── Test set evaluation ──
    test_vi = read_lines(os.path.join(args.data_dir, f"{args.split}.vi"))
    test_en = read_lines(os.path.join(args.data_dir, f"{args.split}.en"))
    print(f"  Test samples: {len(test_vi)} ({args.split})")

    # ── Generate ──
    print(f"\nGenerating translations (mode={args.mode}, beams={args.num_beams})...")

    if args.mode == "saft" and saft_model is not None:
        # Load precomputed PE data
        pe_path = os.path.join(args.data_dir, f"{args.split}_pes.pkl")
        if not os.path.exists(pe_path):
            print(f"  [ERROR] PE file not found: {pe_path}")
            print(f"  Run: python saft_pe_precompute.py --data-dir {args.data_dir}")
            return
        with open(pe_path, 'rb') as f:
            pe_data = pickle.load(f)
        print(f"  PE data loaded: {len(pe_data)} samples")

        predictions = generate_saft(
            saft_model, tokenizer, test_vi, pe_data,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            max_seq=max_seq,
            k_eigenvectors=args.k,
        )
    else:
        predictions = generate_baseline(
            model, tokenizer, test_vi,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            max_seq=max_seq,
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
    print(f"  BLEU:  {bleu.score:.2f}")
    if comet_score: print(f"  COMET: {comet_score:.4f}")
    pe_status = "✓ PE injected" if args.mode == "saft" else "No PE"
    print(f"  PE:    {pe_status}")
    print(f"{'='*50}")

    # ── Save ──
    output_dir = os.path.dirname(args.model_path)
    pred_file = os.path.join(output_dir, f"{args.split}_predictions.txt")
    with open(pred_file, 'w', encoding='utf-8') as f:
        f.writelines(p + '\n' for p in predictions)
    print(f"\nPredictions saved: {pred_file}")

    results = {
        'mode': args.mode, 'split': args.split,
        'bleu': bleu.score, 'comet': comet_score,
        'num_samples': len(test_vi),
        'pe_injected': args.mode == "saft",
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
