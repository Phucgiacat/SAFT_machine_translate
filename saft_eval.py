"""
Evaluate a saved model on test set (BLEU + COMET).

Usage:
    python saft_eval.py \
        --model-path outputs/baseline/best_model \
        --data-dir ../data \
        --mode baseline

    python saft_eval.py \
        --model-path outputs/saft/best_model \
        --data-dir ../data \
        --mode saft
"""

import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
from tqdm.auto import tqdm
import sacrebleu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

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


@torch.no_grad()
def generate_translations(model, tokenizer, vi_texts, amr_texts=None,
                           mode="baseline", batch_size=16, max_new_tokens=256,
                           num_beams=4, max_seq=1280):
    """Generate translations for all samples."""
    model.eval()
    device = next(model.parameters()).device
    predictions = []

    for bs in tqdm(range(0, len(vi_texts), batch_size), desc="Generating"):
        be = min(bs + batch_size, len(vi_texts))
        prompts = []
        for j in range(bs, be):
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


def main():
    parser = argparse.ArgumentParser(description='Evaluate saved model on test set')
    parser.add_argument('--model-path', required=True, help='Path to saved best_model')
    parser.add_argument('--data-dir', default='../data', help='Data directory')
    parser.add_argument('--mode', choices=['baseline', 'saft'], default='baseline')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-beams', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--split', default='tst2013', help='Test split name')
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load model ──
    print(f"\nLoading model from: {args.model_path}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Model loaded: {model.config.hidden_size}d, {dtype}")

    # ── Load test data ──
    test_vi = read_lines(os.path.join(args.data_dir, f"{args.split}.vi"))
    test_en = read_lines(os.path.join(args.data_dir, f"{args.split}.en"))
    test_amr = None
    amr_path = os.path.join(args.data_dir, f"{args.split}.bpe.amr")
    if os.path.exists(amr_path):
        test_amr = read_lines(amr_path)
    print(f"  Test samples: {len(test_vi)} ({args.split})")

    # ── Generate ──
    print(f"\nGenerating translations (mode={args.mode}, beams={args.num_beams})...")
    predictions = generate_translations(
        model, tokenizer, test_vi,
        amr_texts=test_amr if args.mode == "saft" else None,
        mode=args.mode,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
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
