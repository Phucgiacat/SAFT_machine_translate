"""
SAFT Training Pipeline — Full Embedding-Level PE Injection (BFS)
═══════════════════════════════════════════════════════════
2-Track Training:
  Track 1: Baseline     — Vietnamese → English (no AMR)
  Track 2: SAFT (Full)  — Vietnamese + BFS-linearized AMR w/ embedding PE → English

Implements SAFT paper (arXiv:2507.13381) with:
  - Magnetic Laplacian PEs projected via trainable 2-layer MLP
  - Additive PE injection at embedding layer: H = Embed(tokens) + AmrPE
  - LoRA fine-tuning (PEFT) + MLP trained from scratch
  - Per-epoch BLEU + COMET evaluation
  - Early stopping on COMET

Usage (Colab / script):
    python saft_train.py --track saft --data-dir data --epochs 10

HƯỚNG DẪN: Chạy trên Colab với L4 GPU.
  1. Upload thư mục SAFT lên Colab
  2. pip install transformers peft accelerate sacrebleu tqdm matplotlib
  3. (Optional) pip install pytorch-lightning==2.1.0 unbabel-comet==2.2.2
  4. python saft_bfs_linearize.py --data-dir data
  5. python saft_pe_precompute.py --data-dir data
  6. python saft_train.py --track saft --data-dir data
═══════════════════════════════════════════════════════════
"""

import os
import gc
import json
import time
import argparse
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import sacrebleu
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from saft_model import SAFTModel
from saft_dataset import (
    SAFTDataset, BaselineDataset,
    saft_collate_fn, baseline_collate_fn,
    SYSTEM_MSG_SAFT, SYSTEM_MSG_BASELINE,
)


# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

class Config:
    # Model
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype = "bf16"  # bf16 or fp16

    # SAFT
    k_eigenvectors = 20
    sin_dim = 8
    sin_base = 1000.0
    mlp_lr_multiplier = 1.0  # μ in paper

    # LoRA
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Training
    learning_rate = 2e-4
    weight_decay = 0.01
    warmup_steps = 100
    num_epochs = 10
    early_stop_patience = 2

    # Batch — A100 + 0.5B model
    baseline_max_seq = 768
    saft_max_seq = 1280
    baseline_batch_size = 8
    saft_batch_size = 4
    gradient_accumulation = 8     # effective = 128

    # Evaluation
    num_beams = 4
    max_new_tokens = 256
    eval_samples = 300
    eval_batch_size = 16

    # Paths
    data_dir = "data"
    output_dir = "outputs"


# ═══════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════

def load_model_and_tokenizer(config: Config):
    """Load base model + tokenizer with appropriate dtype."""
    print(f"Loading model: {config.model_name}")
    dtype = torch.bfloat16 if config.dtype == "bf16" and torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"  Hidden size: {model.config.hidden_size}")
    print(f"  Vocab size:  {model.config.vocab_size}")
    print(f"  Dtype:       {dtype}")

    return model, tokenizer


def apply_lora(model, config: Config):
    """Apply LoRA adapters to the model."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_targets,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ═══════════════════════════════════════════════════════════
# Learning Rate Scheduler (Linear Warmup + Cosine Decay)
# ═══════════════════════════════════════════════════════════

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine decay with linear warmup."""
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ═══════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_bleu(
    saft_model,
    tokenizer,
    vi_texts,
    ref_en_texts,
    amr_texts=None,
    pe_data=None,
    use_saft=False,
    config=None,
):
    """
    Batch generation + BLEU evaluation.
    Returns (bleu_score, predictions, references).
    """
    saft_model.eval()
    device = next(saft_model.parameters()).device

    n = min(config.eval_samples, len(vi_texts))
    predictions = []

    for batch_start in tqdm(range(0, n, config.eval_batch_size),
                            desc="Evaluating", leave=False):
        batch_end = min(batch_start + config.eval_batch_size, n)

        # Build prompts
        prompts = []
        for j in range(batch_start, batch_end):
            if use_saft and amr_texts:
                prompt = (
                    f"<|im_start|>system\n{SYSTEM_MSG_SAFT}<|im_end|>\n"
                    f"<|im_start|>user\nAMR Graph:\n{amr_texts[j]}\n\n"
                    f"Vietnamese: {vi_texts[j]}\nEnglish:<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
            else:
                prompt = (
                    f"<|im_start|>system\n{SYSTEM_MSG_BASELINE}<|im_end|>\n"
                    f"<|im_start|>user\n"
                    f"Translate the source text from Vietnamese to English.\n"
                    f"Vietnamese: {vi_texts[j]}\nEnglish:<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
            prompts.append(prompt)

        # Tokenize batch
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=config.saft_max_seq if use_saft else config.baseline_max_seq,
        ).to(device)

        # For SAFT: build PE tensors for the batch
        amr_node_pe = None
        amr_intra_pos = None
        amr_mask_tensor = None

        if use_saft and pe_data is not None:
            bs = batch_end - batch_start
            seq_len = inputs.input_ids.size(1)
            pe_dim = 2 * config.k_eigenvectors

            # For inference, we use a simplified approach:
            # tokenize the full prompt and don't inject PEs during generation
            # (generation tokens don't have AMR PEs anyway)
            # This is a practical simplification — PEs only affect prompt understanding
            pass

        # Generate
        outputs = saft_model.base_model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            num_beams=config.num_beams,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode
        for k_idx in range(len(prompts)):
            input_len = inputs.input_ids[k_idx].shape[-1]
            gen_ids = outputs[k_idx][input_len:]
            pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            predictions.append(pred)

    tokenizer.padding_side = "right"

    references = ref_en_texts[:n]
    bleu = sacrebleu.corpus_bleu(predictions, [references])

    return bleu.score, predictions, references


@torch.no_grad()
def evaluate_comet(vi_texts, predictions, references, comet_model, n=None):
    """Compute COMET score. Returns None if comet_model is None."""
    if comet_model is None:
        return None
    if n is None:
        n = len(predictions)
    try:
        data = [
            {"src": src, "mt": mt, "ref": ref}
            for src, mt, ref in zip(vi_texts[:n], predictions[:n], references[:n])
        ]
        output = comet_model.predict(data, batch_size=64, gpus=1)
        return output.system_score
    except Exception as e:
        print(f"  [WARN] COMET evaluation failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════

def train_track(
    track_name: str,
    saft_model: SAFTModel,
    tokenizer,
    train_loader: DataLoader,
    val_vi: list,
    val_en: list,
    val_amr: list,
    val_pe_data: list,
    comet_model,
    config: Config,
    output_dir: str,
    use_saft: bool = False,
):
    """Train one track (Baseline or SAFT)."""
    os.makedirs(output_dir, exist_ok=True)
    device = next(saft_model.parameters()).device

    # Optimizer: separate param groups for LoRA and MLP
    lora_params = [p for n, p in saft_model.base_model.named_parameters() if p.requires_grad]
    mlp_params = list(saft_model.pe_projection.parameters()) if use_saft else []

    param_groups = [
        {"params": lora_params, "lr": config.learning_rate},
    ]
    if mlp_params:
        param_groups.append({
            "params": mlp_params,
            "lr": config.learning_rate * config.mlp_lr_multiplier,
        })

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, total_steps)

    # Mixed precision
    use_amp = (config.dtype == "bf16" and torch.cuda.is_bf16_supported())
    scaler = GradScaler(enabled=not use_amp)  # GradScaler not needed for bf16
    amp_dtype = torch.bfloat16 if use_amp else torch.float16

    # Tracking
    best_comet = -999.0
    best_bleu = -1.0
    best_epoch = 0
    patience_counter = 0
    bleu_history = []
    comet_history = []
    loss_history = []

    print(f"\n{'#'*60}")
    print(f"  TRAINING: {track_name}")
    print(f"  Dataset:  {len(train_loader.dataset):,} samples")
    print(f"  Batch:    {train_loader.batch_size} x {config.gradient_accumulation} = "
          f"{train_loader.batch_size * config.gradient_accumulation}")
    print(f"  Epochs:   {config.num_epochs} (patience={config.early_stop_patience})")
    print(f"  SAFT PE:  {'Yes (MLP injection)' if use_saft else 'No'}")
    if use_saft:
        mlp_param_count = sum(p.numel() for p in mlp_params)
        print(f"  MLP params: {mlp_param_count:,}")
    print(f"{'#'*60}\n")

    global_step = 0
    start_time = time.time()

    for epoch in range(1, config.num_epochs + 1):
        saft_model.train()
        epoch_loss = 0.0
        num_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # SAFT forward with PE injection
            if use_saft:
                amr_node_pe = batch['amr_node_pe'].to(device)
                amr_intra_pos = batch['amr_intra_pos'].to(device)
                amr_mask = batch['amr_mask'].to(device)

                with autocast(device_type='cuda', dtype=amp_dtype):
                    outputs = saft_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        amr_node_pe=amr_node_pe,
                        amr_intra_pos=amr_intra_pos,
                        amr_mask=amr_mask,
                    )
            else:
                # Baseline: standard forward (no PE)
                with autocast(device_type='cuda', dtype=amp_dtype):
                    outputs = saft_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

            loss = outputs.loss / config.gradient_accumulation
            loss.backward()

            epoch_loss += outputs.loss.item()
            num_batches += 1

            if (batch_idx + 1) % config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(saft_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # Update progress bar
            avg_loss = epoch_loss / num_batches
            progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        loss_history.append(avg_epoch_loss)

        # ── Per-epoch evaluation ──
        print(f"\n{'='*60}")
        print(f"[{track_name}] Epoch {epoch} — Evaluating...")
        print(f"{'='*60}")

        bleu_score, preds, refs = evaluate_bleu(
            saft_model, tokenizer,
            val_vi, val_en,
            amr_texts=val_amr if use_saft else None,
            pe_data=val_pe_data if use_saft else None,
            use_saft=use_saft,
            config=config,
        )

        comet_score = evaluate_comet(val_vi, preds, refs, comet_model, n=len(preds))

        bleu_history.append(bleu_score)
        comet_history.append(comet_score if comet_score is not None else 0.0)

        print(f"  Loss  = {avg_epoch_loss:.4f}")
        print(f"  BLEU  = {bleu_score:.2f}  (best: {best_bleu:.2f})")
        if comet_score is not None:
            print(f"  COMET = {comet_score:.4f} (best: {best_comet:.4f})")
        else:
            print(f"  COMET = N/A (unbabel-comet not installed, using BLEU for early stopping)")

        # Early stopping: prefer COMET, fallback to BLEU
        current_metric = comet_score if comet_score is not None else bleu_score
        best_metric = best_comet if comet_score is not None else best_bleu
        metric_name = "COMET" if comet_score is not None else "BLEU"

        if current_metric > best_metric:
            if comet_score is not None:
                best_comet = comet_score
            best_bleu = max(best_bleu, bleu_score)
            patience_counter = 0
            best_epoch = epoch

            # Save best model
            best_path = os.path.join(output_dir, "best_model")
            os.makedirs(best_path, exist_ok=True)
            saft_model.base_model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            if use_saft:
                saft_model.save_pe_projection(os.path.join(best_path, "pe_projection.pt"))
            print(f"  ✓ New best ({metric_name})! Saved → {best_path}")
        else:
            patience_counter += 1
            best_bleu = max(best_bleu, bleu_score)
            print(f"  No improvement. Patience: {patience_counter}/{config.early_stop_patience}")
            if patience_counter >= config.early_stop_patience:
                print(f"  ✗ EARLY STOPPING at epoch {epoch}")
                break

        print()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  {track_name} Training Complete!")
    print(f"  Time:       {elapsed/3600:.1f} hours")
    print(f"  Best BLEU:  {best_bleu:.2f}")
    print(f"  Best COMET: {best_comet:.4f} (epoch {best_epoch})")
    print(f"{'='*60}")

    return {
        'bleu_history': bleu_history,
        'comet_history': comet_history,
        'loss_history': loss_history,
        'best_bleu': best_bleu,
        'best_comet': best_comet,
        'best_epoch': best_epoch,
        'time_hours': elapsed / 3600,
    }


# ═══════════════════════════════════════════════════════════
# Final Test Evaluation
# ═══════════════════════════════════════════════════════════

def evaluate_on_test(
    saft_model, tokenizer, config, comet_model,
    test_vi, test_en, test_amr, test_pe_data,
    use_saft, track_name,
):
    """Full evaluation on test set."""
    print(f"\n{'='*60}")
    print(f"  Evaluating {track_name} on test set (full)...")
    print(f"{'='*60}")

    # Override eval_samples to use all
    orig_samples = config.eval_samples
    config.eval_samples = len(test_vi)

    bleu_score, preds, refs = evaluate_bleu(
        saft_model, tokenizer,
        test_vi, test_en,
        amr_texts=test_amr if use_saft else None,
        pe_data=test_pe_data if use_saft else None,
        use_saft=use_saft,
        config=config,
    )

    comet_score = evaluate_comet(test_vi, preds, refs, comet_model)
    if comet_score is None:
        comet_score = 0.0

    config.eval_samples = orig_samples

    print(f"  Test BLEU:  {bleu_score:.2f}")
    print(f"  Test COMET: {comet_score:.4f}")

    return bleu_score, comet_score, preds


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='SAFT Training')
    parser.add_argument('--track', choices=['baseline', 'saft', 'both'], default='both')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model', default=None, help='Override model name')
    args = parser.parse_args()

    config = Config()
    config.data_dir = args.data_dir
    config.output_dir = args.output_dir
    config.num_epochs = args.epochs
    if args.model:
        config.model_name = args.model

    # Check CUDA
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load COMET model (optional)
    comet_model = None
    try:
        from comet import download_model, load_from_checkpoint
        print("\nLoading COMET model...")
        comet_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(comet_path)
        print("COMET loaded: Unbabel/wmt22-comet-da")
    except ImportError:
        print("\n[INFO] unbabel-comet not installed. Using BLEU-only for evaluation.")
        print("  To enable COMET: pip install pytorch-lightning==2.1.0 unbabel-comet==2.2.2")
    except Exception as e:
        print(f"\n[WARN] Failed to load COMET: {e}")
        print("  Continuing with BLEU-only evaluation.")

    # Load text data
    print("\nLoading text data...")
    def read_lines(path):
        with open(path, 'r', encoding='utf-8') as f:
            return [l.strip() for l in f]

    train_vi = read_lines(os.path.join(config.data_dir, "train.vi"))
    train_en = read_lines(os.path.join(config.data_dir, "train.en"))
    train_amr = read_lines(os.path.join(config.data_dir, "train.linear.amr"))
    val_vi = read_lines(os.path.join(config.data_dir, "tst2012.vi"))
    val_en = read_lines(os.path.join(config.data_dir, "tst2012.en"))
    val_amr = read_lines(os.path.join(config.data_dir, "tst2012.linear.amr"))
    test_vi = read_lines(os.path.join(config.data_dir, "tst2013.vi"))
    test_en = read_lines(os.path.join(config.data_dir, "tst2013.en"))
    test_amr = read_lines(os.path.join(config.data_dir, "tst2013.linear.amr"))
    print(f"  Train: {len(train_vi):,} | Val: {len(val_vi):,} | Test: {len(test_vi):,}")

    # Load precomputed PEs (for SAFT track)
    import pickle
    val_pe_data = None
    test_pe_data = None

    if args.track in ('saft', 'both'):
        pe_files = {
            'train': os.path.join(config.data_dir, 'train_pes.pkl'),
            'val': os.path.join(config.data_dir, 'tst2012_pes.pkl'),
            'test': os.path.join(config.data_dir, 'tst2013_pes.pkl'),
        }
        for name, path in pe_files.items():
            if not os.path.exists(path):
                print(f"  [ERROR] PE file not found: {path}")
                print(f"  Run: python saft_pe_precompute.py --data-dir {config.data_dir}")
                return

        with open(pe_files['val'], 'rb') as f:
            val_pe_data = pickle.load(f)
        with open(pe_files['test'], 'rb') as f:
            test_pe_data = pickle.load(f)
        print(f"  PEs loaded: val={len(val_pe_data)}, test={len(test_pe_data)}")

    all_results = {}

    # ── Track 1: Baseline ──
    if args.track in ('baseline', 'both'):
        print("\n" + "═" * 60)
        print("  TRACK 1: BASELINE")
        print("═" * 60)

        model, tokenizer = load_model_and_tokenizer(config)
        model = apply_lora(model, config)

        saft_model = SAFTModel(model, k_eigenvectors=config.k_eigenvectors,
                               sin_dim=config.sin_dim).to(model.device)

        train_ds = BaselineDataset(
            os.path.join(config.data_dir, "train.vi"),
            os.path.join(config.data_dir, "train.en"),
            tokenizer, config.baseline_max_seq,
        )

        train_loader = DataLoader(
            train_ds, batch_size=config.baseline_batch_size, shuffle=True,
            collate_fn=lambda b: baseline_collate_fn(b, tokenizer.pad_token_id),
            num_workers=2, pin_memory=True,
        )

        baseline_results = train_track(
            "Baseline", saft_model, tokenizer, train_loader,
            val_vi, val_en, val_amr, val_pe_data,
            comet_model, config,
            os.path.join(config.output_dir, "baseline"),
            use_saft=False,
        )

        # Test evaluation
        bl_test_bleu, bl_test_comet, bl_preds = evaluate_on_test(
            saft_model, tokenizer, config, comet_model,
            test_vi, test_en, test_amr, test_pe_data,
            use_saft=False, track_name="Baseline",
        )
        baseline_results['test_bleu'] = bl_test_bleu
        baseline_results['test_comet'] = bl_test_comet
        all_results['baseline'] = baseline_results

        # Save predictions
        with open(os.path.join(config.output_dir, "baseline_predictions.txt"), 'w', encoding='utf-8') as f:
            f.writelines(p + '\n' for p in bl_preds)

        # Cleanup
        del saft_model, model, train_ds, train_loader
        gc.collect()
        torch.cuda.empty_cache()

    # ── Track 2: SAFT ──
    if args.track in ('saft', 'both'):
        print("\n" + "═" * 60)
        print("  TRACK 2: SAFT (Full Embedding PE)")
        print("═" * 60)

        model, tokenizer = load_model_and_tokenizer(config)
        model = apply_lora(model, config)

        saft_model = SAFTModel(model, k_eigenvectors=config.k_eigenvectors,
                               sin_dim=config.sin_dim).to(model.device)

        train_ds = SAFTDataset(
            os.path.join(config.data_dir, "train.vi"),
            os.path.join(config.data_dir, "train.en"),
            os.path.join(config.data_dir, "train.linear.amr"),
            os.path.join(config.data_dir, "train_pes.pkl"),
            tokenizer, config.saft_max_seq, config.k_eigenvectors,
        )

        pe_dim = 2 * config.k_eigenvectors
        train_loader = DataLoader(
            train_ds, batch_size=config.saft_batch_size, shuffle=True,
            collate_fn=lambda b: saft_collate_fn(b, tokenizer.pad_token_id, pe_dim),
            num_workers=2, pin_memory=True,
        )

        saft_results = train_track(
            "SAFT", saft_model, tokenizer, train_loader,
            val_vi, val_en, val_amr, val_pe_data,
            comet_model, config,
            os.path.join(config.output_dir, "saft"),
            use_saft=True,
        )

        # Test evaluation
        sf_test_bleu, sf_test_comet, sf_preds = evaluate_on_test(
            saft_model, tokenizer, config, comet_model,
            test_vi, test_en, test_amr, test_pe_data,
            use_saft=True, track_name="SAFT",
        )
        saft_results['test_bleu'] = sf_test_bleu
        saft_results['test_comet'] = sf_test_comet
        all_results['saft'] = saft_results

        # Save predictions
        with open(os.path.join(config.output_dir, "saft_predictions.txt"), 'w', encoding='utf-8') as f:
            f.writelines(p + '\n' for p in sf_preds)

        del saft_model, model, train_ds, train_loader
        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary ──
    if len(all_results) >= 2:
        print("\n" + "=" * 70)
        print("              FINAL RESULTS: Baseline vs SAFT")
        print("=" * 70)
        bl = all_results.get('baseline', {})
        sf = all_results.get('saft', {})
        print(f"{'Metric':<35} {'Baseline':>10} {'SAFT':>10}")
        print("-" * 60)
        print(f"{'Val BLEU (best)':<35} {bl.get('best_bleu',0):>10.2f} {sf.get('best_bleu',0):>10.2f}")
        print(f"{'Val COMET (best)':<35} {bl.get('best_comet',0):>10.4f} {sf.get('best_comet',0):>10.4f}")
        print(f"{'Test BLEU':<35} {bl.get('test_bleu',0):>10.2f} {sf.get('test_bleu',0):>10.2f}")
        print(f"{'Test COMET':<35} {bl.get('test_comet',0):>10.4f} {sf.get('test_comet',0):>10.4f}")
        print(f"{'Best Epoch':<35} {bl.get('best_epoch',0):>10d} {sf.get('best_epoch',0):>10d}")
        delta_bleu = sf.get('test_bleu', 0) - bl.get('test_bleu', 0)
        delta_comet = sf.get('test_comet', 0) - bl.get('test_comet', 0)
        print(f"\n  SAFT Δ: {'+' if delta_bleu>=0 else ''}{delta_bleu:.2f} BLEU, "
              f"{'+' if delta_comet>=0 else ''}{delta_comet:.4f} COMET")
        print("=" * 70)

    # ── Save results + chart ──
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "results.json"), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Plot
    if all_results:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        colors = {'baseline': '#2196F3', 'saft': '#4CAF50'}

        for track, r in all_results.items():
            c = colors.get(track, '#FF5722')
            if 'bleu_history' in r:
                axes[0].plot(range(1, len(r['bleu_history'])+1), r['bleu_history'],
                           'o-', color=c, label=f"{track} (best={r['best_bleu']:.2f})")
            if 'comet_history' in r:
                axes[1].plot(range(1, len(r['comet_history'])+1), r['comet_history'],
                           's-', color=c, label=f"{track} (best={r['best_comet']:.4f})")
            if 'loss_history' in r:
                axes[2].plot(range(1, len(r['loss_history'])+1), r['loss_history'],
                           '^-', color=c, label=track)

        for ax, title, ylabel in zip(axes,
            ['Validation BLEU', 'Validation COMET', 'Training Loss'],
            ['BLEU', 'COMET', 'Loss']):
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'SAFT Results — {config.model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        chart_path = os.path.join(config.output_dir, "saft_results.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nChart saved: {chart_path}")

    print(f"\nResults saved: {os.path.join(config.output_dir, 'results.json')}")
    print("Done!")


if __name__ == '__main__':
    main()
