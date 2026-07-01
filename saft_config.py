"""
SAFT Configuration — Model Presets
═══════════════════════════════════════════════════════════
Separate config file for different model families (Qwen3, Qwen2.5).
Keeps all model-specific hyperparameters in one place.

Usage:
    python saft_train.py --brand qwen2.5 --track saft
    python saft_train.py --brand qwen3  --track baseline
═══════════════════════════════════════════════════════════
"""


class BaseConfig:
    """Shared hyperparameters across all model brands."""

    # Chat template format: 'chatml' (Qwen) or 'gemma'
    chat_format = "chatml"

    # SAFT PE
    k_eigenvectors = 20
    sin_dim = 8
    sin_base = 1000.0
    mlp_lr_multiplier = 1.0

    # LoRA
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]

    # Training
    learning_rate = 2e-4
    weight_decay = 0.01
    warmup_steps = 100
    num_epochs = 10
    early_stop_patience = 2
    gradient_accumulation = 8

    # AMR chunking
    max_chunks = 3

    # Evaluation
    num_beams = 4
    max_new_tokens = 256
    eval_samples = 300
    eval_batch_size = 16

    # Paths (overridden by CLI)
    data_dir = "data"
    output_dir = "outputs"


class Qwen3Config(BaseConfig):
    """
    Qwen3-0.6B configuration.
    - Architecture: Qwen3 (with hybrid thinking support, but we don't use it)
    - Hidden size: 1024
    - ChatML format: <|im_start|> / <|im_end|>
    - Vocab size: ~151k
    """
    brand = "qwen3"
    model_name = "Qwen/Qwen3-0.6B"
    dtype = "bf16"

    # Batch — tuned for Qwen3-0.6B on L4/T4 GPU
    baseline_max_seq = 768
    saft_max_seq = 1280
    baseline_batch_size = 8
    saft_batch_size = 4

    # Default output
    output_dir = "/content/drive/MyDrive/output_Qwen3-0.6B"


class Qwen25Config(BaseConfig):
    """
    Qwen2.5-0.5B-Instruct configuration.
    - Architecture: Qwen2.5 (no thinking mode)
    - Hidden size: 896
    - ChatML format: <|im_start|> / <|im_end|> (identical to Qwen3)
    - Vocab size: ~151k
    - Trained with default system prompt "You are a helpful assistant."

    Key differences from Qwen3:
    1. Slightly smaller hidden dim (896 vs 1024) → auto-detected, no code change
    2. No hybrid thinking mode → irrelevant (we don't use thinking)
    3. Same ChatML template → prompts are 100% compatible
    4. Same LoRA module names (q_proj, v_proj, etc.)
    """
    brand = "qwen2.5"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype = "bf16"

    # Batch — Qwen2.5-0.5B is slightly smaller, can fit more
    baseline_max_seq = 768
    saft_max_seq = 1280
    baseline_batch_size = 8
    saft_batch_size = 4

    # Default output
    output_dir = "/content/drive/MyDrive/output_Qwen2.5-0.5B"


class Qwen25_1_5BConfig(BaseConfig):
    """
    Qwen2.5-1.5B-Instruct configuration.
    - Architecture: Qwen2.5
    - Hidden size: 1536
    - Larger model → reduce batch size
    """
    brand = "qwen2.5-1.5b"
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype = "bf16"

    # Batch — larger model, smaller batches
    baseline_max_seq = 768
    saft_max_seq = 1024
    baseline_batch_size = 4
    saft_batch_size = 2

    # Default output
    output_dir = "/content/drive/MyDrive/output_Qwen2.5-1.5B"

class Gemma2BConfig(BaseConfig):
    """
    Google Gemma-2-2B-IT configuration.
    - Architecture: Gemma-2 (GQA, RoPE, SwiGLU)
    - Hidden size: 2304
    - Chat template: <start_of_turn>user / <end_of_turn> (NOT ChatML)
    - Context: 8192
    - Vocab size: ~256k
    """
    brand = "gemma-2b"
    model_name = "google/gemma-2-2b-it"
    dtype = "bf16"

    # Gemma uses its own chat template
    chat_format = "gemma"

    # LoRA — test with 2 targets only
    lora_targets = ["q_proj", "k_proj"]

    # Batch — 2B model, reduce batch sizes
    baseline_max_seq = 768
    saft_max_seq = 1024
    baseline_batch_size = 4
    saft_batch_size = 2

    # Default output
    output_dir = "/content/drive/MyDrive/output_Gemma2-2B-en2vi"

class Qwen25BaselineConfig(BaseConfig):
    """
    Deliberately weakened Qwen2.5-0.5B config for ablation study.
    Purpose: Create a lower-bound baseline to show SAFT improvement.

    Differences from full Qwen25Config:
    - LoRA rank 4 (vs 16) → 4× fewer trainable params
    - Only 2 LoRA targets (vs 7) → ~70% fewer adapted modules
    - 2 epochs (vs 10) → undertrained
    - Lower LR (5e-5 vs 2e-4) → slower convergence
    - No early stopping patience → trains exactly 2 epochs
    """
    brand = "qwen2.5-baseline"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype = "bf16"

    # Weak LoRA — minimal adaptation
    lora_r = 4
    lora_alpha = 8
    lora_dropout = 0.1
    lora_targets = ["q_proj", "v_proj"]

    # Undertrained
    learning_rate = 5e-5
    num_epochs = 2
    early_stop_patience = 99  # effectively disabled
    gradient_accumulation = 2

    # Batch
    baseline_max_seq = 768
    saft_max_seq = 1280
    baseline_batch_size = 8
    saft_batch_size = 4

    # Default output
    output_dir = "/content/drive/MyDrive/output_Qwen2.5-0.5B-baseline"


# ── Registry ──
BRAND_CONFIGS = {
    "qwen3": Qwen3Config,
    "qwen2.5": Qwen25Config,
    "qwen2.5-0.5b": Qwen25Config,       # alias
    "qwen2.5-1.5b": Qwen25_1_5BConfig,
    "qwen2.5-baseline": Qwen25BaselineConfig,
    "gemma-2b": Gemma2BConfig,
}


def get_config(brand: str = "qwen3") -> BaseConfig:
    """Get config for a model brand. Case-insensitive."""
    brand_lower = brand.lower().strip()
    if brand_lower not in BRAND_CONFIGS:
        available = ", ".join(sorted(BRAND_CONFIGS.keys()))
        raise ValueError(
            f"Unknown brand '{brand}'. Available: {available}"
        )
    return BRAND_CONFIGS[brand_lower]()
