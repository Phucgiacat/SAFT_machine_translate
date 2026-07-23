import os

def update_file(path, replacements):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old_s, new_s in replacements:
        content = content.replace(old_s, new_s)
        
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

base_dir = r"D:\learning\Thesis\SAFT\SAFT_machine_translate"

# Replace .bpe.amr to .bpe.amr globally first
for filename in os.listdir(base_dir):
    if filename.endswith(".py"):
        update_file(os.path.join(base_dir, filename), [('.bpe.amr', '.bpe.amr')])

# 1. saft_dataset.py
dataset_replacements = [
    (
        'def build_baseline_prompt_parts(vi_text: str, en_text: str = None):\n    """Build prompt parts for Baseline mode. Returns (system, user_content, assistant)."""\n    user_content = f"Translate the source text from Vietnamese to English.\\nVietnamese: {vi_text}\\nEnglish:"\n    return SYSTEM_MSG_BASELINE, user_content, en_text',
        'def build_baseline_prompt_parts(vi_text: str, amr_text: str, en_text: str = None):\n    """Build prompt parts for Baseline mode. Returns (system, user_content, assistant)."""\n    user_content = f"Translate the source text from Vietnamese to English. Using AMR: {amr_text}\\nsource: {vi_text}\\nEnglish:"\n    return SYSTEM_MSG_BASELINE, user_content, en_text'
    ),
    (
        'def tokenize_baseline(\n    tokenizer,\n    vi_text: str,\n    en_text: str,\n    max_seq_length: int = 1280,\n) -> Dict:\n    """Tokenize baseline prompt (no AMR, no PE)."""\n    system, user_content, assistant = build_baseline_prompt_parts(vi_text, en_text)\n\n    # Tokenize in parts for smart truncation (avoid cutting structural tokens)\n    struct_prefix = (\n        f"<|im_start|>system\\n{system}<|im_end|>\\n"\n        f"<|im_start|>user\\n"\n        f"Translate the source text from Vietnamese to English.\\n"\n        f"Vietnamese: "\n    )\n    struct_suffix = f"\\nEnglish:<|im_end|>\\n<|im_start|>assistant\\n"\n\n    prefix_ids = tokenizer.encode(struct_prefix, add_special_tokens=False)\n    vi_ids = tokenizer.encode(vi_text, add_special_tokens=False)\n    suffix_ids = tokenizer.encode(struct_suffix, add_special_tokens=False)\n    response_ids = tokenizer.encode(f"{en_text}<|im_end|>", add_special_tokens=False)\n\n    fixed_overhead = len(prefix_ids) + len(suffix_ids)\n    budget = max_seq_length - fixed_overhead\n\n    if len(vi_ids) + len(response_ids) > budget:\n        if len(response_ids) <= budget - 10:\n            # Keep full response, truncate Vietnamese\n            vi_ids = vi_ids[:budget - len(response_ids)]\n        else:\n            # Both need cutting: prioritize response (70%)\n            resp_budget = max(budget * 7 // 10, min(20, budget))\n            vi_ids = vi_ids[:budget - resp_budget]\n            response_ids = response_ids[:resp_budget]\n\n    prompt_ids = prefix_ids + vi_ids + suffix_ids\n    all_ids = prompt_ids + response_ids\n    if len(all_ids) > max_seq_length:\n        all_ids = all_ids[:max_seq_length]\n\n    seq_len = len(all_ids)\n    labels = [-100] * seq_len\n    for j in range(len(prompt_ids), seq_len):\n        labels[j] = all_ids[j]\n\n    return {\n        \'input_ids\': torch.tensor(all_ids, dtype=torch.long),\n        \'attention_mask\': torch.ones(seq_len, dtype=torch.long),\n        \'labels\': torch.tensor(labels, dtype=torch.long),\n    }',
        'def tokenize_baseline(\n    tokenizer,\n    vi_text: str,\n    amr_text: str,\n    en_text: str,\n    max_seq_length: int = 1280,\n) -> Dict:\n    """Tokenize baseline prompt with AMR text prompt (no PE)."""\n    system, user_content, assistant = build_baseline_prompt_parts(vi_text, amr_text, en_text)\n\n    struct_prefix = f"<|im_start|>system\\n{system}<|im_end|>\\n<|im_start|>user\\n{user_content}<|im_end|>\\n<|im_start|>assistant\\n"\n    prefix_ids = tokenizer.encode(struct_prefix, add_special_tokens=False)\n    response_ids = tokenizer.encode(f"{en_text}<|im_end|>", add_special_tokens=False)\n\n    all_ids = prefix_ids + response_ids\n    if len(all_ids) > max_seq_length:\n        all_ids = all_ids[:max_seq_length]\n\n    seq_len = len(all_ids)\n    labels = [-100] * seq_len\n    for j in range(len(prefix_ids), seq_len):\n        labels[j] = all_ids[j]\n\n    return {\n        \'input_ids\': torch.tensor(all_ids, dtype=torch.long),\n        \'attention_mask\': torch.ones(seq_len, dtype=torch.long),\n        \'labels\': torch.tensor(labels, dtype=torch.long),\n    }'
    ),
    (
        'class BaselineDataset(Dataset):\n    """Dataset for Baseline training (no AMR)."""\n\n    def __init__(\n        self,\n        vi_file: str,\n        en_file: str,\n        tokenizer,\n        max_seq_length: int = 768,\n    ):\n        self.tokenizer = tokenizer\n        self.max_seq_length = max_seq_length\n\n        with open(vi_file, \'r\', encoding=\'utf-8\') as f:\n            self.vi_texts = [l.strip() for l in f]\n        with open(en_file, \'r\', encoding=\'utf-8\') as f:\n            self.en_texts = [l.strip() for l in f]\n\n        n = min(len(self.vi_texts), len(self.en_texts))\n        self.vi_texts = self.vi_texts[:n]\n        self.en_texts = self.en_texts[:n]',
        'class BaselineDataset(Dataset):\n    """Dataset for Baseline training (with AMR text)."""\n\n    def __init__(\n        self,\n        vi_file: str,\n        amr_file: str,\n        en_file: str,\n        tokenizer,\n        max_seq_length: int = 768,\n    ):\n        self.tokenizer = tokenizer\n        self.max_seq_length = max_seq_length\n\n        with open(vi_file, \'r\', encoding=\'utf-8\') as f:\n            self.vi_texts = [l.strip() for l in f]\n        with open(amr_file, \'r\', encoding=\'utf-8\') as f:\n            self.amr_texts = [l.strip() for l in f]\n        with open(en_file, \'r\', encoding=\'utf-8\') as f:\n            self.en_texts = [l.strip() for l in f]\n\n        n = min(len(self.vi_texts), len(self.amr_texts), len(self.en_texts))\n        self.vi_texts = self.vi_texts[:n]\n        self.amr_texts = self.amr_texts[:n]\n        self.en_texts = self.en_texts[:n]'
    ),
    (
        '        result = tokenize_baseline(\n            self.tokenizer, self.vi_texts[idx], self.en_texts[idx],\n            self.max_seq_length,\n        )',
        '        result = tokenize_baseline(\n            self.tokenizer, self.vi_texts[idx], self.amr_texts[idx], self.en_texts[idx],\n            self.max_seq_length,\n        )'
    )
]
update_file(os.path.join(base_dir, 'saft_dataset.py'), dataset_replacements)

# 2. saft_train.py
train_replacements = [
    (
        '        train_ds = BaselineDataset(\n            os.path.join(config.data_dir, "train.vi"),\n            os.path.join(config.data_dir, "train.en"),\n            tokenizer, config.baseline_max_seq,\n        )',
        '        train_ds = BaselineDataset(\n            os.path.join(config.data_dir, "train.vi"),\n            os.path.join(config.data_dir, "train.bpe.amr"),\n            os.path.join(config.data_dir, "train.en"),\n            tokenizer, config.baseline_max_seq,\n        )'
    ),
    (
        '        else:\n            # ── Baseline path: standard string-based generation ──\n            prompts = []\n            for j in range(batch_start, batch_end):\n                prompt = (\n                    f"<|im_start|>system\\n{SYSTEM_MSG_BASELINE}<|im_end|>\\n"\n                    f"<|im_start|>user\\n"\n                    f"Translate the source text from Vietnamese to English.\\n"\n                    f"Vietnamese: {vi_texts[j]}\\nEnglish:<|im_end|>\\n"\n                    f"<|im_start|>assistant\\n"\n                )\n                prompts.append(prompt)',
        '        else:\n            # ── Baseline path: standard text generation ──\n            prompts = []\n            for i in range(batch_start, batch_end):\n                vi = vi_texts[i]\n                amr = amr_texts[i] if amr_texts else ""\n                sys_msg, user_msg, _ = build_baseline_prompt_parts(vi, amr)\n                \n                # Manual formatting (no PE logic)\n                prompt = f"<|im_start|>system\\n{sys_msg}<|im_end|>\\n<|im_start|>user\\n{user_msg}<|im_end|>\\n<|im_start|>assistant\\n"\n                prompts.append(prompt)'
    )
]
update_file(os.path.join(base_dir, 'saft_train.py'), train_replacements)

# 3. saft_eval.py
eval_replacements = [
    (
        '                else:\n                    p = (f"<|im_start|>system\\n{SYSTEM_MSG_BASELINE}<|im_end|>\\n"\n                         f"<|im_start|>user\\n"\n                         f"Translate the source text from Vietnamese to English.\\n"\n                         f"Vietnamese: {vi_texts[j]}\\nEnglish:<|im_end|>\\n"\n                         f"<|im_start|>assistant\\n")\n                prompts.append(p)',
        '                else:\n                    amr = amr_texts[j] if amr_texts else ""\n                    p = (f"<|im_start|>system\\n{SYSTEM_MSG_BASELINE}<|im_end|>\\n"\n                         f"<|im_start|>user\\n"\n                         f"Translate the source text from Vietnamese to English. Using AMR: {amr}\\n"\n                         f"source: {vi_texts[j]}\\nEnglish:<|im_end|>\\n"\n                         f"<|im_start|>assistant\\n")\n                prompts.append(p)'
    ),
    (
        '    else:\n        prompt = (f"<|im_start|>system\\n{SYSTEM_MSG_BASELINE}<|im_end|>\\n"\n                  f"<|im_start|>user\\n"\n                  f"Translate the source text from Vietnamese to English.\\n"\n                  f"Vietnamese: {vi_text}\\nEnglish:<|im_end|>\\n"\n                  f"<|im_start|>assistant\\n")',
        '    else:\n        amr = amr_text if amr_text else ""\n        prompt = (f"<|im_start|>system\\n{SYSTEM_MSG_BASELINE}<|im_end|>\\n"\n                  f"<|im_start|>user\\n"\n                  f"Translate the source text from Vietnamese to English. Using AMR: {amr}\\n"\n                  f"source: {vi_text}\\nEnglish:<|im_end|>\\n"\n                  f"<|im_start|>assistant\\n")'
    )
]
update_file(os.path.join(base_dir, 'saft_eval.py'), eval_replacements)

print("Files successfully updated and restored.")
