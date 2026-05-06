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
from saft_config import get_config, set_chat_format
from saft_eval import load_model, _build_eval_prompt_with_pe
from saft_bfs_linearize import bfs_linearize
from saft_pe_precompute import compute_pes_from_linear
from transformers import BartForConditionalGeneration

def parse_english_to_amr(text, amr_model, amr_tokenizer, device):
    """Parse an English sentence into a Penman AMR string using AMRBART."""
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
def translate_sentence_with_pe(model, tokenizer, english_text, pe_info, config, max_seq=1280):
    """Generate translation using SAFT model with PE injection."""
    model.eval()
    device = next(model.parameters()).device
    
    # Use the utility from saft_eval to build the prompt with PE alignment
    ids, pe, intra, mask = _build_eval_prompt_with_pe(
        tokenizer, english_text, pe_info, max_seq, config.k_eigenvectors
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
    parser.add_argument('--translate', type=str, default=None, help='English sentence to translate')
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
    if not os.path.exists(amrbart_repo):
        print(f"Error: AMRBART repository not found at {amrbart_repo}")
        print("Please clone https://github.com/goodbai-nlp/AMRBART.git and point --amrbart-path to it.")
        return
        
    if amrbart_repo not in sys.path:
        sys.path.insert(0, amrbart_repo)
    
    try:
        from model_interface.tokenization_bart import AMRBartTokenizer
    except ImportError as e:
        print(f"Failed to import AMRBartTokenizer from {amrbart_repo}: {e}")
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
    
    def process_sentence(english_text):
        print(f"\n[1] Parsing to AMR...")
        penman_amr = parse_english_to_amr(english_text, amr_model, amr_tokenizer, device)
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
            model, tokenizer, english_text, pe_info, config, max_seq=config.saft_max_seq
        )
        return translation
        
    if args.translate:
        translation = process_sentence(args.translate)
        print(f"\n🇺🇸 English:    {args.translate}")
        print(f"🇻🇳 Vietnamese: {translation}")
        
    if args.interactive:
        while True:
            try:
                english_text = input("\n🇺🇸 Enter English text (or 'quit'): ").strip()
                if not english_text or english_text.lower() in ('quit', 'q', 'exit'):
                    break
                translation = process_sentence(english_text)
                if translation:
                    print(f"🇻🇳 Vietnamese: {translation}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
