import os
import sys
import pickle
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer
from saft_model import SAFTModel
from saft_config import get_config, BRAND_CONFIGS


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True,
                    help="Thư mục best_model (chứa pe_projection.pt)")
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--split", default="tst2013")
    ap.add_argument("--brand", default="qwen2.5-1.5b",
                    choices=sorted(BRAND_CONFIGS.keys()))
    ap.add_argument("--n-samples", type=int, default=200,
                    help="Số mẫu để lấy thống kê")
    args = ap.parse_args()

    cfg = get_config(args.brand)
    k = cfg.k_eigenvectors
    sin_dim = cfg.sin_dim
    pe_dim = 2 * k

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    print(f"Loading model: {args.model_path}  (dtype={dtype})")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    saft = SAFTModel(base, k_eigenvectors=k, sin_dim=sin_dim)
    pe_path = os.path.join(args.model_path, "pe_projection.pt")
    if os.path.exists(pe_path):
        saft.load_pe_projection(pe_path)
        print(f"  Loaded trained MLP: {pe_path}")
    else:
        print(f"  [WARN] {pe_path} not found → testing untrained MLP.")
    saft.eval()
    device = next(saft.parameters()).device

    emb_layer = saft.get_embedding_layer()
    W = emb_layer.weight.detach().float()
    emb_row_norm = W.norm(dim=1)
    print(f"\nToken embedding: vocab={W.shape[0]}, d_emb={W.shape[1]}")
    print(f"  ||token_emb|| mỗi hàng: mean={emb_row_norm.mean():.4f}  "
          f"median={emb_row_norm.median():.4f}")
    pe_file = os.path.join(args.data_dir, f"{args.split}_pes.pkl")
    if not os.path.exists(pe_file):
        print(f"[ERROR] {pe_file} not found. Run saft_pe_precompute.py first.")
        return
    with open(pe_file, "rb") as f:
        pe_data = pickle.load(f)

    sin_encoder = saft.sin_pe_encoder
    proj = saft.pe_projection

    amr_pe_norms = []
    ratios = []
    n = min(args.n_samples, len(pe_data))
    emb_med = emb_row_norm.median().item()

    with torch.no_grad():
        for i in range(n):
            labels = pe_data[i]["labels"]
            label_pes = pe_data[i]["label_pes"]
            if len(labels) == 0:
                continue
            node_pe = torch.tensor(label_pes, dtype=torch.float32, device=device)
            intra = torch.zeros(len(labels), dtype=torch.long, device=device)
            sin_pe = sin_encoder(intra.unsqueeze(0)).squeeze(0).to(node_pe.dtype) 
            mask = torch.ones(len(labels), device=device)

            amr_pe = proj(
                node_pe.unsqueeze(0).to(proj.net[0].weight.dtype),
                sin_pe.unsqueeze(0).to(proj.net[0].weight.dtype),
                mask.unsqueeze(0),
            ).squeeze(0).float()                          

            norms = amr_pe.norm(dim=1)                      
            amr_pe_norms.extend(norms.cpu().tolist())
            ratios.extend((norms / emb_med).cpu().tolist())

    amr_pe_norms = np.array(amr_pe_norms)
    ratios = np.array(ratios)

    print(f"\n{'='*60}")
    print(f"  KẾT QUẢ ({n} mẫu, {len(amr_pe_norms)} token AMR)")
    print(f"{'='*60}")
    print(f"  ||amr_pe|| (PE đã chiếu): mean={amr_pe_norms.mean():.4f}  "
          f"median={np.median(amr_pe_norms):.4f}  max={amr_pe_norms.max():.4f}")
    print(f"  ||token_emb|| (median)  : {emb_med:.4f}")
    print(f"  Tỉ lệ ||amr_pe|| / ||token_emb||:")
    print(f"     mean   = {ratios.mean()*100:.2f}%")
    print(f"     median = {np.median(ratios)*100:.2f}%")
    print(f"     p90    = {np.percentile(ratios,90)*100:.2f}%")
    print(f"{'='*60}")
    print("\nDIỄN GIẢI:")
    if np.median(ratios) < 0.02:
        print("Tỉ lệ < 2%: PE gần như không đáng kể so với token embedding.")
        print("PE bị model bỏ qua.")
    elif np.median(ratios) < 0.10:
        print("Tỉ lệ 2–10%: PE có hiện diện nhưng yếu. Có thể vẫn bị lấn át.")
    else:
        print("Tỉ lệ > 10%: PE có độ lớn đáng kể. PE đóng góp.")


if __name__ == "__main__":
    main()
