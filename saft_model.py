"""
SAFT Model — Structure-Aware Fine-Tuning
═════════════════════════════════════════════════════════
Custom model wrapper that injects Magnetic Laplacian PEs
into the LLM embedding layer via a trainable 2-layer MLP.

Following SAFT paper (arXiv:2507.13381) Section 3.3:
    AmrPE_j = f_θ( PE(v_i) ‖ SinPE(j) )
    H = Embed(tokens) + AmrPE
═════════════════════════════════════════════════════════
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal PE for intra-node token ordering.
    Following Vaswani et al. (2017), with base=1000.
    Paper Section 3.2, Eq. 5: IntraPE_j = SinPE(j)
    """

    def __init__(self, dim: int = 8, base: float = 1000.0):
        super().__init__()
        self.dim = dim
        self.base = base

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (batch, seq_len) int/long tensor — intra-node position index
        Returns:
            pe: (batch, seq_len, dim) float tensor
        """
        device = positions.device
        dtype = torch.float32
        pe = torch.zeros(*positions.shape, self.dim, device=device, dtype=dtype)

        for i in range(self.dim // 2):
            freq = 1.0 / (self.base ** (2 * i / self.dim))
            pe[..., 2 * i] = torch.sin(positions.float() * freq)
            pe[..., 2 * i + 1] = torch.cos(positions.float() * freq)

        # Handle odd dim
        if self.dim % 2 == 1:
            freq = 1.0 / (self.base ** (2 * (self.dim // 2) / self.dim))
            pe[..., -1] = torch.sin(positions.float() * freq)

        return pe


class AmrPEProjection(nn.Module):
    """
    Two-layer MLP that projects concatenated [node_PE ‖ SinPE] into
    the LLM embedding space.

    Paper Section 3.2, Eq. 6:
        f_θ: ℝ^{2k+d} → ℝ^{d_emb}
        f_θ = Linear → GeLU → Linear
    """

    def __init__(self, pe_dim: int, sin_dim: int, d_emb: int):
        super().__init__()
        input_dim = pe_dim + sin_dim  # 2k + d
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb),
        )
        # Initialize with small weights to avoid disrupting pretrained embeddings
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        node_pe: torch.Tensor,
        sin_pe: torch.Tensor,
        amr_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_pe:  (batch, seq_len, 2k)  — Magnetic Laplacian PE per token
            sin_pe:   (batch, seq_len, d)   — Sinusoidal intra-node PE
            amr_mask: (batch, seq_len)      — 1.0 for AMR concept tokens, 0.0 otherwise

        Returns:
            amr_pe:   (batch, seq_len, d_emb) — projected PE, zero for non-AMR tokens
        """
        x = torch.cat([node_pe, sin_pe], dim=-1)  # (batch, seq, 2k+d)
        projected = self.net(x)                     # (batch, seq, d_emb)
        return projected * amr_mask.unsqueeze(-1)   # zero out non-AMR tokens


class SAFTModel(nn.Module):
    """
    SAFT wrapper around a HuggingFace CausalLM model.

    Injects AMR positional encodings into the embedding layer:
        H = Embed(input_ids) + AmrPE

    The base model (with LoRA) and the MLP projection are trained together.
    """

    def __init__(
        self,
        base_model: nn.Module,
        k_eigenvectors: int = 20,
        sin_dim: int = 8,
        sin_base: float = 1000.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.k = k_eigenvectors
        self.sin_dim = sin_dim

        # Get embedding dimension from model config
        d_emb = base_model.config.hidden_size

        # Trainable MLP projection: ℝ^{2k+d} → ℝ^{d_emb}
        self.pe_projection = AmrPEProjection(
            pe_dim=2 * k_eigenvectors,
            sin_dim=sin_dim,
            d_emb=d_emb,
        )

        # Sinusoidal PE for intra-node token positions
        self.sin_pe_encoder = SinusoidalPositionalEncoding(
            dim=sin_dim, base=sin_base
        )

    def get_embedding_layer(self):
        """Get the token embedding layer from the (possibly PEFT-wrapped) model."""
        return self.base_model.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        amr_node_pe: Optional[torch.Tensor] = None,
        amr_intra_pos: Optional[torch.Tensor] = None,
        amr_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass with optional AMR PE injection.

        Args:
            input_ids:      (batch, seq_len)
            attention_mask:  (batch, seq_len)
            labels:          (batch, seq_len) — -100 for non-response tokens
            amr_node_pe:     (batch, seq_len, 2k) — Magnetic Laplacian PE
            amr_intra_pos:   (batch, seq_len) — intra-node position indices
            amr_mask:        (batch, seq_len) — 1.0 for AMR concept tokens

        Returns:
            HuggingFace CausalLMOutput (with .loss and .logits)
        """
        # Step 1: Get token embeddings
        embed_layer = self.get_embedding_layer()
        inputs_embeds = embed_layer(input_ids)

        # Step 2: Compute and add AMR PE (if provided)
        if amr_node_pe is not None and amr_mask is not None:
            # Compute sinusoidal intra-node PE
            sin_pe = self.sin_pe_encoder(amr_intra_pos)   # (batch, seq, sin_dim)
            sin_pe = sin_pe.to(inputs_embeds.dtype)

            # Project and add PE to embeddings
            amr_node_pe = amr_node_pe.to(inputs_embeds.dtype)
            amr_mask = amr_mask.to(inputs_embeds.dtype)

            amr_pe = self.pe_projection(amr_node_pe, sin_pe, amr_mask)
            inputs_embeds = inputs_embeds + amr_pe

        # Step 3: Forward through the rest of the model with modified embeddings
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs

    def generate(self, input_ids, attention_mask,
                 amr_node_pe=None, amr_intra_pos=None, amr_mask=None,
                 **generate_kwargs):
        """
        Generate with AMR PE injection for the prompt.
        """
        if amr_node_pe is not None and amr_mask is not None:
            embed_layer = self.get_embedding_layer()
            inputs_embeds = embed_layer(input_ids)

            sin_pe = self.sin_pe_encoder(amr_intra_pos)
            sin_pe = sin_pe.to(inputs_embeds.dtype)
            amr_node_pe = amr_node_pe.to(inputs_embeds.dtype)
            amr_mask = amr_mask.to(inputs_embeds.dtype)

            amr_pe = self.pe_projection(amr_node_pe, sin_pe, amr_mask)
            inputs_embeds = inputs_embeds + amr_pe

            return self.base_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generate_kwargs,
            )
        else:
            return self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )

    def save_pe_projection(self, path: str):
        """Save only the MLP projection weights."""
        torch.save(self.pe_projection.state_dict(), path)

    def load_pe_projection(self, path: str):
        """Load MLP projection weights."""
        state_dict = torch.load(path, map_location='cpu')
        self.pe_projection.load_state_dict(state_dict)
