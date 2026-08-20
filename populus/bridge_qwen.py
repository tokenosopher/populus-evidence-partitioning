"""Pinned Qwen2.5-0.5B-Instruct genome behind the 4-method Bridge
interface (embed / forward_embeds / lm_head / d_model), with the
adapter-isolation seam required by REVIEW_run3_v2_ruling.md blocker 4:

  - attach_lora(): ONE shared LoRA (rank 8, attention projections,
    dropout 0, zero-init B) applied to the transformer core only —
    embeddings and LM head are never adapted.
  - forward_embeds(..., adapter_mode="cell"|"base"): "cell" runs with
    the adapter active; "base" runs with the adapter DISABLED — the
    pinned original weights, used by the mouth's h_base path and all
    prior/regression measurements. Bit-identity of the "base" mode
    with the pre-attachment model is verified by
    scripts/bridge_smoke_lora.py and is a standing regression test.

Base weights are loaded frozen; only LoRA matrices are trainable
after attachment (the Bridge module owns the projections/aux head).
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path

import torch
import torch.nn as nn

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PIN = Path(__file__).resolve().parent.parent / "results/bridge/genome_pin.json"

# Frozen LoRA configuration (REVIEW_run3_v2_ruling.md §4: freeze exact
# target modules, rank, alpha, initialization and dropout).
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0


class QwenGenome(nn.Module):
    def __init__(self, dtype=torch.float32, revision: str | None = None):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if revision is None:
            revision = json.loads(PIN.read_text())["revision"]
        self.revision = revision
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID,
                                                 revision=revision)
        # attention backend pinned to "eager" (serialization contract;
        # deterministic and identical across preflight/training)
        causal = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=revision, dtype=dtype,
            attn_implementation="eager")
        causal.eval()
        for p in causal.parameters():
            p.requires_grad_(False)
        self._embed = causal.get_input_embeddings()
        self._head = causal.lm_head
        self.core = causal.model            # Qwen2Model (no LM head)
        self.d_model = causal.config.hidden_size
        self._lora = False

    # ------------------------------------------------------------
    def attach_lora(self):
        """Attach the single shared LoRA to the transformer core.
        Embeddings and LM head stay untouched (they are held outside
        the wrapped core). Idempotent-guarded."""
        assert not self._lora, "LoRA already attached"
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA,
                         lora_dropout=LORA_DROPOUT,
                         target_modules=LORA_TARGETS, bias="none",
                         task_type="FEATURE_EXTRACTION")
        self.core = get_peft_model(self.core, cfg)
        self._lora = True
        return self

    def lora_parameters(self):
        return [p for n, p in self.core.named_parameters()
                if "lora_" in n and p.requires_grad]

    # ------------------------------------------------------------
    def embed(self, ids):
        return self._embed(ids)

    def forward_embeds(self, x, attn_mask, adapter_mode: str = "cell"):
        assert adapter_mode in ("cell", "base"), adapter_mode
        """adapter_mode 'cell': shared LoRA active (if attached).
        'base': adapter disabled — the pinned original computation,
        required for the mouth's h_base, the DP-7 prior vector, and
        every regression/calibration measurement."""
        if self._lora and adapter_mode == "base":
            ctx = self.core.disable_adapter()
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            return self.core(inputs_embeds=x, attention_mask=attn_mask
                             ).last_hidden_state

    def lm_head(self, h):
        return self._head(h)

    def encode(self, texts, device=None, fixed_length: int | None = None):
        """fixed_length: pad every sequence to this exact global
        length (serialization contract) so packet positions never
        depend on batch composition. Truncation is forbidden — any
        text longer than fixed_length raises."""
        if fixed_length is None:
            out = self.tok(texts, return_tensors="pt", padding=True,
                           add_special_tokens=False)
        else:
            out = self.tok(texts, return_tensors="pt",
                           padding="max_length",
                           max_length=fixed_length,
                           add_special_tokens=False)
            lens = [len(self.tok.encode(t, add_special_tokens=False))
                    for t in texts]
            too_long = [t for t, n in zip(texts, lens)
                        if n > fixed_length]
            assert not too_long, \
                f"truncation forbidden; over-length: {too_long[:2]}"
        ids, mask = out["input_ids"], out["attention_mask"]
        if device is not None:
            ids, mask = ids.to(device), mask.to(device)
        return ids, mask
