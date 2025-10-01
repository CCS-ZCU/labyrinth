# wsd_io.py  — lean utilities for Latin/Greek WSD/WSI experiments
# v0.2 (cleaned)
from __future__ import annotations

import math, os, json, inspect, re, random
from collections import defaultdict, Counter
from typing import List, Tuple, Optional, Iterable, Set

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler, DataLoader
from torch import amp as torch_amp
from itertools import zip_longest
import warnings
import math, os, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp as torch_amp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp as torch_amp
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Iterable
import os, json, time, math
import torch, torch.nn as nn, torch.nn.functional as F
from torch import amp as torch_amp
import math, torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp as torch_amp
from typing import Iterable, Optional, Sequence
import math, os, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp as torch_amp


# ------------------------- public API -------------------------
__all__ = [
    # constants
    "MAX_LEN",
    # spans/helpers
    "_ws", "_tokens_and_target_idx", "span_xlmr", "span_labert",
    "add_nav_columns_after_merge",
    # data / sampling
    "WSDDataset", "GroupedBatchSamplerWithReplacement",
    # batching & pooling
    "Collator", "pool_target_span",
    # loss & training
    "SupConLoss", "freeze_bottom_n", "train_encoder_only",
    # small utilities
    "estimate_batches_per_epoch",
]

MAX_LEN = 256

# =================== helpers (strings → tokens/spans) ===================

def _ws(s: str) -> List[str]:
    return s.strip().split()

def _tokens_and_target_idx(left: str, target: str, right: str) -> Tuple[List[str], int]:
    lt = _ws(left); rt = _ws(right)
    return lt + [target] + rt, len(lt)

def _pad_trunc_enc(enc, max_len: int, pad_id: int = 0):
    ids  = enc["input_ids"]
    attn = enc.get("attention_mask", None)

    if not torch.is_tensor(ids):
        ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    if attn is None:
        attn = torch.ones_like(ids, dtype=torch.long)
    elif not torch.is_tensor(attn):
        attn = torch.tensor(attn, dtype=torch.long).unsqueeze(0)

    L = ids.shape[1]
    if L < max_len:
        pad_len = max_len - L
        ids  = torch.cat([ids,  torch.full((1, pad_len), pad_id, dtype=ids.dtype)], dim=1)
        attn = torch.cat([attn, torch.zeros((1, pad_len), dtype=attn.dtype)], dim=1)
    elif L > max_len:
        ids  = ids[:, :max_len]
        attn = attn[:, :max_len]

    enc["input_ids"] = ids
    enc["attention_mask"] = attn
    return enc

# ---- XLM-R: rely on fast-tokenizer word_ids alignment ----
def span_xlmr(tokenizer, tokens: List[str], target_idx: int, max_length: int = MAX_LEN) -> List[int]:
    enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt",
                    padding="max_length", truncation=True, max_length=max_length)
    word_ids = enc.word_ids(0)
    return [] if word_ids is None else [i for i, wid in enumerate(word_ids) if wid == target_idx]

# ---- LaBERT: string path; robust special-token offset detection ----
def _first_occurrence(hay: List[int], needle: List[int]) -> Optional[int]:
    if not needle: return None
    n, m = len(hay), len(needle)
    for i in range(0, n - m + 1):
        if hay[i:i+m] == needle:
            return i
    return None

def span_labert(tokenizer, tokens: List[str], target_idx: int, max_length: int = MAX_LEN) -> List[int]:
    # subword length per whitespace token (no specials)
    wp_lens: List[int] = []
    for tok in tokens:
        try:
            ids = tokenizer.encode(tok, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(tok)
        wp_lens.append(len(ids))

    sent_str = " ".join(tokens)
    enc = tokenizer(sent_str, return_tensors="pt", truncation=True,
                    padding="max_length", max_length=max_length)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
    enc = _pad_trunc_enc(enc, max_length, pad_id=pad_id)
    seq_len = int(enc["attention_mask"][0].sum().item())

    try:
        core_ids = tokenizer.encode(sent_str, add_special_tokens=False)
    except TypeError:
        core_ids = tokenizer.encode(sent_str)

    left_specials = 1
    try:
        built = tokenizer.build_inputs_with_special_tokens(core_ids)
        start = _first_occurrence(built, core_ids)
        if start is not None:
            left_specials = start
    except Exception:
        try:
            with_spec = tokenizer.encode(sent_str, add_special_tokens=True)
            start = _first_occurrence(with_spec, core_ids)
            if start is not None:
                left_specials = start
        except Exception:
            pass

    start_in_core = sum(wp_lens[:target_idx])
    k = wp_lens[target_idx] if target_idx < len(wp_lens) else 0
    if k <= 0:
        return []
    start = left_specials + start_in_core
    end = start + k - 1
    return [i for i in range(start, end + 1) if 0 <= i < seq_len]

def add_nav_columns_after_merge(
    df: pd.DataFrame,
    tok_xlmr,
    tok_labert,
    max_length: int = MAX_LEN,
    lemma_col: str = "lemma",
) -> pd.DataFrame:
    rows = []
    for r in df.itertuples(index=False):
        # surface
        tokens_surf, tidx_surf = _tokens_and_target_idx(r.left_context, r.target_word, r.right_context)
        span_x_surf = span_xlmr(tok_xlmr, tokens_surf, tidx_surf, max_length)
        span_l_surf = span_labert(tok_labert, tokens_surf, tidx_surf, max_length)
        # lemma view
        base_lemma = re.sub(r"\d+$", "", getattr(r, lemma_col))
        tokens_lem, tidx_lem = _tokens_and_target_idx(r.left_context, base_lemma, r.right_context)
        span_x_lem = span_xlmr(tok_xlmr, tokens_lem, tidx_lem, max_length)
        span_l_lem = span_labert(tok_labert, tokens_lem, tidx_lem, max_length)

        row_out = {
            **r._asdict(),
            "tokens": tokens_surf, "target_idx": tidx_surf,
            "xmlr_wp_span": span_x_surf, "labert_wp_span": span_l_surf,
            "xmlr_wp_len": len(span_x_surf), "labert_wp_len": len(span_l_surf),
            "tokens_lemma": tokens_lem, "target_idx_lemma": tidx_lem,
            "xmlr_wp_span_lemma": span_x_lem, "labert_wp_span_lemma": span_l_lem,
            "xmlr_wp_len_lemma": len(span_x_lem), "labert_wp_len_lemma": len(span_l_lem),
        }
        rows.append(row_out)
    return pd.DataFrame(rows)

# =================== dataset & sampler ===================

class WSDDataset(Dataset):
    """
    Rows must have: lemma, sense_id, left_context, right_context, target_word,
    tokens, tokens_lemma (the latter two are only used for compatibility).
    The *contrastive* label is (lemma || sense_id) so positives never cross lemmas.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        keys = (df["lemma"].astype(str) + "||" + df["sense_id"].astype(str)).tolist()
        self.label2id: dict[str, int] = {}
        y = []
        for k in keys:
            if k not in self.label2id:
                self.label2id[k] = len(self.label2id)
            y.append(self.label2id[k])
        self.y = y
        counts = Counter(self.y)
        self.label_counts = counts
        self.singleton_label_ids: Set[int] = {lbl for lbl, c in counts.items() if c == 1}

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        return {
            "tokens": r["tokens"],
            "tokens_lemma": r["tokens_lemma"],
            "lemma": r["lemma"],
            "sense_id": r["sense_id"],
            "y": self.y[idx],
            "left_context": r["left_context"],
            "right_context": r["right_context"],
            "target_word": r["target_word"],
        }

class GroupedBatchSamplerWithReplacement(Sampler[List[int]]):
    """
    Each batch: pick P labels uniformly, then sample K indices *with replacement* per label.
    Length is controlled by batches_per_epoch (so epochs are predictable).
    """
    def __init__(self, labels: Iterable[int], P=8, K=6, seed=42, batches_per_epoch=6000):
        self.P, self.K = P, K
        self.rng = random.Random(seed)
        self.batches_per_epoch = batches_per_epoch
        self.by_lbl = defaultdict(list)
        for i, y in enumerate(labels):
            self.by_lbl[y].append(i)
        self.lbls = [y for y, idxs in self.by_lbl.items() if idxs]
        if len(self.lbls) < min(1, P):
            raise ValueError("Not enough labels with examples.")

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            chosen_lbls = self.rng.sample(self.lbls, min(self.P, len(self.lbls)))
            batch = []
            for y in chosen_lbls:
                pool = self.by_lbl[y]
                for _ in range(self.K):
                    batch.append(self.rng.choice(pool))
            yield batch

    def __len__(self):
        return self.batches_per_epoch


class BalancedPKReplacementSampler(Sampler):
    """
    Each batch: pick P labels, then sample K indices **with replacement** per label.
    Most batches: labels ~ uniform.
    Every `proportional_every`-th batch: labels ~ proportional to class size.
    """
    def __init__(self, labels, P=8, K=4, seed=42, proportional_every=5):
        self.P, self.K = P, K
        self.rng = random.Random(seed)
        self.by_lbl = defaultdict(list)
        for i, y in enumerate(labels):
            self.by_lbl[y].append(i)
        self.lbls = list(self.by_lbl.keys())
        self.sizes = {y: len(self.by_lbl[y]) for y in self.lbls}
        self.proportional_every = max(0, int(proportional_every))

    def __iter__(self):
        step = 0
        while True:
            step += 1
            # choose P labels (unique)
            if self.proportional_every and (step % self.proportional_every == 0):
                weights = [self.sizes[y] for y in self.lbls]
                chosen_lbls = _weighted_sample_without_replacement(
                    self.lbls, weights, self.P, self.rng
                )
                if len(chosen_lbls) < self.P:
                    chosen_lbls = self.rng.sample(self.lbls, min(self.P, len(self.lbls)))
            else:
                chosen_lbls = (
                    self.rng.sample(self.lbls, self.P)
                    if len(self.lbls) >= self.P else self.lbls[:]
                )

            batch = []
            for y in chosen_lbls:
                pool = self.by_lbl[y]
                if not pool:  # defensive
                    continue
                batch.extend(self.rng.choice(pool) for _ in range(self.K))

            if not batch:
                break
            yield batch

    def __len__(self):
        # arbitrary upper bound for DataLoader compatibility
        return 10_000

# =================== collator & pooling ===================

class Collator:
    """
    CPU-only collator. Rebuilds tokens from left/right + target, computes target span.
    Optional: jitter singletons (drop 1 token left & right) to avoid exact duplicates.
    """
    def __init__(
        self,
        tokenizer,
        model_kind: str,           # "xlmr" or "labert"
        use_lemma_view: bool,
        max_len: int = 256,
        *,
        singleton_label_ids: Set[int] | None = None,
        jitter_singletons: bool = True,
    ):
        self.tok = tokenizer
        self.kind = model_kind
        self.use_lemma = use_lemma_view
        self.max_len = max_len
        self.singleton_label_ids = set(singleton_label_ids or set())
        self.jitter_singletons = jitter_singletons

    def __call__(self, batch: List[dict]):
        input_ids, attention_mask, spans, labels = [], [], [], []

        for ex in batch:
            ylbl = ex["y"]
            labels.append(ylbl)

            lt = ex["left_context"].split()
            rt = ex["right_context"].split()
            if self.jitter_singletons and (ylbl in self.singleton_label_ids):
                if lt: lt = lt[:-1]
                if rt: rt = rt[1:]

            target_tok = ex["lemma"] if self.use_lemma else ex["target_word"]
            tokens_current = lt + [target_tok] + rt
            target_idx = len(lt)

            if self.kind == "xlmr":
                enc = self.tok(tokens_current, is_split_into_words=True, return_tensors="pt",
                               truncation=True, padding="max_length", max_length=self.max_len)
                word_ids = enc.word_ids(0)
                span = [] if word_ids is None else [i for i, w in enumerate(word_ids) if w == target_idx]
            else:
                sent_str = " ".join(tokens_current)
                enc = self.tok(sent_str, return_tensors="pt", truncation=True,
                               padding="max_length", max_length=self.max_len)
                pad_id = getattr(self.tok, "pad_token_id", 0) or 0
                enc = _pad_trunc_enc(enc, self.max_len, pad_id=pad_id)
                span = span_labert(self.tok, tokens_current, target_idx, max_length=self.max_len)

            input_ids.append(enc["input_ids"])
            attention_mask.append(enc["attention_mask"])
            spans.append(span)

        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "spans": spans, "labels": labels}

def pool_target_span(hidden_states: torch.Tensor, spans: List[List[int]]) -> torch.Tensor:
    """
    Mean over subword span; fallback to CLS if span is empty.
    hidden_states: [B, L, D] (last layer)
    returns: [B, D]
    """
    cls = hidden_states[:, 0, :]
    out = []
    for b, idxs in enumerate(spans):
        if idxs:
            out.append(hidden_states[b, idxs, :].mean(dim=0))
        else:
            out.append(cls[b])
    return torch.stack(out, dim=0)

# =================== loss & training ===================

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (stable fp32 math under AMP)."""
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.t = temperature

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        if B <= 1:
            return z.sum() * 0.0
        with torch_amp.autocast(device_type="cuda", enabled=False):
            z32 = F.normalize(z.to(torch.float32), dim=1)
            sim = (z32 @ z32.t()) / self.t
            device = z.device
            logits_mask = ~torch.eye(B, dtype=torch.bool, device=device)
            y_i = y.view(-1, 1)
            pos_mask = (y_i == y_i.t()) & logits_mask
            sim_masked = sim.masked_fill(~logits_mask, torch.tensor(-1e9, dtype=sim.dtype, device=device))
            log_prob = sim_masked - torch.logsumexp(sim_masked, dim=1, keepdim=True)
            pos_counts = pos_mask.sum(dim=1)
            valid = pos_counts > 0
            if not valid.any():
                return z.sum() * 0.0
            loss_i = torch.zeros(B, dtype=sim.dtype, device=device)
            loss_i[valid] = -(log_prob[valid] * pos_mask[valid]).sum(dim=1) / pos_counts[valid]
            loss = loss_i.mean()
        return loss.to(z.dtype)

def freeze_bottom_n(model, n=2):
    if hasattr(model, "embeddings"):
        for p in model.embeddings.parameters(): p.requires_grad = False
    enc = getattr(model, "encoder", None)
    if enc is not None and hasattr(enc, "layer"):
        for i, layer in enumerate(enc.layer):
            if i < n:
                for p in layer.parameters(): p.requires_grad = False

def estimate_batches_per_epoch(num_labels: int, P: int, K: int, total_examples: int, passes_per_epoch: float = 1.0) -> int:
    """
    Roughly size an epoch in terms of how many unique examples you want to see.
    """
    batch_size = P * K
    target_examples = int(total_examples * passes_per_epoch)
    return max(1, math.ceil(target_examples / batch_size))

def train_encoder_only(
    model, tokenizer, dataset: WSDDataset,
    *,
    out_dir: str,
    model_kind: str,                 # "xlmr" or "labert"
    use_lemma_view: bool = True,
    P: int = 8, K: int = 4,
    epochs: int = 2,
    lr: float = 2e-5,
    max_len: int = 256,
    grad_accum: int = 1,
    amp: bool = True,
    device: str = "cuda",
    freeze_n: int = 4,
    temperature: float = 0.05,
    num_workers: int = 0,
    log_every: int = 50,
    autocast_dtype: torch.dtype = torch.float16,
    warmup_frac: float = 0.0,
    max_grad_norm: float | None = None,
    drop_empty_in_batch: bool = True,
    batches_per_epoch: Optional[int] = None,
):
    """
    Encoder-only fine-tuning with Supervised Contrastive loss (P×K batches, replacement).
    """
    os.makedirs(out_dir, exist_ok=True)

    # model & freezing
    model.to(device)
    freeze_bottom_n(model, n=freeze_n)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters found (did you freeze too much?).")

    # sampler
    if batches_per_epoch is None:
        batches_per_epoch = estimate_batches_per_epoch(
            num_labels=len(dataset.label2id), P=P, K=K,
            total_examples=len(dataset.df), passes_per_epoch=1.0
        )
    sampler = GroupedBatchSamplerWithReplacement(dataset.y, P=P, K=K, batches_per_epoch=batches_per_epoch)

    # collator
    collate = Collator(
        tokenizer, model_kind=model_kind, use_lemma_view=use_lemma_view, max_len=max_len,
        singleton_label_ids=getattr(dataset, "singleton_label_ids", set()), jitter_singletons=True
    )

    dl = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=(device == "cuda" and num_workers > 0),
        persistent_workers=(num_workers > 0),
    )

    # optim/sched/amp/loss
    optim = torch.optim.AdamW(trainable, lr=lr)
    total_steps = epochs * len(dl)
    if warmup_frac and total_steps > 0:
        from torch.optim.lr_scheduler import LambdaLR
        warmup_steps = max(1, int(warmup_frac * total_steps))
        def lr_lambda(step):
            return step / warmup_steps if step < warmup_steps else 1.0
        sched = LambdaLR(optim, lr_lambda)
    else:
        sched = None

    scaler = torch_amp.GradScaler("cuda", enabled=(amp and device == "cuda"))
    crit = SupConLoss(temperature=temperature)

    model.train()
    step = 0
    loss_ema = None

    for ep in range(1, epochs + 1):
        for i, batch in enumerate(dl, 1):
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels         = batch["labels"].to(device, non_blocking=True)
            spans          = batch["spans"]

            if drop_empty_in_batch:
                keep = [j for j, s in enumerate(spans) if s]
                if len(keep) != len(spans):
                    if not keep: continue
                    input_ids = input_ids[keep]; attention_mask = attention_mask[keep]; labels = labels[keep]
                    spans = [spans[j] for j in keep]

            with torch_amp.autocast(device_type=("cuda" if device == "cuda" else "cpu"),
                                    enabled=(amp and device == "cuda"),
                                    dtype=autocast_dtype):
                outs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
                last = outs.last_hidden_state
                z    = pool_target_span(last, spans)
                loss = crit(z, labels) / grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if i % grad_accum == 0:
                if scaler.is_enabled():
                    if max_grad_norm is not None:
                        scaler.unscale_(optim); torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                    scaler.step(optim); scaler.update()
                else:
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                    optim.step()
                optim.zero_grad(set_to_none=True)
                if sched is not None: sched.step()

            step += 1
            loss_val = float(loss.item() * grad_accum)
            loss_ema = loss_val if loss_ema is None else (0.95 * loss_ema + 0.05 * loss_val)
            if log_every and (step % log_every == 0):
                lr_now = optim.param_groups[0]["lr"]
                print(f"[ep {ep}] step {step}/{total_steps}  loss={loss_val:.4f}  ema={loss_ema:.4f}  lr={lr_now:.2e}")

        # checkpoint per epoch
        ckpt_dir = os.path.join(out_dir, f"epoch_{ep}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        try: tokenizer.save_pretrained(ckpt_dir)
        except Exception: pass

    # final save
    model.save_pretrained(out_dir)
    try: tokenizer.save_pretrained(out_dir)
    except Exception: pass
    with open(os.path.join(out_dir, "train_args.json"), "w") as f:
        json.dump({
            "model_kind": model_kind, "use_lemma_view": use_lemma_view,
            "P": P, "K": K, "epochs": epochs, "lr": lr,
            "max_len": max_len, "grad_accum": grad_accum,
            "freeze_n": freeze_n, "temperature": temperature,
            "amp": amp, "device": device, "warmup_frac": warmup_frac,
            "max_grad_norm": max_grad_norm, "drop_empty_in_batch": drop_empty_in_batch,
            "batches_per_epoch": batches_per_epoch,
        }, f, indent=2)
    print(f"Saved to: {out_dir}")


# =============================================================================
# WiC-style builders and collator (pairs-only, same-source constraint)
# Paste this block AFTER: span_xlmr, span_labert, _pad_trunc_enc
# =============================================================================
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Tuple, Dict, Iterable, Optional
import random

# ---- indexing helpers -------------------------------------------------------
def build_wic_index(
    df: pd.DataFrame,
    source_col: str = "wsd_source",
    lemma_col: str = "lemma",
    sense_col: str = "sense_id",
):
    """
    Returns:
      rows               : df.reset_index(drop=True) to align integer indices
      by_src_lem_sense   : dict[src][lemma][sense] -> list[int row_idx]
      by_src_lem         : dict[src][lemma] -> set(sense)
    """
    rows = df.reset_index(drop=True)
    by_src_lem_sense = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    by_src_lem = defaultdict(lambda: defaultdict(set))
    for i, r in rows.iterrows():
        s = r[source_col]; l = r[lemma_col]; k = r[sense_col]
        by_src_lem_sense[s][l][k].append(i)
        by_src_lem[s][l].add(k)
    return rows, by_src_lem_sense, by_src_lem


def _is_10_1_10(row: pd.Series) -> bool:
    """Return True iff both sides are exactly 10 tokens (so 10–target–10)."""
    return len(str(row["left_context"]).split()) == 10 and len(str(row["right_context"]).split()) == 10


def _aug_views_for_singleton(row: pd.Series) -> Tuple[str, str]:
    """
    Produce two view-specs for a singleton positive:
      viewA = keep LEFT + target (drop RIGHT entirely)
      viewB = keep RIGHT + target (drop LEFT entirely)
    """
    return ("left_only", "right_only")



# ---- dataset: makes pairs with replacement, same-source only ---------------
from itertools import zip_longest
from typing import List, Optional

class WiCPairDataset(torch.utils.data.Dataset):
    """
    Produces WiC-style pairs (i1, i2, y in {0,1}) restricted to the SAME SOURCE.
    - Positives: same (source, lemma, sense). If only 1 item exists and context is 10–1–10,
                 uses ("left_only","right_only") views on the SAME row.
    - Negatives: same (source, lemma), different senses.
    - Sampling is with replacement; `pairs_per_epoch` controls epoch length.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        pos_frac: float = 0.5,
        pairs_per_epoch: int = 20000,
        seed: int = 42,
    ):
        self.rows, self.by_src_lem_sense, self.by_src_lem = build_wic_index(df)
        self.pos_frac = float(pos_frac)
        self.pairs_per_epoch = int(pairs_per_epoch)
        self.rng = random.Random(seed)

        # Buckets
        self._pos_keys = []  # (src, lemma, sense) with >=1 example
        for src, dlem in self.by_src_lem_sense.items():
            for lem, ds in dlem.items():
                for sense, idxs in ds.items():
                    if len(idxs) >= 1:
                        self._pos_keys.append((src, lem, sense))

        self._neg_keys = []  # (src, lemma) with >=2 senses
        for src, dlem in self.by_src_lem.items():
            for lem, senses in dlem.items():
                if len(senses) >= 2:
                    self._neg_keys.append((src, lem))

        if not self._pos_keys:
            raise ValueError("WiCPairDataset: no positives (no sense bucket has examples).")
        if not self._neg_keys:
            print("⚠️ WiCPairDataset: no negatives (no lemma with ≥2 senses within a source).")

        self.pairs: List[PairSpec] = self._make_epoch_pairs()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> PairSpec:
        return self.pairs[idx]

    # ---- public: rebuild pairs for a new epoch ----
    def refresh(self):
        self.pairs = self._make_epoch_pairs()

    # -- sampling helpers --
    def _sample_positive(self) -> Optional[PairSpec]:
        src, lem, sense = self.rng.choice(self._pos_keys)
        idxs = self.by_src_lem_sense[src][lem][sense]
        if len(idxs) >= 2:
            i1, i2 = self.rng.sample(idxs, 2)
            return PairSpec(i1=i1, i2=i2, y=1)
        else:
            # singleton positive: make two complementary views if 10–1–10
            i = idxs[0]
            row = self.rows.iloc[i]
            if _is_10_1_10(row):
                v1, v2 = _aug_views_for_singleton(row)
                return PairSpec(i1=i, i2=i, y=1, view1=v1, view2=v2)
            return None  # resample

    def _sample_negative(self) -> Optional[PairSpec]:
        if not self._neg_keys:
            return None
        src, lem = self.rng.choice(self._neg_keys)
        senses = list(self.by_src_lem[src][lem])
        if len(senses) < 2:
            return None
        s1, s2 = self.rng.sample(senses, 2)
        i1 = self.rng.choice(self.by_src_lem_sense[src][lem][s1])
        i2 = self.rng.choice(self.by_src_lem_sense[src][lem][s2])
        return PairSpec(i1=i1, i2=i2, y=0)

    def _make_epoch_pairs(self) -> List[PairSpec]:
        need = int(self.pairs_per_epoch)
        n_pos_target = int(round(self.pos_frac * need))
        n_neg_target = need - n_pos_target

        pos_list, neg_list = [], []

        # Fill positives
        guard = 0
        while len(pos_list) < n_pos_target and guard < 10 * n_pos_target:
            p = self._sample_positive()
            if p is not None:
                pos_list.append(p)
            guard += 1

        # Fill negatives; if impossible, just take what we can
        guard = 0
        while len(neg_list) < n_neg_target and guard < 10 * n_neg_target:
            n = self._sample_negative()
            if n is not None:
                neg_list.append(n)
            guard += 1

        # Shuffle within pools
        self.rng.shuffle(pos_list)
        self.rng.shuffle(neg_list)

        # Interleave starting with a random pool (pos or neg)
        start_pos = bool(self.rng.getrandbits(1))
        a, b = (pos_list, neg_list) if start_pos else (neg_list, pos_list)

        mixed = []
        for x, y in zip_longest(a, b):
            if x is not None: mixed.append(x)
            if y is not None: mixed.append(y)

        # If still short, top up from the longer pool
        while len(mixed) < need and (pos_list or neg_list):
            if len(pos_list) > len(neg_list) and pos_list:
                mixed.append(pos_list[self.rng.randrange(len(pos_list))])
            elif neg_list:
                mixed.append(neg_list[self.rng.randrange(len(neg_list))])
            else:
                break

        # Trim to exactly need
        if len(mixed) > need:
            mixed = mixed[:need]
        return mixed

# ---- collator: encodes both sides, recomputes spans for each view -----------
class WiCPairCollator:
    """
    Builds encodings for WiC pairs respecting model_kind and lemma/surface view.
    Returns dict:
      - input_ids_1, attention_mask_1, spans_1
      - input_ids_2, attention_mask_2, spans_2
      - labels (0/1)

    Relies on: span_xlmr, span_labert, _pad_trunc_enc (defined in this module).
    """
    def __init__(self, tokenizer, model_kind: str, use_lemma_view: bool, max_len: int = 160):
        self.tok = tokenizer
        self.kind = model_kind            # "xlmr" or "labert"
        self.use_lemma = use_lemma_view
        self.max_len = max_len

    def _encode_one(self, row: pd.Series, view: str):
        # Build context according to the view
        lt = str(row["left_context"]).split()
        rt = str(row["right_context"]).split()
        if view == "left_only":
            rt = []           # keep only left + target
        elif view == "right_only":
            lt = []           # keep only right + target

        # target token form (strip trailing digits if using lemma view)
        if self.use_lemma:
            tgt = re.sub(r"\d+$", "", str(row["lemma"]))
        else:
            tgt = str(row["target_word"])

        # Rebuild the token list to encode
        tokens = lt + [tgt] + rt
        target_idx = len(lt)

        if self.kind == "xlmr":
            # pretokenized → enables .word_ids()
            enc = self.tok(
                tokens,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
            )
            word_ids = enc.word_ids(0)
            span = [] if word_ids is None else [i for i, w in enumerate(word_ids) if w == target_idx]
        else:
            # LaBERT path (string encoder)
            sent = " ".join(tokens)
            try:
                enc = self.tok(
                    sent,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_len,
                )
            except TypeError:
                enc = self.tok(sent, return_tensors="pt", truncation=True, padding=True, max_length=self.max_len)

            pad_id = getattr(self.tok, "pad_token_id", 0) or 0
            enc = _pad_trunc_enc(enc, self.max_len, pad_id=pad_id)
            # span from SAME tokens used for encoding → guarantees alignment
            span = span_labert(self.tok, tokens, target_idx, max_length=self.max_len)

        # Some custom tokenizers omit attention_mask; ensure it exists
        if "attention_mask" not in enc:
            enc["attention_mask"] = torch.ones_like(enc["input_ids"])

        return enc["input_ids"], enc["attention_mask"], span

    def __call__(self, pair_specs: list, rows: Optional[pd.DataFrame] = None):
        if rows is None:
            raise ValueError("WiCPairCollator: `rows` must be the DataFrame used by the dataset.")

        input_ids_1, attn_1, spans_1 = [], [], []
        input_ids_2, attn_2, spans_2 = [], [], []
        labels = []

        for p in pair_specs:
            r1 = rows.iloc[p.i1]; r2 = rows.iloc[p.i2]
            ids1, am1, sp1 = self._encode_one(r1, getattr(p, "view1", "both"))
            ids2, am2, sp2 = self._encode_one(r2, getattr(p, "view2", "both"))
            input_ids_1.append(ids1); attn_1.append(am1); spans_1.append(sp1)
            input_ids_2.append(ids2); attn_2.append(am2); spans_2.append(sp2)
            labels.append(int(p.y))

        def _cat(xs): return torch.cat(xs, dim=0) if xs and torch.is_tensor(xs[0]) else None

        return {
            "input_ids_1": _cat(input_ids_1),
            "attention_mask_1": _cat(attn_1),
            "spans_1": spans_1,
            "input_ids_2": _cat(input_ids_2),
            "attention_mask_2": _cat(attn_2),
            "spans_2": spans_2,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ---- optional: export symbols for clean imports -----------------------------
try:
    __all__  # keep existing if present
except NameError:
    __all__ = []
for _name in [
    "WiCPairDataset", "WiCPairCollator", "PairSpec",
    "build_wic_index", "_is_10_1_10", "_aug_views_for_singleton",
]:
    if _name not in __all__:
        __all__.append(_name)

# =============================================================================
# WiC projection + cosine head (calibrated) + trainer
# =============================================================================


class TargetSpanPooler(nn.Module):
    """Mean over target subtoken span, fallback to CLS if empty."""
    def forward(self, hidden, spans):
        B, L, D = hidden.shape
        cls = hidden[:, 0, :]
        out = []
        for b, idxs in enumerate(spans):
            if isinstance(idxs, (list, tuple)) and len(idxs) > 0:
                out.append(hidden[b, idxs, :].mean(dim=0))
            else:
                out.append(cls[b])
        return torch.stack(out, dim=0)  # [B,D]


def train_wic_head(
    model: WiCPairModel,
    dataloader,
    *,
    device="cuda",
    epochs=2,
    lr=1e-3,
    amp=True,
    log_every=50,
    warmup_frac: float = 0.0,
    pos_weight: float | None = None,  # e.g., if positives are rarer
    max_grad_norm: float | None = 1.0,
    entropy_reg: float = 0.0,         # encourage probability spread if needed
):
    model.to(device)
    model.train()

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    scaler = torch_amp.GradScaler("cuda", enabled=(amp and device == "cuda"))

    if pos_weight is None:
        bce = nn.BCEWithLogitsLoss()
    else:
        # pos_weight > 1 ups weight of positive class
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    # (optional) warmup scheduler
    total_steps = epochs * len(dataloader)
    if warmup_frac and total_steps > 0:
        from torch.optim.lr_scheduler import LambdaLR
        warmup_steps = max(1, int(warmup_frac * total_steps))
        def lr_lambda(step):
            return step / warmup_steps if step < warmup_steps else 1.0
        sched = LambdaLR(opt, lr_lambda)
    else:
        sched = None

    step = 0
    ema = None
    for ep in range(1, epochs + 1):
        for batch in dataloader:
            step += 1
            labels = batch["labels"].float().to(device)
            for k in ("input_ids_1","attention_mask_1","input_ids_2","attention_mask_2"):
                batch[k] = batch[k].to(device, non_blocking=True)

            with torch_amp.autocast(device_type=("cuda" if device == "cuda" else "cpu"),
                                    enabled=(amp and device == "cuda")):
                logits, e1, e2 = model(batch)
                loss = bce(logits, labels)
                if entropy_reg > 0:
                    p = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
                    ent = -(p*torch.log(p) + (1-p)*torch.log(1-p)).mean()
                    loss = loss - entropy_reg * ent

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if max_grad_norm is not None:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                if max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()

            opt.zero_grad(set_to_none=True)
            if sched is not None:
                sched.step()

            val = float(loss.item())
            ema = val if ema is None else 0.95 * ema + 0.05 * val
            if log_every and (step % log_every == 0):
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    print(f"[ep {ep}] step {step}/{total_steps}  "
                          f"loss={val:.4f}  ema={ema:.4f}  mean(p)={probs.mean().item():.3f}")

    return model

# =============================================================================
# Shared projection
# =============================================================================
import torch, torch.nn as nn, torch.nn.functional as F
from torch import amp as torch_amp

class WiCProjection(nn.Module):
    """
    Small projection on top of pooled target embeddings.
    Set proj_dim=None to use identity (no projection).
    Returns L2-normalized vectors (cosine-friendly).
    """
    def __init__(self, in_dim: int, proj_dim: int | None = 256, p_drop: float = 0.1):
        super().__init__()
        if proj_dim is None:
            self.proj = nn.Identity()
            self.out_dim = in_dim
        else:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, proj_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p_drop),
                nn.LayerNorm(proj_dim),
            )
            self.out_dim = proj_dim

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)

# =============================================================================
# WiC pair generator (IterableDataset): fresh, large, and default-full potential
# =============================================================================
from torch.utils.data import IterableDataset
from collections import defaultdict, Counter
from dataclasses import dataclass
import random
import math
import pandas as pd

@dataclass
class PairSpec:
    i1: int
    i2: int
    y: int               # 1 = same sense, 0 = different
    view1: str = "both"  # "both" | "left_only" | "right_only"
    view2: str = "both"



class WiCPairIterableDataset(IterableDataset):
    """
    On-the-fly WiC pair sampler:
      * Always samples within the same source (wsd_source).
      * Positives: same (lemma, sense) within a source. Singletons use 10–1–10 split if available.
      * Negatives: default within-lemma different sense; optionally mix in cross-lemma negatives.

    Default pairs_per_epoch is computed from the dataset size so you use the data's potential
    without hand-tuning.

    Yields: PairSpec objects that your existing WiCPairCollator can encode.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        pos_frac: float = 0.5,
        pairs_per_epoch: int | None = None,
        cross_lemma_neg_frac: float = 0.3,  # 0.0 to disable; 0.3 is a sane default
        seed: int = 42,
        source_col: str = "wsd_source",
        lemma_col: str = "lemma",
        sense_col: str = "sense_id",
    ):
        super().__init__()
        self.rows = df.reset_index(drop=True)
        self.source_col = source_col
        self.lemma_col = lemma_col
        self.sense_col = sense_col

        self.pos_frac = float(pos_frac)
        self.cross_lemma_neg_frac = float(max(0.0, min(1.0, cross_lemma_neg_frac)))
        self.rng = random.Random(seed)

        # ---- index structures ----
        # by source → lemma → sense → [row idx]
        by_src_lem_sense = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        # by source → lemma → set(sense)
        by_src_lem = defaultdict(lambda: defaultdict(set))
        # by source → lemma → all row idx
        by_src_lem_all = defaultdict(lambda: defaultdict(list))

        for i, r in self.rows.iterrows():
            src = r[source_col]; lem = r[lemma_col]; sen = r[sense_col]
            by_src_lem_sense[src][lem][sen].append(i)
            by_src_lem[src][lem].add(sen)
            by_src_lem_all[src][lem].append(i)

        self.by_src_lem_sense = by_src_lem_sense
        self.by_src_lem = by_src_lem
        self.by_src_lem_all = by_src_lem_all

        # buckets
        self.pos_keys = []  # triples (src, lemma, sense) with ≥1
        for src, dlem in self.by_src_lem_sense.items():
            for lem, ds in dlem.items():
                for sen, idxs in ds.items():
                    if len(idxs) >= 1:
                        self.pos_keys.append((src, lem, sen))

        self.neg_within_keys = []  # (src, lemma) with ≥2 senses
        for src, dlem in self.by_src_lem.items():
            for lem, senses in dlem.items():
                if len(senses) >= 2:
                    self.neg_within_keys.append((src, lem))

        self.neg_cross_src_keys = []  # (src, list(lemmas))
        for src, dlem in self.by_src_lem_all.items():
            lems = [lem for lem, idxs in dlem.items() if len(idxs) > 0]
            if len(lems) >= 2:
                self.neg_cross_src_keys.append((src, lems))

        if not self.pos_keys:
            raise ValueError("WiCPairIterableDataset: no positive buckets found.")
        if not (self.neg_within_keys or self.neg_cross_src_keys):
            print("⚠️ WiCPairIterableDataset: no negatives available (need ≥2 senses or ≥2 lemmas per source).")

        # ---- default pairs_per_epoch (use the dataset!) ----
        if pairs_per_epoch is None:
            n = len(self.rows)
            # generous default: 8× examples, but at least 32k and at most 200k
            pairs_per_epoch = min(200_000, max(32_768, 8 * n))
        self.pairs_per_epoch = int(pairs_per_epoch)

    def __len__(self):
        # virtual length for the epoch (DataLoader won't use it strictly for IterableDataset)
        return self.pairs_per_epoch

    # ---- samplers ----
    def _sample_positive(self) -> PairSpec | None:
        src, lem, sen = self.rng.choice(self.pos_keys)
        idxs = self.by_src_lem_sense[src][lem][sen]
        if len(idxs) >= 2:
            i1, i2 = self.rng.sample(idxs, 2)
            return PairSpec(i1=i1, i2=i2, y=1)
        # singleton
        i = idxs[0]
        row = self.rows.iloc[i]
        if _is_10_1_10(row):
            v1, v2 = _aug_views_for_singleton(row)
            return PairSpec(i1=i, i2=i, y=1, view1=v1, view2=v2)
        return None

    def _sample_negative_within(self) -> PairSpec | None:
        if not self.neg_within_keys:
            return None
        src, lem = self.rng.choice(self.neg_within_keys)
        senses = list(self.by_src_lem[src][lem])
        s1, s2 = self.rng.sample(senses, 2)
        i1 = self.rng.choice(self.by_src_lem_sense[src][lem][s1])
        i2 = self.rng.choice(self.by_src_lem_sense[src][lem][s2])
        return PairSpec(i1=i1, i2=i2, y=0)

    def _sample_negative_cross(self) -> PairSpec | None:
        if not self.neg_cross_src_keys or self.cross_lemma_neg_frac <= 0.0:
            return None
        src, lemmas = self.rng.choice(self.neg_cross_src_keys)
        l1, l2 = self.rng.sample(lemmas, 2)
        i1 = self.rng.choice(self.by_src_lem_all[src][l1])
        i2 = self.rng.choice(self.by_src_lem_all[src][l2])
        return PairSpec(i1=i1, i2=i2, y=0)

    def __iter__(self):
        # fresh RNG per worker
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # shard the seed so workers don't mirror each other
            base = self.rng.random()
            self.rng = random.Random(int(base * (worker_info.id + 1) * 1000003))

        n_pairs = self.pairs_per_epoch
        n_pos_target = int(round(self.pos_frac * n_pairs))
        n_neg_target = n_pairs - n_pos_target

        # Positive stream
        pos_emitted = 0
        while pos_emitted < n_pos_target:
            p = self._sample_positive()
            if p is not None:
                yield p
                pos_emitted += 1

        # Negative stream (mix within-lemma and cross-lemma)
        neg_emitted = 0
        while neg_emitted < n_neg_target:
            use_cross = (self.cross_lemma_neg_frac > 0.0) and (self.rng.random() < self.cross_lemma_neg_frac)
            p = self._sample_negative_cross() if use_cross else self._sample_negative_within()
            if p is None:
                # fall back if that bucket is empty
                p = self._sample_negative_within() or self._sample_negative_cross()
            if p is not None:
                yield p
                neg_emitted += 1


def freeze_all(params: Iterable[torch.nn.Parameter], requires_grad: bool):
    for p in params:
        p.requires_grad = requires_grad

# ====== Layer targeting helpers (paste into your module or your notebook) ======
from typing import Iterable

def _matchable_layer_prefixes():
    # Covers common HF stacks: RoBERTa/XLM-R/BERT/DeBERTa-ish names
    return [
        "encoder.layer.",            # e.g., XLM-R, many BERT clones
        "roberta.encoder.layer.",    # RoBERTa-style wrappers
        "bert.encoder.layer.",       # BERT-style wrappers
        "model.encoder.layer.",      # some checkpoints wrap one more level
    ]

def freeze_all_encoder_params(encoder):
    for p in encoder.parameters():
        p.requires_grad = False

def _set_block_trainable(block, train=True):
    for p in block.parameters():
        p.requires_grad = train
    # also LayerNorm submodules
    for m in block.modules():
        if m.__class__.__name__.lower().startswith('layernorm'):
            for p in m.parameters():
                p.requires_grad = train

def unfreeze_encoder_layers(encoder, layers, also_unfreeze_layernorm=True, also_unfreeze_pooler=False):
    # works for HF BERT/RoBERTa style encoders
    enc = getattr(encoder, "encoder", None)
    if enc is None or not hasattr(enc, "layer"):
        return
    for li in layers:
        if li < len(enc.layer):
            _set_block_trainable(enc.layer[li], train=True)
    if also_unfreeze_pooler and hasattr(encoder, "pooler"):
        for p in encoder.pooler.parameters():
            p.requires_grad = True

def _find_transformer_blocks(encoder):
    """
    Return the list-like container of Transformer blocks for common BERT/RoBERTa layouts.
    Supports LaBERT (bert-style) and XLM-R (roberta-style).
    """
    # roberta
    obj = getattr(encoder, "roberta", None)
    if obj is not None and hasattr(obj, "encoder") and hasattr(obj.encoder, "layer"):
        return obj.encoder.layer
    # bert
    obj = getattr(encoder, "bert", None)
    if obj is not None and hasattr(obj, "encoder") and hasattr(obj.encoder, "layer"):
        return obj.encoder.layer
    # fallback: some AutoModels expose .encoder.layer directly
    enc = getattr(encoder, "encoder", None)
    if enc is not None and hasattr(enc, "layer"):
        return enc.layer
    raise AttributeError("Could not find encoder blocks (expected *.encoder.layer)")

def unfreeze_last_n_layers(encoder, n_last: int, train_layernorm: bool = True, train_pooler: bool = False):
    """
    Freeze everything, then unfreeze the last n transformer blocks.
    Optionally keep LayerNorm and pooler trainable (useful for stability).
    """
    # 1) freeze everything
    freeze_all(encoder.parameters(), False)
    freeze_all(encoder.parameters(), True)  # set to True (frozen)
    # 2) unfreeze last n blocks
    blocks = _find_transformer_blocks(encoder)
    if n_last > 0:
        for blk in list(blocks)[-n_last:]:
            freeze_all(blk.parameters(), False)
    # 3) optionally unfreeze LayerNorms globally
    if train_layernorm:
        for m in encoder.modules():
            if isinstance(m, nn.LayerNorm):
                freeze_all(m.parameters(), False)
    # 4) optionally unfreeze pooler if present
    if train_pooler:
        pooler = getattr(getattr(encoder, "roberta", getattr(encoder, "bert", encoder)), "pooler", None)
        if pooler is not None:
            freeze_all(pooler.parameters(), False)

class TargetSpanPooler(nn.Module):
    """Mean over target subtoken span, fallback to CLS if empty."""
    def forward(self, hidden, spans):
        B, L, D = hidden.shape
        cls = hidden[:, 0, :]
        out = []
        for b, idxs in enumerate(spans):
            if isinstance(idxs, (list, tuple)) and len(idxs) > 0:
                out.append(hidden[b, idxs, :].mean(dim=0))
            else:
                out.append(cls[b])
        return torch.stack(out, dim=0)

def _assert_finite(name, t):
    if not torch.is_finite(t).all():
        n_nan = torch.isnan(t).sum().item()
        n_inf = torch.isinf(t).sum().item()
        raise RuntimeError(f"[NaNGuard] {name} has non-finite values: nan={n_nan} inf={n_inf}")

class WiCPairModel(nn.Module):
    """
    Cosine-similarity WiC head with a small projection and *bounded* temperature.
    Fine-tune ready:
      - set freeze_encoder=False
      - optionally call unfreeze_last_n_layers(encoder, n_last)
    """
    def __init__(self, encoder, layer_idx: Optional[int] = -3, proj_dim: int = 256,
                 freeze_encoder: bool = True, init_logit_scale: float = 2.5, scale_max: float = 5.0):
        super().__init__()
        self.encoder = encoder
        self.layer_idx = layer_idx
        self.pool = TargetSpanPooler()
        self.bias = nn.Parameter(torch.zeros(()))  # scalar calibration
        D = encoder.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(D, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.LayerNorm(proj_dim),
        )

        # bounded temperature: scale = scale_max * sigmoid(raw)
        self.logit_scale_raw = nn.Parameter(torch.tensor(float(init_logit_scale)))
        self.scale_max = float(scale_max)

        # remember whether encoder is frozen
        self.encoder_frozen = bool(freeze_encoder)
        if self.encoder_frozen:
            freeze_all(self.encoder.parameters(), True)

    def _scale(self):
        return self.scale_max * torch.sigmoid(self.logit_scale_raw)

    @property
    def logit_scale(self):
        # backward-compat read-only view
        return self._scale().detach()

    def _pick_layer(self, outs):
        if self.layer_idx is None:
            return outs.last_hidden_state
        idx = self.layer_idx
        hs = outs.hidden_states
        if idx < 0:
            idx = len(hs) + idx
        return hs[idx]

    def encode(self, input_ids, attention_mask):
        # grads ON if not frozen, OFF if frozen
        with torch.set_grad_enabled(not self.encoder_frozen):
            outs = self.encoder(input_ids=input_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=True)
            H = self._pick_layer(outs)
        return H

    def forward(self, batch):
        H1 = self.encode(batch["input_ids_1"], batch["attention_mask_1"])
        H2 = self.encode(batch["input_ids_2"], batch["attention_mask_2"])
        _assert_finite("H1", H1);
        _assert_finite("H2", H2)

        z1 = self.pool(H1, batch["spans_1"])
        z2 = self.pool(H2, batch["spans_2"])
        _assert_finite("z1", z1);
        _assert_finite("z2", z2)

        p1 = self.proj(z1)
        p2 = self.proj(z2)
        # sanitize only here; keeps training robust even if 1 sample is odd
        p1 = torch.nan_to_num(p1, nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-1e4, 1e4)
        p2 = torch.nan_to_num(p2, nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-1e4, 1e4)
        _assert_finite("proj1", p1);
        _assert_finite("proj2", p2)

        e1 = p1 / p1.norm(dim=1, keepdim=True).clamp_min(1e-6)
        e2 = p2 / p2.norm(dim=1, keepdim=True).clamp_min(1e-6)
        _assert_finite("e1", e1);
        _assert_finite("e2", e2)

        logits = (e1 * e2).sum(dim=1) * self._scale() + self.bias
        _assert_finite("logits", logits)
        return logits, e1, e2


# =============================================================================
# WiC fine-tuning: compact layer-freeze helpers + model + trainer
# =============================================================================


# ----------------------------- Freeze/Unfreeze -------------------------------

def set_requires_grad(params: Iterable[nn.Parameter], flag: bool) -> None:
    for p in params:
        p.requires_grad = flag

def _find_transformer_blocks(encoder) -> Sequence[nn.Module]:
    """
    Return the list-like container of Transformer blocks for common HF layouts.
    Supports BERT/RoBERTa/XLM-R and similar.
    """
    for root in ("roberta", "bert", None):
        obj = getattr(encoder, root, encoder) if root is not None else encoder
        enc = getattr(obj, "encoder", None)
        if enc is not None and hasattr(enc, "layer"):
            return enc.layer
    raise AttributeError("Could not find encoder blocks (expected *.encoder.layer)")

def freeze_all_encoder_params(encoder) -> None:
    set_requires_grad(encoder.parameters(), False)

def unfreeze_encoder_layers(
    encoder,
    layers: Sequence[int],
    *,
    also_unfreeze_layernorm: bool = True,
    also_unfreeze_pooler: bool = False,
) -> None:
    """Unfreeze specific Transformer blocks (0-based indices)."""
    blocks = _find_transformer_blocks(encoder)
    for i in layers:
        if 0 <= i < len(blocks):
            set_requires_grad(blocks[i].parameters(), True)

    if also_unfreeze_layernorm:
        for m in encoder.modules():
            if isinstance(m, nn.LayerNorm):
                set_requires_grad(m.parameters(), True)

    if also_unfreeze_pooler:
        for root in ("roberta", "bert", None):
            obj = getattr(encoder, root, encoder) if root is not None else encoder
            pooler = getattr(obj, "pooler", None)
            if pooler is not None:
                set_requires_grad(pooler.parameters(), True)
                break

def unfreeze_last_n_layers(
    encoder,
    n_last: int,
    *,
    train_layernorm: bool = True,
    train_pooler: bool = False,
) -> None:
    """Freeze everything, then unfreeze the last n blocks (+optional LN/Pooler)."""
    freeze_all_encoder_params(encoder)
    blocks = _find_transformer_blocks(encoder)
    if n_last > 0:
        for blk in list(blocks)[-n_last:]:
            set_requires_grad(blk.parameters(), True)
    if train_layernorm:
        for m in encoder.modules():
            if isinstance(m, nn.LayerNorm):
                set_requires_grad(m.parameters(), True)
    if train_pooler:
        for root in ("roberta", "bert", None):
            obj = getattr(encoder, root, encoder) if root is not None else encoder
            pooler = getattr(obj, "pooler", None)
            if pooler is not None:
                set_requires_grad(pooler.parameters(), True)
                break


# --------------------------------- Model -------------------------------------

class TargetSpanPooler(nn.Module):
    """Mean over target subtoken span, fallback to CLS if empty."""
    def forward(self, hidden: torch.Tensor, spans: list[list[int]]) -> torch.Tensor:
        # hidden: [B,L,D]
        cls = hidden[:, 0, :]
        out = []
        for b, idxs in enumerate(spans):
            if isinstance(idxs, (list, tuple)) and len(idxs) > 0:
                out.append(hidden[b, idxs, :].mean(dim=0))
            else:
                out.append(cls[b])
        return torch.stack(out, dim=0)  # [B,D]

def _safe_l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)

class WiCPairModel(nn.Module):
    """
    Cosine-similarity WiC head with a small projection and bounded temperature.
    - If you want to fine-tune the encoder, construct with freeze_encoder=False
      and call `unfreeze_last_n_layers(...)` or `unfreeze_encoder_layers(...)`.
    """
    def __init__(
        self,
        encoder,
        layer_idx: Optional[int] = -3,
        proj_dim: int = 256,
        freeze_encoder: bool = True,
        init_logit_scale: float = 2.5,
        scale_max: float = 5.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.layer_idx = layer_idx
        self.encoder_frozen = bool(freeze_encoder)

        D = encoder.config.hidden_size
        self.pool = TargetSpanPooler()
        self.proj = nn.Sequential(
            nn.Linear(D, proj_dim),
            nn.LayerNorm(proj_dim, eps=1e-5),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # bounded temperature: scale = scale_max * sigmoid(raw)
        self.logit_scale_raw = nn.Parameter(torch.tensor(float(init_logit_scale)))
        self.scale_max = float(scale_max)
        self.bias = nn.Parameter(torch.zeros(()))  # scalar calibration

        if self.encoder_frozen:
            freeze_all_encoder_params(self.encoder)

    def _scale(self) -> torch.Tensor:
        return self.scale_max * torch.sigmoid(self.logit_scale_raw)

    @property
    def logit_scale(self) -> torch.Tensor:
        # read-only convenience (detached)
        return self._scale().detach()

    def _pick_layer(self, outs):
        if self.layer_idx is None:
            return outs.last_hidden_state
        idx = self.layer_idx
        hs = outs.hidden_states
        if idx < 0:
            idx = len(hs) + idx
        return hs[idx]

    def encode(self, input_ids, attention_mask) -> torch.Tensor:
        # grads ON only if not frozen
        with torch.set_grad_enabled(not self.encoder_frozen):
            outs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            return self._pick_layer(outs)

    def forward(self, batch):
        H1 = self.encode(batch["input_ids_1"], batch["attention_mask_1"])
        H2 = self.encode(batch["input_ids_2"], batch["attention_mask_2"])
        z1 = self.pool(H1, batch["spans_1"])
        z2 = self.pool(H2, batch["spans_2"])
        e1 = _safe_l2norm(self.proj(z1), dim=-1, eps=1e-6)
        e2 = _safe_l2norm(self.proj(z2), dim=-1, eps=1e-6)
        logits = (e1 * e2).sum(dim=1) * self._scale() + self.bias
        return logits, e1, e2


# -------------------------------- Trainer ------------------------------------

def _print_param_groups(opt: torch.optim.Optimizer) -> None:
    for gi, g in enumerate(opt.param_groups):
        n = sum(p.numel() for p in g["params"])
        print(f"  group {gi}: lr={g['lr']:.2e}  wd={g.get('weight_decay',0)}  n_params={n:,}")

@torch.no_grad()
def _probe_one_batch(model: nn.Module, dataloader, device="cuda") -> None:
    model.eval()
    batch = next(iter(dataloader))
    out = model({
        "input_ids_1": batch["input_ids_1"].to(device),
        "attention_mask_1": batch["attention_mask_1"].to(device),
        "input_ids_2": batch["input_ids_2"].to(device),
        "attention_mask_2": batch["attention_mask_2"].to(device),
        "spans_1": batch["spans_1"],
        "spans_2": batch["spans_2"],
    })
    logits, e1, e2 = out
    for name, t in (("logits", logits), ("e1", e1), ("e2", e2)):
        if not torch.isfinite(t).all():
            raise RuntimeError(f"NaN/Inf found in {name} during probe")
    if not torch.isfinite(torch.sigmoid(logits)).all():
        raise RuntimeError("NaN/Inf after sigmoid during probe")
    model.train()

def _make_groups(model: nn.Module, lr_head: float, lr_encoder: float, weight_decay: float):
    head, enc = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("proj") or ("logit_scale_raw" in n) or (n == "bias"):
            head.append(p)
        else:
            enc.append(p)
    groups = []
    if head: groups.append({"params": head, "lr": lr_head, "weight_decay": weight_decay})
    if enc:  groups.append({"params": enc,  "lr": lr_encoder, "weight_decay": weight_decay})
    return groups

def _cos_lr(base_lr: float, step: int, total_steps: int, warmup_frac: float) -> float:
    warm = int(max(1, round(warmup_frac * total_steps)))
    if step < warm:
        return base_lr * (step / max(1, warm))
    t = (step - warm) / max(1, total_steps - warm)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))

# --- small helpers (place near your other train-time utilities) ---
def _margin_regularizer(cosine: torch.Tensor, y: torch.Tensor, m_pos: float, m_neg: float) -> torch.Tensor:
    """
    Encourage pos pairs to have cos >= m_pos, neg pairs cos <= m_neg.
    cosine: [B] in [-1,1], y: float tensor {0,1}.
    """
    pos_pen = torch.relu(m_pos - cosine) * y
    neg_pen = torch.relu(cosine - m_neg) * (1.0 - y)
    return (pos_pen + neg_pen).mean()

def _temp_penalty(model, l2: float = 1e-4) -> torch.Tensor:
    # Keep the raw temperature near 0 (which maps to mid-range via sigmoid)
    return l2 * (model.logit_scale_raw ** 2)

def _sigmoid_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean()


def train_wic_model(
    model: WiCPairModel,
    dataloader,
    *,
    device: str = "cuda",
    epochs: int = 3,
    lr_head: float = 1e-3,
    lr_encoder: Optional[float] = None,   # set if any encoder params require_grad=True
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    warmup_frac: float = 0.06,
    cosine_decay: bool = True,
    amp: bool = True,
    log_every: int = 100,
    pos_weight: Optional[float] = None,
    entropy_reg: float = 0.0,
    # NEW: regularization knobs (safe defaults)
    margin_pos: float = 0.60,
    margin_neg: float = 0.30,
    margin_weight: float = 0.10,
    temp_l2: float = 1e-4,
    # staged unfreezing / hooks
    unfreeze_at_epoch: Optional[int] = None,  # e.g. 1 → unfreeze at start of epoch 2
    eval_fn=None, eval_every: Optional[int] = None,
    save_dir: Optional[str] = None, save_every: Optional[int] = None,
    probe_first: bool = True,
) -> WiCPairModel:
    model.to(device).train()

    if probe_first:
        _probe_one_batch(model, dataloader, device=device)

    if lr_encoder is None:
        lr_encoder = lr_head * 0.1  # safe fallback if some enc params are trainable
    opt = torch.optim.AdamW(_make_groups(model, lr_head, lr_encoder, weight_decay))
    print("[optimizer] initial groups:")
    _print_param_groups(opt)

    if pos_weight is None:
        bce = nn.BCEWithLogitsLoss()
    else:
        pw = torch.tensor([float(pos_weight)], device=device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    total_steps = epochs * max(1, len(dataloader))
    scaler = torch_amp.GradScaler("cuda", enabled=(amp and device == "cuda"))
    step, ema = 0, None
    t0 = time.time()

    def maybe_eval_save():
        if eval_fn and eval_every and (step % eval_every == 0):
            model.eval()
            try:
                stats = eval_fn(model)
                print(f"[eval@{step}] {stats}")
            finally:
                model.train()
        if save_dir and save_every and (step % save_every == 0):
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, f"wic_model_step{step}.pt"))

    for ep in range(1, epochs + 1):
        # staged unfreezing at epoch boundary (optional)
        if unfreeze_at_epoch is not None and ep == (unfreeze_at_epoch + 1):
            unfrozen = 0
            for n, p in model.named_parameters():
                if (".encoder.layer." in n or ".pooler." in n) and not p.requires_grad:
                    p.requires_grad = True; unfrozen += 1
            if unfrozen:
                opt = torch.optim.AdamW(_make_groups(model, lr_head, lr_encoder, weight_decay))
                print(f"[unfreeze] epoch {ep}: unfroze {unfrozen} params; rebuilt optimizer:")
                _print_param_groups(opt)

        for batch in dataloader:
            step += 1

            # manual cosine schedule per-group (keeps working after opt rebuilds)
            if cosine_decay:
                for gi, g in enumerate(opt.param_groups):
                    base = lr_head if gi == 0 else lr_encoder
                    g["lr"] = _cos_lr(base, step - 1, total_steps, warmup_frac)

            ids1 = batch["input_ids_1"].to(device, non_blocking=True)
            m1   = batch["attention_mask_1"].to(device, non_blocking=True)
            ids2 = batch["input_ids_2"].to(device, non_blocking=True)
            m2   = batch["attention_mask_2"].to(device, non_blocking=True)
            y    = batch["labels"].float().to(device, non_blocking=True)

            with torch_amp.autocast(device_type=("cuda" if device == "cuda" else "cpu"),
                                    enabled=(amp and device == "cuda")):
                # NOTE: we now capture e1,e2 to compute cosine for the margin term
                logits, e1, e2 = model({
                    "input_ids_1": ids1, "attention_mask_1": m1,
                    "input_ids_2": ids2, "attention_mask_2": m2,
                    "spans_1": batch["spans_1"], "spans_2": batch["spans_2"],
                })
                loss = bce(logits, y)

                # optional: entropy regularizer (same sign as your previous code)
                if entropy_reg > 0:
                    loss = loss - entropy_reg * _sigmoid_entropy(logits)

                # NEW: margin regularizer on cosine (geometric guidance)
                cosine = (e1 * e2).sum(dim=1).clamp(-1.0, 1.0)
                if margin_weight > 0:
                    loss = loss + margin_weight * _margin_regularizer(
                        cosine, y, m_pos=margin_pos, m_neg=margin_neg
                    )

                # NEW: tiny L2 on temperature param to avoid runaway scaling
                if temp_l2 > 0:
                    loss = loss + _temp_penalty(model, l2=temp_l2)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if max_grad_norm is not None:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                if max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
            opt.zero_grad(set_to_none=True)

            if log_every and (step % log_every == 0):
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
                    print(f"[ep {ep}] step {step}/{total_steps}  "
                          f"loss={loss.item():.4f}  ema={ema:.4f}  mean(p)={probs.mean().item():.3f}  "
                          f"lr={[g['lr'] for g in opt.param_groups]}")

            if (eval_every and step % eval_every == 0) or (save_every and step % save_every == 0):
                maybe_eval_save()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, "wic_model_final.pt"))
        with open(os.path.join(save_dir, "train_args.json"), "w") as f:
            json.dump({
                "epochs": epochs, "lr_head": lr_head, "lr_encoder": lr_encoder,
                "weight_decay": weight_decay, "max_grad_norm": max_grad_norm,
                "warmup_frac": warmup_frac, "cosine_decay": cosine_decay,
                "amp": amp, "entropy_reg": entropy_reg,
                "margin_pos": margin_pos, "margin_neg": margin_neg,
                "margin_weight": margin_weight, "temp_l2": temp_l2,
                "unfreeze_at_epoch": unfreeze_at_epoch
            }, f, indent=2)

    print(f"Done in {(time.time()-t0)/60:.1f} min.")
    return model

# Keep only ONE _eval_wic in your module. If you already have one, remove this.
@torch.no_grad()
def _eval_wic(model, dataloader, device="cuda", max_batches=100):
    model.eval()
    n_seen, acc_sum, p_sum = 0, 0.0, 0.0
    for i, batch in enumerate(dataloader):
        if i >= max_batches: break
        y = batch["labels"].to(device)
        for k in ("input_ids_1","attention_mask_1","input_ids_2","attention_mask_2"):
            batch[k] = batch[k].to(device, non_blocking=True)
        logits, _, _ = model(batch)
        p = torch.sigmoid(logits)
        pred = (p > 0.5).long()
        acc_sum += float((pred == y).float().mean().item())
        p_sum += float(p.mean().item())
        n_seen += 1
    model.train()
    return {"acc": acc_sum / max(1, n_seen), "mean_p": p_sum / max(1, n_seen)}


def train_wic_model_staged(
    model, train_loader, dev_loader=None, *,
    device="cuda",
    epochs=(1, 1),                # (Stage A, Stage B). Add a third for Stage C if you widen.
    lr_head=1e-3,
    lr_encoder=5e-5,
    weight_decay=0.01,
    warmup_frac=0.06,
    cosine_decay=True,
    amp=True,
    log_every=100,
    eval_every=500,
    entropy_reg=0.05,
    unfreeze_layers_stageB=(6,7,8),  # 0-based: train layers 7–9
    widen_to_last5=False,            # optional Stage C
):
    model.to(device)
    scaler = torch_amp.GradScaler("cuda", enabled=(amp and device=="cuda"))
    bce = nn.BCEWithLogitsLoss()

    def run_phase(phase_name, n_epochs, unfreeze_layers=None, freeze_temp=False):
        # Freeze all, then selectively unfreeze
        freeze_all_encoder_params(model.encoder)
        if unfreeze_layers is not None:
            unfreeze_encoder_layers(model.encoder, unfreeze_layers, also_unfreeze_layernorm=True)

        # Temperature control (avoid early collapse)
        model.logit_scale_raw.requires_grad_(not freeze_temp)

        # Optimizer on CURRENTLY trainable params (head group + any unfrozen encoder params)
        total_steps = n_epochs * len(train_loader)
        opt = torch.optim.AdamW(_make_groups(model, lr_head, lr_encoder, weight_decay))

        step = 0
        for ep in range(1, n_epochs + 1):
            model.train()
            for batch in train_loader:
                step += 1

                # Per-group cosine schedule (survives opt rebuilds)
                if cosine_decay:
                    for gi, g in enumerate(opt.param_groups):
                        base = lr_head if gi == 0 else lr_encoder
                        g["lr"] = _cos_lr(base, (ep - 1) * len(train_loader) + step - 1, total_steps, warmup_frac)

                # Move tensors
                y = batch["labels"].float().to(device)
                for k in ("input_ids_1","attention_mask_1","input_ids_2","attention_mask_2"):
                    batch[k] = batch[k].to(device, non_blocking=True)

                with torch_amp.autocast(device_type=("cuda" if device=="cuda" else "cpu"),
                                        enabled=(amp and device=="cuda")):
                    logits, _, _ = model(batch)
                    p = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
                    loss = bce(logits, y)
                    # gentle entropy regularization
                    ent = -(p*torch.log(p) + (1-p)*torch.log(1-p)).mean()
                    loss = loss - entropy_reg * ent

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                opt.zero_grad(set_to_none=True)

                if log_every and (step % log_every == 0):
                    print(f"[{phase_name}] ep {ep}/{n_epochs} step {step}/{total_steps}  "
                          f"loss={loss.item():.4f}  mean(p)={p.mean().item():.3f}  "
                          f"lr={[g['lr'] for g in opt.param_groups]}")

                if dev_loader is not None and eval_every and (step % eval_every == 0):
                    stats = _eval_wic(model, dev_loader, device=device, max_batches=50)
                    print(f"[{phase_name}][DEV] acc={stats['acc']:.3f}  mean(p)={stats['mean_p']:.3f}")

    # Stage A: head-only (freeze temperature to avoid early collapse)
    run_phase("StageA(head)", n_epochs=epochs[0], unfreeze_layers=None, freeze_temp=True)

    # Stage B: unfreeze specific middle-high layers (default 7–9)
    run_phase("StageB(7-9)", n_epochs=epochs[1], unfreeze_layers=list(unfreeze_layers_stageB), freeze_temp=False)

    # Stage C (optional): widen to last 5 layers
    if widen_to_last5 and len(epochs) >= 3 and epochs[2] > 0:
        last = len(_find_transformer_blocks(model.encoder))
        run_phase("StageC(last5)", n_epochs=epochs[2], unfreeze_layers=list(range(last-5, last)), freeze_temp=False)

    return model
