import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HTTP_PROXY"]= "http://proxy.utwente.nl:3128"
os.environ["HTTPS_PROXY"]= "http://proxy.utwente.nl:3128"
os.environ["http_proxy"]= "http://proxy.utwente.nl:3128"
os.environ["https_proxy"]= "http://proxy.utwente.nl:3128"

import os
import json
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import random
import numpy as np

# -------------------------
# CONFIG
# -------------------------
TRAIN_DATA_PATH = "/home/simonettos/thijs/data_augmentatio_stefano/siamese_unsup_sup/train_clean.json"
VALIDATION_DATA_PATH = "/home/simonettos/thijs/data_augmentatio_stefano/siamese_unsup_sup/val_clean.json"
TEST_DATA_PATH = "/home/simonettos/thijs/data_augmentatio_stefano/siamese_unsup_sup/test_clean.json"

TTP_DESC_PATH   = "/home/simonettos/thijs/data_augmentatio_stefano/mitre/ttp_descriptions2.json"
OUTPUT_DIR      = "/home/simonettos/thijs/data_augmentatio_stefano/siamese_unsup_sup/siamese_sentence_ttp_model_test_only"
BASE_MODEL = "ehsanaghaei/SecureBERT"

BATCH_SIZE = 32
PATIENCE   = 5
SEED       = 42

MAX_LEN_SENT = 192
MAX_LEN_TTP  = 256

LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
MAX_EPOCHS = 50
TEMPERATURE = 0.05

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# REPRODUCIBILITY
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

def normalize_label_list(lbls):
    out = []
    for x in (lbls or []):
        if isinstance(x, str):
            x = x.strip()
            if x:
                out.append(x)
    return out
# -------------------------
# LOAD DATA
# -------------------------
with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open(TTP_DESC_PATH, "r", encoding="utf-8") as f:
    ttp_descriptions = json.load(f)

# allow everything we can actually rank over
allowed_set = set(ttp_descriptions.keys())


# -------------------------
# BUILD TRAIN PAIRS
# -------------------------
# only non empty ttps
train_pairs: List[Tuple[str, str]] = []
for item in train_data:
    sent = item["sentence"].strip()
    for ttp in item["labels"]:
        if ttp in ttp_descriptions and ttp in allowed_set:
            ttp_text = f"{ttp}: {ttp_descriptions[ttp]}"
            train_pairs.append((sent, ttp_text))

print(f"Training pairs: {len(train_pairs)}")
print("train pair sample:", train_pairs[0])
print("train pair sample:", train_pairs[1])
# -------------------------
# BUILD TRAIN PAIRS
# -------------------------
# only non empty ttps
train_pairs: List[Tuple[str, str]] = []
for item in train_data:
    sent = item["sentence"].strip()
    for ttp in item["labels"]:
        if ttp in ttp_descriptions:
            ttp_text = f"{ttp}: {ttp_descriptions[ttp]}"
            train_pairs.append((sent, ttp_text))


# Build sentence -> set(labels) but only allowed labels
sent2labels = defaultdict(set)

for item in train_data:
    sent = (item.get("sentence") or "").strip()
    if not sent:
        continue
    lbls = normalize_label_list(item.get("labels"))
    for ttp in lbls:
        if ttp in allowed_set:
            sent2labels[sent].add(ttp)

# Keep only sentences that still have at least 1 allowed label
train_items = [(sent, sorted(lblset)) for sent, lblset in sent2labels.items() if lblset]
val_items = []
for item in train_data:
    sent = (item.get("sentence") or "").strip()
    if not sent:
        continue
    lbls = normalize_label_list(item.get("labels"))
    for ttp in lbls:
        if ttp in allowed_set:
            sent2labels[sent].add(ttp)

val_items = [(sent, sorted(lblset)) for sent, lblset in sent2labels.items() if lblset]
print("Unique sentences after filtering:", len(train_items))

# Compute "real" (unbalanced) label distribution on this filtered pool
real_counts = Counter()
for _, lbls in train_items:
    real_counts.update(lbls)

print("Allowed labels present in filtered pool:", len(real_counts))
print("Train sentences:", len(train_items))
print("Val sentences  :", len(val_items))

val_counts = Counter()
for _, lbls in val_items:
    val_counts.update(lbls)

print("Real counts (top5):", real_counts.most_common(5))
print("Val  counts (top5):", val_counts.most_common(5))
import math
from collections import Counter
from collections import Counter, defaultdict
import random, math
import numpy as np

def get_sentence(it):
    # supports {"sentence":..., "labels":...} OR (sentence, labels) OR (sentence, labels, ...)
    if isinstance(it, dict):
        return it["sentence"]
    if isinstance(it, (list, tuple)):
        return it[0]
    raise TypeError(f"Unsupported item type: {type(it)}")

def get_labels(it):
    if isinstance(it, dict):
        return it["labels"]
    if isinstance(it, (list, tuple)):
        return it[1]
    raise TypeError(f"Unsupported item type: {type(it)}")

def set_labels(it, new_labels):
    # return same “shape” as input
    if isinstance(it, dict):
        out = dict(it)
        out["labels"] = new_labels
        return out
    if isinstance(it, (list, tuple)):
        # keep first element as sentence, second as labels, keep extras if any
        if len(it) == 2:
            return (it[0], new_labels)
        return (it[0], new_labels, *it[2:])
    raise TypeError(f"Unsupported item type: {type(it)}")

def compute_targets(
    counts: Counter,
    ref_quantile: float = 0.75,
    cap_mult: float = 1.5,
    floor_mult: float = 0.5,
    alpha: float = 0.6,
    min_count: int = 5,
):
    """
    Tempered rebalancing targets that allow BOTH oversampling and downsampling.

    target_l = clamp( ceil(c_l^alpha * ref^(1-alpha)), floor, cap )

    - alpha in (0,1): closer to 1 => gentler change; closer to 0 => stronger pull to ref
    - ref: quantile of counts (e.g., 0.75)
    - cap_mult: max target relative to ref (prevents exploding tail)
    - floor_mult: min target relative to ref (prevents destroying head via heavy downsampling)
    - min_count: absolute minimum target (keeps very rare labels from being downsampled to 0/1)
    """
    vals = np.array(list(counts.values()), dtype=float)
    ref = float(np.quantile(vals, ref_quantile))

    cap = float(ref * cap_mult)
    floor = float(ref * floor_mult)

    targets = {}
    for lbl, c in counts.items():
        c = float(c)

        # tempered pull toward ref
        t = (c ** alpha) * (ref ** (1.0 - alpha))
        t = math.ceil(t)

        # clamp to floor/cap + absolute min_count
        t = max(t, int(min_count), int(math.floor(floor)))
        t = min(t, int(math.ceil(cap)))

        targets[lbl] = int(t)

    return targets, ref, cap, floor


def tempered_oversample_sentences(train_items, targets, seed=42):
    rng = random.Random(seed)

    counts = Counter()
    for it in train_items:
        counts.update(get_labels(it))

    need = {lbl: max(0, targets[lbl] - counts.get(lbl, 0)) for lbl in targets}

    label2idx = defaultdict(list)
    for idx, it in enumerate(train_items):
        for lbl in get_labels(it):
            if lbl in need:
                label2idx[lbl].append(idx)

    augmented = list(train_items)

    max_iters = sum(need.values()) + 1000
    iters = 0
    while True:
        iters += 1
        if iters > max_iters:
            break

        lbl, remaining = max(need.items(), key=lambda x: x[1])
        if remaining <= 0:
            break

        candidates = label2idx.get(lbl, [])
        if not candidates:
            need[lbl] = 0
            continue

        pick_idx = rng.choice(candidates)
        picked = train_items[pick_idx]
        augmented.append(picked)

        for l in get_labels(picked):
            if l in need and need[l] > 0:
                need[l] -= 1

    return augmented


# counts on train (unbalanced) pool
train_counts = Counter()
for it in train_items:
    train_counts.update(get_labels(it))

targets, ref, cap, floor = compute_targets(
    train_counts,
    ref_quantile=0.75,
    cap_mult=1.5,
    floor_mult=0.6,
    alpha=0.6,
    min_count=6,
)
print(ref, cap, floor)

train_items_less_unbal = tempered_oversample_sentences(train_items, targets, seed=SEED)

# Inspect effect
new_counts = Counter()
for it in train_items_less_unbal:
    new_counts.update(get_labels(it))


# print("Reference (q=0.75):", ref_used, "Cap:", cap_used)
print("Original train label min/max:", min(train_counts.values()), max(train_counts.values()))
print("New train label min/max     :", min(new_counts.values()), max(new_counts.values()))

import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# -------------------------
# Helpers: support dict OR tuple items
# -------------------------
def get_labels(it):
    # dict: {"sentence":..., "labels":[...]}
    if isinstance(it, dict):
        return it.get("labels", []) or []
    # tuple/list: (sentence, labels, ...)
    if isinstance(it, (list, tuple)):
        return it[1] if len(it) > 1 else []
    raise TypeError(f"Unsupported item type: {type(it)}")

def norm_labels(lbls):
    out = []
    for x in (lbls or []):
        if isinstance(x, str):
            x = x.strip()
            if x:
                out.append(x)
    return out

def count_labels_from_raw(train_data, allowed_set=None):
    c = Counter()
    for item in train_data:
        lbls = norm_labels(item.get("labels", []) or [])
        for ttp in lbls:
            if allowed_set is None or ttp in allowed_set:
                c[ttp] += 1
    return c

def count_labels_from_items(items, allowed_set=None):
    c = Counter()
    for it in items:
        lbls = norm_labels(get_labels(it))
        for ttp in lbls:
            if allowed_set is None or ttp in allowed_set:
                c[ttp] += 1
    return c

# -------------------------
# Build distributions
# -------------------------
# Original raw distribution but restricted to allowed labels
orig_counts_allowed = count_labels_from_raw(train_data, allowed_set=allowed_set)

# If you have these objects, use them; otherwise comment out what you don't have:
# - items: filtered sentence-level (unique sentences, filtered to allowed, etc.)
# - train_items_unbal: train split before balancing
# - val_items: validation split (unbalanced)
# - train_items_less_unbal: training set after tempered oversampling

# Safe fallbacks:
filtered_counts = count_labels_from_items(train_items, allowed_set=allowed_set) if "items" in globals() else Counter()
train_unbal_counts = count_labels_from_items(train_items, allowed_set=allowed_set) if "train_items_unbal" in globals() else Counter()
val_counts = count_labels_from_items(val_items, allowed_set=allowed_set) if "val_items" in globals() else Counter()
balanced_counts = count_labels_from_items(train_items_less_unbal, allowed_set=allowed_set) if "train_items_less_unbal" in globals() else Counter()

def iter_sentence_label_items(sentence_items):
    """Yields (sentence, labels_list) from either dict-items or tuple-items."""
    for it in sentence_items:
        if isinstance(it, dict):
            sent = (it.get("sentence") or "").strip()
            lbls = it.get("labels", []) or []
        else:
            # tuple/list: (sent, labels, ...)
            sent = (it[0] or "").strip()
            lbls = it[1] if len(it) > 1 else []
        if not sent:
            continue
        # normalize labels
        out = []
        for x in (lbls or []):
            if isinstance(x, str):
                x = x.strip()
                if x:
                    out.append(x)
        if out:
            yield sent, out

allowed_set = set(allowed_set)
print(len(allowed_set), "allowed labels for training pairs.")
source_train_items = train_items  

train_pairs: List[Tuple[str, str]] = []
for sent, lbls in iter_sentence_label_items(source_train_items):
    for ttp in lbls:
        if ttp in allowed_set and ttp in ttp_descriptions:
            ttp_text = f"{ttp}: {ttp_descriptions[ttp]}"
            train_pairs.append((sent, ttp_text))

print(f"Training pairs (from balanced sentences): {len(train_pairs)}")

print("Sample train pair 0:", train_pairs[0])
# 1) Basic sanity: how many items, how many yielded by iterator?
print("source_train_items:", len(source_train_items))

n_yield = 0
n_labels_total = 0
first_few = []
for sent, lbls in iter_sentence_label_items(source_train_items):
    n_yield += 1
    n_labels_total += len(lbls)
    if len(first_few) < 3:
        first_few.append((sent[:80], lbls[:10]))

print("iter_sentence_label_items yielded:", n_yield)
print("total labels across yielded items:", n_labels_total)
print("first few yields:", first_few)

# 2) Check overlap between your labels and the filters
labels_in_data = set()
for _, lbls in iter_sentence_label_items(source_train_items):
    labels_in_data.update(lbls)

print("unique labels in data:", len(labels_in_data))
print("overlap with allowed_set:", len(labels_in_data & allowed_set))
print("overlap with ttp_descriptions:", len(labels_in_data & set(ttp_descriptions.keys())))
print("overlap with BOTH:", len(labels_in_data & allowed_set & set(ttp_descriptions.keys())))

# 3) Show a few labels that are being rejected (helpful to see formatting issues)
rejected_allowed = list(labels_in_data - allowed_set)[:20]
rejected_desc = list(labels_in_data - set(ttp_descriptions.keys()))[:20]
print("example labels not in allowed_set:", rejected_allowed)
print("example labels not in ttp_descriptions:", rejected_desc)


# 1) Basic sanity: how many items, how many yielded by iterator?
print("source_train_items:", len(source_train_items))

n_yield = 0
n_labels_total = 0
first_few = []
for sent, lbls in iter_sentence_label_items(source_train_items):
    n_yield += 1
    n_labels_total += len(lbls)
    if len(first_few) < 3:
        first_few.append((sent[:80], lbls[:10]))

print("iter_sentence_label_items yielded:", n_yield)
print("total labels across yielded items:", n_labels_total)
print("first few yields:", first_few)

# 2) Check overlap between your labels and the filters
labels_in_data = set()
for _, lbls in iter_sentence_label_items(source_train_items):
    labels_in_data.update(lbls)

print("unique labels in data:", len(labels_in_data))
print("overlap with allowed_set:", len(labels_in_data & allowed_set))
print("overlap with ttp_descriptions:", len(labels_in_data & set(ttp_descriptions.keys())))
print("overlap with BOTH:", len(labels_in_data & allowed_set & set(ttp_descriptions.keys())))

# 3) Show a few labels that are being rejected (helpful to see formatting issues)
rejected_allowed = list(labels_in_data - allowed_set)[:20]
rejected_desc = list(labels_in_data - set(ttp_descriptions.keys()))[:20]
print("example labels not in allowed_set:", rejected_allowed)
print("example labels not in ttp_descriptions:", rejected_desc)


with open(VALIDATION_DATA_PATH, "r", encoding="utf-8") as f:
    val_data = json.load(f)   # list of {"sentence":..., "labels":[...]}

# -------------------------
# DATASET + COLLATE
# -------------------------
class PairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.pairs[idx]  # (sent, ttp_text)

@dataclass
class DualCollator:
    tokenizer: object
    max_len_sent: int
    max_len_ttp: int

    def __call__(self, batch: List[Tuple[str, str]]) -> Dict[str, Dict[str, torch.Tensor]]:
        sents = [b[0] for b in batch]
        ttps  = [b[1] for b in batch]

        tok_sent = self.tokenizer(
            sents,
            padding=True,
            truncation=True,
            max_length=self.max_len_sent,
            return_tensors="pt",
        )
        tok_ttp = self.tokenizer(
            ttps,
            padding=True,
            truncation=True,
            max_length=self.max_len_ttp,
            return_tensors="pt",
        )
        return {"sent": tok_sent, "ttp": tok_ttp}

# -------------------------
# MODEL: SecureBERT bi-encoder
# -------------------------
class SecureBertEmbedder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

    @staticmethod
    def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    def encode_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        emb = self.mean_pool(out.last_hidden_state, attention_mask)
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    @torch.no_grad()
    def encode_texts(self, tokenizer, texts: List[str], batch_size: int, max_length: int) -> np.ndarray:
        self.eval()
        all_embs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            tok = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tok = {k: v.to(DEVICE) for k, v in tok.items()}
            embs = self.encode_batch(tok["input_ids"], tok["attention_mask"])
            all_embs.append(embs.detach().cpu().numpy())
        return np.vstack(all_embs)

# -------------------------
# LOSS: In-batch negatives
# -------------------------
class InBatchNegativesLoss(nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        logits = (emb_a @ emb_b.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

# -------------------------
# TOKENIZER / DATALOADER
# -------------------------
# IMPORTANT: use AutoTokenizer for SecureBERT
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

train_ds = PairDataset(train_pairs)
collator = DualCollator(tokenizer=tokenizer, max_len_sent=MAX_LEN_SENT, max_len_ttp=MAX_LEN_TTP)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2,
    pin_memory=(DEVICE == "cuda"),
    collate_fn=collator
)

# -------------------------
# INIT MODEL / OPT / SCHED
# -------------------------
model = SecureBertEmbedder(BASE_MODEL).to(DEVICE)
loss_fn = InBatchNegativesLoss(temperature=TEMPERATURE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps = len(train_loader) * MAX_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

# -------------------------
# PREP VAL TTP TEXTS
# -------------------------
# -------------------------
# PREP EVAL TTP TEXTS (stable order!)
# -------------------------
allowed_set = set(allowed_set)

ttp_ids = sorted([t for t in allowed_set if t in ttp_descriptions])  # this will be all described labels
ttp_texts = [f"{t}: {ttp_descriptions[t]}" for t in ttp_ids]



# -------------------------
# VALIDATION FUNCTION
# -------------------------
def get_val_sentence(item):
    if isinstance(item, dict):
        return (item.get("sentence") or "").strip()
    if isinstance(item, (list, tuple)):
        return (item[0] or "").strip()
    raise TypeError(f"Unsupported val item type: {type(item)}")

def get_val_labels(item):
    if isinstance(item, dict):
        return item.get("labels", []) or []
    if isinstance(item, (list, tuple)):
        return item[1] if len(item) > 1 else []
    raise TypeError(f"Unsupported val item type: {type(item)}")

def validate_on(
    model: SecureBertEmbedder,
    items,
    k_list=(1, 5, 10),
    ttp_batch_size=64,
    sent_batch_size=64,
) -> Dict[str, float]:
    model.eval()

    # Precompute candidate embeddings ONCE for this call
    ttp_embs = model.encode_texts(tokenizer, ttp_texts, batch_size=ttp_batch_size, max_length=MAX_LEN_TTP)

    hits_at_k = {k: 0 for k in k_list}
    mean_recall_at_k = {k: 0.0 for k in k_list}
    mrr = 0.0
    used = 0

    # --- Batch sentence encoding for speed (optional but recommended) ---
    sentences = []
    golds = []

    for item in items:
        sent = get_val_sentence(item)
        raw_lbls = get_val_labels(item)

        gold = {t.strip() for t in raw_lbls if isinstance(t, str) and t.strip()}
        gold = {t for t in gold if t in allowed_set and t in ttp_descriptions}
        if not sent or not gold:
            continue

        sentences.append(sent)
        golds.append(gold)

    if not sentences:
        return {**{f"hit@{k}": 0.0 for k in k_list},
                **{f"mean_recall@{k}": 0.0 for k in k_list},
                "mrr": 0.0,
                "val_used": 0}

    for i in range(0, len(sentences), sent_batch_size):
        batch_sents = sentences[i:i+sent_batch_size]
        batch_golds = golds[i:i+sent_batch_size]

        sent_embs = model.encode_texts(tokenizer, batch_sents, batch_size=sent_batch_size, max_length=MAX_LEN_SENT)
        # sent_embs: [B, D]

        sims = cosine_similarity(sent_embs, ttp_embs)  # [B, num_ttp]
        rankings = np.argsort(-sims, axis=1)          # [B, num_ttp]

        for row in range(rankings.shape[0]):
            gold = batch_golds[row]
            ranking = rankings[row]
            used += 1

            rr = 0.0
            for rank_pos, idx in enumerate(ranking, start=1):
                if ttp_ids[idx] in gold:
                    rr = 1.0 / rank_pos
                    break
            mrr += rr

            for k in k_list:
                topk = [ttp_ids[j] for j in ranking[:k]]
                hits_at_k[k] += int(any(t in gold for t in topk))
                mean_recall_at_k[k] += len(set(topk) & gold) / max(1, len(gold))

    n = max(1, used)
    metrics = {f"hit@{k}": hits_at_k[k] / n for k in k_list}
    metrics.update({f"mean_recall@{k}": mean_recall_at_k[k] / n for k in k_list})
    metrics["mrr"] = mrr / n
    metrics["val_used"] = used
    return metrics




# -------------------------
# SAVE HELPERS
# -------------------------
def save_model(model: SecureBertEmbedder, tokenizer, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    model.backbone.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

# -------------------------
# TRAIN LOOP WITH EARLY STOPPING
# -------------------------
best_score = -1.0
patience_ctr = 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    running = 0.0

    print(f"\nEpoch {epoch}")

    for batch in train_loader:
        tok_sent = {k: v.to(DEVICE) for k, v in batch["sent"].items()}
        tok_ttp  = {k: v.to(DEVICE) for k, v in batch["ttp"].items()}

        emb_sent = model.encode_batch(tok_sent["input_ids"], tok_sent["attention_mask"])
        emb_ttp  = model.encode_batch(tok_ttp["input_ids"], tok_ttp["attention_mask"])

        loss = loss_fn(emb_sent, emb_ttp)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running += loss.item()

    avg_loss = running / max(1, len(train_loader))
    metrics = validate_on(model, val_data)

    print(f"Train loss: {avg_loss:.4f}")
    print("Validation:", ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

    score = metrics["hit@1"]

    if score > best_score:
        best_score = score
        patience_ctr = 0
        save_model(model, tokenizer, OUTPUT_DIR)
        print("✔ New best model saved")
    else:
        patience_ctr += 1
        print(f"No improvement (patience {patience_ctr}/{PATIENCE})")

    if patience_ctr >= PATIENCE:
        print("\nEarly stopping triggered")
        break

print(f"\nBest validation hit@1: {best_score:.4f}")
print(f"Saved to: {OUTPUT_DIR}")

TEST_DATA_PATH = "/home/simonettos/thijs/data_augmentatio_stefano/siamese_unsup_sup/test_clean.json"
with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)

test_metrics = validate_on(model, test_data)
print("Test:", test_metrics)
