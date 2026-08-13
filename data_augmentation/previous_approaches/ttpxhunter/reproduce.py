import json
import numpy as np
import torch
import os
from copy import deepcopy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HTTP_PROXY"]= "http://proxy.utwente.nl:3128"
os.environ["HTTPS_PROXY"]= "http://proxy.utwente.nl:3128"
os.environ["http_proxy"]= "http://proxy.utwente.nl:3128"
os.environ["https_proxy"]= "http://proxy.utwente.nl:3128"

def compute_label_counts(rows):
    c = Counter()
    for r in rows:
        labs = r.get("labels", [])
        if isinstance(labs, list):
            for l in labs:
                if isinstance(l, str) and l.strip():
                    c[l.strip()] += 1
    return c

def load_columnar_json(path):
    with open(path, "r", encoding="utf-8") as f:
        col = json.load(f)

    row_ids = sorted(col["sentence"].keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    rows = []
    for rid in row_ids:
        r = {"_id": str(rid)}
        for k, v in col.items():
            r[k] = v.get(str(rid), None)
        rows.append(r)
    return rows


# -------------------------
# PATHS / SPLIT CONFIG (match main script)
# -------------------------
ORIG_PATH = "/home/simonettos/thijs/classification/classification/datasets/bosch_train.json"
VAL_OUT   = "/home/simonettos/thijs/classification/classification/datasets/bosch_split_val_mainstyle.json"

OUT_PATH_COL = "/home/simonettos/thijs/classification/classification/datasets/bosch_train_ttp_hunter.json"

SEED = 0
VAL_FRAC = 0.20

# If your main loader stratifies on df_original["label"] (only in ARTIFICIAL case),
# set this True AND ensure that column exists and is single-label.
USE_STRATIFY = False
STRATIFY_COL = "label"   # main loader uses "label" in that snippet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MLM_MODEL = "ehsanaghaei/SecureBERT"
SBERT_MODEL = "sentence-transformers/bert-base-nli-mean-tokens"
# -------------------------
# 1) REPRODUCE MAIN SPLIT (row-level train_test_split)
# -------------------------
orig_rows = load_columnar_json(ORIG_PATH)

if USE_STRATIFY:
    # stratify expects a 1D label per row; requires STRATIFY_COL to exist in your JSON
    y = []
    keep = []
    for r in orig_rows:
        lab = r.get(STRATIFY_COL, None)
        if lab is None:
            continue
        y.append(lab)
        keep.append(r)

    train_rows, val_rows = train_test_split(
        keep,
        test_size=VAL_FRAC,
        random_state=SEED,
        stratify=y
    )
else:
    train_rows, val_rows = train_test_split(
        orig_rows,
        test_size=VAL_FRAC,
        random_state=SEED
    )


label_counts = compute_label_counts(train_rows)

q = 0.95
thresh = np.quantile(list(label_counts.values()), q)
minority_labels = {lab for lab, cnt in label_counts.items() if cnt <= thresh}

print(f"Unique labels in train: {len(label_counts)}")
print(f"Minority labels selected: {len(minority_labels)} (definition: <= {thresh})")





def is_minority_row(labels):
    return (
        isinstance(labels, list)
        and any(isinstance(l, str) and l in minority_labels for l in labels)
    )


def is_maskable_token(tok: str) -> bool:
    # Skip special tokens and pure punctuation-like wordpieces
    if tok in tokenizer.all_special_tokens:
        return False
    if tok.startswith("[") and tok.endswith("]"):
        return False
    # many BERT wordpieces for punctuation are like ".", ",", ":", etc.
    # keep alphanumerics and "##subwords"
    has_alnum = any(ch.isalnum() for ch in tok.replace("##", ""))
    return has_alnum

def mlm_topk_at_position(input_ids: torch.Tensor, attention_mask: torch.Tensor, pos: int, topk: int = 5):
    """
    input_ids: shape (1, L)
    pos: token index to predict (should be [MASK] position)
    returns list of (token_id, token_str, prob)
    """
    with torch.no_grad():
        out = mlm(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[0, pos, :]
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, k=topk)

    ids = top.indices.detach().cpu().tolist()
    ps  = top.values.detach().cpu().tolist()

    result = []
    for tid, p in zip(ids, ps):
        tok = tokenizer.convert_ids_to_tokens(int(tid))
        result.append((int(tid), tok, float(p)))
    return result

def generate_candidates_token_level(sentence: str, topk: int = 5, max_len: int = 256):
    """
    Masks each *tokenizer token* (wordpiece), gets MLM topk replacements,
    decodes candidate sentences.

    Returns a list of candidate dicts:
      {"cand": str, "masked_pos": int, "masked_token": str, "replacement": str, "mlm_prob": float}
    """
    enc = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        add_special_tokens=True
    )

    input_ids = enc["input_ids"].to(DEVICE)          # (1, L)
    attn_mask = enc["attention_mask"].to(DEVICE)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())

    cands = []
    L = input_ids.shape[1]

    # skip [CLS]=0 and [SEP]=L-1
    for pos in range(1, L - 1):
        tok = tokens[pos]
        if not is_maskable_token(tok):
            continue

        masked_ids = input_ids.clone()
        masked_ids[0, pos] = MASK_ID

        preds = mlm_topk_at_position(masked_ids, attn_mask, pos, topk=topk)
        for tid, rep_tok, prob in preds:
            cand_ids = masked_ids.clone()
            cand_ids[0, pos] = tid
            cand_text = tokenizer.decode(cand_ids[0], skip_special_tokens=True).strip()

            if not cand_text or cand_text == sentence.strip():
                continue

            cands.append({
                "cand": cand_text,
                "masked_pos": pos,
                "masked_token": tok,
                "replacement": rep_tok,
                "mlm_prob": prob
            })

    return cands

def sort_columnar_json(col_json: dict) -> dict:
    """
    Sorts the inner dict keys (row ids) numerically when possible.
    Falls back to lexicographic for non-numeric ids.
    Applies the same order to all columns.
    """
    # collect all row ids that appear in any column
    all_ids = set()
    for col, mapping in col_json.items():
        if isinstance(mapping, dict):
            all_ids.update(mapping.keys())

    def key_fn(k: str):
        return (0, int(k)) if str(k).isdigit() else (1, str(k))

    ordered_ids = sorted(all_ids, key=key_fn)

    sorted_out = {}
    for col, mapping in col_json.items():
        if not isinstance(mapping, dict):
            sorted_out[col] = mapping
            continue
        sorted_out[col] = {rid: mapping.get(rid) for rid in ordered_ids if rid in mapping}
    return sorted_out



def ttpxhunter_topN_augments(sentence: str, theta: float = 0.975, topk: int = 5, keep_n: int = 5, max_len: int = 256):
    """
    Generate many token-level candidates, compute cosine similarity, keep top-N that pass theta.
    Returns list of dicts sorted by similarity desc. Empty list if none pass.
    """
    # 1) generate candidate sentences
    raw = generate_candidates_token_level(sentence, topk=topk, max_len=max_len)
    if not raw:
        return []

    # de-duplicate candidate strings (keep best metadata later)
    uniq = {}
    for r in raw:
        uniq.setdefault(r["cand"], r)
    cand_texts = list(uniq.keys())

    # 2) embed in batch for speed
    orig_emb = sbert.encode([sentence], convert_to_numpy=True)[0]
    cand_embs = sbert.encode(cand_texts, convert_to_numpy=True)

    scored = []
    for text, emb in zip(cand_texts, cand_embs):
        sim = cosine_sim(orig_emb, emb)
        if sim >= theta:
            meta = uniq[text]
            scored.append({
                "augmented_sentence": text,
                "similarity": sim,
                "masked_pos": meta["masked_pos"],
                "masked_token": meta["masked_token"],
                "replacement": meta["replacement"],
                "mlm_prob": meta["mlm_prob"],
            })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:keep_n]


# -------------------------
# Columnar JSON helpers
# -------------------------


def save_columnar_json(rows, path):
    out = {"sentence": {}, "labels": {}, "doc_title": {}}
    for i, r in enumerate(rows):
        k = str(i)
        out["sentence"][k] = r.get("sentence", "")
        out["labels"][k] = r.get("labels", [])
        out["doc_title"][k] = r.get("doc_title", "")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def rows_to_column_json(rows):
    cols = {}
    for r in rows:
        rid = r["_id"]
        for k, v in r.items():
            if k == "_id":
                continue
            cols.setdefault(k, {})[rid] = v
    return cols


# -------------------------
# 3) AUGMENTATION FUNCTIONS (your originals)
# -------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)



# Optional: save val split for debugging/consistency checks
save_columnar_json(list(val_rows), VAL_OUT)

print(f"Split complete (main-style): train={len(train_rows)} rows, val={len(val_rows)} rows")

# -------------------------
# 2) MODELS
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MLM_MODEL)
mlm = AutoModelForMaskedLM.from_pretrained(MLM_MODEL).to(DEVICE)
mlm.eval()

sbert = SentenceTransformer(SBERT_MODEL, device=DEVICE)

MASK = tokenizer.mask_token
MASK_ID = tokenizer.mask_token_id
# -------------------------
# 4) RUN AUGMENTATION ON *TRAIN SPLIT ONLY*
# -------------------------
THETA = 0.975
TOPK = 5
KEEP_N = 5   # keep up to 5 augmentations per sentence

augmented_rows = []
num_aug = 0
num_considered = 0

for r in tqdm(train_rows):
    sentence = r.get("sentence", "")
    labels = r.get("labels", [])

    if not isinstance(sentence, str) or not sentence.strip():
        continue

    # --- minority selection instead of T1-only ---
    if not is_minority_row(labels):
        continue

    num_considered += 1

    best_list = ttpxhunter_topN_augments(
        sentence,
        theta=THETA,
        topk=TOPK,
        keep_n=KEEP_N
    )
    if not best_list:
        continue

    # Add up to KEEP_N augmented rows
    for j, best in enumerate(best_list):
        new_r = dict(r)
        new_r["_id"] = f'{r["_id"]}__TTPXH_{j}'
        new_r["sentence"] = best["augmented_sentence"]
        new_r["_aug"] = {
            "method": "ttpxhunter_securebert_mlm_topN_tokenlevel",
            "theta": THETA,
            "topk": TOPK,
            "keep_n": KEEP_N,
            "masked_pos": best["masked_pos"],
            "masked_token": best["masked_token"],
            "replacement": best["replacement"],
            "mlm_prob": best["mlm_prob"],
            "similarity": best["similarity"],
            "source_sentence": sentence,
        }
        augmented_rows.append(new_r)
        num_aug += 1

print("Sentences considered (minority rows):", num_considered)
print("Augmented rows generated:", num_aug)

# Keep originals (train split) + augmented
all_rows = list(train_rows) + augmented_rows

col_json = rows_to_column_json(all_rows)
col_json = sort_columnar_json(col_json)

with open(OUT_PATH_COL, "w", encoding="utf-8") as f:
    json.dump(col_json, f, indent=2, ensure_ascii=False)

print("Saved column-oriented JSON to:", OUT_PATH_COL)
