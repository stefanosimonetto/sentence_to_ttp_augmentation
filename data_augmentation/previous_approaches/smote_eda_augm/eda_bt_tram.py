import json, re, random
from copy import deepcopy
from collections import defaultdict

# -------------------------
# CONFIG
# -------------------------
INPUT_JSON   = "/home/simonettos/thijs/classification/classification/datasets/tram_train.json"

# TRAIN_OUT    = "/home/simonettos/thijs/classification/classification/datasets/tram__split_train_eda.json"
VAL_OUT      = "/home/simonettos/thijs/classification/classification/datasets/tram_split_val_eda.json"
AUG_TRAIN_OUT= "/home/simonettos/thijs/classification/classification/datasets/tram_split_train_augmented_eda_bt_T1.json"
SEED = 0
VAL_FRAC = 0.20      # 10% doc_titles go to validation

# How many augmented variants per eligible sentence
N_EDA = 1
N_BT  = 1

# EDA strength (keep small for cyber text)
EDA_ALPHA = 0.10
DEL_PROB  = 0.10

# Back-translation pivot
PIVOT_LANG = "de"

# Augment only sentences that have >=1 label starting with this prefix
AUG_LABEL_PREFIX = "T1"

random.seed(SEED)

# -------------------------
# TEXT CLEANING / PROTECTION
# -------------------------
MD_BOLD = re.compile(r"\*\*(.*?)\*\*")
MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

PATTERNS = {
    "URL": re.compile(r"https?://\S+"),
    "CVE": re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE),
    "ATTACK_TECH": re.compile(r"\bT\d{4}\b"),
    "ATTACK_TAC": re.compile(r"\bTA\d{4}\b"),
    "ATTACK_GRP": re.compile(r"\bG\d{4}\b"),
    "IP": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "HASH": re.compile(r"\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b"),
}

def strip_markdown(text: str) -> str:
    text = MD_BOLD.sub(r"\1", text)
    text = MD_LINK.sub(r"\1", text)  # keep anchor text, drop URL
    return text

def protect_tokens(text: str):
    placeholder_map = {}
    counter = 0

    def _replace(match, kind):
        nonlocal counter
        raw = match.group(0)
        key = f"ZZZ{kind}{counter}ZZZ"
        counter += 1
        placeholder_map[key] = raw
        return key

    out = text
    for kind, pat in PATTERNS.items():
        out = pat.sub(lambda m, k=kind: _replace(m, k), out)

    return out, placeholder_map

def restore_tokens(text: str, placeholder_map: dict) -> str:
    out = text
    for k, v in placeholder_map.items():
        out = out.replace(k, v)
    return out

TOK = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def tokenize(text: str):
    return TOK.findall(text)

def detokenize(tokens):
    out = []
    for i, t in enumerate(tokens):
        if i > 0 and re.match(r"[^\w\s]", t) and t not in ("(", "[", "{"):
            out[-1] = out[-1] + t
        elif t in (")", "]", "}") and out:
            out[-1] = out[-1] + t
        else:
            out.append(t if (not out) else " " + t)
    return "".join(out).strip()

# -------------------------
# EDA (WordNet baseline)
# -------------------------
def setup_wordnet():
    import nltk
    try:
        from nltk.corpus import wordnet
        _ = wordnet.synsets("test")
    except Exception:
        nltk.download("wordnet")
        nltk.download("omw-1.4")
    from nltk.corpus import wordnet
    return wordnet

WORDNET = None
STOP = set(["the","a","an","and","or","of","to","in","on","for","with","is","are","was","were","be","been","by","as","at","it","this","that"])

def random_insertion(tokens, wordnet, n_insert=1):
    """
    Insert synonyms of randomly chosen words into random positions.
    Operates on an already-tokenized list.
    """
    # candidate word indices: alphabetic, not stopword, not placeholder
    cand_idxs = [
        i for i, t in enumerate(tokens)
        if re.match(r"^[A-Za-z]+$", t) and t.lower() not in STOP and not t.startswith("ZZZ")
    ]
    if not cand_idxs:
        return tokens

    out = tokens[:]

    for _ in range(n_insert):
        # try a few times to find a word that has synonyms
        inserted = False
        for _try in range(10):
            i = random.choice(cand_idxs)
            syns = synonyms(out[i], wordnet)
            if syns:
                w = random.choice(syns)
                pos = random.randrange(len(out) + 1)  # insertion position
                out.insert(pos, w)
                inserted = True
                break
        if not inserted:
            # couldn't find a synonym to insert; leave as-is for this insertion
            pass

    return out


def synonyms(word: str, wordnet):
    syns = set()
    for s in wordnet.synsets(word):
        for l in s.lemmas():
            w = l.name().replace("_", " ")
            if w.lower() != word.lower():
                syns.add(w)
    return list(syns)

def eda_augment(text: str, alpha=EDA_ALPHA, del_prob=DEL_PROB):
    global WORDNET
    if WORDNET is None:
        WORDNET = setup_wordnet()

    tokens = tokenize(text)
    word_idxs = [i for i,t in enumerate(tokens)
                 if re.match(r"^[A-Za-z]+$", t) and t.lower() not in STOP and not t.startswith("ZZZ")]
    if not word_idxs:
        return text

    n = max(1, int(len(word_idxs) * alpha))
    aug = tokens[:]

    # synonym replacement
    random.shuffle(word_idxs)
    replaced = 0
    for i in word_idxs:
        if replaced >= n:
            break
        syns = synonyms(aug[i], WORDNET)
        if syns:
            aug[i] = random.choice(syns)
            replaced += 1

    # random deletion
    kept = []
    for t in aug:
        if re.match(r"^[A-Za-z]+$", t) and t.lower() not in STOP and random.random() < del_prob:
            continue
        kept.append(t)
    aug = kept if kept else tokens[:]

    # random insertion
    aug = random_insertion(aug, WORDNET, n_insert=n)
    
    # random swap
    for _ in range(n):
        if len(aug) < 4:
            break
        i, j = random.sample(range(len(aug)), 2)
        aug[i], aug[j] = aug[j], aug[i]

    return detokenize(aug)

# -------------------------
# BACK TRANSLATION (MarianMT)
# Requires: sentencepiece installed in THIS kernel env
# -------------------------
def get_marian_models(pivot="de"):
    from transformers import MarianMTModel, MarianTokenizer
    m1_name = f"Helsinki-NLP/opus-mt-en-{pivot}"
    m2_name = f"Helsinki-NLP/opus-mt-{pivot}-en"

    tok1 = MarianTokenizer.from_pretrained(m1_name)
    mod1 = MarianMTModel.from_pretrained(m1_name)
    tok2 = MarianTokenizer.from_pretrained(m2_name)
    mod2 = MarianMTModel.from_pretrained(m2_name)
    return (tok1, mod1, tok2, mod2)

MARIAN = None

def back_translate(text: str, pivot="de", device="cpu", max_len=256):
    global MARIAN
    if MARIAN is None:
        MARIAN = get_marian_models(pivot)

    tok1, mod1, tok2, mod2 = MARIAN
    mod1.to(device); mod2.to(device)

    def translate(t, tok, model):
        batch = tok([t], return_tensors="pt", truncation=True, max_length=max_len).to(device)
        gen = model.generate(**batch, max_length=max_len)
        return tok.batch_decode(gen, skip_special_tokens=True)[0]

    pivot_text = translate(text, tok1, mod1)
    back_text  = translate(pivot_text, tok2, mod2)
    return back_text

# -------------------------
# DATA IO (columnar JSON <-> row list)
# -------------------------
def load_columnar_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sent = data["sentence"]
    labs = data["labels"]
    docs = data["doc_title"]

    keys = sorted(sent.keys(), key=lambda x: int(x))
    rows = []
    for k in keys:
        rows.append({
            "sentence": sent[k],
            "labels": labs.get(k, []),
            "doc_title": docs.get(k, "")
        })
    return rows

def save_columnar_json(rows, path):
    out = {"sentence": {}, "labels": {}, "doc_title": {}}
    for i, r in enumerate(rows):
        k = str(i)
        out["sentence"][k] = r["sentence"]
        out["labels"][k] = r["labels"]
        out["doc_title"][k] = r["doc_title"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

# -------------------------
# 1) SPLIT TRAIN/VAL BY doc_title (prevents doc leakage)
# -------------------------
from sklearn.model_selection import train_test_split

# -------------------------
# 1) SPLIT TRAIN/VAL LIKE MAIN SCRIPT (row-level)
# -------------------------
rows = load_columnar_json(INPUT_JSON)

# If your main script uses stratify=df_original["label"], you must have a single label per row.
# If you don't, set USE_STRATIFY=False.
USE_STRATIFY = False  # set True only if each row has exactly one label column/value

if USE_STRATIFY:
    # build a single-label series for stratification
    # (adjust this depending on your schema: "label" vs "labels")
    y = []
    for r in rows:
        # if your JSON uses a list under "labels", pick the first (only if it's truly single-label)
        labs = r.get("labels", [])
        y.append(labs[0] if isinstance(labs, list) and len(labs) > 0 else None)

    # sklearn can't stratify with Nones; filter those out consistently
    keep_idx = [i for i, yi in enumerate(y) if yi is not None]
    rows_kept = [rows[i] for i in keep_idx]
    y_kept = [y[i] for i in keep_idx]

    train_rows, val_rows = train_test_split(
        rows_kept,
        test_size=VAL_FRAC,
        random_state=SEED,
        stratify=y_kept
    )

    # Optional: if you want to keep the dropped rows (yi is None) in train (like "no stratify" for them)
    dropped = [rows[i] for i in range(len(rows)) if i not in keep_idx]
    train_rows = list(train_rows) + dropped

else:
    train_rows, val_rows = train_test_split(
        rows,
        test_size=VAL_FRAC,
        random_state=SEED
    )

save_columnar_json(val_rows, VAL_OUT)

print(f"Split complete (row-level): train={len(train_rows)} rows, val={len(val_rows)} rows")


# -------------------------
# 2) AUGMENT ONLY TRAIN, ONLY IF LABEL STARTS WITH 'T1'
# -------------------------
def is_augment_eligible(labels):
    return any(isinstance(l, str) and l.startswith(AUG_LABEL_PREFIX) for l in (labels or []))

aug_train_rows = deepcopy(train_rows)
i=0
for r in train_rows:
    if not is_augment_eligible(r["labels"]):
        continue

    base = strip_markdown(r["sentence"])
    protected, ph_map = protect_tokens(base)
    
    # EDA
    for j in range(N_EDA):
        eda_txt = eda_augment(protected)
        eda_txt = restore_tokens(eda_txt, ph_map)
        aug_train_rows.append({
            "sentence": eda_txt,
            "labels": r["labels"],
            "doc_title": r["doc_title"] + f"__EDA{j}"
        })
        print(f"EDA augmented {i} -> {len(train_rows)}")
        i+=1

    # BT
    for j in range(N_BT):
        bt_txt = back_translate(protected, pivot=PIVOT_LANG, device="cpu")
        # If placeholders got altered/dropped, skip that output
        if any(k not in bt_txt for k in ph_map.keys()):
            continue
        bt_txt = restore_tokens(bt_txt, ph_map)
        aug_train_rows.append({
            "sentence": bt_txt,
            "labels": r["labels"],
            "doc_title": r["doc_title"] + f"__BT{j}"
        })

save_columnar_json(aug_train_rows, AUG_TRAIN_OUT)
print(f"Augmentation complete: train original={len(train_rows)} -> train+aug={len(aug_train_rows)}")
