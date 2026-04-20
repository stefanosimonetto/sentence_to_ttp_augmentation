import json
from transformers import (
    RobertaForSequenceClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
    BertForSequenceClassification
)
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sentence_transformers import models, SentenceTransformer
import pandas as pd
import numpy as np
from mitreattack.stix20 import MitreAttackData
import hashlib
from const import *
import random
from collections import Counter
import math
TAU = 0.5
PREFIXES = ("T1")

DATASET_LABEL_MAP = {
    ExpDataset.BOSCH_TACTICS: BOSCH_TACTICS_LABELS,
    ExpDataset.BOSCH_SOFTWARE: BOSCH_SOFTWARE_LABELS,
    ExpDataset.BOSCH_GROUPS: BOSCH_GROUP_LABELS,
    ExpDataset.BOSCH_TECHNIQUES_10: BOSCH_TECHNIQUES_10_LABELS,
    ExpDataset.BOSCH_TECHNIQUES_25: BOSCH_TECHNIQUES_25_LABELS,
    ExpDataset.BOSCH_TECHNIQUES_50: BOSCH_TECHNIQUES_50_LABELS,
    ExpDataset.BOSCH_TECHNIQUES_53: BOSCH_TECHNIQUES_53_LABELS,
    ExpDataset.BOSCH_TECHNIQUES: BOSCH_TECHNIQUES_LABELS,
    ExpDataset.BOSCH_AUGMENTED_MITRE: BOSCH_TECHNIQUES_LABELS,
    ExpDataset.BOSCH_AUGMENTED_HIERARCHY: BOSCH_TECHNIQUES_LABELS,
    ExpDataset.BOSCH_AUGMENTED_HIERARCHY_EMBED: BOSCH_TECHNIQUES_LABELS,
    ExpDataset.BOSCH_CTI_SENT_ONLY: BOSCH_TECHNIQUES_LABELS,
    ExpDataset.BOSCH_CTI_SENT: BOSCH_TECHNIQUES_LABELS,
    ExpDataset.TRAM_TECHNIQUES: TRAM_TECHNIQUES_LABELS,
    ExpDataset.TRAM_TECHNIQUES_10: TRAM_TECHNIQUES_10_LABELS,
    ExpDataset.TRAM_TECHNIQUES_25: TRAM_TECHNIQUES_25_LABELS,
    ExpDataset.TRAM_TECHNIQUES_SL: TRAM_TECHNIQUES_SL_LABELS,
    ExpDataset.BOSCH_TECHNIQUES_SL: BOSCH_TECHNIQUES_SL_LABELS,
    ExpDataset.BOSCH_ALL_IN: BOSCH_TECHNIQUES_LABELS,
    ExpDataset.TRAM_ALL_IN: TRAM_TECHNIQUES_LABELS,
    ExpDataset.TRAM_AUGMENTED_MITRE: TRAM_TECHNIQUES_LABELS,
    ExpDataset.TRAM_AUGMENTED_OOD: TRAM_TECHNIQUES_LABELS,
    ExpDataset.TRAM_AUGMENTED_HIERARCHY: TRAM_TECHNIQUES_LABELS,
    ExpDataset.TRAM_AUGMENTED_HIERARCHY_EMBED: TRAM_TECHNIQUES_LABELS,
    ExpDataset.TRAM_CTI_SENT: TRAM_TECHNIQUES_LABELS,
    ExpDataset.BOSCH_PREPROCESSES: BOSCH_TECHNIQUES_LABELS,
    ExpDataset.TRAM_PREPROCESSES: TRAM_TECHNIQUES_LABELS,
    ExpDataset.TRAM_AUGMENTED_TILL_MAX: TRAM_TECHNIQUES_LABELS,
    ExpDataset.TRAM_AUGMENTED_CONSISTENTLY: TRAM_TECHNIQUES_LABELS,
    ExpDataset.TRAM_AUGMENTED_PERFORMANCE_DRIVEN: TRAM_TECHNIQUES_LABELS,
    ExpDataset.TRAM_CTI_SENT_ONLY: TRAM_TECHNIQUES_LABELS,
    ExpDataset.BOSCH_AUGMENTED_TILL_MAX: BOSCH_TECHNIQUES_LABELS,
    ExpDataset.BOSCH_AUGMENTED_CONSISTENTLY: BOSCH_TECHNIQUES_LABELS,
    ExpDataset.BOSCH_AUGMENTED_PERFORMANCE_DRIVEN: BOSCH_TECHNIQUES_LABELS,
    ExpDataset.BOSCH_TTP_HUNTER: BOSCH_TECHNIQUES_LABELS,
    ExpDataset.TRAM_TTP_HUNTER: TRAM_TECHNIQUES_LABELS,
    ExpDataset.BOSCH_EDA: BOSCH_TECHNIQUES_LABELS,
    ExpDataset.TRAM_EDA: TRAM_TECHNIQUES_LABELS,
    ExpDataset.BOSCH_ARTIFICIAL: BOSCH_TECHNIQUES_LABELS,
    ExpDataset.TRAM_ARTIFICIAL: TRAM_TECHNIQUES_LABELS
}


class UnsupportedModel(Exception):
    pass


class MultiLabelTTPSentenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, labels, max_len=MAX_LEN, threshold=TAU):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.threshold = threshold
        self.labels = labels
        self.lb = LabelBinarizer()
        self.lb.fit(self.labels)

    def __getitem__(self, index):
        sentence = self.data.iloc[index].sentence
        labels = self.data.iloc[index].labels
        encoding = self.tokenizer(
            sentence,
            # return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )

        labels = labels if len(labels) > 0 else ["NO_TTP"]
        encoded_labels = np.clip(self.lb.transform(labels).sum(axis=0), 0.0, 1.0)
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item["labels"] = torch.as_tensor(encoded_labels)
        return item

    def __len__(self):
        return self.len


class SingleLabelTTPSentenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, labels, max_len=MAX_LEN, threshold=TAU):
        self.len = len(dataframe)
        self.data = dataframe.explode("labels").reset_index(drop=True)
        # self.data = self.data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.threshold = threshold
        self.labels = labels
        self.lb = LabelBinarizer()
        self.lb.fit(self.labels)

    def __getitem__(self, index):
        sentence = self.data.iloc[index].sentence
        labels = self.data.iloc[index].labels
        encoding = self.tokenizer(
            sentence,
            # return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )
        labels = [labels] if labels in self.labels else ["NO_TTP"]
        encoded_labels = np.clip(self.lb.transform(labels).sum(axis=0), 0, 1)
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item["labels"] = torch.as_tensor(encoded_labels)
        return item

    def __len__(self):
        return self.len


class BoschAllDataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, BOSCH_ALL_LABELS)


class BoschTechniquesDataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, BOSCH_TECHNIQUES_LABELS)


class Bosch10TechniquesDataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, BOSCH_TECHNIQUES_10_LABELS)


class Bosch25TechniquesDataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, BOSCH_TECHNIQUES_25_LABELS)


class Bosch50TechniquesDataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, BOSCH_TECHNIQUES_50_LABELS)


class Bosch53TechniquesDataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, BOSCH_TECHNIQUES_53_LABELS)


class BoschTacticsDataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, BOSCH_TACTICS_LABELS)


class BoschGroupsDataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, BOSCH_GROUP_LABELS)


class BoschSoftwareDataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, BOSCH_SOFTWARE_LABELS)


class TramDataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, TRAM_TECHNIQUES_LABELS)

class BoschDataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, BOSCH_TECHNIQUES_LABELS)

class Tram10Dataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, TRAM_TECHNIQUES_10_LABELS)

class Tram25Dataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, TRAM_TECHNIQUES_25_LABELS)


class BoschTechniquesDatasetSL(SingleLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, BOSCH_TECHNIQUES_SL_LABELS)


class TramDatasetSL(SingleLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, TRAM_TECHNIQUES_SL_LABELS)


def available_models():
    print("Models",MODELS, "MODEL_SENTENCE_SIM",MODEL_SENTENCE_SIM)
    return MODELS + MODEL_SENTENCE_SIM


def model_name_to_folder_name(model_name):
    return model_name.replace("/", "-")



def ensure_labels_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures df has a 'labels' column where each row is a list of labels.
    Supports common read_json shapes:
    - record style: df has 'labels' column already
    - dict style: json is {"labels": {id: [...]}} resulting df where 'labels' may be in index/columns
    """
    if "labels" in df.columns:
        # normalize: ensure list
        df = df.copy()
        df["labels"] = df["labels"].apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else list(x)))
        return df

    # Handle dict-style JSON: {"labels": {id: [...]}}
    # pd.read_json often yields a DF where the key "labels" becomes a column OR an index entry
    if "labels" in df.index:
        # df.loc["labels"] is a Series (columns -> values) containing the dict, sometimes nested
        row = df.loc["labels"]
        # if it's a Series of length 1 containing the dict
        if isinstance(row, pd.Series) and len(row) == 1 and isinstance(row.iloc[0], dict):
            labels_dict = row.iloc[0]
        elif isinstance(row, dict):
            labels_dict = row
        else:
            # sometimes row is a Series where each element is itself dict-ish; try to merge
            # best-effort fallback
            possible = row.to_dict()
            # unwrap one layer if needed
            if len(possible) == 1 and isinstance(next(iter(possible.values())), dict):
                labels_dict = next(iter(possible.values()))
            else:
                labels_dict = possible

        out = pd.DataFrame({
            "doc_id": list(labels_dict.keys()),
            "labels": list(labels_dict.values())
        })
        out["labels"] = out["labels"].apply(lambda x: x if isinstance(x, list) else list(x))
        return out

    raise ValueError("Could not locate labels. Check df.columns and df.index after read_json().")

def filter_labels_df(df: pd.DataFrame, prefixes=PREFIXES) -> pd.DataFrame:
    df = df.copy()
    df["labels"] = df["labels"].apply(
        lambda labs: [l for l in labs if isinstance(l, str) and l.startswith(prefixes)]
    )
    return df

def count_labels_df(df: pd.DataFrame) -> Counter:
    c = Counter()
    for labs in df["labels"]:
        c.update(labs)
    return c

def filter_labels_list(labels, prefixes=PREFIXES):
    if not isinstance(labels, list):
        return []
    return [l for l in labels if isinstance(l, str) and l.startswith(prefixes)]

def count_labels_from_df(df, label_col="labels"):
    c = Counter()
    for labs in df[label_col]:
        if isinstance(labs, list):
            c.update(labs)
    return c

def augment_till_max(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    prefixes=PREFIXES,
    seed = None,
) -> tuple[pd.DataFrame, Counter]:
    """
    For each label in df2, if df1 count < max_count_df1, sample rows from df2 that contain that label
    and append them to df1 until that label surpasses max_count_df1 (or df2 runs out).
    Returns augmented df1 and final counts.
    """
    rng = random.Random(seed)

    df1 = filter_labels_df(df1, prefixes)
    df2 = filter_labels_df(df2, prefixes)

    counts1 = count_labels_df(df1)
    counts2 = count_labels_df(df2)

    if not counts1:
        raise ValueError("df1 has no labels after filtering. Check prefixes or data format.")

    max_count_df1 = max(counts1.values())

    # Precompute which rows in df2 contain each label (fast lookup)
    # Build a mapping label -> list of row indices in df2 that contain it
    label_to_indices = {}
    for idx, labs in df2["labels"].items():
        for l in labs:
            label_to_indices.setdefault(l, []).append(idx)

    rows_to_add = []

    for label in counts2.keys():
        current = counts1.get(label, 0)
        if current >= max_count_df1:
            continue

        needed = (max_count_df1 - current)
        candidate_indices = label_to_indices.get(label, [])
        if not candidate_indices:
            continue

        # sample without replacement
        k = min(needed, len(candidate_indices))
        sampled = rng.sample(candidate_indices, k)
        rows_to_add.extend(sampled)

        # Update counts1 incrementally to reduce oversampling across labels
        for idx in sampled:
            counts1.update(df2.at[idx, "labels"])

        # Update max_count_df1? In your original logic you kept it fixed as original max.
        # We'll keep it fixed to match your previous behavior.

    if rows_to_add:
        df_added = df2.loc[rows_to_add].copy()
        df_aug = pd.concat([df1, df_added], ignore_index=True)
    else:
        df_aug = df1.copy()

    final_counts = count_labels_df(df_aug)
    return df_aug, final_counts


def augment_add_fraction_of_max(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    add_fraction_of_max: float = 0.40,
    prefixes=PREFIXES,
    seed= None,
    max_passes: int = 10,
) -> tuple[pd.DataFrame, Counter]:
    """
    For every label present in df1 (after filtering), try to ADD at least:
        delta = ceil(add_fraction_of_max * max_count_df1)
    more occurrences, by sampling rows from df2 (without replacement).

    Multi-label-aware:
    - A sampled row can contribute to multiple labels at once.
    - Each df2 row is added at most once globally (no duplicates).

    "When possible":
    - If df2 doesn't contain enough candidate rows for some label, we add as many as possible.

    Strategy:
    - Precompute label -> candidate df2 indices.
    - Maintain remaining_needed[label] = delta (how many more occurrences we want to add for that label).
    - Greedy multi-pass sampling: repeatedly prioritize labels with the highest remaining need.

    Returns:
      (augmented_df, final_label_counts)
    """
    if not (0 < add_fraction_of_max <= 1.0):
        raise ValueError("add_fraction_of_max must be in (0, 1].")

    rng = random.Random(seed)

    # Filter labels
    df1 = filter_labels_df(df1, prefixes)
    df2 = filter_labels_df(df2, prefixes)

    # Counts in df1
    counts1 = count_labels_df(df1)
    if not counts1:
        raise ValueError("df1 has no labels after filtering. Check prefixes or data format.")

    max_count_df1 = max(counts1.values())
    delta = int(math.ceil(max_count_df1 * add_fraction_of_max))

    # Labels we will try to boost (labels already present in df1)
    labels_to_boost = list(counts1.keys())

    # remaining_needed: we want +delta occurrences for each label (not a final target)
    remaining_needed = {lbl: delta for lbl in labels_to_boost}

    # Precompute label -> list of df2 row indices containing it
    label_to_indices: dict[str, list] = {}
    for idx, labs in df2["labels"].items():
        for l in labs:
            if l in remaining_needed:  # only store for labels we care about
                label_to_indices.setdefault(l, []).append(idx)

    used_indices: set = set()
    rows_to_add: list = []

    # Helper: apply a chosen df2 row, decreasing needs for any labels it contains
    def apply_row(idx):
        labs = df2.at[idx, "labels"]
        for l in labs:
            if l in remaining_needed and remaining_needed[l] > 0:
                remaining_needed[l] -= 1

    # Multi-pass greedy:
    # each pass: iterate labels in order of most remaining need, pick rows for them
    for _pass in range(max_passes):
        progress = 0

        # labels still needing boosts, sorted by remaining need desc
        active_labels = [l for l, need in remaining_needed.items() if need > 0]
        if not active_labels:
            break

        active_labels.sort(key=lambda l: remaining_needed[l], reverse=True)

        for label in active_labels:
            need = remaining_needed[label]
            if need <= 0:
                continue

            candidates = label_to_indices.get(label, [])
            if not candidates:
                continue

            # only unused candidates
            available = [i for i in candidates if i not in used_indices]
            if not available:
                continue

            # sample up to "need" rows for this label
            k = min(need, len(available))
            sampled = rng.sample(available, k)

            for idx in sampled:
                used_indices.add(idx)
                rows_to_add.append(idx)
                apply_row(idx)
                progress += 1

        # If we couldn't add anything in this pass, no further progress is possible
        if progress == 0:
            break

    if rows_to_add:
        df_added = df2.loc[rows_to_add].copy()
        df_aug = pd.concat([df1, df_added], ignore_index=True)
    else:
        df_aug = df1.copy()

    final_counts = count_labels_df(df_aug)
    return df_aug, final_counts



def augment_performance_driven(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    perf_df: pd.DataFrame,
    prefixes=PREFIXES,
    seed=None,
    # ---- how many rows to add overall ----
    total_budget=None,   # if None -> computed from fraction_of_df1
    fraction_of_df1: float = 0.30,     # used when total_budget=None
    # ---- what "weakness" means ----
    score_mode: str = "fnr",           # "fnr", "low_f1", "low_recall", "combo"
    f1_col: str = "f1",
    # ---- guards ----
    min_support: int = 0,              # ignore labels with (TP+FN) < min_support in perf_df
    max_per_label_add= None,  # cap to avoid runaway
) -> tuple[pd.DataFrame, Counter, pd.DataFrame]:
    """
    Performance-driven augmentation: allocate more df2 samples to labels where the model is weak.

    perf_df must have columns: ["label","TP","FP","FN","TN"].
    Optionally, if score_mode="low_f1", it should have an f1 column (or provide f1_col).

    Parameters
    ----------
    total_budget:
        Total number of df2 rows to add. If None, uses fraction_of_df1 * len(df1).
    fraction_of_df1:
        Used only if total_budget=None.
    score_mode:
        - "fnr": prioritize high false-negative-rate (FN/(TP+FN))  => recall weakness
        - "low_recall": same as fnr
        - "low_f1": prioritize low F1
        - "combo": 0.6*FNR + 0.4*(1-F1) (requires f1)
    max_per_label_add:
        Optional cap per label (in terms of how many rows we attempt to sample for that label).
    Returns
    -------
    df_aug, final_counts, allocation_df
    """
    rng = random.Random(seed)

    # Filter labels
    df1 = filter_labels_df(df1, prefixes)
    df2 = filter_labels_df(df2, prefixes)

    counts1 = count_labels_df(df1)
    if not counts1:
        raise ValueError("df1 has no labels after filtering. Check prefixes or data format.")

    # Budget
    if total_budget is None:
        total_budget = int(math.ceil(fraction_of_df1 * len(df1)))
    total_budget = max(0, int(total_budget))

    # --- build label -> candidate indices in df2 ---
    label_to_indices: dict[str, list] = {}
    for idx, labs in df2["labels"].items():
        for l in labs:
            label_to_indices.setdefault(l, []).append(idx)

    # --- compute weakness score per label from perf_df ---
    pdf = perf_df.copy()

    # Ensure required columns
    for col in ["label", "TP", "FP", "FN", "TN"]:
        if col not in pdf.columns:
            raise ValueError(f"perf_df missing required column: {col}")

    # Support = positives in truth
    pdf["support"] = pdf["TP"] + pdf["FN"]

    # Ignore tiny support labels (optional)
    pdf = pdf[pdf["support"] >= min_support].copy()

    # Compute recall + FNR
    pdf["recall"] = pdf["TP"] / (pdf["TP"] + pdf["FN"] + 1e-12)
    pdf["fnr"]    = pdf["FN"] / (pdf["TP"] + pdf["FN"] + 1e-12)

    if score_mode in ("low_f1", "combo") and f1_col not in pdf.columns:
        raise ValueError(f"score_mode='{score_mode}' requires perf_df column '{f1_col}'")

    if score_mode in ("fnr", "low_recall"):
        pdf["score"] = pdf["fnr"]
    elif score_mode == "low_f1":
        pdf["score"] = 1.0 - pdf[f1_col].clip(0, 1)
    elif score_mode == "combo":
        pdf["score"] = 0.6 * pdf["fnr"] + 0.4 * (1.0 - pdf[f1_col].clip(0, 1))
    else:
        raise ValueError("score_mode must be one of: 'fnr','low_recall','low_f1','combo'")

    # Only augment labels that exist in df1 (common practice)
    pdf = pdf[pdf["label"].isin(counts1.keys())].copy()

    # If nothing to do
    if pdf.empty or total_budget == 0:
        final_counts = count_labels_df(df1)
        alloc_df = pdf.assign(allocated=0, available=0)
        return df1.copy(), final_counts, alloc_df

    # Make sure we only allocate to labels that have candidates in df2
    pdf["available"] = pdf["label"].apply(lambda l: len(label_to_indices.get(l, [])))
    pdf = pdf[pdf["available"] > 0].copy()

    if pdf.empty:
        final_counts = count_labels_df(df1)
        alloc_df = perf_df.copy()
        alloc_df["allocated"] = 0
        alloc_df["available"] = 0
        return df1.copy(), final_counts, alloc_df

    # Normalize scores into allocation weights
    pdf["score"] = pdf["score"].clip(lower=0)
    score_sum = float(pdf["score"].sum())

    if score_sum == 0.0:
        # All scores are zero -> distribute uniformly across candidate labels
        pdf["weight"] = 1.0 / len(pdf)
    else:
        pdf["weight"] = pdf["score"] / score_sum

    # Initial allocation (rounded)
    pdf["allocated"] = (pdf["weight"] * total_budget).round().astype(int)

    # Ensure at least 1 for labels with nonzero score (optional)
    # pdf.loc[pdf["score"] > 0, "allocated"] = pdf.loc[pdf["score"] > 0, "allocated"].clip(lower=1)

    # Apply caps: availability and max_per_label_add
    if max_per_label_add is not None:
        pdf["allocated"] = pdf["allocated"].clip(upper=int(max_per_label_add))
    pdf["allocated"] = pdf[["allocated", "available"]].min(axis=1).astype(int)

    # Fix rounding drift to match total_budget as best as possible
    allocated_sum = int(pdf["allocated"].sum())
    remaining = total_budget - allocated_sum

    if remaining > 0:
        # Greedily give extra to highest-score labels with remaining availability
        pdf = pdf.sort_values("score", ascending=False).copy()
        i = 0
        while remaining > 0 and i < len(pdf):
            cap = pdf.iloc[i]["available"] - pdf.iloc[i]["allocated"]
            if max_per_label_add is not None:
                cap = min(cap, int(max_per_label_add) - int(pdf.iloc[i]["allocated"]))
            if cap > 0:
                add = min(int(cap), remaining)
                pdf.iat[i, pdf.columns.get_loc("allocated")] += add
                remaining -= add
            i += 1

    # --- sample rows according to allocation ---
    used_indices: set = set()
    rows_to_add: list = []

    # Iterate labels from highest score down (so we “spend” budget on worst labels first)
    pdf = pdf.sort_values("score", ascending=False).copy()

    for _, row in pdf.iterrows():
        label = row["label"]
        need = int(row["allocated"])
        if need <= 0:
            continue

        candidates = label_to_indices.get(label, [])
        if not candidates:
            continue

        # only unused
        available = [idx for idx in candidates if idx not in used_indices]
        if not available:
            continue

        k = min(need, len(available))
        sampled = rng.sample(available, k)

        rows_to_add.extend(sampled)
        used_indices.update(sampled)

        # Update counts1 so multi-label rows reduce future oversampling effects
        for idx in sampled:
            counts1.update(df2.at[idx, "labels"])

    # Build augmented dataframe
    if rows_to_add:
        df_added = df2.loc[rows_to_add].copy()
        df_aug = pd.concat([df1, df_added], ignore_index=True)
    else:
        df_aug = df1.copy()

    final_counts = count_labels_df(df_aug)
    allocation_df = pdf[["label", "TP", "FP", "FN", "TN", "support", "recall", "fnr", "score", "available", "allocated"]].copy()

    return df_aug, final_counts, allocation_df

def load_model_for_embedding(model_name):
    if model_name not in available_models():
        raise UnsupportedModel

    print(f"Loading model: {model_name} ...")

    if model_name == "s2w-ai/DarkBERT":
        with open(".darkbert_token", "r") as f:
            token = f.readline().strip()
        transformer = models.Transformer(model_name, model_args={"token": token})
    elif model_name == "priyankaranade/cybert":
        transformer = models.Transformer("local/CyBERT-Base-MLM-v1.1")
    else:
        transformer = models.Transformer(model_name)

    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(), pooling_mode="mean"
    )
    model = SentenceTransformer(modules=[transformer, pooling])
    return model


def load_untrained_model(model_name, dataset_name, multi_label=True):
    dataset = ExpDataset(dataset_name)
    labels = DATASET_LABEL_MAP[dataset]
    num_labels = len(labels)
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}

    if model_name not in available_models() + ["scibert_multi_label_model"]:
        raise UnsupportedModel

    if multi_label == True:
        kwargs = {"problem_type": "multi_label_classification"}
    else:
        kwargs = {"problem_type": "single_label_classification"}

    print(f"Loading model: {model_name} ...")

    if model_name == "s2w-ai/DarkBERT":
        with open(".darkbert_token", "r") as f:
            token = f.readline().strip()
        return RobertaForSequenceClassification.from_pretrained(
            model_name,
            token=token,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            **kwargs,
        ), AutoTokenizer.from_pretrained(model_name, token=token)
    elif model_name == "priyankaranade/cybert":
        return AutoModelForSequenceClassification.from_pretrained(
            "local/CyBERT-Base-MLM-v1.1",
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        ), AutoTokenizer.from_pretrained("local/CyBERT-Base-MLM-v1.1")
    elif model_name == "tram_multi_label_model":
        tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", max_length=512)
        bert = BertForSequenceClassification.from_pretrained(
            "local/tram_finetuned",
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        return bert, tokenizer
    elif model_name == "scibert_multi_label_model":
        tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", max_length=512)
        bert = BertForSequenceClassification.from_pretrained("local/scibert_multi_label_model")
        return bert, tokenizer
    else:
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            **kwargs,
        ), AutoTokenizer.from_pretrained(model_name)


def load_finetuned_model(model_path):
    return AutoModelForSequenceClassification.from_pretrained(
        model_path
    ), AutoTokenizer.from_pretrained(model_path)


def load_datasets_for_tuning_embedding_threshold(dataset_name):
    dataset_name = ExpDataset(dataset_name)
    if dataset_name == ExpDataset.BOSCH_TECHNIQUES:
        df = pd.read_json("datasets/bosch_train.json")
    elif dataset_name == ExpDataset.TRAM_TECHNIQUES:
        df = pd.read_json("datasets/tram_train.json")
    else:
        print("[!] dataset not found")
        exit()
    print("Dataset: {}".format(df.shape))
    return df


def load_datasets_for_testing_embedding_threshold(dataset_name):
    dataset_name = ExpDataset(dataset_name)
    if dataset_name == ExpDataset.BOSCH_TECHNIQUES:
        df = pd.read_json("datasets/bosch_test.json")
    elif dataset_name == ExpDataset.TRAM_TECHNIQUES:
        df = pd.read_json("datasets/tram_test.json")
    else:
        print("[!] dataset not found")
        exit()
    print("Dataset: {}".format(df.shape))
    return df

def text_hash(s: str) -> str:
    return hashlib.sha256(
        s.strip().lower().encode("utf-8")
    ).hexdigest()

def load_datasets_for_finetuning_sentence_model():
    mitre_attack_data = MitreAttackData("datasets/enterprise-attack.json")
    techniques = mitre_attack_data.get_techniques(remove_revoked_deprecated=True)
    print(f"Retrieved {len(techniques)} ATT&CK techniques ...")
    ttps = []
    for t in techniques:
        technique_id = [
            e for e in t.external_references if e.source_name == "mitre-attack"
        ][0].external_id
        content = f"{t.name}:\n{t.description}"
        ttps.append({"id": technique_id, "sentence": content})

    ttps = sorted(ttps, key=lambda x: x["id"])
    df = pd.concat([pd.read_json("datasets/tram_train.json").rename(columns={"doc_title": "document"}), pd.read_json("datasets/bosch_train.json")]).reset_index(drop=True)
    return ttps

# Function to load data from a given file path
def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


# Function to filter labels that start with 'T1' or 'TA'
def filter_labels(data, prefix=('T1')):
    filtered_labels = {}
    for key, item in data.get("labels", {}).items():
        filtered_labels[key] = [label for label in item if label.startswith(prefix) ]
    return filtered_labels


# Function to count labels in the dataset
def count_labels(data, prefix=('T1')):
    label_counts = Counter()
    for item in data.get("labels", {}).values():
        for label in item:
            if label.startswith(prefix):
                label_counts.update([label])
    return label_counts


# Function to augment labels from data2 into data1
def augment_data(data1, data2, label_counts_data1, label_counts_data2, max_count_data1, prefix=('T1')):
    labels_data1 = data1.get("labels", [])
    labels_data2 = data2.get("labels", [])

    if not isinstance(labels_data1, list):
        labels_data1 = list(labels_data1.values())  # Convert from dictionary if necessary
    if not isinstance(labels_data2, list):
        labels_data2 = list(labels_data2.values())  # Convert from dictionary if necessary

    for label in label_counts_data2:
        current_count = label_counts_data1.get(label, 0)
        augmentation_count = label_counts_data2.get(label, 0)

        # Only augment if the current count is less than the max count in data1
        if current_count < max_count_data1:
            # Calculate how much we need to surpass the largest class
            needed = max_count_data1 - current_count + 1  # Ensure surpassing the max count

            # Get all augmented items for this label, filtered by the prefix
            augmented_items = [item for item in labels_data2 if label in item]

            # Avoid sampling more than available
            needed = min(needed, len(augmented_items))  # Ensure we don't sample more than available

            # Add the required samples to labels_data1
            labels_data1.extend(random.sample(augmented_items, needed))

    return labels_data1

def load_augmented(dataset_name, batch_size, tokenizer, test_size=0.2, random_state=0, per_document=False):
    PREFIXES = ("T1")
    if dataset_name == ExpDataset.TRAM_ARTIFICIAL:
        df_train_augm = pd.read_json("datasets/tram_only_augmented_artificial_columnar.json")
        df_original = pd.read_json("datasets/tram_train_no_val_overlap_hmcat.json")
        df_val = pd.read_json("datasets/tram_split_val_hmcat.json")
        #the duplicates between train and val are dropped later
        df_train = pd.concat([df_original, df_train_augm]) 
        df_test = pd.read_json("datasets/tram_test.json")

    elif dataset_name == ExpDataset.BOSCH_ARTIFICIAL:
        df_train_augm = pd.read_json("datasets/bosch_only_augmented_artificial_columnar.json")
        df_original = pd.read_json("datasets/bosch_train_no_val_overlap_hmcat.json")
        df_val = pd.read_json("datasets/bosch_split_val_hmcat.json")
        #the duplicates between train and val are dropped later
        df_train = pd.concat([df_original, df_train_augm]) 
        df_test = pd.read_json("datasets/bosch_test.json")

    elif dataset_name == ExpDataset.TRAM_EDA:
        df_train = pd.read_json("datasets/tram_split_train_augmented_eda_bt_T1.json")
        df_val = pd.read_json("datasets/tram_split_val_eda.json")
        #the duplicates between train and val are dropped later
        df_test = pd.read_json("datasets/tram_test.json")

    elif dataset_name == ExpDataset.BOSCH_EDA:
        df_train = pd.read_json("datasets/bosch_split_train_augmented_eda_bt_T1.json")
        df_val = pd.read_json("datasets/bosch_split_val_eda.json")
        #the duplicates between train and val are dropped later
        df_test = pd.read_json("datasets/bosch_test.json")

    elif dataset_name == ExpDataset.TRAM_TTP_HUNTER:
        df_train = pd.read_json("datasets/tram_train_ttp_hunter.json")
        df_val = pd.read_json("datasets/tram_split_val_mainstyle.json")
        #the duplicates between train and val are dropped later
        df_test = pd.read_json("datasets/tram_test.json")

    elif dataset_name == ExpDataset.BOSCH_TTP_HUNTER:
        df_train = pd.read_json("datasets/bosch_train_ttp_hunter.json")
        df_val = pd.read_json("datasets/bosch_split_val_mainstyle.json")
        #the duplicates between train and val are dropped later
        df_test = pd.read_json("datasets/bosch_test.json")

    elif dataset_name == ExpDataset.TRAM_AUGMENTED_MITRE:
        # df_train_augm = pd.read_json("datasets/tram_train_augmented_mitre.json")  
        df_train_augm = pd.read_json("datasets/tram_augmented_hierarchy_embeddings.json") 
        df_original = pd.read_json("datasets/tram_train.json")
        df_train_orig, df_val = train_test_split(df_original, test_size=test_size, random_state=random_state)
        df_train = pd.concat([df_train_orig, df_train_augm]) 
        df_test = pd.read_json("datasets/tram_test.json") 
    
    elif dataset_name == ExpDataset.TRAM_CTI_SENT_ONLY:
        df_train_augm = pd.read_json("/home/simonettos/thijs/data_augmentatio_stefano/combined_6th_tram.json")   
        df_original = pd.read_json("datasets/tram_train.json")
        df_train_orig, df_val = train_test_split(df_original, test_size=test_size, random_state=random_state)
        data1 = ensure_labels_column(df_train_orig)
        data2 = ensure_labels_column(df_train_augm)
        df_train = pd.concat([data1, data2]) 
        df_test = pd.read_json("datasets/tram_test.json") 

    elif dataset_name == ExpDataset.TRAM_AUGMENTED_TILL_MAX:
        # df_train_augm = pd.read_json("datasets/tram_train_augmented_mitre.json")  
        df_train_augm = pd.read_json("/home/simonettos/thijs/data_augmentatio_stefano/combined_6th_tram.json")  
        df_original = pd.read_json("datasets/tram_train.json")
        df_train_orig, df_val = train_test_split(df_original, test_size=test_size, random_state=random_state)
        data1 = ensure_labels_column(df_train_orig)
        data2 = ensure_labels_column(df_train_augm)
        aug_df, final_label_counts = augment_till_max(data1, data2, prefixes=PREFIXES, seed=None)
        df_train = aug_df
        df_test = pd.read_json("datasets/tram_test.json") 

    elif dataset_name == ExpDataset.TRAM_AUGMENTED_CONSISTENTLY:
        # df_train_augm = pd.read_json("datasets/tram_train_augmented_mitre.json")  
        add_fraction_of_max = 0.40 
        df_train_augm = pd.read_json("/home/simonettos/thijs/data_augmentatio_stefano/combined_6th_tram.json")  
        df_original = pd.read_json("datasets/tram_train.json")
        df_train_orig, df_val = train_test_split(df_original, test_size=test_size, random_state=random_state)
        data1 = ensure_labels_column(df_train_orig)
        data2 = ensure_labels_column(df_train_augm)
        aug_df, final_label_counts = augment_add_fraction_of_max(data1, data2, add_fraction_of_max=add_fraction_of_max, prefixes=PREFIXES, seed=None)
        df_train = aug_df
        df_test = pd.read_json("datasets/tram_test.json") 

    elif dataset_name == ExpDataset.TRAM_AUGMENTED_PERFORMANCE_DRIVEN:
        # df_train_augm = pd.read_json("datasets/tram_train_augmented_mitre.json")   
        df_train_augm = pd.read_json("/home/simonettos/thijs/data_augmentatio_stefano/combined_6th_tram.json")  
        df_original = pd.read_json("datasets/tram_train.json")
        df_train_orig, df_val = train_test_split(df_original, test_size=test_size, random_state=random_state)
        data1 = ensure_labels_column(df_train_orig)
        data2 = ensure_labels_column(df_train_augm)
        cm_all_df = pd.read_csv("/home/simonettos/thijs/classification/classification/extra/all_docs_cm_tram.csv")
        perf_df = (cm_all_df.groupby("label")[["TP", "FP", "FN", "TN"]].sum().reset_index())
        eps = 1e-12
        perf_df["precision"] = perf_df["TP"] / (perf_df["TP"] + perf_df["FP"] + eps)
        perf_df["recall"]    = perf_df["TP"] / (perf_df["TP"] + perf_df["FN"] + eps)
        perf_df["f1"]        = 2 * perf_df["precision"] * perf_df["recall"] / (perf_df["precision"] + perf_df["recall"] + eps)
        print("Worst by recall:")
        print(perf_df.sort_values("recall").head(50)[["label","TP","FP","FN","precision","recall","f1"]])
        print("\nWorst by f1:")
        print(perf_df.sort_values("f1").head(50)[["label","TP","FP","FN","precision","recall","f1"]])

        fraction_of_df1 = 0.40
        total_budget = int(math.ceil(fraction_of_df1 * len(data1)))

        aug_df, final_counts, allocation_df = augment_performance_driven(
            df1=data1,
            df2=data2,
            perf_df=perf_df,          # computed on held-out validation (or CV) split
            prefixes=PREFIXES,
            seed=None,
            total_budget=total_budget,
            score_mode="fnr",       
            min_support=1,           # stabilize per-label weakness estimates
            max_per_label_add=math.ceil(0.4 * total_budget),  # cap by budget share
        )
        df_train = aug_df
        df_test = pd.read_json("datasets/tram_test.json") 

    elif dataset_name == ExpDataset.BOSCH_AUGMENTED_TILL_MAX:
        # df_train_augm = pd.read_json("datasets/bosch_train_augmented_mitre.json")  
        df_train_augm = pd.read_json("/home/simonettos/thijs/data_augmentatio_stefano/combined_6th_bosch.json")   
        df_original = pd.read_json("datasets/bosch_train.json")
        df_train_orig, df_val = train_test_split(df_original, test_size=test_size, random_state=random_state)
        data1 = ensure_labels_column(df_train_orig)
        data2 = ensure_labels_column(df_train_augm)
        aug_df, final_label_counts = augment_till_max(data1, data2, prefixes=PREFIXES, seed=None)
        df_train = aug_df
        df_test = pd.read_json("datasets/bosch_test.json") 

    elif dataset_name == ExpDataset.BOSCH_AUGMENTED_CONSISTENTLY:
        # df_train_augm = pd.read_json("datasets/bosch_train_augmented_mitre.json") 
        add_fraction_of_max = 0.30
        df_train_augm = pd.read_json("/home/simonettos/thijs/data_augmentatio_stefano/combined_6th_bosch.json")  
        df_original = pd.read_json("datasets/bosch_train.json")
        df_train_orig, df_val = train_test_split(df_original, test_size=test_size, random_state=random_state)
        data1 = ensure_labels_column(df_train_orig)
        data2 = ensure_labels_column(df_train_augm)
        aug_df, final_label_counts = augment_add_fraction_of_max(data1, data2, add_fraction_of_max=add_fraction_of_max, prefixes=PREFIXES, seed=None)
        df_train = aug_df
        df_test = pd.read_json("datasets/bosch_test.json") 

    elif dataset_name == ExpDataset.BOSCH_AUGMENTED_PERFORMANCE_DRIVEN:
        # df_train_augm = pd.read_json("datasets/bosch_train_augmented_mitre.json") 
        df_train_augm = pd.read_json("/home/simonettos/thijs/data_augmentatio_stefano/combined_6th_bosch.json")  
        df_original = pd.read_json("datasets/bosch_train.json")
        df_train_orig, df_val = train_test_split(df_original, test_size=test_size, random_state=random_state)
        data1 = ensure_labels_column(df_train_orig)
        data2 = ensure_labels_column(df_train_augm)
        cm_all_df = pd.read_csv("/home/simonettos/thijs/classification/classification/extra/all_docs_cm_bosch.csv")
        perf_df = (cm_all_df.groupby("label")[["TP", "FP", "FN", "TN"]].sum().reset_index())
        eps = 1e-12
        perf_df["precision"] = perf_df["TP"] / (perf_df["TP"] + perf_df["FP"] + eps)
        perf_df["recall"]    = perf_df["TP"] / (perf_df["TP"] + perf_df["FN"] + eps)
        perf_df["f1"]        = 2 * perf_df["precision"] * perf_df["recall"] / (perf_df["precision"] + perf_df["recall"] + eps)
        print("Worst by recall:")
        print(perf_df.sort_values("recall").head(50)[["label","TP","FP","FN","precision","recall","f1"]])
        print("\nWorst by f1:")
        print(perf_df.sort_values("f1").head(50)[["label","TP","FP","FN","precision","recall","f1"]])

        fraction_of_df1 = 0.20
        total_budget = int(math.ceil(fraction_of_df1 * len(data1)))

        aug_df, final_counts, allocation_df = augment_performance_driven(
            df1=data1,
            df2=data2,
            perf_df=perf_df,          # computed on held-out validation (or CV) split
            prefixes=PREFIXES,
            seed=None,
            total_budget=total_budget,
            score_mode="fnr",       # more stable than pure FNR
            min_support=1,           # stabilize per-label weakness estimates
            max_per_label_add=math.ceil(0.2 * total_budget),  # cap by budget share
        )
        df_train = aug_df
        df_test = pd.read_json("datasets/bosch_test.json") 

    elif dataset_name == ExpDataset.TRAM_AUGMENTED_HIERARCHY:
        add_fraction_of_max = 0.40
        df_train_augm_mitre = pd.read_json("datasets/tram_train_augmented_mitre.json")
        df_train_augm = pd.read_json("datasets/tram_augmented_hierarchy.json")   
        df_original = pd.read_json("datasets/tram_train.json")
        df_train_orig, df_val = train_test_split(df_original, test_size=test_size, random_state=random_state)
        df_original = ensure_labels_column(df_original)
        df_train_augm = ensure_labels_column(df_train_augm)
        df_train_augm_mitre = ensure_labels_column(df_train_augm_mitre)
        print("Original training set size:", len(df_train_orig))
        print("Hierarchy augmented set size:", len(df_train_augm))
        print("Mitre augmented set size:", len(df_train_augm_mitre))
        # df_train = pd.concat([df_train_orig, df_train_augm, df_train_augm_mitre]) 
        aug_df, _ = augment_add_fraction_of_max(df_train_orig, df_train_augm_mitre, add_fraction_of_max=add_fraction_of_max, prefixes=PREFIXES, seed=None)
        df_train, __ = augment_add_fraction_of_max(aug_df, df_train_augm, add_fraction_of_max=add_fraction_of_max, prefixes=PREFIXES, seed=None)
        print("Final training set size:", len(df_train))
        df_test = pd.read_json("datasets/tram_test.json") 

    elif dataset_name == ExpDataset.TRAM_AUGMENTED_HIERARCHY_EMBED:
        add_fraction_of_max = 0.40
        df_train_augm_mitre = pd.read_json("datasets/tram_train_augmented_mitre.json")
        df_train_augm = pd.read_json("datasets/tram_augmented_hierarchy_embeddings.json")   
        df_original = pd.read_json("datasets/tram_train.json")
        df_train_orig, df_val = train_test_split(df_original, test_size=test_size, random_state=random_state)
        df_train_orig = ensure_labels_column(df_train_orig)
        df_train_augm = ensure_labels_column(df_train_augm)
        df_train_augm_mitre = ensure_labels_column(df_train_augm_mitre)
        print("Original training set size:", len(df_train_orig))
        print("Hierarchy augmented set size:", len(df_train_augm))
        print("Mitre augmented set size:", len(df_train_augm_mitre))
        # augm_concat = pd.concat([df_train_augm,df_train_augm_mitre ], axis=0, ignore_index=True)
        aug_df, _ = augment_add_fraction_of_max(df_train_orig, df_train_augm_mitre, add_fraction_of_max=add_fraction_of_max, prefixes=PREFIXES, seed=None)
        df_train, __ = augment_add_fraction_of_max(aug_df, df_train_augm, add_fraction_of_max=add_fraction_of_max, prefixes=PREFIXES, seed=None)
        print("Final training set size:", len(df_train))
        df_test = pd.read_json("datasets/tram_test.json")

    elif dataset_name == ExpDataset.TRAM_CTI_SENT:
        add_fraction_of_max = 0.40
        df_train_augm_mitre = pd.read_json("datasets/tram_train_augmented_mitre.json")
        df_train_augm = pd.read_json("/home/simonettos/thijs/data_augmentatio_stefano/combined_6th_tram.json")   
        df_original = pd.read_json("datasets/tram_train.json")
        df_train_orig, df_val = train_test_split(df_original, test_size=test_size, random_state=random_state)
        df_train_orig = ensure_labels_column(df_train_orig)
        df_train_augm = ensure_labels_column(df_train_augm)
        df_train_augm_mitre = ensure_labels_column(df_train_augm_mitre)
        print("Original training set size:", len(df_train_orig))
        print("Hierarchy augmented set size:", len(df_train_augm))
        print("Mitre augmented set size:", len(df_train_augm_mitre))
        # df_train = pd.concat([df_train_orig, df_train_augm, df_train_augm_mitre]) 
        aug_df, _ = augment_add_fraction_of_max(df_train_orig, df_train_augm_mitre, add_fraction_of_max=add_fraction_of_max, prefixes=PREFIXES, seed=None)
        df_train, __ = augment_add_fraction_of_max(aug_df, df_train_augm, add_fraction_of_max=add_fraction_of_max, prefixes=PREFIXES, seed=None)
        print("Final training set size:", len(df_train))
        df_test = pd.read_json("datasets/tram_test.json")

    elif dataset_name == ExpDataset.TRAM_ALL_IN:
        add_fraction_of_max = 0.40
        df_train_augm_mitre = pd.read_json("datasets/tram_train_augmented_mitre.json")
        df_train_augm1 = pd.read_json("/home/simonettos/thijs/data_augmentatio_stefano/combined_6th_tram.json")   
        df_train_augm2 = pd.read_json("datasets/tram_augmented_hierarchy_embeddings.json")   
        df_original = pd.read_json("datasets/tram_train.json")
        df_train_orig, df_val = train_test_split(df_original, test_size=test_size, random_state=random_state)
        df_train_orig = ensure_labels_column(df_train_orig)
        df_train_augm1 = ensure_labels_column(df_train_augm1)
        df_train_augm2 = ensure_labels_column(df_train_augm2)
        df_train_augm_mitre = ensure_labels_column(df_train_augm_mitre)
        print("Original training set size:", len(df_train_orig))
        # print("Hierarchy augmented set size:", len(df_train_augm1))
        print("Mitre augmented set size:", len(df_train_augm_mitre))
        # data1 = pd.concat([df_train_orig, df_train_augm2, df_train_augm_mitre]) 
        aug_df2, _ = augment_add_fraction_of_max(df_train_orig, df_train_augm_mitre, add_fraction_of_max=add_fraction_of_max, prefixes=PREFIXES, seed=None)
        aug_df2, __ = augment_add_fraction_of_max(aug_df2, df_train_augm1, add_fraction_of_max=add_fraction_of_max, prefixes=PREFIXES, seed=None)
        df_train, ___ = augment_add_fraction_of_max(aug_df2, df_train_augm2, add_fraction_of_max=add_fraction_of_max, prefixes=PREFIXES, seed=None)
        print("Final training set size:", len(df_train))
        df_test = pd.read_json("datasets/tram_test.json")

    elif dataset_name == ExpDataset.BOSCH_AUGMENTED_MITRE:
        # df_train_augm = pd.read_json("datasets/bosch_train_augmented_mitre.json")
        df_train_augm = pd.read_json("datasets/bosch_augmented_hierarchy_embeddings.json") 
        df_original = pd.read_json("datasets/bosch_train.json")
        df_train_orig, df_val = train_test_split(df_original, test_size=test_size, random_state=random_state)
        df_train_orig = ensure_labels_column(df_train_orig)
        df_train_augm = ensure_labels_column(df_train_augm)
        df_train = pd.concat([df_train_orig, df_train_augm]) 
        df_test = pd.read_json("datasets/bosch_test.json")

    elif dataset_name == ExpDataset.BOSCH_CTI_SENT_ONLY:
        df_train_augm = pd.read_json("/home/simonettos/thijs/data_augmentatio_stefano/combined_6th_bosch.json")
        df_original = pd.read_json("datasets/bosch_train.json")
        df_train_orig, df_val = train_test_split(df_original, test_size=test_size, random_state=random_state)
        df_train_orig = ensure_labels_column(df_train_orig)
        df_train_augm = ensure_labels_column(df_train_augm)
        df_train = pd.concat([df_train_orig, df_train_augm]) 
        df_test = pd.read_json("datasets/bosch_test.json")

    elif dataset_name == ExpDataset.BOSCH_AUGMENTED_HIERARCHY:
        add_fraction_of_max = 0.30
        df_train_augm_mitre = pd.read_json("datasets/bosch_train_augmented_mitre.json")
        df_train_augm = pd.read_json("datasets/bosch_augmented_hierarchy.json")
        df_original = pd.read_json("datasets/bosch_train.json")
        df_train_orig, df_val = train_test_split(df_original, test_size=test_size, random_state=random_state)
        df_train_orig = ensure_labels_column(df_train_orig)
        df_train_augm = ensure_labels_column(df_train_augm)
        df_train_augm_mitre = ensure_labels_column(df_train_augm_mitre)
        print("Original training set size:", len(df_train_orig))
        print("Hierarchy augmented set size:", len(df_train_augm))
        print("Mitre augmented set size:", len(df_train_augm_mitre))
        # df_train = pd.concat([df_train_orig, df_train_augm, df_train_augm_mitre]) 
        aug_df, _ = augment_add_fraction_of_max(df_train_orig, df_train_augm_mitre, add_fraction_of_max=add_fraction_of_max, prefixes=PREFIXES, seed=None)
        df_train, __ = augment_add_fraction_of_max(aug_df, df_train_augm, add_fraction_of_max=add_fraction_of_max, prefixes=PREFIXES, seed=None)
        print("Final training set size:", len(df_train))
        df_test = pd.read_json("datasets/bosch_test.json")

    elif dataset_name == ExpDataset.BOSCH_AUGMENTED_HIERARCHY_EMBED:
        add_fraction_of_max = 0.30
        df_train_augm_mitre = pd.read_json("datasets/bosch_train_augmented_mitre.json")
        df_train_augm = pd.read_json("datasets/bosch_augmented_hierarchy_embeddings.json")
        df_original = pd.read_json("datasets/bosch_train.json")
        df_train_orig, df_val = train_test_split(df_original, test_size=test_size, random_state=random_state)
        df_train_orig = ensure_labels_column(df_train_orig)
        df_train_augm = ensure_labels_column(df_train_augm)
        df_train_augm_mitre = ensure_labels_column(df_train_augm_mitre)
        print("Original training set size:", len(df_train_orig))
        print("Hierarchy augmented set size:", len(df_train_augm))
        print("Mitre augmented set size:", len(df_train_augm_mitre))
        # df_train = pd.concat([df_train_orig, df_train_augm, df_train_augm_mitre]) 
        aug_df, _ = augment_add_fraction_of_max(df_train_orig, df_train_augm_mitre, add_fraction_of_max=add_fraction_of_max, prefixes=PREFIXES, seed=None)
        df_train, __ = augment_add_fraction_of_max(aug_df, df_train_augm, add_fraction_of_max=add_fraction_of_max, prefixes=PREFIXES, seed=None)
        print("Final training set size:", len(df_train))
        df_test = pd.read_json("datasets/bosch_test.json")

    elif dataset_name == ExpDataset.BOSCH_CTI_SENT:
        add_fraction_of_max = 0.30
        df_train_augm_mitre = pd.read_json("datasets/bosch_train_augmented_mitre.json")
        df_train_augm = pd.read_json("/home/simonettos/thijs/data_augmentatio_stefano/combined_6th_bosch_global.json")
        df_original = pd.read_json("datasets/bosch_train.json")
        df_train_orig, df_val = train_test_split(df_original, test_size=test_size, random_state=random_state)
        df_train_orig = ensure_labels_column(df_train_orig)
        df_train_augm = ensure_labels_column(df_train_augm)
        df_train_augm_mitre = ensure_labels_column(df_train_augm_mitre)
        print("Original training set size:", len(df_train_orig))
        print("Hierarchy augmented set size:", len(df_train_augm))
        print("Mitre augmented set size:", len(df_train_augm_mitre))
        # df_train = pd.concat([df_train_orig, df_train_augm, df_train_augm_mitre]) 
        aug_df, _ = augment_add_fraction_of_max(df_train_orig, df_train_augm_mitre, add_fraction_of_max=add_fraction_of_max, prefixes=PREFIXES, seed=None)
        df_train, __ = augment_add_fraction_of_max(aug_df, df_train_augm, add_fraction_of_max=add_fraction_of_max, prefixes=PREFIXES, seed=None)
        print("Final training set size:", len(df_train))
        df_test = pd.read_json("datasets/bosch_test.json")

    elif dataset_name == ExpDataset.BOSCH_ALL_IN:
        add_fraction_of_max = 0.30
        df_train_augm_mitre = pd.read_json("datasets/bosch_train_augmented_mitre.json")
        df_train_augm1 = pd.read_json("/home/simonettos/thijs/data_augmentatio_stefano/combined_6th_bosch_global.json")
        df_train_augm2 = pd.read_json("datasets/bosch_augmented_hierarchy.json")
        df_original = pd.read_json("datasets/bosch_train.json")
        df_train_orig, df_val = train_test_split(df_original, test_size=test_size, random_state=random_state)
        df_train_orig = ensure_labels_column(df_train_orig)
        df_train_augm1 = ensure_labels_column(df_train_augm1)
        df_train_augm2 = ensure_labels_column(df_train_augm2)
        df_train_augm_mitre = ensure_labels_column(df_train_augm_mitre)
        print("Original training set size:", len(df_train_orig))
        # print("Hierarchy augmented set size:", len(df_train_augm1))
        print("Mitre augmented set size:", len(df_train_augm_mitre))
        # aug_df1 = pd.concat([df_train_orig, df_train_augm2, df_train_augm_mitre]) 
        aug_df2, _ = augment_add_fraction_of_max(df_train_orig, df_train_augm_mitre, add_fraction_of_max=add_fraction_of_max, prefixes=PREFIXES, seed=None)
        aug_df2, __ = augment_add_fraction_of_max(aug_df2, df_train_augm1, add_fraction_of_max=add_fraction_of_max, prefixes=PREFIXES, seed=None)
        df_train, ___ = augment_add_fraction_of_max(aug_df2, df_train_augm2, add_fraction_of_max=add_fraction_of_max, prefixes=PREFIXES, seed=None)
        print("Final training set size:", len(df_train))
        df_test = pd.read_json("datasets/bosch_test.json")

    elif dataset_name == ExpDataset.BOSCH_PREPROCESSES:
        df_original = pd.read_json("datasets/bosch_train_preprocessed.json")
        df_train, df_val = train_test_split(df_original, test_size=test_size, random_state=random_state)
        df_test = pd.read_json("datasets/bosch_test_preprocessed.json")
    elif dataset_name == ExpDataset.TRAM_PREPROCESSES:
        df_original = pd.read_json("datasets/tram_train_preprocessed.json")
        df_train, df_val = train_test_split(df_original, test_size=test_size, random_state=random_state)
        df_test = pd.read_json("datasets/tram_test_preprocessed.json")
    # df_original = pd.read_json("datasets/tram_train.json")
    # _, df_val = train_test_split(df_original, test_size=test_size, random_state=random_state)
    # df_test = pd.read_json("datasets/tram_test.json")
    # train_set = TramDataset(df_train, tokenizer)
    # val_set = TramDataset(df_val, tokenizer)
    before = len(df_train)
    # 1) Remove exact duplicates in training set

    for df in (df_train, df_val, df_test):
        df["_s"] = df["sentence"].apply(text_hash)
        df["_l"] = df["labels"].apply(lambda x: tuple(sorted(set(x))) if isinstance(x, list) else x)
    df_train = df_train.drop_duplicates(subset=["_s", "_l"], keep="first")
    after = len(df_train)

    if before != after:
        print(f"[INFO] Removed {before - after} duplicate training samples")

    df_train = (
        df_train
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    print(f"Training set size after augmentation: {len(df_train)} samples")
    print(f"Validation set size: {len(df_val)} samples")
    print(f"Test set size: {len(df_test)} samples")
    if "tram" in str(dataset_name).lower():
        train_set = TramDataset(df_train, tokenizer)
        val_set = TramDataset(df_val, tokenizer)

    elif "bosch" in str(dataset_name).lower():
        train_set = BoschDataset(df_train, tokenizer)
        val_set = BoschDataset(df_val, tokenizer)

    train_params = {"batch_size": batch_size, "shuffle": True, "num_workers": 0}
    val_params = {"batch_size": batch_size, "shuffle": False, "num_workers": 0}
    test_params = {"batch_size": batch_size, "shuffle": False, "num_workers": 0}

    training_loader = DataLoader(train_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)
    print("Per-document test sets:", per_document)
    if per_document:
        if "doc_title" in df_test.columns:
            grouped = df_test.groupby('doc_title')
        else:
            grouped = df_test.groupby('document')
        df_list = {k: group for k, group in grouped}
        if "tram" in str(dataset_name).lower():
            test_sets = {k: TramDataset(d, tokenizer) for k,d in df_list.items()}
        elif "bosch" in str(dataset_name).lower():
            test_sets = {k: BoschDataset(d, tokenizer) for k,d in df_list.items()}
        print(f"[INFO] Created {len(test_sets)} per-document test sets")
        return None, None, {k: DataLoader(d, **test_params) for k,d in test_sets.items()}
    else:
        print("[INFO] Creating single test set for all documents")
        test_set = TramDataset(df_test, tokenizer)
        testing_loader = DataLoader(test_set, **test_params)
    
    return training_loader, val_loader, testing_loader


def load_datasets(dataset_name, batch_size, tokenizer, test_size=0.2, random_state=0, per_document=False):
    dataset_name = ExpDataset(dataset_name)
    dataset_class = None
    df_train_name = None
    df_test_name = None
    if dataset_name == ExpDataset.BOSCH_TECHNIQUES:
        df_train_name = "datasets/bosch_train.json"
        df_test_name = "datasets/bosch_test.json"
        dataset_class = BoschTechniquesDataset
    elif dataset_name == ExpDataset.BOSCH_TACTICS: 
        df_train_name = "datasets/bosch_train.json"
        df_test_name = "datasets/bosch_test.json"
        dataset_class = BoschTacticsDataset
    elif dataset_name == ExpDataset.BOSCH_GROUPS: 
        df_train_name = "datasets/bosch_train.json"
        df_test_name = "datasets/bosch_test.json"
        dataset_class = BoschGroupsDataset
    elif dataset_name == ExpDataset.BOSCH_SOFTWARE: 
        df_train_name = "datasets/bosch_train.json"
        df_test_name = "datasets/bosch_test.json"
        dataset_class = BoschSoftwareDataset
    elif dataset_name == ExpDataset.BOSCH_TECHNIQUES_10: 
        df_train_name = "datasets/bosch_train.json"
        df_test_name = "datasets/bosch_test.json"
        dataset_class = Bosch10TechniquesDataset
    elif dataset_name == ExpDataset.BOSCH_TECHNIQUES_25: 
        df_train_name = "datasets/bosch_train.json"
        df_test_name = "datasets/bosch_test.json"
        dataset_class = Bosch25TechniquesDataset
    elif dataset_name == ExpDataset.BOSCH_TECHNIQUES_50: 
        df_train_name = "datasets/bosch_train.json"
        df_test_name = "datasets/bosch_test.json"
        dataset_class = Bosch50TechniquesDataset
    elif dataset_name == ExpDataset.BOSCH_TECHNIQUES_53: 
        df_train_name = "datasets/bosch_train.json"
        df_test_name = "datasets/bosch_test.json"
        dataset_class = Bosch53TechniquesDataset
    elif dataset_name == ExpDataset.BOSCH_TECHNIQUES_SL: 
        df_train_name = "datasets/bosch_train.json"
        df_test_name = "datasets/bosch_test.json"
        dataset_class = BoschTechniquesDatasetSL
    elif dataset_name == ExpDataset.TRAM_TECHNIQUES:
        df_train_name = "datasets/tram_train.json"
        df_test_name = "datasets/tram_test.json"
        dataset_class = TramDataset
    elif dataset_name == ExpDataset.TRAM_TECHNIQUES_10:
        df_train_name = "datasets/tram_train.json"
        df_test_name = "datasets/tram_test.json"
        dataset_class = Tram10Dataset
    elif dataset_name == ExpDataset.TRAM_TECHNIQUES_25:
        df_train_name = "datasets/tram_train.json"
        df_test_name = "datasets/tram_test.json"
        dataset_class = Tram25Dataset
    elif dataset_name == ExpDataset.TRAM_TECHNIQUES_SL:
        df_train_name = "datasets/tram_train.json"
        df_test_name = "datasets/tram_test.json"
        dataset_class = TramDatasetSL
    elif dataset_name == ExpDataset.BOSCH_ALL:
        df_train_name = "datasets/bosch_train.json"
        df_test_name = "datasets/bosch_test.json"
        dataset_class = BoschAllDataset
    elif dataset_name in [ExpDataset.TRAM_AUGMENTED_MITRE, ExpDataset.TRAM_AUGMENTED_HIERARCHY, 
                          ExpDataset.TRAM_AUGMENTED_HIERARCHY_EMBED,
                          ExpDataset.BOSCH_AUGMENTED_MITRE, ExpDataset.BOSCH_AUGMENTED_HIERARCHY, 
                          ExpDataset.BOSCH_AUGMENTED_HIERARCHY_EMBED, ExpDataset.BOSCH_PREPROCESSES, 
                          ExpDataset.TRAM_PREPROCESSES,
                          ExpDataset.TRAM_AUGMENTED_TILL_MAX, ExpDataset.TRAM_AUGMENTED_CONSISTENTLY,
                          ExpDataset.TRAM_AUGMENTED_PERFORMANCE_DRIVEN, ExpDataset.BOSCH_AUGMENTED_TILL_MAX, ExpDataset.BOSCH_AUGMENTED_CONSISTENTLY,
                          ExpDataset.BOSCH_AUGMENTED_PERFORMANCE_DRIVEN,
                          ExpDataset.TRAM_CTI_SENT, ExpDataset.BOSCH_CTI_SENT, ExpDataset.TRAM_ALL_IN, ExpDataset.BOSCH_ALL_IN,
                          ExpDataset.TRAM_CTI_SENT_ONLY, ExpDataset.BOSCH_CTI_SENT_ONLY, 
                          ExpDataset.TRAM_ARTIFICIAL, ExpDataset.BOSCH_ARTIFICIAL, ExpDataset.TRAM_EDA, ExpDataset.BOSCH_EDA, 
                          ExpDataset.TRAM_TTP_HUNTER, ExpDataset.BOSCH_TTP_HUNTER]:
        return load_augmented(dataset_name, batch_size, tokenizer, test_size, random_state, per_document)
    
    
    df = pd.read_json(df_train_name)
    df_train, df_val = train_test_split(df, test_size=test_size, random_state=random_state)
    df_test = pd.read_json(df_test_name)
    for df in (df_train, df_val, df_test):
        df["text_hash"] = df["sentence"].apply(text_hash)

    # 1) Remove exact duplicates in training set
    before = len(df_train)
    df_train = df_train.drop_duplicates(subset="text_hash", keep="first")
    after = len(df_train)

    if before != after:
        print(f"[INFO] Removed {before - after} duplicate training samples")

    df_train = (
        df_train
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    print(f"Training set size after augmentation: {len(df_train)} samples")
    print(f"Validation set size: {len(df_val)} samples")
    print(f"Test set size: {len(df_test)} samples")
    train_set = dataset_class(df_train, tokenizer)
    val_set = dataset_class(df_val, tokenizer)

    train_params = {"batch_size": batch_size, "shuffle": True, "num_workers": 0}
    val_params = {"batch_size": batch_size, "shuffle": False, "num_workers": 0}
    test_params = {"batch_size": batch_size, "shuffle": False, "num_workers": 0}

    training_loader = DataLoader(train_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    if per_document:
        if "doc_title" in df_test.columns:
            grouped = df_test.groupby('doc_title')
        else:
            grouped = df_test.groupby('document')
        df_list = {k: group for k, group in grouped}
        test_sets = {k: dataset_class(d, tokenizer) for k,d in df_list.items()}
        return None, None, {k: DataLoader(d, **test_params) for k,d in test_sets.items()}
    else:
        test_set = dataset_class(df_test, tokenizer)
        testing_loader = DataLoader(test_set, **test_params)

    return training_loader, val_loader, testing_loader
