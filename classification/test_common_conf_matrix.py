
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, classification_report
from collections import defaultdict
import numpy as np
import pandas as pd

from collections import Counter
import os
import json

def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den != 0 else 0.0

def f1_from_pr(precision: float, recall: float) -> float:
    return safe_div(2 * precision * recall, precision + recall)

def calc_set_metrics(true_items, pred_items):
    true_set = set(true_items)
    pred_set = set(pred_items)

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = f1_from_pr(precision, recall)
    accuracy = safe_div(tp, tp + fp + fn)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }

def calc_results_per_document_and_label(results, labels_map=None, min_label_support=1):
    if labels_map is not None:
        label_indices = sorted(labels_map.keys())
        idx_to_label = {i: labels_map[i] for i in label_indices}
    else:
        first_doc = next(iter(results.values()))
        first_vec = first_doc["labels"][0]
        n_labels = len(first_vec)
        idx_to_label = {i: str(i) for i in range(n_labels)}
        label_indices = list(idx_to_label.keys())

    ordered_indices = sorted(label_indices)
    ordered_labels = [idx_to_label[i] for i in ordered_indices]
    idx_to_pos = {idx: pos for pos, idx in enumerate(ordered_indices)}
    n_labels = len(ordered_indices)

    doc_rows = []
    tp = np.zeros(n_labels, dtype=np.int64)
    fp = np.zeros(n_labels, dtype=np.int64)
    fn = np.zeros(n_labels, dtype=np.int64)
    tn = np.zeros(n_labels, dtype=np.int64)
    true_support = np.zeros(n_labels, dtype=np.int64)
    pred_support = np.zeros(n_labels, dtype=np.int64)
    pair_confusion_counter = Counter()

    for doc_id, doc_payload in results.items():
        true_sentences = doc_payload["labels"]
        pred_sentences = doc_payload["preds"]

        all_true_ttps = []
        all_pred_ttps = []

        for true_values, pred_values in zip(true_sentences, pred_sentences):
            true_values = np.asarray(true_values)
            pred_values = np.asarray(pred_values)

            for original_idx in ordered_indices:
                pos = idx_to_pos[original_idx]
                t = int(true_values[original_idx] == 1)
                p = int(pred_values[original_idx] == 1)

                if t == 1:
                    true_support[pos] += 1
                if p == 1:
                    pred_support[pos] += 1

                if t == 1 and p == 1:
                    tp[pos] += 1
                elif t == 0 and p == 1:
                    fp[pos] += 1
                elif t == 1 and p == 0:
                    fn[pos] += 1
                else:
                    tn[pos] += 1

            true_i = np.where(true_values == 1)[0]
            pred_i = np.where(pred_values == 1)[0]

            true_ttps = [idx_to_label[i] for i in true_i if i in idx_to_label]
            pred_ttps = [idx_to_label[i] for i in pred_i if i in idx_to_label]

            all_true_ttps.extend(true_ttps)
            all_pred_ttps.extend(pred_ttps)

            true_set = set(true_ttps)
            pred_set = set(pred_ttps)
            missed = true_set - pred_set
            extra = pred_set - true_set

            for t_lbl in missed:
                for p_lbl in extra:
                    pair_confusion_counter[(t_lbl, p_lbl)] += 1

        doc_metrics = calc_set_metrics(all_true_ttps, all_pred_ttps)
        true_set = set(all_true_ttps)
        pred_set = set(all_pred_ttps)

        doc_rows.append({
            "doc_id": doc_id,
            "true": sorted(true_set),
            "pred": sorted(pred_set),
            "n_true_labels": len(true_set),
            "n_pred_labels": len(pred_set),
            "tp": doc_metrics["tp"],
            "fp": doc_metrics["fp"],
            "fn": doc_metrics["fn"],
            "precision": doc_metrics["precision"],
            "recall": doc_metrics["recall"],
            "f1": doc_metrics["f1"],
            "accuracy": doc_metrics["accuracy"],
            "missing_labels": sorted(true_set - pred_set),
            "extra_labels": sorted(pred_set - true_set),
            "correct_labels": sorted(true_set & pred_set),
        })

    doc_df = pd.DataFrame(doc_rows)

    label_rows = []
    for pos, label_name in enumerate(ordered_labels):
        precision = safe_div(tp[pos], tp[pos] + fp[pos])
        recall = safe_div(tp[pos], tp[pos] + fn[pos])
        f1 = f1_from_pr(precision, recall)
        support = true_support[pos]

        if support >= min_label_support:
            label_rows.append({
                "label": label_name,
                "support_true": int(true_support[pos]),
                "support_pred": int(pred_support[pos]),
                "tp": int(tp[pos]),
                "fp": int(fp[pos]),
                "fn": int(fn[pos]),
                "tn": int(tn[pos]),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fn_rate": safe_div(fn[pos], tp[pos] + fn[pos]),
                "fp_rate": safe_div(fp[pos], fp[pos] + tn[pos]),
            })

    label_df = pd.DataFrame(label_rows)
    if not label_df.empty:
        label_df = label_df.sort_values(by=["support_true", "f1"], ascending=[False, False]).reset_index(drop=True)

    pair_confusion_rows = [
        {"true_label_missed": t, "pred_label_extra": p, "count": c}
        for (t, p), c in pair_confusion_counter.items()
    ]
    pair_confusion_df = pd.DataFrame(pair_confusion_rows)
    if not pair_confusion_df.empty:
        pair_confusion_df = pair_confusion_df.sort_values("count", ascending=False).reset_index(drop=True)

    return doc_df, label_df, pair_confusion_df

def sanitize_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in s)

def save_detailed_analysis(analysis_dir, model_name, dataset_name, results, labels_map, min_label_support=5):
    os.makedirs(analysis_dir, exist_ok=True)
    base_name = sanitize_filename(f"{dataset_name}__{model_name}")

    doc_df, label_df, pair_conf_df = calc_results_per_document_and_label(
        results=results,
        labels_map=labels_map,
        min_label_support=min_label_support,
    )

    doc_df.to_csv(os.path.join(analysis_dir, f"{base_name}__doc_metrics.csv"), index=False)
    label_df.to_csv(os.path.join(analysis_dir, f"{base_name}__label_metrics.csv"), index=False)
    pair_conf_df.to_csv(os.path.join(analysis_dir, f"{base_name}__pair_confusion.csv"), index=False)

    return doc_df

    
def calculate_metrics(row, metric: str):
    # unique all ids
    unique_prediction_list = list(set(row['pred']))
    unique_true_label_list = list(set(row['true']))

    if len(unique_prediction_list) == 0 and len(unique_true_label_list) == 0:  # if empty labels are correct
        return 1.

    # Initialize variables for true positives, false positives, and false negatives
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Iterate through each item in the prediction list
    for item in unique_prediction_list:
        # Check if the item is in the true label list
        if item in unique_true_label_list:
            # If the item is in both lists, it's a true positive
            true_positives += 1
        else:
            # If the item is not in the true label list, it's a false positive
            false_positives += 1

    # Calculate false negatives
    false_negatives = len(unique_true_label_list) - true_positives

    # Calculate Jaccard Index or Critical Success Index
    accuracy = true_positives / (true_positives + false_positives + false_negatives)

    # Calculate precision
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives != 0 else 0

    # Calculate recall
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives != 0 else 0

    # Calculate F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    if metric == 'f1':
        return f1
    if metric == 'accuracy':
        return accuracy
    if metric == 'precision':
        return precision
    if metric == 'recall':
        return recall

import os
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix

import os
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix

# def calc_results_per_document(
#     results,
#     id2label,                     # can be dict[int,str] or list[str]
#     save_dir='/home/simonettos/thijs/classification/classification/extra',                # e.g. ".../eval/conf_mats/"
#     save_stacked_csv_path='/home/simonettos/thijs/classification/classification/extra/all_docs_cm_bosch.csv',   # e.g. ".../eval/conf_mats/all_docs_cm.csv"
# ):
#     """
#     Computes per-document metrics AND saves per-document multi-label confusion matrices.

#     Assumes:
#       results[doc]["labels"] and results[doc]["preds"] are sequences of binary vectors
#       (one vector per sentence/segment inside the document), shape = (n_samples, n_labels).

#     id2label:
#       - dict {i: label} or list[label]
#     """

#     # ---- normalize id2label to a list where index = class index ----
#     if isinstance(id2label, dict):
#         # ensure correct order even if keys aren't sorted
#         labels_map = [id2label[i] for i in sorted(id2label.keys())]
#     else:
#         labels_map = list(id2label)

#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)

#     out = defaultdict(list)
#     stacked_rows = []

#     for doc in results:
#         true = results[doc]["labels"]
#         preds = results[doc]["preds"]

#         # Convert to arrays (n_samples x n_labels)
#         y_true_doc = np.asarray(true)
#         y_pred_doc = np.asarray(preds)

#         # Safety checks
#         if y_true_doc.ndim != 2 or y_pred_doc.ndim != 2:
#             raise ValueError(
#                 f"Expected 2D arrays per document (n_samples x n_labels). "
#                 f"Got shapes: true={y_true_doc.shape}, pred={y_pred_doc.shape} for doc={doc}"
#             )
#         if y_true_doc.shape != y_pred_doc.shape:
#             raise ValueError(
#                 f"Shape mismatch for doc={doc}: true={y_true_doc.shape}, pred={y_pred_doc.shape}"
#             )

#         # Human-readable lists for your existing metrics
#         all_true_ttps, all_pred_ttps = [], []
#         for t_vec, p_vec in zip(y_true_doc, y_pred_doc):
#             true_i = np.where(t_vec == 1)[0]
#             pred_i = np.where(p_vec == 1)[0]
#             all_true_ttps.extend([labels_map[i] for i in true_i])
#             all_pred_ttps.extend([labels_map[i] for i in pred_i])

#         out["doc"].append(doc)
#         out["true"].append(all_true_ttps)
#         out["pred"].append(all_pred_ttps)

#         # ---- per-document per-label confusion (TN FP FN TP) ----
#         cm_doc = multilabel_confusion_matrix(y_true_doc, y_pred_doc)  # (n_labels, 2, 2)

#         doc_rows = []
#         for i, label in enumerate(labels_map):
#             tn, fp, fn, tp = cm_doc[i].ravel()
#             row = {
#                 "doc": doc,
#                 "label": label,
#                 "TP": int(tp),
#                 "FP": int(fp),
#                 "FN": int(fn),
#                 "TN": int(tn),
#             }
#             doc_rows.append(row)
#             stacked_rows.append(row)

#         cm_doc_df = pd.DataFrame(doc_rows)

#         # Save per-doc matrix
#         if save_dir:
#             safe_doc = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(doc))
#             cm_doc_df.to_csv(os.path.join(save_dir, f"{safe_doc}_cm.csv"), index=False)

#     out_df = pd.DataFrame(out)

#     # Your metrics (assumes calculate_metrics works on row with true/pred lists)
#     out_df["precision"] = out_df.apply(lambda x: calculate_metrics(x, "precision"), axis=1)
#     out_df["recall"]    = out_df.apply(lambda x: calculate_metrics(x, "recall"), axis=1)
#     out_df["accuracy"]  = out_df.apply(lambda x: calculate_metrics(x, "accuracy"), axis=1)
#     out_df["f1"]        = out_df.apply(lambda x: calculate_metrics(x, "f1"), axis=1)

#     cm_all_df = pd.DataFrame(stacked_rows)

#     # Save stacked CSV (all docs in one file)
#     if save_stacked_csv_path:
#         os.makedirs(os.path.dirname(save_stacked_csv_path), exist_ok=True)
#         cm_all_df.to_csv(save_stacked_csv_path, index=False)

#     return out_df


def calc_results_per_document(results, labels_map=None):
    # we will save the results inside here
    out = defaultdict(list)
    for doc in results:
        true = results[doc]["labels"]
        predictions = results[doc]["preds"]
        all_true_ttps = []
        all_pred_ttps = []
        
        if labels_map:
            for true_values, prediction_values in zip(true, predictions):
                true_i, = np.where(true_values == 1)
                pred_i, = np.where(prediction_values == 1)
                true_ttps = [labels_map[i] for i in true_i]
                pred_ttps = [labels_map[i] for i in pred_i]
                all_true_ttps.extend(true_ttps)
                all_pred_ttps.extend(pred_ttps)
        else:
            all_true_ttps.extend(true)
            all_pred_ttps.extend(predictions)

        # out['doc_title'].append(doc)
        out['true'].append(all_true_ttps)
        out['pred'].append(all_pred_ttps)

        out_df = pd.DataFrame(out)
        out_df["precision"] = out_df.apply(lambda x: calculate_metrics(x, "precision"), axis=1)
        out_df["recall"] = out_df.apply(lambda x: calculate_metrics(x, "recall"), axis=1)
        out_df["accuracy"] = out_df.apply(lambda x: calculate_metrics(x, "accuracy"), axis=1)
        out_df["f1"] = out_df.apply(lambda x: calculate_metrics(x, "f1"), axis=1)
        # Debug: inspect predictions per document
        true_set = set(all_true_ttps)
        pred_set = set(all_pred_ttps)

        missing = true_set - pred_set       # false negatives
        extra   = pred_set - true_set       # false positives
        correct = true_set & pred_set       # true positives
        print("\n")

    return out_df


def calc_results_per_sentence(results):
    out = defaultdict(list)
    true = results['labels']
    pred = results['preds']
    
    micro_f1 = f1_score(true, pred, average="micro", zero_division=0.0)
    macro_f1 = f1_score(true, pred, average="macro", zero_division=0.0)
    samples_f1 = f1_score(true, pred, average="samples", zero_division=0.0)
    weighted_f1 = f1_score(true, pred, average="weighted", zero_division=0.0)
    accuracy = accuracy_score(true, pred)
    micro_precision = precision_score(true, pred, average="micro", zero_division=0.0)
    macro_precision = precision_score(true, pred, average="macro", zero_division=0.0)
    samples_precision = precision_score(true, pred, average="samples", zero_division=0.0)
    weighted_precision = precision_score(true, pred, average="weighted", zero_division=0.0)
    micro_recall = recall_score(true, pred, average="micro", zero_division=0.0)
    macro_recall = recall_score(true, pred, average="macro", zero_division=0.0)
    samples_recall = recall_score(true, pred, average="samples", zero_division=0.0)
    weighted_recall = recall_score(true, pred, average="weighted", zero_division=0.0)

    # out['doc_title'].append("per_sentence")
    out['micro_f1'].append(micro_f1)
    out['macro_f1'].append(macro_f1)
    out['samples_f1'].append(samples_f1)
    out['weighted_f1'].append(weighted_f1)
    out['micro_precision'].append(micro_precision)
    out['macro_precision'].append(macro_precision)
    out['samples_precision'].append(samples_precision)
    out['weighted_precision'].append(weighted_precision)
    out['micro_recall'].append(micro_recall)
    out['macro_recall'].append(macro_recall)
    out['samples_recall'].append(samples_recall)
    out['weighted_recall'].append(weighted_recall)
    out['accuracy'].append(accuracy)
    out_df = pd.DataFrame(out)
    return out_df
