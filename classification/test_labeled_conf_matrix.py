# [markdown]
# # Test "Supervised" BERT Models

import torch
import loader
from const import ExpDataset
from typing import Optional
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from train_common import TAU
from collections import defaultdict
from test_common_conf_matrix import calc_results_per_document, calc_results_per_sentence, save_detailed_analysis, calc_results_per_document_and_label, sanitize_filename


def get_innermost_dirs(base_dir):
    innermost_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if not dirs:
            rel_path = os.path.relpath(root, base_dir)
            innermost_dirs.append(rel_path)
    return [os.path.join(base_dir, dir) for dir in innermost_dirs]


# [markdown]
# ## Test functions
# The following snippet contains all the functions that, given a model and a set of (or a single) data loaders, will return the labels and predictions.
def run_test_multi_label_per_document(model, data_loaders):
    model.to(device)
    model.eval()

    docs = {}

    for doc_name, data_loader in tqdm(data_loaders.items()):
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in data_loader:

                input_ids = torch.as_tensor(batch["input_ids"]).to(device)
                attn_mask = torch.as_tensor(batch["attention_mask"]).to(device)
                labels = torch.as_tensor(batch["labels"]).to(device)
                all_labels.extend(labels.cpu().numpy())
                out = model(input_ids=input_ids, attention_mask=attn_mask)
                probs = out.logits.sigmoid()
                preds = torch.where(probs > TAU, 1.0, 0.0)
                all_preds.extend(preds.cpu().numpy())

        all_labels = np.clip(np.sum(all_labels, axis=0, dtype=int), 0, 1).reshape(1, -1)
        all_preds = np.clip(np.sum(all_preds, axis=0, dtype=int), 0, 1).reshape(1, -1)

        docs[doc_name] = {"labels": all_labels, "preds": all_preds}

    return docs


def run_test_multi_label_per_sentence(model, data_loader):
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(data_loader):

            input_ids = torch.as_tensor(batch["input_ids"]).to(device)
            attn_mask = torch.as_tensor(batch["attention_mask"]).to(device)
            labels = torch.as_tensor(batch["labels"]).to(device)
            all_labels.extend(labels.cpu().numpy())
            out = model(input_ids=input_ids, attention_mask=attn_mask)
            probs = out.logits.sigmoid()
            preds = torch.where(probs > TAU, 1.0, 0.0)
            all_preds.extend(preds.cpu().numpy())

    docs = {"labels": all_labels, "preds": all_preds}

    return docs


def run_test_single_label_per_sentence(model, data_loader):
    raise NotImplementedError


def run_test_single_label_per_document(model, data_loaders):
    raise NotImplementedError


# [markdown]
# ## Test on single experimental setup
# Each setup is saved inside a folder named `"fine_tuned/{problem_type}/{conf\_id}\_{dataset}\_{model\_name}"`.
# Select the folder by its name and run the tests!


def run_test(model_dir, model_name, dataset_name, per_document=True):
    model, tokenizer = loader.load_finetuned_model(model_dir)
    print("Dataset_loader loading...", dataset_name)
    _, _, data_loaders = loader.load_datasets(
        dataset_name, 16, tokenizer, per_document=per_document
    )

    if (
        dataset_name == ExpDataset.BOSCH_TECHNIQUES_SL.value
        or dataset_name == ExpDataset.TRAM_TECHNIQUES_SL.value
    ):
        is_single_label = True
    else:
        is_single_label = False

    if per_document and is_single_label:
        test_f = run_test_single_label_per_document
    elif per_document and not is_single_label:
        test_f = run_test_multi_label_per_document
    elif not per_document and is_single_label:
        test_f = run_test_single_label_per_sentence
    else:
        test_f = run_test_multi_label_per_sentence

    print("Setup:")
    print(
        f"\tdataset_name={dataset_name}\n\tmodel_name={model_name}\n\tis_single_label={is_single_label}\n\tper_document={per_document}"
    )
    print(f"\ttest_function={test_f}")

    results = test_f(model, data_loaders)

    if per_document:
        out_df = calc_results_per_document(results, model.config.id2label)
    else:
        out_df = calc_results_per_sentence(results)

    # MINIMAL CHANGE: also return raw results + label map
    return out_df, results, model.config.id2label


def retrieve_setup_from_folder_name(folder):
    import os, json

    single_label = "single_label" in folder
    dataset = "_".join(os.path.basename(folder).split("_")[1:3])

    name = os.path.basename(folder)

    if "tram_artificial" in name:
        dataset = "tram_artificial"
    elif "bosch_artificial" in name:
        dataset = "bosch_artificial"
    elif "tram_eda" in name:
        dataset = "tram_eda"
    elif "bosch_eda" in name:
        dataset = "bosch_eda"
    elif "tram_ttp_hunter" in name:
        dataset = "tram_ttp_hunter"
    elif "bosch_ttp_hunter" in name:
        dataset = "bosch_ttp_hunter"
    elif "tram_augmented_hierarchy_embeddings" in name:
        dataset = "tram_augmented_hierarchy_embeddings"
    elif "tram_augmented_hierarchy" in name:
        dataset = "tram_augmented_hierarchy"
    elif "tram_train_augmented_mitre" in name:
        dataset = "tram_train_augmented_mitre"
    elif "tram_preprocessed" in name:
        dataset = "tram_preprocessed"
    elif "tram_augmented_till_max" in name:
        dataset = "tram_augmented_till_max"
    elif "tram_augmented_consistently" in name:
        dataset = "tram_augmented_consistently"
    elif "tram_augmented_performance_driven" in name:
        dataset = "tram_augmented_performance_driven"
    elif "bosch_preprocessed" in name:
        dataset = "bosch_preprocessed"
    elif "bosch_train_augmented_mitre" in name:
        dataset = "bosch_train_augmented_mitre"
    elif "bosch_cti_sent_only" in name:
        dataset = "bosch_cti_sent_only"
    elif "bosch_cti_sent" in name:
        dataset = "bosch_cti_sent"
    elif "tram_cti_sent_only" in name:
        dataset = "tram_cti_sent_only"
    elif "tram_cti_sent" in name:
        dataset = "tram_cti_sent"
    elif "bosch_all_in" in name:
        dataset = "bosch_all_in"
    elif "tram_all_in" in name:
        dataset = "tram_all_in"
    elif "bosch_augmented_hierarchy_embeddings" in name:
        dataset = "bosch_augmented_hierarchy_embeddings"
    elif "bosch_augmented_hierarchy" in name:
        dataset = "bosch_augmented_hierarchy"
    elif "bosch_augmented_till_max" in name:
        dataset = "bosch_augmented_till_max"
    elif "bosch_augmented_consistently" in name:    
        dataset = "bosch_augmented_consistently"
    elif "bosch_augmented_performance_driven" in name:
        dataset = "bosch_augmented_performance_driven"
    elif "bosch_t" in name:
        dataset = "bosch_t"
    elif name.startswith("tram_"):
        dataset = name.split("_")[1]
    elif dataset[:4] == "tram" and dataset not in [
        "tram",
        "tram_10",
        "tram_25",
        "tram_sl",
        "tram_artificial",
        "tram_ood",
        "tram_ood_rebalanced",
    ]:
        dataset = "tram"

    try:
        with open(os.path.join(folder, "model_params.json"), "r") as f:
            params = json.load(f)
        batch_size = params.get("batch_size")
        freeze_layers = params.get("freeze_layers")
        learning_rate = params.get("learning_rate")
        # if pos_weight is missing, it's 25. the experiments were run with a previous version of the
        # code that didn't contain the pos_weight parameter, but it was generated inside the function.
        end_factor = params.get("end_factor")
        pos_weight = params.get("pos_weight")
        print(f"dataset: {dataset}\n\tbatch_size: {batch_size}\n\tfreeze_layers: {freeze_layers}\n\tlearning_rate: {learning_rate}\n\tpos_weight: {pos_weight}\n\tend_factor: {end_factor}\n\tsingle_label: {single_label}")
        config_path = os.path.join(folder, "config.json")
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            model_name = cfg.get("_name_or_path")
            # Guard against useless values you might see in practice
            if model_name in (None, "", ".", "./"):
                model_name = None
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            model_name = None

        # Fallback: infer from folder name
        # Example: 17_tram_markusbayer-CySecBERT  -> markusbayer/CySecBERT
        if not model_name:
            last_part = name.split("_")[-1]
            model_name = last_part.replace("-", "/")

        print(f"\tmodel_name: {model_name}")
        print("Retrieved setup for folder: %s" % folder)
        print(f"\tmodel_name: {model_name}\n\tdataset: {dataset}\n\tbatch_size: {batch_size}\n\tfreeze_layers: {freeze_layers}\n\tlearning_rate: {learning_rate}\n\tpos_weight: {pos_weight}\n\tend_factor: {end_factor}\n\tsingle_label: {single_label}")
        return {
            "model_name": model_name,
            "dataset": dataset,
            "batch_size": batch_size,
            "freeze_layers": freeze_layers,
            "learning_rate": learning_rate,
            "pos_weight": pos_weight,
            "end_factor": end_factor,
            "is_single_label": single_label,
        }
    except:
        print("missing config for: %s" % folder)

import json
from pathlib import Path

def resolve_model_name(model_dir: str) -> Optional[str]:
    p = Path(model_dir)

    # 1) Try: config.json -> _name_or_path
    config_path = p / "config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        model_name = cfg.get("_name_or_path")
        if model_name in (None, "", ".", "./"):
            model_name = None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        model_name = None

    # 2) Fallback: infer from folder name
    # Example: 17_tram_markusbayer-CySecBERT  -> markusbayer/CySecBERT
    if not model_name:
        name = p.name
        last_part = name.split("_")[-1]
        model_name = last_part.replace("-", "/") or None

    return model_name

# device selection: you can choose gpu 0 with cuda:0 and gpu 1 with cuda:1
# device = "cuda:1" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="test_labeled",
        description="This code tests supervised BERT models stored inside a folder on their corresponding datasets.",
    )

    parser.add_argument(
        "model_dir",
        type=str,
        help='The base directory containing the models to test (e.g. "fine_tuned/tram_swipe").',
    )
    parser.add_argument(
        "outfile", type=str, help="The output file to save the results."
    )
    parser.add_argument(
        "device",
        type=str,
        help="The device to use for testing (e.g., 'cuda:0' or 'cpu').",
        default="cuda:0",
    )
    parser.add_argument(
        "--show-baseline",
        action="store_true",
        help="This flag will show the baseline results for the TRAM dataset.",
    )
    parser.add_argument(
        "--remove-dupl-models",
        action="store_true",
        help="This flag will remove duplicate models from the results (e.g., two RoBERTa trained with different hyper-parameters)",
    )
    parser.add_argument(
    "--analysis-dir",
    type=str,
    default="./analysis",
    help="Directory to save per-document and per-label analysis."
    )
    parser.add_argument(
        "--min-label-support",
        type=int,
        default=3,
        help="Minimum label support for per-label analysis."
    )
    args = parser.parse_args()

    device = args.device
    model_dir = args.model_dir
    outfile = args.outfile
    remove_dupl_models = args.remove_dupl_models

    selected_models = get_innermost_dirs(model_dir)

    setups = {}
    for f in selected_models:
        setups[f] = retrieve_setup_from_folder_name(f)

    # [markdown]
    # ### Collect test results per document
    final_results = {
        "model": [],
        "dataset": [],
        "freeze_layers": [],
        "batch_size": [],
        "learning_rate": [],
        "pos_weight": [],
        "end_factor": [],
        "accuracy_mean": [],
        "precision_mean": [],
        "recall_mean": [],
        "f1_mean": [],
    }

    for i, s in enumerate(setups):
        print("Evaluating setup %d/%d: %s" % (i + 1, len(setups), s))
        try:
            model_dir = s
            model_name = resolve_model_name(model_dir)
            dataset_name = setups[s]["dataset"]
            freeze_layers = setups[s]["freeze_layers"]
            pos_weight = setups[s]["pos_weight"]
            end_factor = setups[s]["end_factor"]
            learning_rate = setups[s]["learning_rate"]
            batch_size = setups[s]["batch_size"]
            result_df, raw_results, labels_map = run_test(model_dir, model_name, dataset_name)

            f1_mean = result_df["f1"].mean()
            accuracy_mean = result_df["accuracy"].mean()
            precision_mean = result_df["precision"].mean()
            recall_mean = result_df["recall"].mean()
            if args.analysis_dir is not None:
                os.makedirs(args.analysis_dir, exist_ok=True)
                save_detailed_analysis(
                    analysis_dir=args.analysis_dir,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    results=raw_results,
                    labels_map=labels_map,
                    min_label_support=args.min_label_support,
    )
            final_results["model"].append(model_name)
            final_results["dataset"].append(dataset_name)
            final_results["freeze_layers"].append(freeze_layers)
            final_results["pos_weight"].append(pos_weight)
            final_results["end_factor"].append(end_factor)
            final_results["learning_rate"].append(learning_rate)
            final_results["batch_size"].append(batch_size)
            final_results["f1_mean"].append(f1_mean)
            final_results["accuracy_mean"].append(accuracy_mean)
            final_results["precision_mean"].append(precision_mean)
            final_results["recall_mean"].append(recall_mean)
            print(f"{model_name}: {f1_mean}")
        except:
            print("\033[91m [!] Skipping %s ! \033[0m" % s)
            #print what is missing
            print(setups[s])
            print("\n")
            # print(result_df)
            print("\n")
            print(final_results)

    with pd.option_context("display.max_rows", None):
        final_df = pd.DataFrame(final_results)
        final_df = final_df.sort_values(by=["dataset", "model", "f1_mean"])
        
        final_df["f1_mean"] = (final_df["f1_mean"] * 100).round(2)
        final_df["precision_mean"] = (final_df["precision_mean"] * 100).round(2)
        final_df["recall_mean"] = (final_df["recall_mean"] * 100).round(2)
        if remove_dupl_models:
            out_table = final_df.loc[
                final_df.groupby("model")["f1_mean"].idxmax()
            ].reset_index(drop=True)[
                ["model", "dataset","f1_mean", "precision_mean", "recall_mean"]
            ]
        else:
            out_table = final_df[["model", "dataset","f1_mean", "precision_mean", "recall_mean"]]
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        print(out_table)
        final_df.to_csv(outfile)

    # [markdown]
    # ## Test Baseline
    if args.show_baseline:
        model_name = "baseline_tram"
        model, tokenizer = loader.load_untrained_model(
            "scibert_multi_label_model", "tram"
        )
        _, _, data_loaders = loader.load_datasets(
            "tram", 16, tokenizer, per_document=True
        )
        results = run_test_multi_label_per_document(model, data_loaders)
        doc_df, label_df, pair_conf_df = calc_results_per_document_and_label(
            results,
            model.config.id2label,
            min_label_support=args.min_label_support,
        )

        print(f"Baseline results for {model_name} on TRAM dataset:")
        print(
            f"TRAM(original), f1_mean: {(doc_df.f1.mean()*100).round(2)}, precision_mean: {(doc_df.precision.mean()*100).round(2)}, recall_mean: {(doc_df.recall.mean()*100).round(2)}"
        )

        if args.analysis_dir is not None:
            os.makedirs(args.analysis_dir, exist_ok=True)
            save_detailed_analysis(
                analysis_dir=args.analysis_dir,
                model_name=model_name,
                dataset_name="tram",
                results=results,
                labels_map=model.config.id2label,
                min_label_support=args.min_label_support,
            )