import torch
import numpy as np

import time

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
import random
import os
import sys
from transformers import get_scheduler
from typing import Optional
import loader
from train_common import *
from torch import nn

#Models ['bert-base-uncased', 'bert-base-cased', 'FacebookAI/roberta-base', 'FacebookAI/xlm-roberta-base', 'FacebookAI/roberta-large', 'FacebookAI/xlm-roberta-large', 's2w-ai/DarkBERT', 'jackaduma/SecBERT', 'jackaduma/SecRoBERTa', 'markusbayer/CySecBERT', 'allenai/scibert_scivocab_cased', 'allenai/scibert_scivocab_uncased', 'priyankaranade/cybert', 'tram_multi_label_model', 'ehsanaghaei/SecureBERT'] MODEL_SENTENCE_SIM ['sentence-transformers/all-mpnet-base-v2', 'basel/ATTACK-BERT', 'qcri-cs/SentSecBert_10k']
def get_base_encoder(model):
    return getattr(model, "bert", None) or getattr(model, "roberta", None)
def get_inner_encoder(model):
    return getattr(model, "bert", None) or getattr(model, "roberta", None) or model.base_model

@torch.no_grad()
def embed_loader(model, loader, device):
    base = get_base_encoder(model).to(device)
    base.eval()

    X_list, Y_list = [], []
    for batch in tqdm(loader, desc="Embedding"):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["labels"].cpu().numpy()

        out = base(input_ids=ids, attention_mask=mask, return_dict=True)
        # CLS embedding (works for BERT/Roberta)
        cls = out.last_hidden_state[:, 0, :].detach().cpu().numpy()

        X_list.append(cls)
        Y_list.append(y)

    X = np.vstack(X_list)
    Y = np.vstack(Y_list).astype(np.float32)

    return X, Y
from sklearn.neighbors import NearestNeighbors

def mlsmote_embeddings(X, Y, target_per_label=0, k=5, seed=42):
    """
    X: (N, D) embeddings
    Y: (N, L) multi-hot labels
    target_per_label: if 0 -> balance up to max label count
    """
    rng = np.random.default_rng(seed)
    N, D = X.shape
    L = Y.shape[1]

    counts = Y.sum(axis=0)
    max_c = int(counts.max())
    if target_per_label <= 0:
        target_per_label = max_c

    X_new = [X]
    Y_new = [Y]

    for lab in range(L):
        idx = np.where(Y[:, lab] == 1)[0]
        c = len(idx)
        if c < 2:
            continue

        need = target_per_label - c
        if need <= 0:
            continue

        # kNN within positives of this label
        X_pos = X[idx]
        nn = NearestNeighbors(n_neighbors=min(k, c), metric="euclidean")
        nn.fit(X_pos)

        synth_X = []
        synth_Y = []
        for _ in range(need):
            i_local = rng.integers(0, c)
            x_i = X_pos[i_local]

            neigh = nn.kneighbors([x_i], return_distance=False)[0]
            # pick a neighbor (avoid self if possible)
            j_local = int(rng.choice(neigh[1:] if len(neigh) > 1 else neigh))
            x_j = X_pos[j_local]

            lam = rng.random()
            x_s = x_i + lam * (x_j - x_i)

            # label assignment: OR of the two originals (good default)
            yi = Y[idx[i_local]]
            yj = Y[idx[j_local]]
            y_s = np.logical_or(yi, yj).astype(np.int8)

            synth_X.append(x_s)
            synth_Y.append(y_s)

        X_new.append(np.array(synth_X, dtype=np.float32))
        Y_new.append(np.array(synth_Y, dtype=np.int8))

    Xb = np.vstack(X_new).astype(np.float32)
    Yb = np.vstack(Y_new).astype(np.int8)
    return Xb, Yb
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch

class EmbDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return {"emb": self.X[i], "labels": self.Y[i]}

class MLPHead(nn.Module):
    def __init__(self, in_dim, num_labels, hidden=512, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, num_labels),
        )
    def forward(self, emb):
        return self.net(emb)

def start_training(model, training_loader, val_loader, output_dir, model_params):
    learning_rate = model_params["learning_rate"]
    freeze_layers = model_params["freeze_layers"]
    pos_weight = model_params["pos_weight"]
    end_factor = model_params.get("end_factor", 0.7)
    model_file = os.path.join(output_dir, "model_chkp")
    epochs = model_params["epochs"]

    inner = get_inner_encoder(model).to(device)
    for p in inner.parameters():
        p.requires_grad = False
    inner.eval()

    # OPTIONAL: freeze encoder if you only want head training
    if int(freeze_layers) != 0:
        # your freezing code assumes BERT/Roberta-style encoder.layer
        inner_mdl = getattr(model, "bert", None) or getattr(model, "roberta", None)
        for param in inner_mdl.embeddings.parameters():
            param.requires_grad = False
        for param in inner_mdl.encoder.layer[:freeze_layers].parameters():
            param.requires_grad = False

    # Create head ONCE
    hidden = inner.config.hidden_size
    head = MLPHead(in_dim=hidden, num_labels=model.num_labels).to(device)
    print("[SMOTE] Precomputing train embeddings...")
    Xtr, Ytr = embed_loader(model, training_loader, device)

    print("[SMOTE] Applying MLSMOTE in embedding space...")
    Xtr_sm, Ytr_sm = mlsmote_embeddings(Xtr, Ytr, target_per_label=0, k=5, seed=SEED_VAL)

    train_emb_loader = DataLoader(
        EmbDataset(Xtr_sm, Ytr_sm),
        batch_size=training_loader.batch_size,
        shuffle=True
    )

    print("[SMOTE] Precomputing val embeddings...")
    Xva, Yva = embed_loader(model, val_loader, device)

    val_emb_loader = DataLoader(
        EmbDataset(Xva, Yva),
        batch_size=val_loader.batch_size,
        shuffle=False
    )

    # Choose what to optimize
    # If encoder frozen: optimizer = AdamW(head.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate)
    total_steps = len(train_emb_loader) * epochs

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=end_factor, total_iters=total_steps
    )

    if pos_weight != 0:
        pos_weight_t = torch.full((model.num_labels,), pos_weight).to(device)
    else:
        pos_weight_t = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    best_eval_loss = float("inf")
    step_no_improv = 0

    for epoch_i in range(epochs):
        model.train()
        head.train()

        tr_loss = 0.0
        tr_accuracy = 0.0
        tr_f1 = 0.0

        for batch in tqdm(train_emb_loader, desc="Training"):
            emb = batch["emb"].to(device, dtype=torch.float32)
            labels = batch["labels"].to(device, dtype=torch.float32)

            optimizer.zero_grad()
            logits = head(emb)
            loss = criterion(logits, labels)

            tr_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > TAU).float()

            tr_accuracy += accuracy_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy())
            tr_f1 += f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(),
                            average="weighted", zero_division=0)

            torch.nn.utils.clip_grad_norm_(head.parameters(), MAX_GRAD_NORM)

            loss.backward()
            optimizer.step()
            scheduler.step()


        tr_loss /= len(train_emb_loader)
        tr_accuracy /= len(train_emb_loader)
        tr_f1 /= len(train_emb_loader)

        # ---- Validation (same path!)
        model.eval()
        head.eval()

        eval_loss = 0.0
        eval_accuracy = 0.0
        eval_f1 = 0.0
        eval_preds, eval_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_emb_loader, desc="Validation"):
                emb = batch["emb"].to(device, dtype=torch.float32)
                labels = batch["labels"].to(device, dtype=torch.float32)

                logits = head(emb)
                loss = criterion(logits, labels)
                eval_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs > TAU).float()

                eval_labels.extend(labels.cpu().numpy())
                eval_preds.extend(preds.cpu().numpy())

                eval_accuracy += accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
                eval_f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(),
                                    average="weighted", zero_division=0)

        eval_loss /= len(val_emb_loader)
        eval_accuracy /= len(val_emb_loader)
        eval_f1 /= len(val_emb_loader)

        # Save best (save both!)
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            step_no_improv = 0
            torch.save({"model": model.state_dict(), "head": head.state_dict()}, model_file)
        else:
            step_no_improv += 1
            if step_no_improv >= PATIENCE:
                break


if __name__ == "__main__":
    config_file = sys.argv[1]
    device = sys.argv[2]
    index_start = int(sys.argv[3])
    n_to_read = int(sys.argv[4])

    if len(sys.argv) < 1:
        print("usage: python train_multilabel.py CONFIG_FILE")

    if device == "cpu":
        print("[!] Running on CPU!")

    exp_configs = parse_config(config_file)
    exp_configs = exp_configs[index_start:index_start+n_to_read]

    for conf_id, model_name, dataset_name, model_params in exp_configs:
        try:
            print("TRAIN MULTILABEL:")
            print(
                f"conf_id: {conf_id}, model_name: {model_name}, dataset_name: {dataset_name}"
            )
            print(f"model_params:\n{model_params}")
            mdl, tokenizer = loader.load_untrained_model(model_name, dataset_name)
            mdl.to(device)
            batch_size = model_params["batch_size"]
            sanitized_name = loader.model_name_to_folder_name(model_name)
            folder_name = "%s_%s_%s" % (conf_id, dataset_name, sanitized_name)
            output_dir = os.path.join("fine_tuned", "multi_label", folder_name)
            model_file = os.path.join(output_dir, "model_chkp")
            os.makedirs(output_dir, exist_ok=True)
            # # display_df_stats(training_stats)
            train_emb_loader, val_emb_loader, _ = loader.load_datasets(
                dataset_name, batch_size, tokenizer, test_size=0.2
            )
            start_training(mdl, train_emb_loader, val_emb_loader, output_dir, model_params)
            ckpt = torch.load(model_file, map_location=device)
            mdl.load_state_dict(ckpt["model"])
            # recreate head with correct in_dim:
            inner = get_inner_encoder(mdl)
            hidden = inner.config.hidden_size
            head = MLPHead(in_dim=hidden, num_labels=mdl.num_labels).to(device)
            head.load_state_dict(ckpt["head"])

            mdl.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(head.state_dict(), os.path.join(output_dir, "head.pt"))
            os.remove(model_file)
        except Exception as e:
            raise e
            print("[!] ERROR! Skipping")
