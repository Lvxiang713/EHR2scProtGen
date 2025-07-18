# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 14:44:58 2025

@author: LvXiang
"""
from torch.utils.data import DataLoader, Subset, random_split
from datasets.dataset import EHRSingleCellDataset
import torch
import pandas as pd
import os
from collections import defaultdict
import copy
 
def prepare_data(cfg: dict):
    """
    Load CSV/XLSX data, split into train/val/test loaders,
    build global EHR tensors and diagnosis dicts.
    
    Returns:
      train_ld, val_ld, test_ld: DataLoader objects
      global_ehr_tensor:  EHR embeddings for training donors
      test_global_ehr_tensor: EHR embeddings for test donors
      train_diag: mapping donor_ID → diagnosis for training set
      test_diag:  mapping donor_ID → diagnosis for test set
      ds: the raw dataset object (for metadata)
      """
    ehr_csv   = cfg['data']['ehr_csv']
    sc_csv    = cfg['data']['sc_csv']
    diag_csv  = cfg['data']['diag_csv']
    N0        = int(cfg['initial_split'])
    ratio     = float(cfg['train_val_ratio'])
    bs        = int(cfg['batch_size'])
    ds = EHRSingleCellDataset(ehr_csv, sc_csv)
    # Split: first N0 → train/val, rest → test
    split_ds    = Subset(ds, list(range(N0)))
    train_len   = int(ratio * N0)
    val_len     = N0 - train_len
    train_ds, val_ds = random_split(split_ds, [train_len, val_len])
    test_ds     = Subset(ds, list(range(N0, len(ds))))

    train_ld = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_ld   = DataLoader(val_ds,   batch_size=bs, shuffle=False)
    test_ld  = DataLoader(test_ds,  batch_size=bs, shuffle=False)

    # Build global EHR tensors & diagnosis maps
    train_donors = {ds.samples[i][0] for i in train_ds.indices}
    test_donors  = {ds.samples[i][0] for i in test_ds.indices}

    train_ehr = [torch.from_numpy(ds.ehr_dict[d])
                 for d in sorted(train_donors, key=lambda x: ds.donor2label[x])]
    test_ehr  = [torch.from_numpy(ds.ehr_dict[d])
                 for d in sorted(test_donors,  key=lambda x: ds.donor2label[x])]

    global_ehr_tensor      = torch.stack(train_ehr, dim=0)
    test_global_ehr_tensor = torch.stack(test_ehr,  dim=0)

    diag_df = pd.read_excel(diag_csv)
    train_diag = {
        row['donor_ID']: row['clinical_diagnosis']
        for _, row in diag_df.iterrows() if row['donor_ID'] in train_donors
    }
    test_diag = {
        row['donor_ID']: row['clinical_diagnosis']
        for _, row in diag_df.iterrows() if row['donor_ID'] in test_donors
    }

    return (train_ld, val_ld, test_ld,
            global_ehr_tensor, test_global_ehr_tensor,
            train_diag, test_diag, ds)


def generate_weighted_labels(labels, train_donor2diagnosis,soft_value=0.6):
    """
    Create soft labels for classification:
      - true class index gets soft_value
      - other donors with same diagnosis split remaining weight
      - donors with different diagnosis get zero
    """
    weighted_labels = torch.zeros(labels.size(0), len(train_donor2diagnosis)).cuda()
    diagnosis_to_donors = defaultdict(list)
    for idx, (donor_id, diagnosis) in enumerate(train_donor2diagnosis.items()):
        diagnosis_to_donors[diagnosis].append(idx)
    for idx,i in enumerate(labels):
        diagnosis = train_donor2diagnosis[list(train_donor2diagnosis.keys())[i]]
        same_col = copy.deepcopy(diagnosis_to_donors[diagnosis])  
        weighted_labels[idx,i]=soft_value
        same_col.remove(i.cpu().item())
        weighted_labels[idx,same_col] = (1-soft_value)/(len(same_col)-1)
    return weighted_labels

def evaluate_diagnosis_classification(model, loader,
                                       train_diag, global_ehr,
                                       device, top_k=3, test_diag=None):
    model.eval()
    with torch.no_grad():
        t_emb = model.encode_text(global_ehr.to(device))
        t_emb = t_emb / t_emb.norm(dim=1, keepdim=True)

    donor_ids = sorted(train_diag.keys())
    t1d_idx, ctrl_idx = [], []
    for i, d in enumerate(donor_ids):
        (t1d_idx if train_diag[d]=="T1D" else ctrl_idx).append(i)

    t1d_feat = t_emb[t1d_idx] if t1d_idx else None
    ctrl_feat = t_emb[ctrl_idx] if ctrl_idx else None

    mapping = {"T1D":0, "T1D control":1}
    confusion = torch.zeros(2,2, dtype=torch.int32)
    correct = total = 0

    for sc, ehr, _, donors in loader:
        sc = sc.to(device)
        with torch.no_grad():
            i_emb = model.encode_image(sc)
            i_emb = i_emb / i_emb.norm(dim=1, keepdim=True)
        for b, did in enumerate(donors):
            ground = (test_diag or train_diag)[did]

            # only compute if feature tensor exists
            if t1d_feat is not None and t1d_feat.size(0) > 0:
                sims_t = (i_emb[b:b+1] @ t1d_feat.T).sort().values[...,-top_k:]
                sim_t = sims_t.mean().item()
            else:
                sim_t = 0.0

            if ctrl_feat is not None and ctrl_feat.size(0) > 0:
                sims_c = (i_emb[b:b+1] @ ctrl_feat.T).sort().values[...,-top_k:]
                sim_c = sims_c.mean().item()
            else:
                sim_c = 0.0

            pred = "T1D" if sim_t >= sim_c else "T1D control"
            correct += (pred == ground)
            total += 1
            confusion[mapping[ground], mapping[pred]] += 1

    acc = correct / total if total > 0 else 0.0
    return acc, confusion

def fit(model, train_ld, val_ld, test_ld,
                        global_ehr, test_global_ehr,
                        train_diag, test_diag,
                        criterion, optimizer, cfg, device):
    """
    Train/validation loop with early stopping:
      - At each epoch, compute contrastive+classification loss on train set
      - Evaluate diagnostic accuracy on train/val
      - Save best model by validation loss; stop after patience epochs
    Returns:
      model: state_dict loaded with best checkpoint
    """
    best_val = float('inf')
    no_imp   = 0
    ckpt     = cfg.get('checkpoint_dir', './CLIPcheckpointDir')
    os.makedirs(ckpt, exist_ok=True)

    for epoch in range(int(cfg['num_epochs'])):
        # ——— Training ———
        model.train()
        run_loss = 0.0
        correct_clf = 0
        total_clf   = 0

        for sc, ehr, labels, _ in train_ld:
            sc, ehr, labels = sc.to(device), ehr.to(device), labels.to(device)
            img_e, _ = model(sc, ehr)
            gt = model.encode_text(global_ehr.to(device))
            gt = gt / gt.norm(dim=1, keepdim=True)
            logits = model.logit_scale.exp() * img_e @ gt.T
            soft_lbl = generate_weighted_labels(labels,
                                                train_diag,
                                                float(cfg['soft_value']))
            loss = criterion(logits, soft_lbl)
            preds = logits.argmax(dim=1)
            correct_clf += (preds == labels).sum().item()
            total_clf   += labels.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * labels.size(0)
        train_loss = run_loss / len(train_ld.dataset)
        # train_acc  = correct_clf / total_clf if total_clf > 0 else 0.0
        train_diag_acc, _ = evaluate_diagnosis_classification(
            model, train_ld, train_diag, global_ehr, device, top_k=1
        )
        # ——— Validation ———
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val   = 0
        with torch.no_grad():
            gt = model.encode_text(global_ehr.to(device))
            gt = gt / gt.norm(dim=1, keepdim=True)
            for sc, ehr, labels, _ in val_ld:
                sc, ehr, labels = sc.to(device), ehr.to(device), labels.to(device)
                img_e, _ = model(sc, ehr)
                logits = model.logit_scale.exp() * img_e @ gt.T

                soft_lbl = generate_weighted_labels(labels,
                                                    train_diag,
                                                    float(cfg['soft_value']))
                val_loss += criterion(logits, soft_lbl).item() * labels.size(0)

                preds = logits.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val   += labels.size(0)
        val_loss = val_loss / len(val_ld.dataset)
        # val_acc  = correct_val / total_val if total_val > 0 else 0.0
        val_diag_acc, _ = evaluate_diagnosis_classification(
            model, val_ld, train_diag, global_ehr, device, top_k=1
        )
        # ——— Print metrics ———
        print(f"Epoch {epoch+1:03d}: "
              f"Train Loss: {train_loss:.4f}, Diagn Acc: {train_diag_acc:.4f} | "
              f"Val Loss: {val_loss:.4f},  Diagn Acc: {val_diag_acc:.4f}")
        # ——— Check for improvement & early stop ———
        if val_loss < best_val:
            best_val = val_loss
            no_imp   = 0
            torch.save(model.state_dict(), os.path.join(ckpt, "bestCL_model.pth"))
            print("  Saved bestCL_model.pth")
        else:
            no_imp += 1
            if no_imp >= int(cfg['early_stop_patience']):
                print("Early stopping triggered")
                break
    # —— final test evaluation omitted for brevity —— 
    return model