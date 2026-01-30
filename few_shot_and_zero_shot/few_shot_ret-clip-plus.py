# -*- coding: utf-8 -*-
"""
Few-shot evaluation script for RET-CLIP models.

Supports:
  - Zero-shot baseline
  - CLIP-Adapter (train adapter on support set, evaluate on query/test)
  - TIP-Adapter (training-free, cache-based fusion)
  - TIP-Adapter-F (learn alpha/beta on support set)
"""

import os
import argparse
import json
import random
from tqdm import tqdm
import numpy as np
from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score, f1_score

from RET_CLIP_PLUS.clip.model import CLIP
from RET_CLIP_PLUS.clip import tokenize

from PIL import Image

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

BATCH_SIZE = 4
NUM_WORKERS = 4


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def _preprocess_text(text):
    # adapt the text to Chinese BERT vocab
    text = text.lower().replace("“", "\"").replace("”", "\"")
    return text


class RETFoundDataset(data.Dataset):
    def __init__(self, root, split, imsize=224, drop_others=False):
        self.data = []
        if 'PAPILA' in root:
            dr_folder_list = ['anormal', 'bsuspectglaucoma', 'cglaucoma']
        elif 'Glaucoma_fundus' in root:
            dr_folder_list = ['anormal_control', 'bearly_glaucoma', 'cadvanced_glaucoma']
        elif 'Retina' in root:
            dr_folder_list = ['anormal', 'bcataract', 'cglaucoma', 'ddretina_disease']
            if drop_others:
                dr_folder_list = dr_folder_list[:-1]
        elif 'OCTID' in root:
            dr_folder_list = ['ANormal', 'ARMD', 'CSR', 'Diabetic_retinopathy', 'Macular_Hole']
        elif 'JSIEC' in root:
            dr_folder_list = ['0.0.Normal', '20.Massive hard exudates',
                              '0.1.Tessellated fundus', '21.Yellow-white spots-flecks',
                              '0.2.Large optic cup', '22.Cotton-wool spots',
                              '0.3.DR1', '23.Vessel tortuosity',
                              '1.0.DR2', '24.Chorioretinal atrophy-coloboma',
                              '1.1.DR3', '25.Preretinal hemorrhage',
                              '10.0.Possible glaucoma', '26.Fibrosis',
                              '10.1.Optic atrophy', '27.Laser Spots',
                              '11.Severe hypertensive retinopathy', '28.Silicon oil in eye',
                              '12.Disc swelling and elevation', '29.0.Blur fundus without PDR',
                              '13.Dragged Disc', '29.1.Blur fundus with suspected PDR',
                              '14.Congenital disc abnormality', '3.RAO',
                              '15.0.Retinitis pigmentosa', '4.Rhegmatogenous RD',
                              '15.1.Bietti crystalline dystrophy', '5.0.CSCR',
                              '16.Peripheral retinal degeneration and break', '5.1.VKH disease',
                              '17.Myelinated nerve fiber', '6.Maculopathy',
                              '18.Vitreous particles', '7.ERM',
                              '19.Fundus neoplasm', '8.MH',
                              '2.0.BRVO', '9.Pathological myopia',
                              '2.1.CRVO']
        elif 'IDRiD' in root:
            dr_folder_list = ['anoDR', 'bmildDR', 'cmoderateDR', 'dsevereDR', 'eproDR']
        elif 'REFUGE1' in root:
            dr_folder_list = ['A_Non-Glaucoma', 'B_Glaucoma']
        elif 'FIVES' in root:
            dr_folder_list = ['Normal', 'AMD', 'DR', 'Glaucoma']
        elif 'ODIR' in root:
            dr_folder_list = ['A', 'C', 'D', 'G', 'H', 'M', 'N']
        elif 'MESSIDOR2' or 'APTOS2019' in root:
            dr_folder_list = ['anodr', 'bmilddr', 'cmoderatedr', 'dseveredr', 'eproliferativedr']

        for lbl, lbl_name in enumerate(dr_folder_list):
            img_files = os.listdir(os.path.join(root, split, lbl_name))
            for img_f in img_files:
                img_fpath = os.path.join(root, split, lbl_name, img_f)
                if drop_others and 'MESSIDOR2' in root:
                    # merge bmilddr (1), cmoderatedr (2), dseveredr (3)
                    if lbl in [1, 2, 3]:
                        merged_lbl = 1
                    elif lbl == 0:
                        merged_lbl = 0
                    else:  # lbl == 4
                        merged_lbl = 2
                    lbl_to_use = merged_lbl
                else:
                    lbl_to_use = lbl
                self.data.append({'img_fpath': img_fpath, 'label': lbl_to_use})
        self.split = split
        self.transform = transforms.Compose([
            transforms.Resize((imsize, imsize)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711))
        ])

    def __getitem__(self, index):
        entry = self.data[index]
        img = pil_loader(entry['img_fpath'])
        if self.transform is not None:
            img = self.transform(img)
        return img, entry['label']

    def __len__(self):
        return len(self.data)


# ------------------ Adapters -------------------
class ResidualAdapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super().__init__()
        self.ratio = 0.2
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_res = self.fc(x)
        x = self.ratio * x_res + (1 - self.ratio) * x
        return x


class CLIPAdapterHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.adapter = ResidualAdapter(dim)

    def forward(self, logits_scale, img_feats, text_feats):
        img_feats = F.normalize(self.adapter(img_feats), dim=-1)
        text_feats = F.normalize(text_feats, dim=-1)
        return logits_scale * img_feats @ text_feats.t()


class TIPAdapter(nn.Module):
    def __init__(self, dim: int, num_classes: int, alpha=1.0, beta=5.0, zst=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.zst = zst
        self.cache_keys = None
        self.cache_labels_onehot = None
        self.num_classes = num_classes

    def build_cache(self, feats, labels):
        keys = F.normalize(feats, dim=-1)
        N = keys.size(0)
        oh = torch.zeros(N, self.num_classes, device=feats.device)
        oh[torch.arange(N), labels] = 1.0
        self.cache_keys = keys
        self.cache_labels_onehot = oh

    def forward(self, logits_scale, feats, zs_logits):
        sim = feats @ self.cache_keys.t()
        cache_logits = torch.exp((-1) * (self.beta - self.beta * sim)) @ self.cache_labels_onehot
        # cache_logits = logits_scale * cache_logits
        logits = self.alpha * cache_logits + zs_logits / self.zst
        return logits


class TIPAdapterF(TIPAdapter):
    def __init__(self, dim: int, num_classes: int, init_alpha=1.0, init_beta=5.0, init_zst=1.0):
        super().__init__(dim, num_classes, init_alpha, init_beta, init_zst)
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.beta = nn.Parameter(torch.tensor(init_beta))
        self.zst = nn.Parameter(torch.tensor(init_zst))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath",
        default='/home/ubuntu/nfs/8T1/dujw/clip/datapath_clip/datasets/FIVES',
        type=str,
        help="Dataset root containing train/val/test"
    )
    parser.add_argument(
        "--resume",
        default='/home/ubuntu/nfs/8T2/dujw/experiments/Both_original_and_MAE_as_input/checkpoints/best/epoch2.pt',
        type=str,
        help="Path to checkpoint"
    )
    parser.add_argument(
        "--vision-config",
        default='../RET_CLIP_PLUS/clip/model_configs/ViT-B-16.json',
        type=str,
    )
    parser.add_argument(
        "--text-config",
        default='../RET_CLIP_PLUS/clip/model_configs/RoBERTa-wwm-ext-base-chinese.json',
        type=str
    )
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--epochs-adapter", type=int, default=40)
    parser.add_argument("--epochs-tipf", type=int, default=40)
    parser.add_argument("--output", type=str,
                        default="/home/ubuntu/nfs/8T1/dujw/clip/downstream_train_valid_test/few_shot/fives_ret-clip-plus-test.json",
                        help="Output JSON file")
    return parser.parse_args()


def build_text_features(model, text_classes, context_length, device):
    weights = []
    with torch.no_grad():
        for cname_list in text_classes:
            texts = [_preprocess_text(name) for name in cname_list]
            texts = tokenize(texts, context_length=context_length).to(device)
            class_emb, _, _ = model.encode_text(text=texts)
            class_emb /= class_emb.norm(dim=-1, keepdim=True)
            class_emb = class_emb.mean(dim=0)
            class_emb /= class_emb.norm()
            weights.append(class_emb)
        weights = torch.stack(weights, dim=1).to(device)
    return weights


def split_fewshot(dataset, k_shot, seed=SEED):
    rng = random.Random(seed)
    cls_to_indices = {}
    for idx, (_, y) in enumerate(dataset):
        cls_to_indices.setdefault(y, []).append(idx)
    support_idx, query_idx = [], []
    for c, idxs in cls_to_indices.items():
        rng.shuffle(idxs)
        support_idx += idxs[:k_shot]
        query_idx += idxs[k_shot:]
    return support_idx, query_idx


def evaluate_auc_map_aca(logits, labels, num_classes):
    logits = logits.cpu().numpy()
    labels = labels.cpu().numpy()
    auc = roc_auc_score(labels, logits, multi_class='ovr')
    mAP = average_precision_score(labels, logits)
    cm = confusion_matrix(labels, logits.argmax(axis=1), labels=list(range(num_classes)))
    class_acc = cm.diagonal() / cm.sum(axis=1)
    aca = np.mean(class_acc)
    return auc, mAP, aca


def evaluate_metrics(logits, labels, num_classes):
    logits = logits.cpu().numpy()
    labels = labels.cpu().numpy()

    if num_classes == 2:
        auc = roc_auc_score(labels, logits[:, 1])
        mAP = average_precision_score(labels, logits[:, 1])

        preds_label = logits.argmax(axis=1)
        acc = accuracy_score(labels, preds_label)
        return {"auc": auc, "mAP": mAP, "acc": acc}
    else:
        auc = roc_auc_score(labels, logits, multi_class='ovr')
        mAP = average_precision_score(labels, logits)
        cm = confusion_matrix(labels, logits.argmax(axis=1), labels=list(range(num_classes)))
        class_acc = cm.diagonal() / cm.sum(axis=1)
        aca = np.mean(class_acc)
        return {"auc": auc, "mAP": mAP, "aca": aca}


def main():
    args = parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    with open(args.vision_config, 'r') as fv, open(args.text_config, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info['vision_layers'], str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])
        for k, v in json.load(ft).items():
            model_info[k] = v

    model = CLIP(**model_info).to(device)
    ckpt = torch.load(args.resume, map_location="cpu")
    sd = {k.replace('module.', ''): v for k, v in ckpt.items() if "bert.pooler" not in k}
    model.load_state_dict(sd, strict=True)
    model.eval()

    # Datasets
    train_ds = RETFoundDataset(args.datapath, split="train", imsize=224, drop_others=True)
    test_ds = RETFoundDataset(args.datapath, split="test", imsize=224, drop_others=True)

    num_classes = len(set([y for _, y in test_ds]))

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Iterate k-shot values
    ks = [1, 5, 10]
    results_all = {str(k): {} for k in ks}

    # Define text classes (adjust to your dataset)
    text_classes = [
        ['正常眼底'],
        ['年龄相关性黄斑病变可疑', '年龄相关性黄斑病变'],
        ['糖尿病视网膜病变可疑', '糖尿病视网膜病变'],
        ['青光眼可疑', '青光眼'],
    ]

    text_features = build_text_features(model, text_classes, context_length=200, device=device)

    for k_shot in ks:
        # Few-shot split from train
        support_idx, _ = split_fewshot(train_ds, k_shot)
        support_ds = torch.utils.data.Subset(train_ds, support_idx)
        support_loader = torch.utils.data.DataLoader(support_ds, batch_size=k_shot, shuffle=True, drop_last=False,
                                                     num_workers=NUM_WORKERS)

        # Zero-shot baseline
        zs_logits_all, zs_labels_all = [], []
        with torch.no_grad():
            for img, y in tqdm(test_loader, desc=f"[k={k_shot}] Zero-shot"):
                img, y = img.to(device), y.to(device)
                feats = model.get_img_feats_global(img)
                feats = F.normalize(feats, dim=-1)
                logits = model.logit_scale.exp() * feats @ text_features
                zs_logits_all.append(logits.softmax(dim=-1))
                zs_labels_all.append(y)
        zs_logits_all = torch.cat(zs_logits_all, dim=0)
        zs_labels_all = torch.cat(zs_labels_all, dim=0)
        metrics_zs = evaluate_metrics(zs_logits_all, zs_labels_all, num_classes)
        results_all[str(k_shot)]["zero_shot"] = metrics_zs

        # CLIP-Adapter
        adapter = CLIPAdapterHead(dim=text_features.size(0)).to(device)
        opt = torch.optim.AdamW(adapter.parameters(), lr=1e-3, eps=1e-4)
        for ep in range(args.epochs_adapter):
            for img, y in support_loader:
                img, y = img.to(device), y.to(device)
                with torch.no_grad():
                    feats = F.normalize(model.get_img_feats_global(img), dim=-1)
                logits = adapter(model.logit_scale.exp(), feats, text_features.t())
                loss = F.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        ad_logits_all, ad_labels_all = [], []
        with torch.no_grad():
            for img, y in tqdm(test_loader, desc=f"[k={k_shot}] CLIP-Adapter"):
                img, y = img.to(device), y.to(device)
                feats = F.normalize(model.get_img_feats_global(img), dim=-1)
                logits = adapter(model.logit_scale.exp(), feats, text_features.t())
                ad_logits_all.append(logits.softmax(dim=-1))
                ad_labels_all.append(y)
        ad_logits_all = torch.cat(ad_logits_all, dim=0)
        ad_labels_all = torch.cat(ad_labels_all, dim=0)
        metrics_ad = evaluate_metrics(ad_logits_all, ad_labels_all, num_classes)
        results_all[str(k_shot)]["clip_adapter"] = metrics_ad

        # TIP-Adapter
        support_feats, support_labels = [], []
        with torch.no_grad():
            for img, y in support_loader:
                img, y = img.to(device), y.to(device)
                feats = F.normalize(model.get_img_feats_global(img), dim=-1)
                support_feats.append(feats)
                support_labels.append(y)
        support_feats = torch.cat(support_feats, dim=0)
        support_labels = torch.cat(support_labels, dim=0)
        tip = TIPAdapter(dim=support_feats.size(1), num_classes=num_classes).to(device)
        tip.build_cache(support_feats, support_labels)
        tip_logits_all, tip_labels_all = [], []
        with torch.no_grad():
            for img, y in tqdm(test_loader, desc=f"[k={k_shot}] TIP-Adapter"):
                img, y = img.to(device), y.to(device)
                feats = F.normalize(model.get_img_feats_global(img), dim=-1)
                zs_logits = model.logit_scale.exp() * feats @ text_features
                logits = tip(model.logit_scale.exp(), feats, zs_logits)
                tip_logits_all.append(logits.softmax(dim=-1))
                tip_labels_all.append(y)
        tip_logits_all = torch.cat(tip_logits_all, dim=0)
        tip_labels_all = torch.cat(tip_labels_all, dim=0)
        metrics_tip = evaluate_metrics(tip_logits_all, tip_labels_all, num_classes)
        results_all[str(k_shot)]["tip_adapter"] = metrics_tip

        # TIP-Adapter-F
        tipf = TIPAdapterF(dim=support_feats.size(1), num_classes=num_classes).to(device)
        tipf.build_cache(support_feats, support_labels)
        opt = torch.optim.AdamW([p for p in tipf.parameters() if p.requires_grad], lr=5e-3, eps=1e-4)
        for ep in range(args.epochs_tipf):
            for img, y in support_loader:
                img, y = img.to(device), y.to(device)
                with torch.no_grad():
                    feats = F.normalize(model.get_img_feats_global(img), dim=-1)
                    zs_logits = model.logit_scale.exp() * feats @ text_features
                logits = tipf(model.logit_scale.exp(), feats, zs_logits)
                loss = F.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        tipf_logits_all, tipf_labels_all = [], []
        with torch.no_grad():
            for img, y in tqdm(test_loader, desc=f"[k={k_shot}] TIP-Adapter-F"):
                img, y = img.to(device), y.to(device)
                feats = F.normalize(model.get_img_feats_global(img), dim=-1)
                zs_logits = model.logit_scale.exp() * feats @ text_features
                logits = tipf(model.logit_scale.exp(), feats, zs_logits)
                tipf_logits_all.append(logits.softmax(dim=-1))
                tipf_labels_all.append(y)
        tipf_logits_all = torch.cat(tipf_logits_all, dim=0)
        tipf_labels_all = torch.cat(tipf_labels_all, dim=0)
        metrics_tipf = evaluate_metrics(tipf_logits_all, tipf_labels_all, num_classes)
        results_all[str(k_shot)]["tip_adapter_f"] = metrics_tipf

        # Save results
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results_all, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {out_path.resolve()}")
        print(json.dumps(results_all, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
