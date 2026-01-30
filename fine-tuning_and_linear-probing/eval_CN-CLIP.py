import os
from PIL import Image
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import json
import math
from tqdm import tqdm
from time import gmtime, strftime
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score, \
    multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
import numpy as np
import logging
from cn_clip.training.logger import setup_primary_logging, setup_worker_logging
from cn_clip.clip.model import CLIP
from timm.loss import LabelSmoothingCrossEntropy

LINEAR_PROBING = True  # linear probing
DATA_DIR = '/home/ubuntu/nfs/8T1/dujw/clip/datapath_clip/datasets/FIVES'
RANK = 0
BATCH_SIZE = 16
NUM_WORKERS = 16


def _convert_to_rgb(image):
    return image.convert('RGB')


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class RETFoundDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.data = []
        if 'PAPILA' in root:
            dr_folder_list = ['anormal', 'bsuspectglaucoma', 'cglaucoma']
        elif 'Glaucoma_fundus' in root:
            dr_folder_list = ['anormal_control', 'bearly_glaucoma', 'cadvanced_glaucoma']
        elif 'Retina' in root:
            dr_folder_list = ['anormal', 'bcataract', 'cglaucoma', 'ddretina_disease']
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
        elif 'FIVES' in root:
            dr_folder_list = ['Normal', 'AMD', 'DR', 'Glaucoma']
        else:
            dr_folder_list = ['anodr', 'bmilddr', 'cmoderatedr', 'dseveredr', 'eproliferativedr']

        for lbl, lbl_name in enumerate(dr_folder_list):
            img_files = os.listdir(os.path.join(root, split, lbl_name))
            for img_f in img_files:
                img_fpath = os.path.join(root, split, lbl_name, img_f)
                self.data.append({'img_fpath': img_fpath, 'label': lbl})
        self.transform = transform

    def __getitem__(self, index):
        entry = self.data[index]
        img = pil_loader(entry['img_fpath'])
        if self.transform is not None:
            img = self.transform(img)
        return img, entry['label']

    def __len__(self):
        return len(self.data)


def eval_multiLabelCls_ViT(model, device):
    num_classes = 4
    resolution = 224

    train_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.RandomRotation(degrees=30, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomResizedCrop((resolution, resolution), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0., hue=0.),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
        transforms.RandomErasing(p=0.2)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    dataset_train = RETFoundDataset(root=DATA_DIR, split='train', transform=train_transform)
    dataset_val = RETFoundDataset(root=DATA_DIR, split='val', transform=val_transform)
    dataset_test = RETFoundDataset(root=DATA_DIR, split='test', transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=False
    )

    # for ViT-B-16
    classifier = torch.nn.Sequential(
        torch.nn.Linear(512, num_classes),
        torch.nn.Softmax()
    ).to(device)

    exclude = lambda n: "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n: not exclude(n)
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

    # linear probing
    if LINEAR_PROBING:
        for param in model.parameters():
            param.requires_grad = False

        for param in classifier.parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW(
            [
                {"params": classifier.parameters(), "lr": 1e-3, "weight_decay": 0.01, "betas": (0.9, 0.99)},
            ],
            eps=1e-6,
        )
    # fine-tuning
    else:
        for param in model.parameters():
            param.requires_grad = True

        for param in classifier.parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW(
            [
                {"params": gain_or_bias_params, "lr": 2e-6, "weight_decay": 0.01, "betas": (0.9, 0.98)},
                {"params": rest_params, "lr": 2e-6, "weight_decay": 0.01, "betas": (0.9, 0.98)},
                {"params": classifier.parameters(), "lr": 1e-3, "weight_decay": 0.01, "betas": (0.9, 0.99)},
            ],
            eps=1e-6,
        )

    from timm.data.mixup import Mixup
    mixup_fn = Mixup(
        mixup_alpha=0.2, cutmix_alpha=0.2, cutmix_minmax=None,
        prob=1.0, switch_prob=0.5, mode="batch",
        label_smoothing=0.1, num_classes=num_classes)

    criterion = LabelSmoothingCrossEntropy()

    best_auc = 0
    best_map = 0
    best_both = 0
    best_epoch = 0

    best_auc_test = 0
    best_map_test = 0
    best_both_test = 0
    best_epoch_test = 0

    best_aca = 0
    best_aca_test = 0

    for epoch in range(50):
        model.train()
        classifier.train()
        data_iter = iter(train_loader)
        for step in tqdm(range(len(data_iter))):
            image, label = next(data_iter)

            optimizer.zero_grad()

            imgs = image.to(device)
            label = label.to(device)

            if LINEAR_PROBING:
                with torch.no_grad():
                    image_features = model.encode_image(imgs)
            else:
                image_features = model.encode_image(imgs)

            probs = classifier(image_features)
            loss = criterion(probs, label)

            loss.backward()
            optimizer.step()
        logging.info(f"LR: {optimizer.param_groups[0]['lr']:6f} | ")

        with torch.no_grad():
            model.eval()
            classifier.eval()
            data_iter = iter(val_loader)
            preds = []
            labels = []
            for step in tqdm(range(len(data_iter))):
                image, label = next(data_iter)

                imgs = image.to(device)

                image_features = model.encode_image(imgs)

                pred = classifier(image_features)
                preds.append(pred)
                labels.append(label)

        preds = torch.cat(preds, dim=0).cpu().numpy()
        labels = torch.cat(labels, dim=0).cpu().numpy()
        auc = roc_auc_score(labels, preds, multi_class='ovr')
        map = average_precision_score(labels, preds)

        cm = confusion_matrix(labels, preds.argmax(axis=1), labels=list(range(num_classes)))
        class_acc = cm.diagonal() / cm.sum(axis=1)
        aca = np.mean(class_acc)

        if auc > best_auc:
            best_auc = auc
        if map > best_map:
            best_map = map
        if auc + map > best_both:
            best_both = auc + map
            best_epoch = epoch + 1
            best_aca = aca

        print(f"Epoch {epoch + 1}: AUC = {auc:.4f}, mAP = {map:.4f}, ACA = {aca:.4f}, Loss = {loss:.4f}")
        logging.info(
            f"Validation | Epoch {epoch + 1} | AUC: {auc:.6f} | mAP: {map:.6f} | ACA: {aca:.6f} | Loss: {loss:.6f}")

        with torch.no_grad():
            model.eval()
            classifier.eval()
            data_iter = iter(test_loader)
            preds_test = []
            labels_test = []
            for step in tqdm(range(len(data_iter))):
                image, label = next(data_iter)

                imgs = image.to(device)

                image_features = model.encode_image(imgs)

                pred_test = classifier(image_features)
                preds_test.append(pred_test)
                labels_test.append(label)

        preds_test = torch.cat(preds_test, dim=0).cpu().numpy()
        labels_test = torch.cat(labels_test, dim=0).cpu().numpy()
        auc_test = roc_auc_score(labels_test, preds_test, multi_class='ovr')
        map_test = average_precision_score(labels_test, preds_test)

        cm_test = confusion_matrix(labels_test, preds_test.argmax(axis=1), labels=list(range(num_classes)))
        class_acc_test = cm_test.diagonal() / cm_test.sum(axis=1)
        aca_test = np.mean(class_acc_test)

        logging.info(f"---TEST--- | AUC: {auc_test:.6f} | MAP: {map_test:.6f} | ACA: {aca_test:.6f}")

        if auc_test > best_auc_test:
            best_auc_test = auc_test
        if map_test > best_map_test:
            best_map_test = map_test
        if auc_test + map_test > best_both_test:
            best_both_test = auc_test + map_test
            best_epoch_test = epoch + 1
            best_aca_test = aca_test
    return best_auc, best_map, best_epoch, best_auc_test, best_map_test, best_epoch_test, best_aca, best_aca_test


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i in range(5):
        clip_resume = "/home/ubuntu/nfs/8T1/dujw/clip/datapath_clip/pretrained_weights/clip_cn_vit-b-16.pt"

        vision_model_config_file = "../cn_clip/clip/model_configs/ViT-B-16.json"
        print('Loading vision model config from', vision_model_config_file)
        assert os.path.exists(vision_model_config_file), "The vision_model_config_file does not exist!"

        text_model_config_file = "../cn_clip/clip/model_configs/RoBERTa-wwm-ext-base-chinese.json"
        print('Loading text model config from', text_model_config_file)
        assert os.path.exists(text_model_config_file), "The text_model_config_file does not exist!"

        with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
            model_info = json.load(fv)
            if isinstance(model_info['vision_layers'], str):
                model_info['vision_layers'] = eval(model_info['vision_layers'])
            for k, v in json.load(ft).items():
                model_info[k] = v

        model = CLIP(**model_info)
        model.set_grad_checkpointing()
        checkpoint = torch.load(clip_resume, map_location="cpu")
        sd = {k.replace('module.', ''): v for k, v in checkpoint["state_dict"].items() if "bert.pooler" not in k}
        # Load the state dict
        model.load_state_dict(sd, strict=True)

        for param in model.parameters():
            param.data = param.data.float()
        model = model.float().to(device=device)

        time_suffix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        log_path = '/home/ubuntu/nfs/8T1/dujw/clip/downstream_train_valid_test/FIVES'
        log_path = os.path.join(log_path, "cn_clip_test_{}.log".format(time_suffix))
        log_level = logging.INFO
        log_queue = setup_primary_logging(log_path, log_level, RANK)
        setup_worker_logging(RANK, log_queue, log_level)

        best_auc, best_map, best_epoch, best_auc_test, best_map_test, best_epoch_test, best_aca, best_aca_test = eval_multiLabelCls_ViT(
            model,
            device=device)
        logging.info(
            f"Best epoch_valid: {best_epoch:.6f} | "
            f"Best Valid AUC: {best_auc:.6f} | "
            f"Best Valid MAP: {best_map:.6f} | "
            f"Best Valid ACA: {best_aca:.6f} | "
        )
        logging.info(
            f"Best epoch_test: {best_epoch_test:.6f} | "
            f"Best Test AUC: {best_auc_test:.6f} | "
            f"Best Test MAP: {best_map_test:.6f} | "
            f"Best Test ACA: {best_aca_test:.6f} | "
        )


if __name__ == "__main__":
    main()
