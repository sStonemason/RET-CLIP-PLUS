import os
from PIL import Image
import torch
from torch import optim
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
from time import gmtime, strftime
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score, \
    multilabel_confusion_matrix
import logging
from RET_CLIP_PLUS.training.logger import setup_primary_logging, setup_worker_logging
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from dinov2.models import vision_transformer as vits

LINEAR_PROBING = False  # linear probing
DATA_DIR = '/home/ubuntu/nfs/8T1/dujw/clip/datapath_clip/datasets/REFUGE1'
RANK = 0
BATCH_SIZE = 16
NUM_WORKERS = 16


def _convert_to_rgb(image):
    return image.convert('RGB')


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class EvalDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.data = []

        if 'REFUGE1' in root:
            dr_folder_list = ['A_Non-Glaucoma', 'B_Glaucoma']

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
    num_classes = 2
    resolution = 224

    train_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.RandomRotation(degrees=30, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomResizedCrop((resolution, resolution), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0., hue=0.),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
        transforms.RandomErasing(p=0.1)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    dataset_train = EvalDataset(root=DATA_DIR, split='train', transform=train_transform)
    dataset_val = EvalDataset(root=DATA_DIR, split='val', transform=val_transform)
    dataset_test = EvalDataset(root=DATA_DIR, split='test', transform=val_transform)

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

    # for ViT
    classifier = torch.nn.Sequential(
        torch.nn.Linear(768, num_classes),
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
            mixup_alpha=0.2, cutmix_alpha=0., cutmix_minmax=None,
            prob=1., switch_prob=0., mode="elem",
            label_smoothing=0., num_classes=num_classes)

        criterion = SoftTargetCrossEntropy()

        best_auc = 0
        best_map = 0
        best_both = 0
        best_epoch = 0

        best_auc_test = 0
        best_map_test = 0
        best_both_test = 0
        best_epoch_test = 0

        best_acc = 0
        best_acc_test = 0
        best_f1 = 0
        best_f1_test = 0

        for epoch in range(100):
            model.train()
            classifier.train()
            data_iter = iter(train_loader)
            for step in tqdm(range(len(data_iter))):
                image, label = next(data_iter)

                optimizer.zero_grad()
                imgs = image.to(device)
                label = label.to(device)

                imgs, label = mixup_fn(imgs, label)
                if LINEAR_PROBING:
                    with torch.no_grad():
                        image_features = model(imgs)
                else:
                    image_features = model(imgs)

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

                    image_features = model(imgs)

                    logits = classifier(image_features)
                    pred = torch.softmax(logits, dim=-1)
                    preds.append(pred)
                    labels.append(label)

            preds = torch.cat(preds, dim=0).cpu().numpy()
            labels = torch.cat(labels, dim=0).cpu().numpy()
            auc = roc_auc_score(labels, preds[:, 1])
            map = average_precision_score(labels, preds[:, 1])

            preds_label = preds.argmax(axis=1)
            acc = accuracy_score(labels, preds_label)
            f1 = f1_score(labels, preds_label)

            if auc > best_auc:
                best_auc = auc
            if map > best_map:
                best_map = map
            if acc > best_acc:
                best_acc = acc
            if f1 > best_f1:
                best_f1 = f1
            if auc + map > best_both:
                best_both = auc + map
                best_epoch = epoch + 1

            print(
                f"Epoch {epoch + 1}: AUC = {auc:.4f}, mAP = {map:.4f}, ACC = {acc:.4f}, F1 = {f1:.4f}, Loss = {loss:.4f}")
            logging.info(
                f"Validation | Epoch {epoch + 1} | AUC: {auc:.6f} | mAP: {map:.6f} | ACC: {acc:.6f} | F1: {f1:.6f} | Loss: {loss:.6f}"
            )

            with torch.no_grad():
                model.eval()
                classifier.eval()
                data_iter = iter(test_loader)
                preds_test = []
                labels_test = []
                for step in tqdm(range(len(data_iter))):
                    image, label = next(data_iter)

                    imgs = image.to(device)

                    image_features = model(imgs)

                    logits = classifier(image_features)
                    pred_test = torch.softmax(logits, dim=-1)
                    preds_test.append(pred_test)
                    labels_test.append(label)

            preds_test = torch.cat(preds_test, dim=0).cpu().numpy()
            labels_test = torch.cat(labels_test, dim=0).cpu().numpy()
            auc_test = roc_auc_score(labels_test, preds_test[:, 1])
            map_test = average_precision_score(labels_test, preds_test[:, 1])

            preds_test_label = preds_test.argmax(axis=1)
            acc_test = accuracy_score(labels_test, preds_test_label)
            f1_test = f1_score(labels_test, preds_test_label)

            logging.info(
                f"---TEST--- | AUC: {auc_test:.6f} | MAP: {map_test:.6f} | ACC: {acc_test:.6f} | F1: {f1_test:.6f}")

            if auc_test > best_auc_test:
                best_auc_test = auc_test
            if map_test > best_map_test:
                best_map_test = map_test
            if acc_test > best_acc_test:
                best_acc_test = acc_test
            if f1_test > best_f1_test:
                best_f1_test = f1_test
            if auc_test + map_test > best_both_test:
                best_both_test = auc_test + map_test
                best_epoch_test = epoch + 1
        return best_auc, best_map, best_epoch, best_auc_test, best_map_test, best_epoch_test, best_acc, best_acc_test, best_f1, best_f1_test


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i in range(5):
        dino_resume = '/home/ubuntu/nfs/8T1/dujw/clip/datapath_clip/pretrained_weights/dinov2_vitb14_pretrain.pth'

        checkpoint = torch.load(dino_resume, map_location="cpu")

        vit_kwargs = dict(
            img_size=518,
            patch_size=14,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
            num_register_tokens=0,
            interpolate_antialias=False,
            interpolate_offset=0.1,
        )
        model = vits.__dict__["vit_base"](**vit_kwargs)

        model.set_grad_checkpointing()
        sd = checkpoint
        # Load the state dict
        model.load_state_dict(sd, strict=True)

        for param in model.parameters():
            param.data = param.data.float()
        model = model.float().to(device=device)

        time_suffix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        log_path = '/home/ubuntu/nfs/8T1/dujw/clip/downstream_train_valid_test/REFUGE1'
        log_path = os.path.join(log_path, "dinov2_ft_{}.log".format(time_suffix))
        log_level = logging.INFO
        log_queue = setup_primary_logging(log_path, log_level, RANK)
        setup_worker_logging(RANK, log_queue, log_level)

        best_auc, best_map, best_epoch, best_auc_test, best_map_test, best_epoch_test, best_acc, best_acc_test, best_f1, best_f1_test = eval_multiLabelCls_ViT(
            model,
            device=device)
        logging.info(
            f"Best epoch_valid: {best_epoch:.6f} | "
            f"Best Valid AUC: {best_auc:.6f} | "
            f"Best Valid MAP: {best_map:.6f} | "
            f"Best Valid ACC: {best_acc:.6f} | "
            f"Best Valid F1: {best_f1:.6f} | "
        )
        logging.info(
            f"Best epoch_test: {best_epoch_test:.6f} | "
            f"Best Test AUC: {best_auc_test:.6f} | "
            f"Best Test MAP: {best_map_test:.6f} | "
            f"Best Test ACC: {best_acc_test:.6f} | "
            f"Best Test F1: {best_f1_test:.6f} | "
        )


if __name__ == "__main__":
    main()
