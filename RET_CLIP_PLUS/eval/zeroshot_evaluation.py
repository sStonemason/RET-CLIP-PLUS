# -*- coding: utf-8 -*-
'''
This script performs zero-shot evaluation on ImageNet-1K. (with single-GPU)
'''

import os
import argparse
from pathlib import Path
import json
from tqdm import tqdm
from PIL import Image
import pandas as pd

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from RET_CLIP_PLUS.clip.model import convert_weights, CLIP
from RET_CLIP_PLUS.clip import tokenize
from RET_CLIP_PLUS.eval.data import get_zeroshot_dataset, _preprocess_text

# from cn_clip_test.cn_clip.clip.model import CLIP
# from cn_clip_test.cn_clip.clip import tokenize
# from cn_clip_test.cn_clip.eval.data import get_zeroshot_dataset, _preprocess_text

# from RET_CLIP.clip.model import convert_weights, CLIP
# from RET_CLIP.clip import tokenize
# from RET_CLIP.eval.data import get_zeroshot_dataset, _preprocess_text

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score, \
    multilabel_confusion_matrix, confusion_matrix
import numpy as np

# DATA_DIR = '/home/ubuntu/nfs/8T1/dujw/clip/downstream_train_valid_test/zero_shot/datasets/RFMiD'
# LABEL_DIR = '/home/ubuntu/nfs/8T1/dujw/clip/downstream_train_valid_test/zero_shot/datasets/RFMiD'
# DATA_DIR = '/home/ubuntu/nfs/8T1/dujw/clip/downstream_train_valid_test/zero_shot/datasets/Retina'
# LABEL_DIR = '/home/ubuntu/nfs/8T1/dujw/clip/downstream_train_valid_test/zero_shot/datasets/Retina'
# DATA_DIR = '/home/ubuntu/nfs/8T1/dujw/clip/datapath_clip/datasets/RETFound-Split/Retina/Retina'
# DATA_DIR = '/home/ubuntu/nfs/8T1/dujw/clip/datapath_clip/datasets/REFUGE1'
# DATA_DIR = '/home/ubuntu/nfs/8T1/dujw/clip/datapath_clip/datasets/FIVES'
# DATA_DIR = '/home/ubuntu/nfs/8T1/dujw/clip/datapath_clip/datasets/RETFound-Split/MESSIDOR2'
DATA_DIR='/home/ubuntu/nfs/8T1/dujw/clip/downstream_train_valid_test/zero_shot/datasets/ODIR'
# DATA_DIR = '/home/ubuntu/nfs/8T1/dujw/clip/datapath_clip/datasets/ODIR_single_eye'
BATCH_SIZE = 32
NUM_WORKERS = 4


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class RETFoundDataset(data.Dataset):
    def __init__(self, root, split, imsize=224, drop_others=True):
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
        else:
            dr_folder_list = ['anodr', 'bmilddr', 'cmoderatedr', 'dseveredr', 'eproliferativedr']

        for lbl, lbl_name in enumerate(dr_folder_list):
            img_files = os.listdir(os.path.join(root, split, lbl_name))
            for img_f in img_files:
                img_fpath = os.path.join(root, split, lbl_name, img_f)
                if drop_others:
                    # 将 bmilddr (1), cmoderatedr (2), dseveredr (3) 合并为1
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
        if self.split == 'train':
            self.transform = self.transforms_valid(resolution=imsize)
        else:
            self.transform = self.transforms_valid(resolution=imsize)

    def transforms_train(self, resolution):
        transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.RandomRotation(degrees=30, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop((resolution, resolution), scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0., hue=0.),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711)),
            transforms.RandomErasing(p=0.1)
        ])
        return transform

    def transforms_valid(self, resolution):
        transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        return transform

    def __getitem__(self, index):
        entry = self.data[index]
        img = pil_loader(entry['img_fpath'])
        if self.transform is not None:
            img = self.transform(img)
        return img, entry['label']

    def __len__(self):
        return len(self.data)


class ODIRDataset(data.Dataset):
    def __init__(self, data_dir, split='train', imsize=224, drop_others=False):

        self.split = split
        self.data_dir = data_dir
        self.label_dir = data_dir
        if self.split is not None:
            assert os.path.isdir(data_dir), "The data directory {} of {} split does not exist!".format(data_dir, split)
            self.image_dir = '{}/{}'.format(data_dir, self.split)
        else:
            self.image_dir = data_dir

        self.imsize = imsize

        if self.split == 'train':
            self.transform = self.transforms_train(resolution=imsize)
        elif self.split == 'valid' or 'test':
            self.transform = self.transforms_valid(resolution=imsize)
        self.len = 0

        if self.split == 'train':
            labelfile = self.label_dir + '/TrainingSet_zs.csv'
        elif self.split == 'valid':
            labelfile = self.label_dir + '/TrainingSet_zs.csv'
        elif self.split == 'test':
            labelfile = self.label_dir + '/TestSet_zs.csv'

        label_df = pd.read_csv(labelfile)

        self.imgfiles = []
        self.imglabels = []
        if self.split is not None:
            for _, _, files in os.walk(self.image_dir):
                for filename in files:
                    self.len += 1
                    label = torch.tensor(label_df.loc[label_df['img'] == filename].values[0][1:].astype(float)).float()
                    if drop_others:
                        if label[-1] == 1 and label.sum() == 1:
                            continue
                        else:
                            label = label[:-1]
                    self.imgfiles.append("{}/{}".format(self.image_dir, filename))
                    self.imglabels.append(label)

    def transforms_train(self, resolution):
        transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.RandomRotation(degrees=30, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop((resolution, resolution), scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0., hue=0.),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711)),
            transforms.RandomErasing(p=0.2)
        ])
        return transform

    def transforms_valid(self, resolution):
        transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        return transform

    def get_imgs(self, img_path, transform=None):
        if self.split == 'train':
            img = Image.open(img_path)
            if transform is not None:
                img = transform(img)
        elif self.split == 'valid' or 'test':
            img = Image.open(img_path).convert('RGB')
            if transform is not None:
                img = transform(img)
        return img

    def __getitem__(self, index):

        filepath = self.imgfiles[index]
        image = self.get_imgs(filepath, transform=self.transform)
        label = self.imglabels[index]

        return image, label

    def __len__(self):
        return len(self.imgfiles)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vision-model",
        choices=["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"],
        default="ViT-B-16",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--text-model",
        choices=["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese", "RBT3-chinese"],
        default="RoBERTa-wwm-ext-base-chinese",
        help="Name of the text backbone to use.",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )
    parser.add_argument(
        "--label-file",
        type=str,
        default='/home/ubuntu/nfs/8T1/dujw/clip/downstream_train_valid_test/zero_shot/datasets/ODIR/label_cn.txt',
        help="file for labels",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default='/home/ubuntu/nfs/8T1/dujw/clip/downstream_train_valid_test/zero_shot/datasets/Retina/test',
        required=False,
        help="Path to the test set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ODIR",
        help="Specified dataset.",
    )
    parser.add_argument(
        "--index",
        type=str,
        default="",
        help="Specify image paths.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/home/ubuntu/nfs/8T1/dujw/clip/downstream_train_valid_test/zero_shot/save_predictions",
        help="Specified dataset.",
    )
    # parser.add_argument(
    #     "--imagenet-val",
    #     type=str,
    #     required=True,
    #     help="Path to imagenet val set for conducting zero shot evaluation.",
    # )
    parser.add_argument(
        "--img-batch-size", type=int, default=BATCH_SIZE, help="Image batch size."
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=200,
        help="The maximum length of input text (include [CLS] & [SEP] tokens)."
    )
    parser.add_argument(
        "--resume",
        default='/home/ubuntu/nfs/8T2/dujw/experiments/Both_original_and_MAE_as_input/checkpoints/best/epoch2.pt',
        # default='/home/ubuntu/nfs/8T2/dujw/ret-clip.pt',
        # default='/home/ubuntu/nfs/8T2/dujw/experiments/revision-stage1/checkpoints/best/epoch9.pt',
        # default='/home/ubuntu/nfs/8T1/yangsz/6disease/AD_text_to_image/text2image-main/cn_clip/data/experiments/ALBEF_ita_loss/231220/no_mod_notonlymomentum_weightedmax_v2_lr3e-5_queue768_momentum0.75_addbertbnpost/checkpoints/epoch5.pt',
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=NUM_WORKERS, help="Number of workers for ImageNet dataloader."
    )
    args = parser.parse_args()

    return args


def zero_shot_classifier_p(model, classnames, args):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [_preprocess_text(name) for name in classname]  # format with class
            texts = tokenize(texts, context_length=args.context_length).to(args.gpu)  # tokenize
            class_embeddings, _, _ = model.encode_text(text=texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.gpu)
    return zeroshot_weights


def zero_shot_classifier_l(model, classnames, templates, args):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [_preprocess_text(template(classname)) for template in templates]  # format with class
            texts = tokenize(texts, context_length=args.context_length).to(args.gpu)  # tokenize
            _, class_embeddings, _ = model.encode_text(text=texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.gpu)
    return zeroshot_weights


def zero_shot_classifier_r(model, classnames, templates, args):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [_preprocess_text(template(classname)) for template in templates]  # format with class
            texts = tokenize(texts, context_length=args.context_length).to(args.gpu)  # tokenize
            _, _, class_embeddings = model.encode_text(text=texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.gpu)
    return zeroshot_weights


def zero_shot_classifier(model, classnames, templates, args):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [_preprocess_text(template(classname)) for template in templates]  # format with class
            texts = tokenize(texts, context_length=args.context_length).to(args.gpu)  # tokenize
            class_embeddings = model.encode_text(text=texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.gpu)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    total_logits = []
    total_targets = []
    with torch.no_grad():
        top1, top5, n = 0.0, 0.0, 0.0
        for images, target in tqdm(dataloader):
            images = images.to(args.gpu)
            target = target.to(args.gpu)
            total_targets.append(target)

            # predict
            # _, _, image_features = model(images, images, None)
            _, image_features, _ = model(images, images, None)
            # image_features, _, _ = model(images, images, None)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_features @ classifier).softmax(dim=-1)
            total_logits.append(logits)

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 1))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    outputs = torch.cat(total_logits, dim=0).cpu().numpy()
    targets = torch.cat(total_targets, dim=0).cpu().numpy()

    auc = roc_auc_score(targets, outputs, multi_class='ovr')
    map = average_precision_score(targets, outputs)

    if getattr(args, "index", ""):
        print("Use index to rearrange the logits...")
        with open(args.index, "r", encoding="utf-8") as f:
            index = json.load(f)
            print(index)
        outputs = outputs[index]
        targets = targets[index]
        print(targets)

    top1 = top1 / n
    top5 = top5 / n

    return top1, top5, auc, map, outputs


def run_for_my_dataloader(model, classifier, args):
    num_classes = 7
    # train_dataset = RFMiDDataset(data_dir=DATA_DIR, label_dir=LABEL_DIR, split='train', imsize=224)
    # val_dataset = RFMiDDataset(data_dir=DATA_DIR, label_dir=LABEL_DIR, split='valid', imsize=224)
    #
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True,
    #                                            num_workers=NUM_WORKERS)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False,
    #                                          num_workers=NUM_WORKERS)
    # test_dataset = RFMiDDataset(data_dir=DATA_DIR, label_dir=LABEL_DIR, split='test', imsize=224)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False,
    #                                           num_workers=NUM_WORKERS)

    # train_dataset = RetinaDataset(data_dir=DATA_DIR, label_dir=LABEL_DIR, split='train', imsize=224)
    # val_dataset = RetinaDataset(data_dir=DATA_DIR, label_dir=LABEL_DIR, split='valid', imsize=224)
    # test_dataset = RetinaDataset(data_dir=DATA_DIR, label_dir=LABEL_DIR, split='test', imsize=224)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False,
    #                                           num_workers=NUM_WORKERS)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True,
    #                                            num_workers=NUM_WORKERS)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False,
    #                                          num_workers=NUM_WORKERS)

    # train_dataset = ODIRDataset(data_dir=DATA_DIR, label_dir=LABEL_DIR, split='train', imsize=224)
    # val_dataset = ODIRDataset(data_dir=DATA_DIR, label_dir=LABEL_DIR, split='valid', imsize=224)
    # test_dataset = ODIRDataset(data_dir=DATA_DIR, label_dir=LABEL_DIR, split='test', imsize=224)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False,
    #                                           num_workers=NUM_WORKERS)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True,
    #                                            num_workers=NUM_WORKERS)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False,
    #                                          num_workers=NUM_WORKERS)
    # train_dataset = RETFoundDataset(root=DATA_DIR, split='train', imsize=224, drop_others=True)
    # val_dataset = RETFoundDataset(root=DATA_DIR, split='val', imsize=224, drop_others=True)
    # test_dataset = RETFoundDataset(root=DATA_DIR, split='test', imsize=224, drop_others=True)
    test_dataset = ODIRDataset(data_dir=DATA_DIR, split='test', imsize=224, drop_others=False)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False,
    #                                            num_workers=NUM_WORKERS)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False,
    #                                          num_workers=NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False,
                                              num_workers=NUM_WORKERS)

    total_logits = []
    total_targets = []
    with torch.no_grad():
        # for img, label in tqdm(train_loader):
        #     img = img.to(args.gpu)
        #     total_targets.append(label)
        #
        #     # _, images, target = next(test_iter)
        #     # images = images.to(args.gpu)
        #     # target = target.to(args.gpu)
        #     # total_targets.append(target)
        #
        #     # predict
        #     # image_features = model.get_img_feats_zs(img)
        #     image_features = model.get_img_feats_global(img)
        #     # _, image_features, _ = model(images, images, None)
        #     # image_features, _, _ = model(images, images, None)
        #     image_features /= image_features.norm(dim=-1, keepdim=True)
        #     logits = (model.logit_scale.exp() * image_features @ classifier).softmax(dim=-1)
        #     # logits = (10 * image_features @ classifier).sigmoid()
        #     total_logits.append(logits)
        #
        # for img, label in tqdm(val_loader):
        #     img = img.to(args.gpu)
        #     total_targets.append(label)
        #
        #     # _, images, target = next(test_iter)
        #     # images = images.to(args.gpu)
        #     # target = target.to(args.gpu)
        #     # total_targets.append(target)
        #
        #     # predict
        #     # image_features = model.get_img_feats_zs(img)
        #     image_features = model.get_img_feats_global(img)
        #     # _, image_features, _ = model(images, images, None)
        #     # image_features, _, _ = model(images, images, None)
        #     image_features /= image_features.norm(dim=-1, keepdim=True)
        #     logits = (model.logit_scale.exp() * image_features @ classifier).softmax(dim=-1)
        #     # logits = (10 * image_features @ classifier).sigmoid()
        #     total_logits.append(logits)

        for img, label in tqdm(test_loader):
            img = img.to(args.gpu)
            total_targets.append(label)

            # _, images, target = next(test_iter)
            # images = images.to(args.gpu)
            # target = target.to(args.gpu)
            # total_targets.append(target)

            # predict
            # image_features = model.get_img_feats_zs(img)
            image_features = model.get_img_feats_global(img)
            # _, image_features, _ = model(images, images, None)
            # image_features, _, _ = model(images, images, None)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = (model.logit_scale.exp() * image_features @ classifier).softmax(dim=-1)
            # logits = (10 * image_features @ classifier).sigmoid()
            total_logits.append(logits)

    outputs = torch.cat(total_logits, dim=0).cpu().numpy()
    targets = torch.cat(total_targets, dim=0).cpu().numpy()

    # auc0 = roc_auc_score(targets[:, 0], outputs[:, 0], multi_class='ovr')
    # auc1 = roc_auc_score(targets[:, 1], outputs[:, 1], multi_class='ovr')
    # auc2 = roc_auc_score(targets[:, 2], outputs[:, 2], multi_class='ovr')
    # auc3 = roc_auc_score(targets[:, 3], outputs[:, 3], multi_class='ovr')
    # auc4 = roc_auc_score(targets[:, 4], outputs[:, 4], multi_class='ovr')
    # auc5 = roc_auc_score(targets[:, 5], outputs[:, 5], multi_class='ovr')
    # auc6 = roc_auc_score(targets[:, 6], outputs[:, 6], multi_class='ovr')

    # map0 = average_precision_score(targets[:, 0], outputs[:, 0])
    # map1 = average_precision_score(targets[:, 1], outputs[:, 1])
    # map2 = average_precision_score(targets[:, 2], outputs[:, 2])
    # map3 = average_precision_score(targets[:, 3], outputs[:, 3])
    # map4 = average_precision_score(targets[:, 4], outputs[:, 4])
    # map5 = average_precision_score(targets[:, 5], outputs[:, 5])
    # map6 = average_precision_score(targets[:, 6], outputs[:, 6])

    # auc = roc_auc_score(targets, outputs, multi_class='ovr')
    auc = roc_auc_score(targets, outputs, multi_class='ovr')
    map = average_precision_score(targets, outputs)

    cm = confusion_matrix(targets.argmax(axis=1), outputs.argmax(axis=1), labels=list(range(num_classes)))
    class_acc = cm.diagonal() / cm.sum(axis=1)
    aca = np.mean(class_acc)

    # 二分类
    # auc = roc_auc_score(targets, outputs[:, 1])
    # map = average_precision_score(targets, outputs[:, 1])
    #
    # preds_label = outputs.argmax(axis=1)
    # acc = accuracy_score(targets, preds_label)

    # 多分类
    # preds_binary = (outputs > 0.5).astype(int)
    # macro_f1 = f1_score(targets, preds_binary, average='macro', zero_division=0)

    return auc, map, aca, outputs


if __name__ == "__main__":
    args = parse_args()

    # Log params.
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")

    args.gpu = 2
    torch.cuda.set_device(args.gpu)

    # Initialize the model.
    vision_model_config_file = "../clip/model_configs/ViT-B-16.json"
    print('Loading vision model config from', vision_model_config_file)
    assert os.path.exists(vision_model_config_file), "The vision_model_config_file does not exist!"

    text_model_config_file = "../clip/model_configs/RoBERTa-wwm-ext-base-chinese.json"
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
    # convert_weights(model)
    #
    # # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    # if args.precision == "amp" or args.precision == "fp32":
    #     convert_models_to_fp32(model)
    # model.cuda(args.gpu)
    # if args.precision == "fp16":
    #     convert_weights(model)

    # # Get eval data.
    # print("Preparing zeroshot dataset.")
    # data = {}
    # print(f"{model_info['image_resolution']}")
    # data[args.dataset] = get_zeroshot_dataset(
    #     args, image_transform(model_info["image_resolution"])
    # )

    # Resume from a checkpoint.
    print("Begin to load model checkpoint from {}.".format(args.resume))
    assert os.path.exists(args.resume), "The checkpoint file {} not exists!".format(args.resume)
    # Map model to be loaded to specified single gpu.
    loc = "cuda:{}".format(args.gpu)
    checkpoint = torch.load(args.resume, map_location="cpu")
    sd = {k.replace('module.', ''): v for k, v in checkpoint.items() if "bert.pooler" not in k}
    # sd = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items() if "bert.pooler" not in k}
    # Load the state dict
    model.load_state_dict(sd, strict=False)

    for param in model.parameters():
        param.data = param.data.float()
    model = model.float().to(device=args.gpu)

    # Compute ensembled class embeddings
    print('Building zero-shot classifier')

    model.eval()

    # f = open(args.label_file, "r", encoding="utf8")
    # classnames = [line.strip() for line in f.readlines()]
    #
    # templates = [
    #     lambda c: f"{c}",
    # ]

    # text_classes = [
    #     ['Normal fundus', 'Normal'],
    #     ['Suspected cataract', 'Cataract'],
    #     ['Suspected glaucoma', 'Glaucoma'],
    # ]

    # text_classes = [
    #     ['正常眼底'],
    #     # ['年龄相关性黄斑病变可疑', '年龄相关性黄斑病变'],
    #     ['糖尿病视网膜病变可疑', '糖尿病视网膜病变'],
    #     # ['白内障可疑', '白内障'],
    #     ['青光眼可疑', '青光眼'],
    # ]
    text_classes = [
        ['正常眼底'],
        ['糖尿病视网膜病变可疑', '糖尿病视网膜病变'],
        ['青光眼可疑', '青光眼'],
        ['白内障可疑', '白内障'],
        ['年龄相关性黄斑病变可疑', '年龄相关性黄斑病变'],
        ['高血压性视网膜病变'],
        ['病理性近视'],
    ]
    # text_classes = [
    #     ['正常眼底'],
    #     ['双眼轻度糖尿病视网膜病变'],
    #     ['双眼中度非增生性糖尿病视网膜病变', '双眼中度非增殖性糖尿病视网膜病变'],
    #     ['双眼重度非增生性糖尿病视网膜病变', '双眼重度非增殖性糖尿病视网膜病变'],
    #     ['双眼增生性糖尿病视网膜病变', '双眼增殖性糖尿病视网膜病变'],
    # ]

    # text_classes = [
    #     ['正常眼底', '无明显异常'],
    #     ['轻度糖尿病视网膜病变', '糖尿病视网膜病变Ⅰ期'],
    #     ['中度非增殖性糖尿病视网膜病变', '糖尿病视网膜病变Ⅱ期'],
    #     ['重度非增殖性糖尿病视网膜病变', '糖尿病视网膜病变Ⅲ期'],
    #     ['P糖尿病视网膜病变','新生血管'],
    # ]

    # text_classes = [
    #     ['正常眼底', '无明显异常'],
    #     ['非增殖性糖尿病视网膜病变', '糖尿病视网膜病变Ⅰ期', '糖尿病视网膜病变Ⅱ期', '糖尿病视网膜病变Ⅲ期',
    #      '轻度糖尿病视网膜病变', '中度非增殖性糖尿病视网膜病变', '重度非增殖性糖尿病视网膜病变'],
    #     ['P糖尿病视网膜病变'],
    # ]

    text_classes = [
        ['正常眼底'],
        ['轻度糖尿病视网膜病变', '糖尿病视网膜病变Ⅰ期'],
        ['中度非增殖性糖尿病视网膜病变', '糖尿病视网膜病变Ⅱ期', '硬性渗出，出血'],
        ['重度非增殖性糖尿病视网膜病变', '糖尿病视网膜病变Ⅲ期', '棉絮斑'],
        ['P糖尿病视网膜病变', '新生血管'],
    ]

    # text_classes = [
    #     ['正常眼底'],
    #     ['糖尿病视网膜病变Ⅰ期'],
    #     ['糖尿病视网膜病变Ⅱ期'],
    #     ['糖尿病视网膜病变Ⅲ期'],
    #     ['P糖尿病视网膜病变'],
    # ]

    # text_classes = [
    #     ['双眼正常眼底'],
    #     ['双眼年龄相关性黄斑病变'],
    #     ['双眼糖尿病视网膜病变'],
    #     # ['白内障'],
    #     ['双眼青光眼'],
    # ]

    # text_classes = [
    #     ['双眼正常眼底'],
    #     ['双眼轻度糖尿病视网膜病变'],
    #     ['双眼中度非增生性糖尿病视网膜病变'],
    #     ['双眼重度非增生性糖尿病视网膜病变'],
    #     ['双眼增生性糖尿病视网膜病变'],
    # ]

    # Make inference and evaluation
    print('Using classifier')
    classifier = zero_shot_classifier_p(model, text_classes, args)
    # classifier_l = zero_shot_classifier_l(model, classnames, templates, args)
    # classifier_r = zero_shot_classifier_r(model, classnames, templates, args)
    # stacked_tensors = torch.stack([classifier_p, classifier_l, classifier_r])
    # classifier = torch.mean(stacked_tensors, dim=0)
    results = {}
    # top1, top5, auc, map, logits = run(model, classifier, data[args.dataset].dataloader, args)
    auc, map, aca, logits = run_for_my_dataloader(model, classifier, args)


    def json_prec_dump(data, prec=6):
        return json.dumps(
            json.loads(json.dumps(data), parse_float=lambda x: round(float(x), prec))
        )


    # print(logits.size())
    output_dict = {
        "model_name": "RET-CLIP-" + args.vision_model,
        "dataset_name": args.dataset,
        "num_trainable_params": 0,
        "num_params": sum(x.numel() for x in model.parameters()),
        "num_visual_params": sum(x.numel() for x in model.visual.parameters()),
        "num_backbone_params": sum(x.numel() for x in model.parameters()),
        "n_shot": 0,
        "rnd_seeds": [123],
        "predictions": [logits.tolist()],
    }
    json_string = json_prec_dump(output_dict)
    with open(os.path.join(args.save_dir, f"{args.dataset}.json"), "w", encoding="utf-8") as w:
        w.write(json_string)

    # results["zeroshot-top1"] = top1
    # results["zeroshot-top5"] = top5
    results["zeroshot-auc"] = auc
    results["zeroshot-map"] = map
    results["zeroshot-aca"] = aca

    print('Result:')
    print(", ".join(["{}: {}".format(k, v) for k, v in results.items()]))
    print('Finished.')
