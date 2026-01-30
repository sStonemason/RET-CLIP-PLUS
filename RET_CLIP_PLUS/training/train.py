import copy
import os
import time
import json
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.cuda.amp import autocast
import torch.distributed.nn
import torch.distributed as dist
import torch.nn.functional as F

from RET_CLIP_PLUS.clip.model import convert_state_dict
from eval_RFMiD import eval_multiLabelCls_ViT, eval_multiLabelCls_RN50


def is_master(args):
    return args.rank == 0


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def attention_fn(query, context, temp1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    # ih, iw = context.size(2), context.size(3)
    sourceL = context.size(2)

    # --> batch x sourceL x ndf
    # context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query)
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax(dim=-1)(attn)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)

    attn = attn * temp1
    attn = nn.Softmax(dim=-1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext


def local_loss(
        img_features, words_emb, cap_lens, temp1=4.0, temp2=5.0, temp3=10.0, agg="sum"
):
    batch_size = img_features.shape[0]

    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(words_emb.shape[0]):

        # Get the i-th text description
        words_num = cap_lens[i] - 1  # 25
        # TODO: remove [SEP]
        # word = words_emb[i, :, 1:words_num+1].unsqueeze(0).contiguous()    # [1, 512, 25]
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, 512, 25]
        word = word.repeat(batch_size, 1, 1)  # [256, 512, 25]
        context = img_features  # [256, 512, 2]

        weiContext = attention_fn(
            word, context, temp1
        )  # [256, 512, 25]

        word = word.transpose(1, 2).contiguous()  # [256, 25, 512]
        weiContext = weiContext.transpose(1, 2).contiguous()  # [256, 25, 512]

        word = word.view(batch_size * words_num, -1)  # [256*25, 512]
        weiContext = weiContext.view(batch_size * words_num, -1)  # [256*25, 512]

        row_sim = cosine_similarity(word, weiContext)
        row_sim = row_sim.view(batch_size, words_num)  # [256, 25]

        row_sim.mul_(temp2).exp_()
        if agg == "sum":
            row_sim = row_sim.sum(dim=1, keepdim=True)  # [256, 1]
        else:
            row_sim = row_sim.mean(dim=1, keepdim=True)  # [256, 1]
        row_sim = torch.log(row_sim)

        similarities.append(row_sim)

    similarities = torch.cat(similarities, 1)  #
    similarities = similarities * temp3
    similarities1 = similarities.transpose(0, 1)  # [256, 256]

    labels = Variable(torch.LongTensor(range(batch_size))).to(similarities.device)

    loss0 = nn.CrossEntropyLoss()(similarities, labels)  # labels: arange(batch_size)
    loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    loss = (loss0 + loss1) / 2
    return loss


def get_loss(epoch, model, img_l, img_r, img_p, texts, loss_img, loss_txt, eos_indices, args, ):
    stage_1_epochs = 2  # stage_1: CLIP (epoch 0-1)
    stage_2_epochs = 3  # stage_2: CLIP + MAE (epoch 2-4)
    # stage_3: epoch >= 5, VisionFlow

    is_stage_1 = epoch < stage_1_epochs
    is_stage_2 = stage_1_epochs <= epoch < (stage_1_epochs + stage_2_epochs)
    is_stage_3 = epoch >= (stage_1_epochs + stage_2_epochs)

    # =====================================================
    # stage_1: CLIP (epoch 0-1)
    # =====================================================
    if is_stage_1:
        image_features, left_features, right_features, text_features, text_features_left, text_features_right, \
            logit_scale, logit_scale_left, logit_scale_right = model.module.forward_CLIP(img_l, img_r, texts)
        MAE_loss = torch.tensor(0.0).cuda()

        logit_scale = logit_scale.mean()
        logit_scale_left = logit_scale_left.mean()
        logit_scale_right = logit_scale_right.mean()

        if args.aggregate:
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            if args.gather_with_grad:
                all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
                all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
                all_left_features = torch.cat(torch.distributed.nn.all_gather(left_features), dim=0)
                all_text_features_left = torch.cat(torch.distributed.nn.all_gather(text_features_left), dim=0)
                all_right_features = torch.cat(torch.distributed.nn.all_gather(right_features), dim=0)
                all_text_features_right = torch.cat(torch.distributed.nn.all_gather(text_features_right), dim=0)
            else:
                gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
                gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
                dist.all_gather(gathered_image_features, image_features)
                dist.all_gather(gathered_text_features, text_features)
                all_image_features = torch.cat(
                    [image_features] + gathered_image_features[:rank] + gathered_image_features[rank + 1:])
                all_text_features = torch.cat(
                    [text_features] + gathered_text_features[:rank] + gathered_text_features[rank + 1:])

            logits_per_image = logit_scale * all_image_features @ all_text_features.t()
            logits_per_text = logits_per_image.t()
            logits_per_left_image = logit_scale_left * all_left_features @ all_text_features_left.t()
            logits_per_left_text = logits_per_left_image.t()
            logits_per_right_image = logit_scale_right * all_right_features @ all_text_features_right.t()
            logits_per_right_text = logits_per_right_image.t()
        else:
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            logits_per_left_image = logit_scale_left * left_features @ text_features_left.t()
            logits_per_left_text = logits_per_left_image.t()
            logits_per_right_image = logit_scale_right * right_features @ text_features_right.t()
            logits_per_right_text = logits_per_right_image.t()

        # ground truth
        ground_truth = torch.arange(len(logits_per_image)).long().cuda(args.local_device_rank, non_blocking=True)
        ground_truth_left = torch.arange(len(logits_per_left_image)).long().cuda(args.local_device_rank,
                                                                                 non_blocking=True)
        ground_truth_right = torch.arange(len(logits_per_right_image)).long().cuda(args.local_device_rank,
                                                                                   non_blocking=True)

        # CLIP losses
        global_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        left_loss = (loss_img(logits_per_left_image, ground_truth_left) + loss_txt(logits_per_left_text,
                                                                                   ground_truth_left)) / 2
        right_loss = (loss_img(logits_per_right_image, ground_truth_right) + loss_txt(logits_per_right_text,
                                                                                      ground_truth_right)) / 2

        final_CLIP_loss = (left_loss + right_loss + global_loss) / 3.0
        final_MAE_loss = torch.tensor(0.0).cuda()
        total_loss = final_CLIP_loss

        acc = None
        if args.report_training_batch_acc:
            i2t_acc = (logits_per_image.argmax(-1) == ground_truth).sum() / len(logits_per_image)
            t2i_acc = (logits_per_text.argmax(-1) == ground_truth).sum() / len(logits_per_text)
            acc = {"i2t": i2t_acc, "t2i": t2i_acc}

        return total_loss, final_MAE_loss, final_CLIP_loss, acc, MAE_loss, global_loss, left_loss, right_loss

    # =====================================================
    # stage_2: CLIP + MAE (epoch 2-4)
    # =====================================================
    elif is_stage_2:
        MAE_loss = model.module.forward_MAE(img_p)
        image_features, left_features, right_features, text_features, text_features_left, text_features_right, \
            logit_scale, logit_scale_left, logit_scale_right = model.module.forward_CLIP(img_l, img_r, texts)

        logit_scale = logit_scale.mean()
        logit_scale_left = logit_scale_left.mean()
        logit_scale_right = logit_scale_right.mean()

        if args.aggregate:
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            if args.gather_with_grad:
                all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
                all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
                all_left_features = torch.cat(torch.distributed.nn.all_gather(left_features), dim=0)
                all_text_features_left = torch.cat(torch.distributed.nn.all_gather(text_features_left), dim=0)
                all_right_features = torch.cat(torch.distributed.nn.all_gather(right_features), dim=0)
                all_text_features_right = torch.cat(torch.distributed.nn.all_gather(text_features_right), dim=0)

                all_MAE_loss = MAE_loss.clone()
                dist.all_reduce(all_MAE_loss, op=dist.ReduceOp.SUM)
                all_MAE_loss /= world_size
            else:
                gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
                gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
                dist.all_gather(gathered_image_features, image_features)
                dist.all_gather(gathered_text_features, text_features)
                all_image_features = torch.cat(
                    [image_features] + gathered_image_features[:rank] + gathered_image_features[rank + 1:])
                all_text_features = torch.cat(
                    [text_features] + gathered_text_features[:rank] + gathered_text_features[rank + 1:])

            logits_per_image = logit_scale * all_image_features @ all_text_features.t()
            logits_per_text = logits_per_image.t()
            logits_per_left_image = logit_scale_left * all_left_features @ all_text_features_left.t()
            logits_per_left_text = logits_per_left_image.t()
            logits_per_right_image = logit_scale_right * all_right_features @ all_text_features_right.t()
            logits_per_right_text = logits_per_right_image.t()
        else:
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            logits_per_left_image = logit_scale_left * left_features @ text_features_left.t()
            logits_per_left_text = logits_per_left_image.t()
            logits_per_right_image = logit_scale_right * right_features @ text_features_right.t()
            logits_per_right_text = logits_per_right_image.t()

        # ground truth
        ground_truth = torch.arange(len(logits_per_image)).long().cuda(args.local_device_rank, non_blocking=True)
        ground_truth_left = torch.arange(len(logits_per_left_image)).long().cuda(args.local_device_rank,
                                                                                 non_blocking=True)
        ground_truth_right = torch.arange(len(logits_per_right_image)).long().cuda(args.local_device_rank,
                                                                                   non_blocking=True)

        # CLIP losses
        global_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        left_loss = (loss_img(logits_per_left_image, ground_truth_left) + loss_txt(logits_per_left_text,
                                                                                   ground_truth_left)) / 2
        right_loss = (loss_img(logits_per_right_image, ground_truth_right) + loss_txt(logits_per_right_text,
                                                                                      ground_truth_right)) / 2

        final_CLIP_loss = (left_loss + right_loss + global_loss) / 3.0
        final_MAE_loss = MAE_loss * 10

        if args.aggregate:
            world_size = dist.get_world_size()
            dist.all_reduce(final_MAE_loss, op=dist.ReduceOp.SUM)
            final_MAE_loss /= world_size

        total_loss = final_CLIP_loss + final_MAE_loss

        acc = None
        if args.report_training_batch_acc:
            i2t_acc = (logits_per_image.argmax(-1) == ground_truth).sum() / len(logits_per_image)
            t2i_acc = (logits_per_text.argmax(-1) == ground_truth).sum() / len(logits_per_text)
            acc = {"i2t": i2t_acc, "t2i": t2i_acc}

        return total_loss, final_MAE_loss, final_CLIP_loss, acc, MAE_loss, global_loss, left_loss, right_loss

    # =====================================================
    # stage_3: epoch >= 5, VisionFlow
    # =====================================================
    else:  # is_stage_3
        image_features, left_features, right_features, MAE_loss, text_features, text_features_left, text_features_right, \
            logit_scale, logit_scale_left, logit_scale_right = model.module.forward_VisionFlow(img_p, texts)
        logit_scale = logit_scale.mean()
        logit_scale_left = logit_scale_left.mean()
        logit_scale_right = logit_scale_right.mean()

        if args.aggregate:
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            if args.gather_with_grad:
                all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
                all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
                all_left_features = torch.cat(torch.distributed.nn.all_gather(left_features), dim=0)
                all_text_features_left = torch.cat(torch.distributed.nn.all_gather(text_features_left), dim=0)
                all_right_features = torch.cat(torch.distributed.nn.all_gather(right_features), dim=0)
                all_text_features_right = torch.cat(torch.distributed.nn.all_gather(text_features_right), dim=0)

                all_MAE_loss = MAE_loss.clone()
                dist.all_reduce(all_MAE_loss, op=dist.ReduceOp.SUM)
                all_MAE_loss /= world_size
            else:
                gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
                gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
                dist.all_gather(gathered_image_features, image_features)
                dist.all_gather(gathered_text_features, text_features)
                all_image_features = torch.cat(
                    [image_features] + gathered_image_features[:rank] + gathered_image_features[rank + 1:])
                all_text_features = torch.cat(
                    [text_features] + gathered_text_features[:rank] + gathered_text_features[rank + 1:])

            logits_per_image = logit_scale * all_image_features @ all_text_features.t()
            logits_per_text = logits_per_image.t()
            logits_per_left_image = logit_scale_left * all_left_features @ all_text_features_left.t()
            logits_per_left_text = logits_per_left_image.t()
            logits_per_right_image = logit_scale_right * all_right_features @ all_text_features_right.t()
            logits_per_right_text = logits_per_right_image.t()
        else:
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            logits_per_left_image = logit_scale_left * left_features @ text_features_left.t()
            logits_per_left_text = logits_per_left_image.t()
            logits_per_right_image = logit_scale_right * right_features @ text_features_right.t()
            logits_per_right_text = logits_per_right_image.t()

        # ground truth
        ground_truth = torch.arange(len(logits_per_image)).long().cuda(args.local_device_rank, non_blocking=True)
        ground_truth_left = torch.arange(len(logits_per_left_image)).long().cuda(args.local_device_rank,
                                                                                 non_blocking=True)
        ground_truth_right = torch.arange(len(logits_per_right_image)).long().cuda(args.local_device_rank,
                                                                                   non_blocking=True)

        # CLIP losses
        global_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        left_loss = (loss_img(logits_per_left_image, ground_truth_left) + loss_txt(logits_per_left_text,
                                                                                   ground_truth_left)) / 2
        right_loss = (loss_img(logits_per_right_image, ground_truth_right) + loss_txt(logits_per_right_text,
                                                                                      ground_truth_right)) / 2

        final_CLIP_loss = (left_loss + right_loss + global_loss) / 3.0
        final_MAE_loss = MAE_loss * 10

        if args.aggregate:
            world_size = dist.get_world_size()
            dist.all_reduce(final_MAE_loss, op=dist.ReduceOp.SUM)
            final_MAE_loss /= world_size

        total_loss = final_CLIP_loss + final_MAE_loss

        acc = None
        if args.report_training_batch_acc:
            i2t_acc = (logits_per_image.argmax(-1) == ground_truth).sum() / len(logits_per_image)
            t2i_acc = (logits_per_text.argmax(-1) == ground_truth).sum() / len(logits_per_text)
            acc = {"i2t": i2t_acc, "t2i": t2i_acc}

        return total_loss, final_MAE_loss, final_CLIP_loss, acc, MAE_loss, global_loss, left_loss, right_loss


def freeze_vision_bn(args, model):
    # freeze bn running mean and variance
    if 'RN' in args.vision_model:
        RN_visual_modules = model.module.visual.modules() if isinstance(model,
                                                                        nn.parallel.DistributedDataParallel) else model.visual.modules()
        for m in RN_visual_modules:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def train(model, data, epoch, optimizer, scaler, scheduler, args, global_trained_steps, teacher_model=None):
    # os.environ["WDS_EPOCH"] = str(epoch)

    model.train()
    if args.freeze_vision:
        freeze_vision_bn(args, model)

    dataloader, sampler = data['train'].dataloader, data['train'].sampler

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    loss_img = loss_img.cuda(args.local_device_rank)
    loss_txt = loss_txt.cuda(args.local_device_rank)

    if sampler is not None:
        sampler.set_epoch(epoch)

    num_steps_per_epoch = dataloader.num_batches // args.accum_freq
    data_iter = iter(dataloader)

    if args.accum_freq > 1:
        accum_img_l, accum_img_r, accum_texts, accum_image_features, accum_text_features = [], [], [], [], []
        if args.distllation:
            teacher_accum_image_features = []

    end = time.time()
    epoch_trained_steps = 0
    for i in range(0, dataloader.num_batches):
        batch = next(data_iter)

        i_accum = i // args.accum_freq
        step = num_steps_per_epoch * epoch + i_accum
        # reach the args.max_steps, exit training:
        if step >= args.max_steps:
            logging.info("Stopping training due to step {} has reached max_steps {}".format(step,
                                                                                            args.max_steps // args.accum_freq))
            return epoch_trained_steps
        scheduler(step)

        optimizer.zero_grad()

        img_l, img_r, img_l_p, img_r_p, texts, eos_indices = batch
        img_p = torch.cat((img_l_p, img_r_p), dim=0)

        img_l = img_l.cuda(args.local_device_rank, non_blocking=True)
        img_r = img_r.cuda(args.local_device_rank, non_blocking=True)
        img_p = img_p.cuda(args.local_device_rank, non_blocking=True)
        texts = texts.cuda(args.local_device_rank, non_blocking=True)
        eos_indices = eos_indices.cuda(args.local_device_rank, non_blocking=True)

        data_time = time.time() - end

        m = model.module

        # with automatic mixed precision.
        if args.precision == "amp":
            with autocast():
                if args.distllation:
                    total_loss, acc = get_loss(model, img_l, img_r, texts, loss_img, loss_txt, eos_indices, args,
                                               teacher_model=teacher_model)
                else:
                    total_loss, final_MAE_loss, final_CLIP_loss, acc, MAE_loss, global_loss, left_loss, right_loss = get_loss(
                        epoch, model, img_l,
                        img_r,
                        img_p,
                        texts,
                        loss_img, loss_txt,
                        eos_indices,
                        args)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()

        else:
            if args.distllation:
                total_loss, acc = get_loss(model, img_l, img_r, texts, loss_img, loss_txt, eos_indices, args,
                                           teacher_model=teacher_model)
            else:
                total_loss, acc = get_loss(model, img_l, img_r, texts, loss_img, loss_txt, eos_indices, args)
            total_loss.backward()
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)
        m.logit_scale_left.data = torch.clamp(m.logit_scale_left.data, 0, 4.6052)
        m.logit_scale_right.data = torch.clamp(m.logit_scale_right.data, 0, 4.6052)

        batch_time = time.time() - end
        end = time.time()

        epoch_trained_steps += 1

        if is_master(args) and ((step + 1) % args.log_interval) == 0:
            batch_size = len(img_l) * args.accum_freq
            num_samples = (i_accum + 1) * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * (i_accum + 1) / num_steps_per_epoch

            logging.info(
                f"Global Steps: {step + 1}/{args.max_steps} | " +
                f"Train Epoch: {epoch + 1} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)] | " +
                f"Loss: {total_loss.item():.6f} | " +
                f"Global_Loss: {global_loss.item():.6f} | " +
                f"Left_Loss: {left_loss.item():.6f} | " +
                f"Right_Loss: {right_loss.item():.6f} | " +
                f"MAE_Loss: {MAE_loss.item():.6f} | " +
                (f"Image2Text Acc: {acc['i2t'].item() * 100:.2f} | " if args.report_training_batch_acc else "") +
                (f"Text2Image Acc: {acc['t2i'].item() * 100:.2f} | " if args.report_training_batch_acc else "") +
                f"Data Time: {data_time:.3f}s | " +
                f"Batch Time: {batch_time:.3f}s | " +
                f"LR: {optimizer.param_groups[0]['lr']:5f} | " +
                f"logit_scale: {m.logit_scale.data:.3f} | " +
                # f"logit_scale_left: {m.logit_scale_left.data:.3f} | " +
                # f"logit_scale_right: {m.logit_scale_right.data:.3f} | " +
                f"Global Batch Size: {batch_size * args.world_size}"
            )

        if args.val_data is not None and args.valid_step_interval is not None and (
                (step + 1) % args.valid_step_interval) == 0:
            assert "val" in data, "Error: Valid dataset has not been built."
            if not args.use_flash_attention:
                evaluate(model, data, epoch, args, step + 1)
            else:
                # fp16 is needed in flash attention
                with autocast():
                    evaluate(model, data, epoch, args, step + 1)
            # set model back to train mode
            model.train()
            if args.freeze_vision:
                freeze_vision_bn(args, model)

        if args.should_save and args.save_step_frequency > 0 and ((step + 1) % args.save_step_frequency) == 0:
            save_path = os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}_{step + 1}.pt")
            t1 = time.time()
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "name": args.name,
                    "state_dict": model.state_dict() if not args.use_flash_attention else convert_state_dict(
                        model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            logging.info(
                "Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, epoch + 1,
                                                                                             step + 1,
                                                                                             time.time() - t1))

            # Save the latest params
            t1 = time.time()
            save_path = os.path.join(args.checkpoint_path, f"epoch_latest.pt")
            # torch.save(
            #     {
            #         "epoch": epoch + 1,
            #         "step": step + 1,
            #         "name": args.name,
            #         "state_dict": model.state_dict() if not args.use_flash_attention else convert_state_dict(
            #             model.state_dict()),
            #         "optimizer": optimizer.state_dict(),
            #     },
            #     save_path,
            # )
            logging.info(
                "Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, epoch + 1,
                                                                                             step + 1,
                                                                                             time.time() - t1))

    return epoch_trained_steps


def evaluate(model, data, epoch, args, steps, vision_model):
    logging.info("Begin to eval on validation set (epoch {} @ {} steps)...".format(epoch + 1, steps))

    model.eval()
    copy_model = copy.deepcopy(model)
    # use RFMID dataset for evaluation
    if vision_model in ['RN50']:
        epoch_main, best_auc, best_map, best_epoch, best_auc_test, best_map_test, best_epoch_test = eval_multiLabelCls_RN50(
            copy_model, epoch + 1,
            cuda_rank=args.local_device_rank)
        logging.info(
            f"MultiLabelCls Validation Result (epoch {epoch_main}) | "
            f"Best epoch_valid: {best_epoch:.6f} | "
            f"Best Valid AUC: {best_auc:.6f} | "
            f"Best Valid MAP: {best_map:.6f} | "
        )
        logging.info(
            f"MultiLabelCls Validation Result (epoch {epoch_main}) | "
            f"Best epoch_test: {best_epoch_test:.6f} | "
            f"Best Test AUC: {best_auc_test:.6f} | "
            f"Best Test MAP: {best_map_test:.6f} | "
        )
    else:
        epoch_main, best_auc, best_map, best_epoch, best_auc_test, best_map_test, best_epoch_test = eval_multiLabelCls_ViT(
            copy_model, epoch + 1,
            cuda_rank=args.local_device_rank)
        logging.info(
            f"MultiLabelCls Validation Result (epoch {epoch_main}) | "
            f"Best epoch_valid: {best_epoch:.6f} | "
            f"Best Valid AUC: {best_auc:.6f} | "
            f"Best Valid MAP: {best_map:.6f} | "
        )
        logging.info(
            f"MultiLabelCls Validation Result (epoch {epoch_main}) | "
            f"Best epoch_test: {best_epoch_test:.6f} | "
            f"Best Test AUC: {best_auc_test:.6f} | "
            f"Best Test MAP: {best_map_test:.6f} | "
        )


def cosineSimilarityLoss(feature1, feature2):
    scale_factor_h = feature1.shape[0] / feature2.size(0)
    scale_factor_w = feature1.shape[1] / feature2.size(1)

    feature2_interpolated = F.interpolate(feature2.unsqueeze(0).unsqueeze(0),
                                          size=(feature1.shape[0], feature1.shape[1]),
                                          mode='bilinear',
                                          align_corners=False)
    feature2_interpolated = feature2_interpolated.squeeze(0).squeeze(0)

    cosine_sim = F.cosine_similarity(feature1, feature2_interpolated, dim=1)
    similarity_loss = 1 - cosine_sim.mean()
    return similarity_loss
