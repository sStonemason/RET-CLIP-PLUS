from collections import OrderedDict
from typing import Tuple, Union
from itertools import repeat
import collections.abc

import math
import random
import logging
import numpy as np
import torch
import torch.nn.functional as F
import timm
from torch import nn
from torch.utils.checkpoint import checkpoint

import importlib.util

if importlib.util.find_spec('flash_attn'):
    FlashMHA = importlib.import_module('flash_attn.flash_attention').FlashMHA

from RET_CLIP_PLUS.clip import _tokenizer
from RET_CLIP_PLUS.clip.configuration_bert import BertConfig
from RET_CLIP_PLUS.clip.modeling_bert import BertModel

from timm.models.vision_transformer import PatchEmbed, Block
from functools import partial
from RET_CLIP_PLUS.clip.pos_embed import get_2d_sincos_pos_embed


def upsample_pos_emb(emb, new_size):
    # upsample the pretrained embedding for higher resolution
    # emb size NxD
    first = emb[:1, :]
    emb = emb[1:, :]
    N, D = emb.size(0), emb.size(1)
    size = int(np.sqrt(N))
    assert size * size == N
    # new_size = size * self.upsample
    emb = emb.permute(1, 0)
    emb = emb.view(1, D, size, size).contiguous()
    emb = F.interpolate(emb, size=new_size, mode='bilinear', )
    emb = emb.view(D, -1).contiguous()
    emb = emb.permute(1, 0)
    emb = torch.cat([first, emb], 0)
    emb = nn.parameter.Parameter(emb.half())
    return emb


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        # FIXME support for non-transformer
        pass

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, use_flash_attention: bool = False):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head) if not use_flash_attention else FlashMHA(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.use_flash_attention = use_flash_attention

    def attention(self, x: torch.Tensor, need_weights=False):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if self.use_flash_attention:
            # Batch first is needed for FlashAttention. See https://github.com/HazyResearch/flash-attention/issues/84 for more information.
            return self.attn(x.transpose(1, 0))[0].transpose(1, 0)
        else:
            if need_weights:
                return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask,
                                 average_attn_weights=True)
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)

    def forward(self, x: torch.Tensor, return_attn=False):
        if return_attn:
            attn_output, attn_weights = self.attention(self.ln_1(x), need_weights=True)
            x = x + attn_output
            x = x + self.mlp(self.ln_2(x))
            return x, attn_weights
        attn_output, attn_weights = self.attention(self.ln_1(x))
        # x = x + self.attention(self.ln_1(x))
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,
                 use_flash_attention: bool = False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False
        self.ln_post = LayerNorm(width)
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask, use_flash_attention) for _ in range(layers)])
        print('transformer int finished')

    def process_feature(self, x):
        # 对每个特征应用 permute 和 LayerNorm
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        return x

    def get_last_attn(self, x):
        for i, blk in enumerate(self.resblocks):
            if i < len(self.resblocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attn=True)

    def get_grad_cam(self, x):
        attn_weights = []
        with torch.no_grad():
            layers = self.layers - 1
            for i in range(layers):
                x, attn_weight = self.resblocks[i](x, return_attn=True)
                attn_weights.append(attn_weight)
        return x, attn_weights

    def forward(self, x: torch.Tensor):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for r in self.resblocks:
                x = checkpoint(r, x)
            return x
        return self.resblocks(x)


class bMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 grad=1.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.grad = grad

    def forward(self, x):
        x = self.drop(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 use_flash_attention: bool = False):
        super().__init__()
        self.input_resolution = input_resolution
        self.grid_size = (self.input_resolution // patch_size, self.input_resolution // patch_size)
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, use_flash_attention=use_flash_attention)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def get_last_attn(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        attn = self.transformer.get_last_attn(x)
        return attn

    def get_grad_cam(self, x, H, W):
        self.positional_embedding_new = upsample_pos_emb(self.positional_embedding, (H // 16, W // 16))
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding_new.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, attn_weight = self.transformer.get_grad_cam(x)
        return x, attn_weight

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.0):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]

        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        if mask_ratio != 0:
            cls_token = x[:, 0].unsqueeze(1)
            x, mask, ids_restore = self.random_masking(x[:, 1:, :], mask_ratio)
            x = torch.cat([cls_token, x], dim=1)
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            # x = torch.cat(x, dim=-1)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_post(x)
            return x, mask, ids_restore
        else:
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_post(x)
            if self.proj is not None:
                x = x[:, 0, :] @ self.proj
            return x

    def forward_intermediate_outputs(self, x: torch.Tensor, mask_ratio: float = 0.75, n=12,
                                     selected_levels=[3, 5, 7, 11]):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]

        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        if mask_ratio != 0:
            cls_token = x[:, 0].unsqueeze(1)
            x, mask, ids_restore = self.random_masking(x[:, 1:, :], mask_ratio)
            x = torch.cat([cls_token, x], dim=1)
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND
            output = []
            for i, blk in enumerate(self.transformer.resblocks):
                x = blk(x)
                if len(self.transformer.resblocks) - i <= n:
                    output.append(self.ln_post(x.permute(1, 0, 2)))
            features = [output[idx][:, 1:] for idx in selected_levels]
            feature = torch.cat(features, dim=-1)
            return feature, mask, ids_restore
        else:
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND
            output = []
            for i, blk in enumerate(self.transformer.resblocks):
                x = blk(x)
                if len(self.transformer.resblocks) - i <= n:
                    output.append(self.ln_post(x.permute(1, 0, 2)))
            features = [output[idx][:, 1:] for idx in selected_levels]
            feature = torch.cat(features, dim=-1)
        return feature

    def forward_all_tokens(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]

        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        return x[:, 1:] @ self.proj


class MAE_decoder(nn.Module):
    def __init__(self, embed_dim=768, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, patch_size=16,
                 num_patches=196,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=True):
        super().__init__()

        self.grad_checkpointing = False

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * 3, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(196 ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        # print(x.shape)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for r in self.decoder_blocks:
                x = checkpoint(r, x)
        else:
            x = self.decoder_blocks(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.squeeze(x, dim=self.dim)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,

                 # text
                 vocab_size: int,
                 text_attention_probs_dropout_prob: float,
                 text_hidden_act: str,
                 text_hidden_dropout_prob: float,
                 text_hidden_size: int,
                 text_initializer_range: float,
                 text_intermediate_size: int,
                 text_max_position_embeddings: int,
                 text_num_attention_heads: int,
                 text_num_hidden_layers: int,
                 text_type_vocab_size: int,
                 tokenizer=_tokenizer,
                 # vision head width, added this param for ViT-H
                 vision_head_width: int = 64,
                 use_flash_attention: bool = False,
                 ):
        super().__init__()

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // vision_head_width
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // vision_head_width
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                use_flash_attention=use_flash_attention
            )

        self.bert_config = BertConfig(
            vocab_size_or_config_json_file=vocab_size,
            hidden_size=text_hidden_size,
            num_hidden_layers=text_num_hidden_layers,
            num_attention_heads=text_num_attention_heads,
            intermediate_size=text_intermediate_size,
            hidden_act=text_hidden_act,
            hidden_dropout_prob=text_hidden_dropout_prob,
            attention_probs_dropout_prob=text_attention_probs_dropout_prob,
            max_position_embeddings=text_max_position_embeddings,
            type_vocab_size=text_type_vocab_size,
            initializer_range=text_initializer_range,
            layer_norm_eps=1e-12,
            use_flash_attention=use_flash_attention
        )
        self.bert = BertModel(self.bert_config)

        self.text_projection = nn.Sequential(nn.Linear(text_hidden_size, text_hidden_size),
                                             nn.ReLU(),
                                             nn.Linear(text_hidden_size, embed_dim))
        self.text_projection_left = nn.Sequential(nn.Linear(text_hidden_size, text_hidden_size),
                                                  nn.ReLU(),
                                                  nn.Linear(text_hidden_size, embed_dim))
        self.text_projection_right = nn.Sequential(nn.Linear(text_hidden_size, text_hidden_size),
                                                   nn.ReLU(),
                                                   nn.Linear(text_hidden_size, embed_dim))

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_left = nn.Parameter(torch.ones([]) * np.log(90.017))
        self.logit_scale_right = nn.Parameter(torch.ones([]) * np.log(90.017))

        self.global_feature_mapping = nn.Linear(2 * embed_dim, embed_dim, bias=False)
        self.left_feature_mapping = nn.Linear(embed_dim, embed_dim, bias=False)
        self.right_feature_mapping = nn.Linear(embed_dim, embed_dim, bias=False)

        self.tokenizer = tokenizer
        self.MAE_decoder = MAE_decoder()

        self.initialize_parameters()

    def initialize_parameters(self):
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(90.017))
        # self.logit_scale_single = nn.Parameter(torch.ones([]) * np.log(90.017))
        self.logit_scale_left = nn.Parameter(torch.ones([]) * np.log(90.017))
        self.logit_scale_right = nn.Parameter(torch.ones([]) * np.log(90.017))

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        if self.text_projection is not None:
            for module in self.text_projection.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=self.bert_config.hidden_size ** -0.5)

        if self.text_projection_left is not None:
            for module in self.text_projection_left.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=self.bert_config.hidden_size ** -0.5)

        if self.text_projection_right is not None:
            for module in self.text_projection_right.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=self.bert_config.hidden_size ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.bert.set_grad_checkpointing(enable)
        self.MAE_decoder.grad_checkpointing = enable

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * 3)
        imgs: (N, 3, H, W)
        """
        p = 16
        h = w = int(x.shape[1] ** 0.5)  # 假设patch的数量为 h*w
        assert h * w == x.shape[1], "x 的第二个维度应该是完整的patch网格"

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def get_img_feats(self, img):
        return self.visual(img.type(self.dtype), mask_ratio=0)

    def get_img_feats_global(self, img):
        vision_feature = torch.cat(
            (
                self.visual(img.type(self.dtype), mask_ratio=0),
                self.visual(img.type(self.dtype), mask_ratio=0)),
            dim=1)
        return self.global_feature_mapping(vision_feature)

    def get_feats_gradcam(self, img):
        return self.right_feature_mapping(self.visual(img.type(self.dtype), mask_ratio=0))

    def encode_image(self, img_l, img_r, mask_ratio=0):
        if img_r is None:
            if isinstance(self.visual, ModifiedResNet):
                # mask_ratio > 0 (FLIP strategy) is currently only implemented for VisualTransformer.
                vision_feature, _, _, _ = self.visual(img_l.type(self.dtype))
                return vision_feature
            vision_feature = self.visual(img_l.type(self.dtype))
            return vision_feature
        if img_l is None:
            if isinstance(self.visual, ModifiedResNet):
                # mask_ratio > 0 (FLIP strategy) is currently only implemented for VisualTransformer.
                vision_feature, _, _, _ = self.visual(img_r.type(self.dtype))
                return vision_feature
            vision_feature = self.visual(img_r.type(self.dtype))
            return vision_feature
        if isinstance(self.visual, ModifiedResNet):
            # mask_ratio > 0 (FLIP strategy) is currently only implemented for VisualTransformer.
            left_feature, _, _, _ = self.visual(img_l.type(self.dtype))
            right_feature, _, _, _ = self.visual(img_r.type(self.dtype))
            vision_feature = torch.cat(
                (left_feature, right_feature), dim=1)

            return self.global_feature_mapping(vision_feature), self.single_feature_mapping(
                left_feature), self.single_feature_mapping(right_feature)

        left_feature = self.visual(img_l.type(self.dtype), mask_ratio)
        right_feature = self.visual(img_r.type(self.dtype), mask_ratio)
        vision_feature = torch.cat(
            (left_feature, right_feature), dim=1)
        return self.left_feature_mapping(left_feature), self.right_feature_mapping(
            right_feature), self.global_feature_mapping(vision_feature)

    def get_attn_map(self, img,
                     output_dir='/home/ubuntu/nfs/8T1/dujw/clip/downstream_train_valid_test/Feature_maps/ReVision'):
        attn = self.visual.get_last_attn(img.type(self.dtype))
        import os
        import matplotlib.pyplot as plt
        import torchvision
        nh = attn.shape[1]
        w_featmap = 14
        h_featmap = 14
        os.makedirs(output_dir, exist_ok=True)

        bs = img.shape[0]
        for i in range(bs):
            # torchvision.utils.save_image(torchvision.utils.make_grid(img[i], normalize=True, scale_each=True),
            #                              os.path.join(output_dir, "img" + str(i) + ".png"))

            attentions = attn[i, :, 0, 1:].reshape(nh, -1)
            attentions = attentions.reshape(nh, w_featmap, h_featmap)
            attentions = attentions.detach()
            attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=16, mode="nearest")[
                0].cpu().numpy()
            attn_mean = np.mean(attentions, axis=0)
            fname = os.path.join(output_dir, "img" + str(i) + "_attn-map" + "_mean" + ".png")
            plt.imsave(fname=fname, arr=attn_mean, format='png')

    def get_grad_cam(self, img, H, W):
        vision_feature, attn_weight_list = self.visual.get_grad_cam(img.type(self.dtype), H, W)
        return vision_feature, attn_weight_list

    def encode_image_mae_input(self, pred, mask_ratio=0):
        bs = int(pred.shape[0] / 2)
        left_feature = self.visual(pred[:bs].type(self.dtype), mask_ratio)
        right_feature = self.visual(pred[bs:].type(self.dtype), mask_ratio)
        vision_feature = torch.cat(
            (left_feature, right_feature), dim=1)

        return self.left_feature_mapping(left_feature), self.right_feature_mapping(
            right_feature), self.global_feature_mapping(
            vision_feature)
        # return left_feature, right_feature, self.global_feature_mapping(
        #     vision_feature)

    def encode_text(self, text):
        pad_index = self.tokenizer.vocab['[PAD]']
        attn_mask = text.ne(pad_index).type(self.dtype)
        x = self.bert(text, attention_mask=attn_mask)[0].type(self.dtype)  # [batch_size, seq_length, hidden_size]

        text = self.text_projection(x[:, 0, :])
        text_left = self.text_projection_left(x[:, 0, :])
        text_right = self.text_projection_right(x[:, 0, :])

        return text, text_left, text_right

    def MAE_encoder(self, x, mask_ratio=0.75):
        print('MAE mask_ratio: ')
        print(mask_ratio)
        # latent, mask, ids_restore = self.visual.forward_intermediate_outputs(x, mask_ratio=mask_ratio)
        latent, mask, ids_restore = self.visual(x, mask_ratio=mask_ratio)
        return latent, mask, ids_restore

    def MAE_loss(self, imgs, pred, mask, norm_pix_loss=False):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        print('MAE_loss:')
        print(imgs.shape)
        print(pred.shape)
        target = self.patchify(imgs)
        if norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_MAE(self, img_p, mask_ratio=0.75):
        latent, mask, ids_restore = self.MAE_encoder(img_p, mask_ratio=mask_ratio)
        pred = self.MAE_decoder(latent, ids_restore)  # [N, L, p*p*3]
        MAE_loss = self.MAE_loss(img_p, pred, mask, norm_pix_loss=True)
        return MAE_loss

    def forward_VisionFlow(self, img_p, text=None, mask_ratio=0.75):
        latent, mask, ids_restore = self.MAE_encoder(img_p, mask_ratio=mask_ratio)
        pred = self.MAE_decoder(latent, ids_restore)  # [N, L, p*p*3]
        patches = self.patchify(img_p)  # [N, L, patch_dim]
        mask_unsqueeze = mask.unsqueeze(-1)  # [N, L, 1]
        pred = patches * (1 - mask_unsqueeze) + pred * mask_unsqueeze  # 逐patch融合
        MAE_loss = self.MAE_loss(img_p, pred, mask, norm_pix_loss=True)

        pred = self.unpatchify(pred)
        left_features_rec, right_features_rec, image_features_rec = self.encode_image_mae_input(pred)

        image_features_rec = image_features_rec / image_features_rec.norm(dim=-1, keepdim=True)
        left_features_rec = left_features_rec / left_features_rec.norm(dim=-1, keepdim=True)
        right_features_rec = right_features_rec / right_features_rec.norm(dim=-1, keepdim=True)

        text_features, text_features_left, text_features_right = self.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_left = text_features_left / text_features_left.norm(dim=-1, keepdim=True)
        text_features_right = text_features_right / text_features_right.norm(dim=-1, keepdim=True)

        return image_features_rec, left_features_rec, right_features_rec, MAE_loss, text_features, text_features_left, text_features_right, self.logit_scale.exp(), self.logit_scale_left.exp(), self.logit_scale_right.exp()

    def forward_CLIP(self, img_l, img_r, text=None, mask_ratio=0):
        assert img_l is not None or img_r is not None or text is not None, "text and images cannot all be None!"

        left_features, right_features, image_features = self.encode_image(img_l, img_r, mask_ratio)
        text_features, text_features_left, text_features_right = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        left_features = left_features / left_features.norm(dim=-1, keepdim=True)
        right_features = right_features / right_features.norm(dim=-1, keepdim=True)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_left = text_features_left / text_features_left.norm(dim=-1, keepdim=True)
        text_features_right = text_features_right / text_features_right.norm(dim=-1, keepdim=True)

        return image_features, left_features, right_features, text_features, text_features_left, text_features_right, self.logit_scale.exp(), self.logit_scale_left.exp(), self.logit_scale_right.exp()

    def get_similarity(self, img_l, img_r, text):
        image_features, _ = self.encode_image(img_l, img_r)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    def forward_last_layer(self, image_features, text_features):
        x, attn_weight = self.visual.transformer.resblocks[-1](image_features,
                                                               return_attn=True)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.visual.ln_post(x)
        # x = torch.mean(x[:, 1:, :], dim=1)
        x = x[:, 0, :]

        if self.visual.proj is not None:
            x = x @ self.visual.proj

        image_features = self.right_feature_mapping(x)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.logit_scale_right.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        # shape = [global_batch_size, global_batch_size]
        logits_per_image = logits_per_image.softmax(dim=-1)

        return logits_per_image, attn_weight


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        if isinstance(l, BertModel):
            l.to(torch.half)

        for name in ["proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                # if isinstance(attr, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                #     attr.weight.data = attr.weight.data.half()
                #     if attr.bias is not None:
                #         attr.bias.data = attr.bias.data.half()
                if attr is not None:
                    attr.data = attr.data.half()  # 2023.10.14 should be 'attr.data'
        if isinstance(l, nn.Sequential):
            l.half()

        # for name in ["text_projection", "proj"]:
        #     if hasattr(l, name):
        #         attr = getattr(l, name)
        #         # if isinstance(attr, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        #         #     attr.weight.data = attr.weight.data.half()
        #         #     if attr.bias is not None:
        #         #         attr.bias.data = attr.bias.data.half()
        #         if attr is not None:
        #             attr.data = attr.data.half()  # 2023.10.14 should be 'attr.data'

    # model.apply(_convert_weights_to_fp16)
    for param in model.parameters():
        param.data = param.data.half()
        if param.grad:
            param.grad.data = param.grad.data.float()
    # model.half()


def restore_model(model, clip_state_dict: dict, bert_state_dict: dict, use_flash_attention: bool):
    merged_state_dict = {}

    # use clip_state_dict to initialize the image encoder & logit scale
    if clip_state_dict is not None:
        for k, v in clip_state_dict.items():
            if k.startswith("visual") or k == "logit_scale":
                merged_state_dict[k] = v

    # use bert_state_dict to initialize the text encoder
    if bert_state_dict is not None:
        for k, v in bert_state_dict.items():
            if k.startswith("bert") and "bert.pooler" not in k:
                merged_state_dict[k] = v

    # adapt flash attention
    if use_flash_attention:
        merged_state_dict = convert_state_dict(merged_state_dict)

    convert_weights(model)
    resize_pos_embed(merged_state_dict, model)
    model.load_state_dict(merged_state_dict, strict=False)
    return model.eval()


def convert_state_dict(state_dict):
    """Adapt to Flash Attention"""
    if not state_dict:
        return state_dict

    prefix = 'module.' if list(state_dict.keys())[0].startswith('module') else ''

    if f'{prefix}visual.transformer.resblocks.0.attn.in_proj_weight' in state_dict:
        for k in list(state_dict.keys()):
            if 'attn.in_proj_weight' in k:
                state_dict[k.replace('attn.in_proj_weight', 'attn.Wqkv.weight')] = state_dict.pop(k)
            elif 'attn.in_proj_bias' in k:
                state_dict[k.replace('attn.in_proj_bias', 'attn.Wqkv.bias')] = state_dict.pop(k)
    elif f'{prefix}visual.transformer.resblocks.0.attn.Wqkv.weight' in state_dict:
        for k in list(state_dict.keys()):
            if 'attn.Wqkv.weight' in k:
                state_dict[k.replace('attn.Wqkv.weight', 'attn.in_proj_weight')] = state_dict.pop(k)
            elif 'attn.Wqkv.bias' in k:
                state_dict[k.replace('attn.Wqkv.bias', 'attn.in_proj_bias')] = state_dict.pop(k)

    if f'{prefix}bert.encoder.layer.0.attention.self.query.weight' in state_dict:
        i = 0
        while f'{prefix}bert.encoder.layer.{i}.attention.self.query.weight' in state_dict:
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.weight'] = torch.cat(
                (state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.query.weight'),
                 state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.key.weight'),
                 state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.value.weight'))
            )
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.bias'] = torch.cat(
                (state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.query.bias'),
                 state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.key.bias'),
                 state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.value.bias'))
            )
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.out_proj.weight'] = \
                state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.output.dense.weight')
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.out_proj.bias'] = \
                state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.output.dense.bias')
            i += 1
    elif f'{prefix}bert.encoder.layer.0.attention.self.Wqkv.weight' in state_dict:
        i = 0
        while f'{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.weight' in state_dict:
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.query.weight'], \
                state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.key.weight'], \
                state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.value.weight'] = \
                torch.chunk(state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.weight'), chunks=3)
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.query.bias'], \
                state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.key.bias'], \
                state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.value.bias'] = \
                torch.chunk(state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.bias'), chunks=3)
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.output.dense.weight'] = \
                state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.out_proj.weight')
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.output.dense.bias'] = \
                state_dict.pop(f'module.bert.encoder.layer.{i}.attention.self.out_proj.bias')
            i += 1

    return state_dict


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', seq_dim=1, prefix=""):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get(prefix + 'visual.positional_embedding', None)
    model = model.module if hasattr(model, 'module') else model
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        align_corners=True,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict[prefix + 'visual.positional_embedding'] = new_pos_embed


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)

# from collections import OrderedDict
# from typing import Tuple, Union
# from itertools import repeat
# import collections.abc
#
# import math
# import logging
# import numpy as np
# import torch
# import torch.nn.functional as F
# import timm
# from torch import nn
# from torch.utils.checkpoint import checkpoint
#
# import importlib.util
#
# if importlib.util.find_spec('flash_attn'):
#     FlashMHA = importlib.import_module('flash_attn.flash_attention').FlashMHA
#
# from RET_CLIP_PLUS.clip import _tokenizer
# from RET_CLIP_PLUS.clip.configuration_bert import BertConfig
# from RET_CLIP_PLUS.clip.modeling_bert import BertModel
#
# from timm.models.vision_transformer import PatchEmbed, Block
# from functools import partial
# from RET_CLIP_PLUS.clip.pos_embed import get_2d_sincos_pos_embed
#
#
# class RestNetBasicBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride):
#         super(RestNetBasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         output = self.conv1(x)
#         output = F.relu(self.bn1(output))
#         output = self.conv2(output)
#         output = self.bn2(output)
#         return F.relu(x + output)
#
#
# class RestNetDownBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride):
#         super(RestNetDownBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.extra = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
#             nn.BatchNorm2d(out_channels)
#         )
#
#     def forward(self, x):
#         extra_x = self.extra(x)
#         output = self.conv1(x)
#         out = F.relu(self.bn1(output))
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         return F.relu(extra_x + out)
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1):
#         super().__init__()
#
#         # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
#         self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#
#         self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#
#         self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
#
#         self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = None
#         self.stride = stride
#
#         if stride > 1 or inplanes != planes * Bottleneck.expansion:
#             # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
#             self.downsample = nn.Sequential(OrderedDict([
#                 ("-1", nn.AvgPool2d(stride)),
#                 ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
#                 ("1", nn.BatchNorm2d(planes * self.expansion))
#             ]))
#
#     def forward(self, x: torch.Tensor):
#         identity = x
#
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.relu(self.bn2(self.conv2(out)))
#         out = self.avgpool(out)
#         out = self.bn3(self.conv3(out))
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#         return out
#
#
# class AttentionPool2d(nn.Module):
#     def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
#         super().__init__()
#         self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
#         self.num_heads = num_heads
#
#     def forward(self, x):
#         x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
#         x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
#         x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
#         x, _ = F.multi_head_attention_forward(
#             query=x, key=x, value=x,
#             embed_dim_to_check=x.shape[-1],
#             num_heads=self.num_heads,
#             q_proj_weight=self.q_proj.weight,
#             k_proj_weight=self.k_proj.weight,
#             v_proj_weight=self.v_proj.weight,
#             in_proj_weight=None,
#             in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
#             bias_k=None,
#             bias_v=None,
#             add_zero_attn=False,
#             dropout_p=0,
#             out_proj_weight=self.c_proj.weight,
#             out_proj_bias=self.c_proj.bias,
#             use_separate_proj_weight=True,
#             training=self.training,
#             need_weights=False
#         )
#
#         return x[0]
#
#
# class ModifiedResNet(nn.Module):
#     """
#     A ResNet class that is similar to torchvision's but contains the following changes:
#     - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
#     - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
#     - The final pooling layer is a QKV attention instead of an average pool
#     """
#
#     def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
#         super().__init__()
#         self.output_dim = output_dim
#         self.input_resolution = input_resolution
#
#         # the 3-layer stem
#         self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(width // 2)
#         self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(width // 2)
#         self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(width)
#         self.avgpool = nn.AvgPool2d(2)
#         self.relu = nn.ReLU(inplace=True)
#
#         # residual layers
#         self._inplanes = width  # this is a *mutable* variable used during construction
#         self.layer1 = self._make_layer(width, layers[0])
#         self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
#         self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
#         self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
#
#         embed_dim = width * 32  # the ResNet feature dimension
#         self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)
#
#     def _make_layer(self, planes, blocks, stride=1):
#         layers = [Bottleneck(self._inplanes, planes, stride)]
#
#         self._inplanes = planes * Bottleneck.expansion
#         for _ in range(1, blocks):
#             layers.append(Bottleneck(self._inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     @torch.jit.ignore
#     def set_grad_checkpointing(self, enable=True):
#         # FIXME support for non-transformer
#         pass
#
#     def forward(self, x):
#         def stem(x):
#             for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
#                 x = self.relu(bn(conv(x)))
#             x = self.avgpool(x)
#             return x
#
#         x = x.type(self.conv1.weight.dtype)
#         x = stem(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.attnpool(x)
#
#         return x
#
#
# class LayerNorm(nn.LayerNorm):
#     """Subclass torch's LayerNorm to handle fp16."""
#
#     def forward(self, x: torch.Tensor):
#         orig_type = x.dtype
#         ret = super().forward(x.type(torch.float32))
#         return ret.type(orig_type)
#
#
# class QuickGELU(nn.Module):
#     def forward(self, x: torch.Tensor):
#         return x * torch.sigmoid(1.702 * x)
#
#
# class ResidualAttentionBlock(nn.Module):
#     def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, use_flash_attention: bool = False):
#         super().__init__()
#
#         self.attn = nn.MultiheadAttention(d_model, n_head) if not use_flash_attention else FlashMHA(d_model, n_head)
#         self.ln_1 = LayerNorm(d_model)
#         self.mlp = nn.Sequential(OrderedDict([
#             ("c_fc", nn.Linear(d_model, d_model * 4)),
#             ("gelu", QuickGELU()),
#             ("c_proj", nn.Linear(d_model * 4, d_model))
#         ]))
#         self.ln_2 = LayerNorm(d_model)
#         self.attn_mask = attn_mask
#         self.use_flash_attention = use_flash_attention
#
#     def attention(self, x: torch.Tensor):
#         self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
#         if self.use_flash_attention:
#             # Batch first is needed for FlashAttention. See https://github.com/HazyResearch/flash-attention/issues/84 for more information.
#             return self.attn(x.transpose(1, 0))[0].transpose(1, 0)
#         else:
#             return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
#
#     def forward(self, x: torch.Tensor):
#         x = x + self.attention(self.ln_1(x))
#         x = x + self.mlp(self.ln_2(x))
#         return x
#
#
# class Transformer(nn.Module):
#     def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,
#                  use_flash_attention: bool = False):
#         super().__init__()
#         self.width = width
#         self.layers = layers
#         self.grad_checkpointing = False
#         self.resblocks = nn.Sequential(
#             *[ResidualAttentionBlock(width, heads, attn_mask, use_flash_attention) for _ in range(layers)])
#         print('transformer int finished')
#
#     def forward(self, x: torch.Tensor):
#         if self.grad_checkpointing and not torch.jit.is_scripting():
#             for r in self.resblocks:
#                 x = checkpoint(r, x)
#             return x
#         return self.resblocks(x)
#
#
# class VisualTransformer(nn.Module):
#     def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
#                  use_flash_attention: bool = False):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.grid_size = (self.input_resolution // patch_size, self.input_resolution // patch_size)
#         self.output_dim = output_dim
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
#
#         scale = width ** -0.5
#         self.class_embedding = nn.Parameter(scale * torch.randn(width))
#         self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
#         self.ln_pre = LayerNorm(width)
#
#         self.transformer = Transformer(width, layers, heads, use_flash_attention=use_flash_attention)
#
#         self.ln_post = LayerNorm(width)
#         self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
#
#     @torch.jit.ignore
#     def set_grad_checkpointing(self, enable=True):
#         self.transformer.grad_checkpointing = enable
#
#     # def random_masking(self, x, mask_ratio):
#     #     N, L, D = x.shape  # batch, length, dim
#     #     len_keep = int((L - 1) * (1 - mask_ratio))
#     #
#     #     noise = torch.rand(N, L - 1, device=x.device)
#     #     ids_shuffle = torch.argsort(noise, dim=1) + torch.ones(N, L - 1, device=x.device,
#     #                                                            dtype=int)
#     #     ids_keep = ids_shuffle[:, :len_keep]
#     #
#     #     x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
#     #
#     #     x0 = x[:, 0, :]
#     #     x0 = x0.reshape(N, 1, D)
#     #     x_masked_add = torch.cat([x0, x_masked], axis=1)
#     #     return x_masked_add
#
#     def random_masking(self, x, mask_ratio):
#         """
#         Perform per-sample random masking by per-sample shuffling.
#         Per-sample shuffling is done by argsort random noise.
#         x: [N, L, D], sequence
#         """
#         N, L, D = x.shape  # batch, length, dim
#         len_keep = int(L * (1 - mask_ratio))
#
#         noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
#
#         # sort noise for each sample
#         ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
#         ids_restore = torch.argsort(ids_shuffle, dim=1)
#
#         # keep the first subset
#         ids_keep = ids_shuffle[:, :len_keep]
#         x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
#
#         # generate the binary mask: 0 is keep, 1 is remove
#         mask = torch.ones([N, L], device=x.device)
#         mask[:, :len_keep] = 0
#         # unshuffle to get the binary mask
#         mask = torch.gather(mask, dim=1, index=ids_restore)
#
#         return x_masked, mask, ids_restore
#
#     def forward(self, x: torch.Tensor, mask_ratio: float = 0.0, mae_input=False):
#         if mae_input == False:
#             x = self.conv1(x)  # shape = [*, width, grid, grid]
#             x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#
#             x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
#         else:
#             x = x
#         x = torch.cat(
#             [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
#              x], dim=1)  # shape = [*, grid ** 2 + 1, width]
#         x = x + self.positional_embedding.to(x.dtype)
#         if mask_ratio != 0:
#             x, mask, ids_restore = self.random_masking(x[:, 1:, :], mask_ratio)
#             x = self.ln_pre(x)
#             x = x.permute(1, 0, 2)  # NLD -> LND
#             x = self.transformer(x)
#             x = x.permute(1, 0, 2)  # LND -> NLD
#             # 2024-5-8 MAE
#             latent = self.ln_post(x)
#             return latent, mask, ids_restore
#         else:
#             x = self.ln_pre(x)
#             x = x.permute(1, 0, 2)  # NLD -> LND
#             x = self.transformer(x)
#             x = x.permute(1, 0, 2)  # LND -> NLD
#             x = self.ln_post(x)
#             if self.proj is not None:
#                 x = x[:, 0, :] @ self.proj
#             return x
#
#
# class MAE_decoder(nn.Module):
#     def __init__(self, embed_dim=768, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, patch_size=16,
#                  num_patches=196,
#                  mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=True):
#         super().__init__()
#
#         self.grad_checkpointing = False
#         # MAE decoder specifics
#         self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
#
#         self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
#
#         self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
#                                               requires_grad=False)  # fixed sin-cos embedding
#
#         self.decoder_blocks = nn.ModuleList([
#             Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
#             for i in range(decoder_depth)])
#
#         self.decoder_norm = norm_layer(decoder_embed_dim)
#         self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * 3, bias=True)  # decoder to patch
#         # --------------------------------------------------------------------------
#         self.norm_pix_loss = norm_pix_loss
#         self.initialize_weights()
#
#     def initialize_weights(self):
#         # initialization
#         # initialize (and freeze) pos_embed by sin-cos embedding
#         decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
#                                                     int(196 ** .5), cls_token=True)
#         self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
#
#         # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
#         torch.nn.init.normal_(self.mask_token, std=.02)
#
#         # initialize nn.Linear and nn.LayerNorm
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def forward(self, x: torch.Tensor, ids_restore):
#         # embed tokens
#         x = self.decoder_embed(x)
#         # print(x.shape)
#
#         # append mask tokens to sequence
#         mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
#         x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
#         x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
#         x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
#         # print(x.shape)
#
#         # add pos embed
#         x = x + self.decoder_pos_embed
#
#         # apply Transformer blocks
#         if self.grad_checkpointing and not torch.jit.is_scripting():
#             for r in self.decoder_blocks:
#                 x = checkpoint(r, x)
#
#         for blk in self.decoder_blocks:
#             x = blk(x)
#         x = self.decoder_norm(x)
#
#         # predictor projection
#         x = self.decoder_pred(x)
#
#         # remove cls token
#         x = x[:, 1:, :]
#
#         return x
#
#
# class Squeeze(nn.Module):
#     def __init__(self, dim=None):
#         super(Squeeze, self).__init__()
#         self.dim = dim
#
#     def forward(self, x):
#         return torch.squeeze(x, dim=self.dim)
#
#
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.fc2(x)
#         return x
#
#
# class SentenceAttentionPool(nn.Module):
#     def __init__(self, spacial_dim=256, embed_dim=512, num_heads=8, output_dim: int = None, pos_embed=True):
#         super().__init__()
#         self.pos_embed = pos_embed
#         if self.pos_embed:
#             self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
#         self.num_heads = num_heads
#
#     def forward(self, x):
#         # X: [B, N, C]
#         x = x.permute(1, 0, 2)
#         x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (N+1)BC
#         if self.pos_embed:
#             x = x + self.positional_embedding[: x.size(0), None, :].to(x.dtype)  # (L+1)NC
#         x, _ = F.multi_head_attention_forward(
#             query=x, key=x, value=x,
#             embed_dim_to_check=x.shape[-1],
#             num_heads=self.num_heads,
#             q_proj_weight=self.q_proj.weight,
#             k_proj_weight=self.k_proj.weight,
#             v_proj_weight=self.v_proj.weight,
#             in_proj_weight=None,
#             in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
#             bias_k=None,
#             bias_v=None,
#             add_zero_attn=False,
#             dropout_p=0,
#             out_proj_weight=self.c_proj.weight,
#             out_proj_bias=self.c_proj.bias,
#             use_separate_proj_weight=True,
#             training=self.training,
#             need_weights=False
#         )
#         return x[0]
#
#
# class SimpleSentenceAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads=8, max_seq_length=100):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.max_seq_length = max_seq_length
#         self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads)
#         self.norm = nn.LayerNorm(self.embed_dim)
#         self.dropout = nn.Dropout(0.1)
#
#         # 初始化位置嵌入
#         self.position_embeddings = nn.Parameter(torch.randn(self.max_seq_length, self.embed_dim))
#
#     def forward(self, sentence):
#         # sentence 形状: [seq_len, 1, embed_dim]
#         sentence.permute(1, 0, 2)
#         seq_length = sentence.size(0)
#
#         # 检查序列长度并应用位置嵌入
#         position_embeddings = self.position_embeddings[:seq_length, :]  # 切片以适配输入长度
#         sentence = sentence + position_embeddings.unsqueeze(1)  # 增加批次维度并相加
#
#         # 进行自注意力操作
#         attn_output, _ = self.attention(sentence, sentence, sentence)
#         sentence = sentence + self.dropout(attn_output)
#         sentence = self.norm(sentence)
#
#         # 返回句子级表示, 取平均
#         return sentence.mean(dim=0)  # 返回句子级表示
#
#
# class SimplifiedCrossAttention(nn.Module):
#     def __init__(self, embed_dim, drop_rate=0):
#         super(SimplifiedCrossAttention, self).__init__()
#         self.embed_dim = embed_dim
#
#         self.query = nn.Linear(embed_dim, embed_dim)
#         self.key = nn.Linear(embed_dim, embed_dim)
#         self.value = nn.Linear(embed_dim, embed_dim)
#         self.dropout = nn.Dropout(drop_rate)
#
#     def forward(self, image_tensor, text_tensor):
#         # 图像作为查询向量，直接使用
#         query_layer = self.query(image_tensor)  # [1, 1024]
#
#         # 文本作为键和值
#         key_layer = self.key(text_tensor)  # [K, 1024]
#         value_layer = self.value(text_tensor)  # [K, 1024]
#
#         # 计算注意力分数
#         attention_scores = query_layer @ key_layer.T  # [1, 1024] @ [1024, K] = [1, K]
#         attention_scores = attention_scores / math.sqrt(self.embed_dim)
#
#         # 应用 sigmoid 来获取注意力权重
#         attention_probs = F.sigmoid(attention_scores)
#         attention_probs = self.dropout(attention_probs)
#
#         # 应用注意力权重到值上
#         context_layer = attention_probs @ value_layer  # [1, K] @ [K, 1024] = [1, 1024]
#
#         return context_layer
#
#
# class SPB(nn.Module):
#     def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, temp=0.9):
#         super().__init__()
#         self.dim = dim
#         self.n_embed = n_embed
#         self.decay = decay
#         self.eps = eps
#         self.embed = nn.Embedding(n_embed, dim)
#         self.temp = temp
#         self.curr_temp = temp
#
#     def set_temp(self, epoch, max_epoch, strategy="fixed"):
#         if strategy == "fixed":
#             self.curr_temp = self.temp
#         elif strategy == "linear":
#             self.curr_temp = self.temp - 0.9 * self.temp * epoch / max_epoch
#         elif strategy == "exp":
#             self.curr_temp = self.temp * (0.1 ** (epoch / max_epoch))
#
#     def forward(self, input):
#         flatten = input.reshape(-1, self.dim)
#         dist = flatten @ self.embed.weight.T
#         # self.gt_dist = dist
#         soft_one_hot = F.gumbel_softmax(dist, tau=self.curr_temp, dim=1, hard=False)
#         output = soft_one_hot @ self.embed.weight
#         # embed_ind = soft_one_hot.argmax(1)
#         # recon_loss = (output - flatten).abs().mean()
#         # loss = recon_loss
#         # self.dist = dist
#         return output
#
#     # @torch.no_grad()
#     # def query(self, input):
#     #     flatten = input.reshape(-1, self.dim)
#     #     logits = flatten @ self.embed.weight.T
#     #     soft_one_hot = F.gumbel_softmax(logits, tau=self.curr_temp, dim=1, hard=False)
#     #     output = soft_one_hot @ self.embed.weight
#     #     # recon_loss = (output - flatten).abs().mean()
#     #     embed_ind = soft_one_hot.argmax(1)
#     #     return output, embed_ind
#
#
# class CLIP(nn.Module):
#     def __init__(self,
#                  embed_dim: int,
#                  # vision
#                  image_resolution: int,
#                  vision_layers: Union[Tuple[int, int, int, int], int],
#                  vision_width: int,
#                  vision_patch_size: int,
#
#                  # text
#                  vocab_size: int,
#                  text_attention_probs_dropout_prob: float,
#                  text_hidden_act: str,
#                  text_hidden_dropout_prob: float,
#                  text_hidden_size: int,
#                  text_initializer_range: float,
#                  text_intermediate_size: int,
#                  text_max_position_embeddings: int,
#                  text_num_attention_heads: int,
#                  text_num_hidden_layers: int,
#                  text_type_vocab_size: int,
#                  tokenizer=_tokenizer,
#                  # vision head width, added this param for ViT-H
#                  vision_head_width: int = 64,
#                  use_flash_attention: bool = False,
#                  ):
#         super().__init__()
#
#         if isinstance(vision_layers, (tuple, list)):
#             vision_heads = vision_width * 32 // vision_head_width
#             self.visual = ModifiedResNet(
#                 layers=vision_layers,
#                 output_dim=embed_dim,
#                 heads=vision_heads,
#                 input_resolution=image_resolution,
#                 width=vision_width
#             )
#         else:
#             vision_heads = vision_width // vision_head_width
#             self.visual = VisualTransformer(
#                 input_resolution=image_resolution,
#                 patch_size=vision_patch_size,
#                 width=vision_width,
#                 layers=vision_layers,
#                 heads=vision_heads,
#                 output_dim=embed_dim,
#                 use_flash_attention=use_flash_attention
#             )
#
#         self.bert_config = BertConfig(
#             vocab_size_or_config_json_file=vocab_size,
#             hidden_size=text_hidden_size,
#             num_hidden_layers=text_num_hidden_layers,
#             num_attention_heads=text_num_attention_heads,
#             intermediate_size=text_intermediate_size,
#             hidden_act=text_hidden_act,
#             hidden_dropout_prob=text_hidden_dropout_prob,
#             attention_probs_dropout_prob=text_attention_probs_dropout_prob,
#             max_position_embeddings=text_max_position_embeddings,
#             type_vocab_size=text_type_vocab_size,
#             initializer_range=text_initializer_range,
#             layer_norm_eps=1e-12,
#             use_flash_attention=use_flash_attention
#         )
#         self.bert = BertModel(self.bert_config)
#
#         # self.ViT_MAE = VisualTransformer(
#         #         input_resolution=image_resolution,
#         #         patch_size=vision_patch_size,
#         #         width=vision_width,
#         #         layers=vision_layers,
#         #         heads=vision_heads,
#         #         output_dim=embed_dim,
#         #         use_flash_attention=use_flash_attention
#         #     )
#
#         # self.cross_attention = SimplifiedCrossAttention(embed_dim=embed_dim)
#         # # self.self_attention = SimpleSentenceAttention(embed_dim=embed_dim)
#         # self.self_attention = SentenceAttentionPool(spacial_dim=100, embed_dim=embed_dim)
#         # self.self_attention_global = SentenceAttentionPool(spacial_dim=50, embed_dim=embed_dim)
#
#         # self.text_projection = nn.Parameter(torch.empty(text_hidden_size, embed_dim))
#         # self.text_projection_left = nn.Parameter(torch.empty(text_hidden_size, embed_dim))
#         # self.text_projection_right = nn.Parameter(torch.empty(text_hidden_size, embed_dim))
#
#         self.text_projection = nn.Sequential(nn.Linear(text_hidden_size, text_hidden_size),
#                                              nn.ReLU(),
#                                              nn.Linear(text_hidden_size, embed_dim))
#         self.text_projection_left = nn.Sequential(nn.Linear(text_hidden_size, text_hidden_size),
#                                                   nn.ReLU(),
#                                                   nn.Linear(text_hidden_size, embed_dim))
#         self.text_projection_right = nn.Sequential(nn.Linear(text_hidden_size, text_hidden_size),
#                                                    nn.ReLU(),
#                                                    nn.Linear(text_hidden_size, embed_dim))
#
#         # self.text_projection = nn.Sequential(nn.Linear(text_hidden_size, embed_dim),
#         #                                      nn.ReLU(),
#         #                                      nn.Linear(embed_dim, embed_dim))
#         # self.text_projection_left = nn.Sequential(nn.Linear(text_hidden_size, embed_dim),
#         #                                           nn.ReLU(),
#         #                                           nn.Linear(embed_dim, embed_dim))
#         # self.text_projection_right = nn.Sequential(nn.Linear(text_hidden_size, embed_dim),
#         #                                            nn.ReLU(),
#         #                                            nn.Linear(embed_dim, embed_dim))
#
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#         # self.logit_scale_single = nn.Parameter(torch.ones([]) * np.log(90.017))
#         self.logit_scale_left = nn.Parameter(torch.ones([]) * np.log(90.017))
#         self.logit_scale_right = nn.Parameter(torch.ones([]) * np.log(90.017))
#
#         # self.global_feature_mapping = nn.Linear(2 * embed_dim, embed_dim)
#         # self.single_feature_mapping = nn.Linear(embed_dim, embed_dim)
#         self.global_feature_mapping = nn.Linear(2 * embed_dim, embed_dim, bias=False)
#         self.left_feature_mapping = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.right_feature_mapping = nn.Linear(embed_dim, embed_dim, bias=False)
#
#         self.tokenizer = tokenizer
#         self.MAE_decoder = MAE_decoder()
#
#         # self.sentence_bank = SPB(embed_dim, 256)
#
#         # self.predictor = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2),
#         #                                nn.ReLU(inplace=True),  # hidden layer
#         #                                nn.Linear(embed_dim // 2, embed_dim))  # output layer # used for simsiam loss
#
#         self.initialize_parameters()
#
#     def initialize_parameters(self):
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#         # self.logit_scale_single = nn.Parameter(torch.ones([]) * np.log(90.017))
#         self.logit_scale_left = nn.Parameter(torch.ones([]) * np.log(90.017))
#         self.logit_scale_right = nn.Parameter(torch.ones([]) * np.log(90.017))
#
#         if isinstance(self.visual, ModifiedResNet):
#             if self.visual.attnpool is not None:
#                 std = self.visual.attnpool.c_proj.in_features ** -0.5
#                 nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
#
#             for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
#                 for name, param in resnet_block.named_parameters():
#                     if name.endswith("bn3.weight"):
#                         nn.init.zeros_(param)
#
#         # if self.text_projection is not None:
#         #     nn.init.normal_(self.text_projection, std=self.bert_config.hidden_size ** -0.5)
#         #
#         # if self.text_projection_left is not None:
#         #     nn.init.normal_(self.text_projection_left, std=self.bert_config.hidden_size ** -0.5)
#         #
#         # if self.text_projection_right is not None:
#         #     nn.init.normal_(self.text_projection_right, std=self.bert_config.hidden_size ** -0.5)
#
#         if self.text_projection is not None:
#             for module in self.text_projection.modules():
#                 if isinstance(module, nn.Linear):
#                     nn.init.normal_(module.weight, std=self.bert_config.hidden_size ** -0.5)
#
#         if self.text_projection_left is not None:
#             for module in self.text_projection_left.modules():
#                 if isinstance(module, nn.Linear):
#                     nn.init.normal_(module.weight, std=self.bert_config.hidden_size ** -0.5)
#
#         if self.text_projection_right is not None:
#             for module in self.text_projection_right.modules():
#                 if isinstance(module, nn.Linear):
#                     nn.init.normal_(module.weight, std=self.bert_config.hidden_size ** -0.5)
#
#     @torch.jit.ignore
#     def set_grad_checkpointing(self, enable=True):
#         self.visual.set_grad_checkpointing(enable)
#         # self.ViT_MAE.set_grad_checkpointing(enable)
#         self.bert.set_grad_checkpointing(enable)
#         self.MAE_decoder.grad_checkpointing = enable
#
#     @property
#     def dtype(self):
#         return self.visual.conv1.weight.dtype
#
#     def get_local_features(self, texts, comma_indices):
#         """
#             处理文本批次，每个样本根据逗号位置索引被分割成句子，并应用自注意力模型。
#
#             Args:
#             - texts (Tensor): 输入的批次数据，形状为 (batch_size, seq_len, embed_dim)
#             - comma_indices (list of lists): 每个样本中逗号和[SEP]的索引列表
#
#             Returns:
#             - Tensor: 批次中所有句子的特征表示
#         """
#         batch_size = texts.shape[0]
#         sentence_features = []
#
#         for i in range(batch_size):
#             indices = comma_indices[i]
#             start_idx = 1
#             features = []
#
#             # 遍历每个逗号位置，切分句子并应用注意力
#             for end_idx in indices:
#                 sentence = texts[i, start_idx:end_idx, :]  # 切分句子
#                 # sentence = sentence.unsqueeze(0)  # 增加批次维度以适配模型输入
#                 # sentence_feature = self.self_attention(sentence)  # 应用自注意力模型
#                 sentence_feature = torch.mean(sentence, dim=0, keepdim=True)
#                 features.append(sentence_feature)
#                 start_idx = end_idx + 1
#
#             # 将所有句子的特征合并
#             sentence_features.append(torch.cat(features, dim=0))
#
#         return sentence_features
#
#     def get_global_features(self, local_features):
#         batch_size = len(local_features)
#         global_features = []
#         for i in range(batch_size):
#             sentences = local_features[i].unsqueeze(0)
#             global_feature = self.self_attention_global(sentences)
#             global_features.append(global_feature)
#         global_features = torch.stack(global_features)
#         return global_features
#
#     def proto_local_features(self, local_features):
#         batch_size = len(local_features)
#         proto_local_features = []
#         for i in range(batch_size):
#             sample = local_features[i]
#             proto_local_feature = self.sentence_bank(sample)
#             proto_local_features.append(proto_local_feature)
#
#         return proto_local_features
#
#     def patchify(self, imgs):
#         """
#         imgs: (N, 3, H, W)
#         x: (N, L, patch_size**2 *3)
#         """
#         p = 16
#         assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
#
#         h = w = imgs.shape[2] // p
#         x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
#         x = torch.einsum('nchpwq->nhwpqc', x)
#         x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
#         return x
#
#     def encode_image(self, img_l, img_r, mask_ratio=0, mae_input=False):
#         if img_r is None:
#             if isinstance(self.visual, ModifiedResNet):
#                 # mask_ratio > 0 (FLIP strategy) is currently only implemented for VisualTransformer.
#                 vision_feature, _, _, _ = self.visual(img_l.type(self.dtype))
#                 return vision_feature
#             vision_feature = self.visual(img_l.type(self.dtype), mask_ratio)
#             return vision_feature
#         if img_l is None:
#             if isinstance(self.visual, ModifiedResNet):
#                 # mask_ratio > 0 (FLIP strategy) is currently only implemented for VisualTransformer.
#                 vision_feature, _, _, _ = self.visual(img_r.type(self.dtype))
#                 return vision_feature
#             vision_feature = self.visual(img_r.type(self.dtype), mask_ratio)
#             return vision_feature
#         if isinstance(self.visual, ModifiedResNet):
#             # mask_ratio > 0 (FLIP strategy) is currently only implemented for VisualTransformer.
#             left_feature, _, _, _ = self.visual(img_l.type(self.dtype))
#             right_feature, _, _, _ = self.visual(img_r.type(self.dtype))
#             vision_feature = torch.cat(
#                 (left_feature, right_feature), dim=1)
#
#             return self.global_feature_mapping(vision_feature), self.single_feature_mapping(
#                 left_feature), self.single_feature_mapping(right_feature)
#         if mae_input == False:
#             print('lr-level visual encoder mask ratio: ')
#             print(mask_ratio)
#             left_feature = self.visual(img_l.type(self.dtype), mask_ratio)
#             right_feature = self.visual(img_r.type(self.dtype), mask_ratio)
#             # vision_feature = torch.cat(
#             #     (left_feature, right_feature), dim=1)
#             return self.left_feature_mapping(
#                 left_feature), self.right_feature_mapping(right_feature)
#             # return self.global_feature_mapping(vision_feature)
#         else:
#             print('p-level visual encoder mask ratio: ')
#             print(mask_ratio)
#             left_feature = self.visual(img_l.type(self.dtype), mask_ratio, mae_input=False)
#             right_feature = self.visual(img_r.type(self.dtype), mask_ratio, mae_input=False)
#             vision_feature = torch.cat(
#                 (left_feature, right_feature), dim=1)
#
#             # return self.left_feature_mapping(
#             #     left_feature), self.right_feature_mapping(right_feature), self.global_feature_mapping(vision_feature)
#         return self.global_feature_mapping(vision_feature)
#
#     def encode_text(self, text):
#
#         def process_indices(indices):
#             # 初始化结果列表
#             processed_indices = []
#
#             # 遍历索引列表
#             i = 0
#             while i < len(indices):
#                 # 如果索引为1，则删除
#                 if indices[i] == 1:
#                     del indices[i]
#                 # 如果当前索引与下一个索引连续，则保留连续数字的最后一个
#                 elif i < len(indices) - 1 and indices[i] + 1 == indices[i + 1]:
#                     i += 1
#                 # 否则将当前索引添加到结果列表中
#                 else:
#                     processed_indices.append(indices[i])
#                     i += 1
#
#             return processed_indices
#
#         def find_commas_and_sep(batch):
#             batch_indices = []
#             for sample in batch:
#                 # 找到每个样本中所有中文逗号的位置
#                 comma_indices = (sample == _tokenizer.vocab['，']).nonzero(as_tuple=True)[0].tolist()
#
#                 # 找到每个样本中 [sep] 的位置
#                 sep_index = (sample == _tokenizer.vocab['[SEP]']).nonzero(as_tuple=True)[0].tolist()
#
#                 comma_indices.extend(sep_index)
#
#                 comma_indices = process_indices(comma_indices)
#
#                 # 将这个样本的索引列表添加到批次的索引列表中
#                 batch_indices.append(comma_indices)
#
#             return batch_indices
#
#         comma_indices = find_commas_and_sep(text)
#
#         pad_index = self.tokenizer.vocab['[PAD]']
#         attn_mask = text.ne(pad_index).type(self.dtype)
#         x = self.bert(text, attention_mask=attn_mask)[0].type(self.dtype)  # [batch_size, seq_length, hidden_size]
#
#         # local_features = self.get_local_features(x @ self.text_projection_local, comma_indices)
#
#         # proto_local_features = self.proto_local_features(local_features)
#         # # if torch.isnan(local_features).any():
#         # #     print("local_feats张量中存在 NaN")
#         # # 找出包含 NaN 的张量的索引
#         # nan_tensor_indices = [i for i, tensor in enumerate(local_features) if torch.isnan(tensor).any()]
#         #
#         # # 输出包含 NaN 的张量的索引
#         # logging.info("包含 NaN 的张量的索引:", nan_tensor_indices)
#
#         # text_features = self.get_global_features(local_features).squeeze(1)
#         # if torch.isnan(text_features).any():
#         #     print("text张量中存在 NaN")
#         #     print(comma_indices[nan_tensor_indices[0]])
#         #     print(x[nan_tensor_indices[0]])
#         #     print(text[nan_tensor_indices[0]])
#
#         # print('仅双眼版本')
#
#         # text_features = self.text_projection(x[:, 0, :])
#         # text = x[:, 0, :] @ self.text_projection
#         # text_left = x[:, 0, :] @ self.text_projection_left
#         # text_right = x[:, 0, :] @ self.text_projection_right
#         text = self.text_projection(x[:, 0, :])
#         text_left = self.text_projection_left(x[:, 0, :])
#         text_right = self.text_projection_right(x[:, 0, :])
#         # text_left = self.text_projection_left(x)
#         # text_right = self.text_projection_right(x)
#         #
#         # local_features_left = self.get_local_features(text_left, comma_indices)
#         # local_features_right = self.get_local_features(text_right, comma_indices)
#         return text, text_left, text_right
#         # return text_features
#
#     def MAE_encoder(self, x, mask_ratio=0.75):
#         print('MAE mask_ratio: ')
#         print(mask_ratio)
#         # latent, mask, ids_restore = self.ViT_MAE(x, mask_ratio=mask_ratio)
#         latent, mask, ids_restore = self.visual(x, mask_ratio=mask_ratio)
#         return latent, mask, ids_restore
#
#     def MAE_loss(self, imgs, pred, mask, norm_pix_loss=False):
#         """
#         imgs: [N, 3, H, W]
#         pred: [N, L, p*p*3]
#         mask: [N, L], 0 is keep, 1 is remove,
#         """
#         print('MAE_loss:')
#         print(imgs.shape)
#         print(pred.shape)
#         target = self.patchify(imgs)
#         if norm_pix_loss:
#             mean = target.mean(dim=-1, keepdim=True)
#             var = target.var(dim=-1, keepdim=True)
#             target = (target - mean) / (var + 1.e-6) ** .5
#
#         loss = (pred - target) ** 2
#         loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
#
#         loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
#         return loss
#
#     def cross_representation(self, imgs_l, imgs_r, local_features):
#         batch_size = imgs_l.shape[0]
#         cross_imgs_l = []
#         cross_imgs_r = []
#         for i in range(batch_size):
#             sentences = local_features[i]
#             img_l = imgs_l[i, :].unsqueeze(0)
#             cross_img_l = self.cross_attention(img_l, sentences)
#             img_r = imgs_r[i, :].unsqueeze(0)
#             cross_img_r = self.cross_attention(img_r, sentences)
#
#             cross_imgs_l.append(cross_img_l)
#             cross_imgs_r.append(cross_img_r)
#         cross_imgs_l = torch.stack(cross_imgs_l).squeeze(1)
#         cross_imgs_r = torch.stack(cross_imgs_r).squeeze(1)
#         return cross_imgs_l, cross_imgs_r
#
#     def simsiam_loss_func(self, x, y, predictor, flag='image'):
#         p_x = predictor(x)
#         p_y = predictor(y)
#         z_x = x.detach()
#         z_y = y.detach()
#         return - (F.cosine_similarity(p_x, z_y, dim=-1).mean() + F.cosine_similarity(p_y, z_x, dim=-1).mean()) * 0.5
#
#     def simsiam_loss(self, text_to_local_image_embed, local_image_embed):
#         '''
#         The convolutions in encoder may cause overlap between the receptive fields of the patches, a simple negative sampling strategy is not applicable.
#         '''
#         image_loss = self.simsiam_loss_func(text_to_local_image_embed, local_image_embed, self.predictor, flag='image')
#         return image_loss
#
#     def forward(self, img_l, img_r, text, img_p_l=None, img_p_r=None, mask_ratio=0):
#         assert img_l is not None or img_r is not None or text is not None, "text and images cannot all be None!"
#
#         if img_l is None and img_r is None:
#             return self.encode_text(text)
#         elif text is None and img_r is None:
#             return self.encode_image(img_l=img_l, img_r=None)
#         elif text is None and img_l is None:
#             return self.encode_image(img_l=None, img_r=img_r)
#         elif text is None:
#             return self.encode_image(img_l, img_r)
#         assert img_l is not None and img_r is not None, "both images is required!"
#         # print('---double input---')
#         # print('simsiam-init')
#         # print(img_l.shape)
#         # print(img_r.shape)
#
#         left_features, right_features = self.encode_image(img_l, img_r, mask_ratio)
#         # image_features = self.encode_image(img_l, img_r, mask_ratio)
#         text_features, text_features_left, text_features_right = self.encode_text(text)
#         # text_features = self.encode_text(text)
#
#         imgs = torch.cat((img_l, img_r), dim=0)
#         latent, mask, ids_restore = self.MAE_encoder(imgs, mask_ratio=0.75)
#         pred = self.MAE_decoder(latent, ids_restore)  # [N, L, p*p*3]
#         MAE_loss = self.MAE_loss(imgs, pred, mask, norm_pix_loss=True)
#
#         # return image_features, text_features, MAE_loss, self.logit_scale.exp()
#
#         # print('pred.shape:')
#         # print(pred.shape)
#         # img_l_mae = pred[:128]
#         # img_r_mae = pred[128:]
#
#         # print(img_l_mae.shape)
#         # print(img_r_mae.shape)
#         print('经典图像增强的输入')
#         image_features = self.encode_image(img_p_l, img_p_r, mask_ratio=0,
#                                            mae_input=True)
#         # left_features, right_features = self.encode_image(img_l_mae, img_r_mae, mask_ratio=0,
#         #                                                   mae_input=True)
#
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         left_features = left_features / left_features.norm(dim=-1, keepdim=True)
#         right_features = right_features / right_features.norm(dim=-1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)
#         text_features_left = text_features_left / text_features_left.norm(dim=-1, keepdim=True)
#         text_features_right = text_features_right / text_features_right.norm(dim=-1, keepdim=True)
#
#         return image_features, text_features, text_features_left, text_features_right, left_features, right_features, self.logit_scale.exp(), self.logit_scale_left.exp(), self.logit_scale_right.exp(), MAE_loss
#
#     def get_similarity(self, img_l, img_r, text):
#         image_features, _ = self.encode_image(img_l, img_r)
#         text_features = self.encode_text(text)
#
#         # normalized features
#         image_features = image_features / image_features.norm(dim=1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=1, keepdim=True)
#
#         # cosine similarity as logits
#         logit_scale = self.logit_scale.exp()
#         logits_per_image = logit_scale * image_features @ text_features.t()
#         logits_per_text = logits_per_image.t()
#
#         # shape = [global_batch_size, global_batch_size]
#         return logits_per_image, logits_per_text
#
#
# def convert_models_to_fp32(model):
#     for p in model.parameters():
#         p.data = p.data.float()
#         if p.grad:
#             p.grad.data = p.grad.data.float()
#
#
# def convert_weights(model: nn.Module):
#     """Convert applicable model parameters to fp16"""
#
#     def _convert_weights_to_fp16(l):
#         if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
#             l.weight.data = l.weight.data.half()
#             if l.bias is not None:
#                 l.bias.data = l.bias.data.half()
#
#         if isinstance(l, nn.MultiheadAttention):
#             for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
#                 tensor = getattr(l, attr)
#                 if tensor is not None:
#                     tensor.data = tensor.data.half()
#
#         if isinstance(l, BertModel):
#             l.to(torch.half)
#
#         for name in ["proj"]:
#             if hasattr(l, name):
#                 attr = getattr(l, name)
#                 # if isinstance(attr, (nn.Conv1d, nn.Conv2d, nn.Linear)):
#                 #     attr.weight.data = attr.weight.data.half()
#                 #     if attr.bias is not None:
#                 #         attr.bias.data = attr.bias.data.half()
#                 if attr is not None:
#                     attr.data = attr.data.half()  # 2023.10.14 should be 'attr.data'
#         if isinstance(l, nn.Sequential):
#             l.half()
#
#         # for name in ["text_projection", "proj"]:
#         #     if hasattr(l, name):
#         #         attr = getattr(l, name)
#         #         # if isinstance(attr, (nn.Conv1d, nn.Conv2d, nn.Linear)):
#         #         #     attr.weight.data = attr.weight.data.half()
#         #         #     if attr.bias is not None:
#         #         #         attr.bias.data = attr.bias.data.half()
#         #         if attr is not None:
#         #             attr.data = attr.data.half()  # 2023.10.14 should be 'attr.data'
#
#     # model.apply(_convert_weights_to_fp16)
#     for param in model.parameters():
#         param.data = param.data.half()
#         if param.grad:
#             param.grad.data = param.grad.data.float()
#     # model.half()
#
#
# def restore_model(model, clip_state_dict: dict, bert_state_dict: dict, use_flash_attention: bool):
#     merged_state_dict = {}
#
#     # use clip_state_dict to initialize the image encoder & logit scale
#     if clip_state_dict is not None:
#         for k, v in clip_state_dict.items():
#             if k.startswith("visual") or k == "logit_scale":
#                 merged_state_dict[k] = v
#
#     # use bert_state_dict to initialize the text encoder
#     if bert_state_dict is not None:
#         for k, v in bert_state_dict.items():
#             if k.startswith("bert") and "bert.pooler" not in k:
#                 merged_state_dict[k] = v
#
#     # adapt flash attention
#     if use_flash_attention:
#         merged_state_dict = convert_state_dict(merged_state_dict)
#
#     convert_weights(model)
#     resize_pos_embed(merged_state_dict, model)
#     model.load_state_dict(merged_state_dict, strict=False)
#     return model.eval()
#
#
# def convert_state_dict(state_dict):
#     """Adapt to Flash Attention"""
#     if not state_dict:
#         return state_dict
#
#     prefix = 'module.' if list(state_dict.keys())[0].startswith('module') else ''
#
#     if f'{prefix}visual.transformer.resblocks.0.attn.in_proj_weight' in state_dict:
#         for k in list(state_dict.keys()):
#             if 'attn.in_proj_weight' in k:
#                 state_dict[k.replace('attn.in_proj_weight', 'attn.Wqkv.weight')] = state_dict.pop(k)
#             elif 'attn.in_proj_bias' in k:
#                 state_dict[k.replace('attn.in_proj_bias', 'attn.Wqkv.bias')] = state_dict.pop(k)
#     elif f'{prefix}visual.transformer.resblocks.0.attn.Wqkv.weight' in state_dict:
#         for k in list(state_dict.keys()):
#             if 'attn.Wqkv.weight' in k:
#                 state_dict[k.replace('attn.Wqkv.weight', 'attn.in_proj_weight')] = state_dict.pop(k)
#             elif 'attn.Wqkv.bias' in k:
#                 state_dict[k.replace('attn.Wqkv.bias', 'attn.in_proj_bias')] = state_dict.pop(k)
#
#     if f'{prefix}bert.encoder.layer.0.attention.self.query.weight' in state_dict:
#         i = 0
#         while f'{prefix}bert.encoder.layer.{i}.attention.self.query.weight' in state_dict:
#             state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.weight'] = torch.cat(
#                 (state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.query.weight'),
#                  state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.key.weight'),
#                  state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.value.weight'))
#             )
#             state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.bias'] = torch.cat(
#                 (state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.query.bias'),
#                  state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.key.bias'),
#                  state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.value.bias'))
#             )
#             state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.out_proj.weight'] = \
#                 state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.output.dense.weight')
#             state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.out_proj.bias'] = \
#                 state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.output.dense.bias')
#             i += 1
#     elif f'{prefix}bert.encoder.layer.0.attention.self.Wqkv.weight' in state_dict:
#         i = 0
#         while f'{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.weight' in state_dict:
#             state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.query.weight'], \
#                 state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.key.weight'], \
#                 state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.value.weight'] = \
#                 torch.chunk(state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.weight'), chunks=3)
#             state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.query.bias'], \
#                 state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.key.bias'], \
#                 state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.value.bias'] = \
#                 torch.chunk(state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.bias'), chunks=3)
#             state_dict[f'{prefix}bert.encoder.layer.{i}.attention.output.dense.weight'] = \
#                 state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.out_proj.weight')
#             state_dict[f'{prefix}bert.encoder.layer.{i}.attention.output.dense.bias'] = \
#                 state_dict.pop(f'module.bert.encoder.layer.{i}.attention.self.out_proj.bias')
#             i += 1
#
#     return state_dict
#
#
# def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', seq_dim=1, prefix=""):
#     # Rescale the grid of position embeddings when loading from state_dict
#     old_pos_embed = state_dict.get(prefix + 'visual.positional_embedding', None)
#     model = model.module if hasattr(model, 'module') else model
#     if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
#         return
#     grid_size = to_2tuple(model.visual.grid_size)
#     extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
#     new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
#     if new_seq_len == old_pos_embed.shape[0]:
#         return
#
#     if extra_tokens:
#         pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
#     else:
#         pos_emb_tok, pos_emb_img = None, old_pos_embed
#     old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))
#
#     logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
#     pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
#     pos_emb_img = F.interpolate(
#         pos_emb_img,
#         size=grid_size,
#         mode=interpolation,
#         align_corners=True,
#     )
#     pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
#     if pos_emb_tok is not None:
#         new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
#     else:
#         new_pos_embed = pos_emb_img
#     state_dict[prefix + 'visual.positional_embedding'] = new_pos_embed
#
#
# # From PyTorch internals
# def _ntuple(n):
#     def parse(x):
#         if isinstance(x, collections.abc.Iterable):
#             return x
#         return tuple(repeat(x, n))
#
#     return parse
#
#
# to_1tuple = _ntuple(1)
# to_2tuple = _ntuple(2)
# to_3tuple = _ntuple(3)
# to_4tuple = _ntuple(4)
# to_ntuple = lambda n, x: _ntuple(n)(x)
