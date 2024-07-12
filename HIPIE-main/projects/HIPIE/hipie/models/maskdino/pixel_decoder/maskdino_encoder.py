# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified by Feng Li and Hao Zhang.
import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
import fvcore.nn.weight_init as weight_init

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch.cuda.amp import autocast

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm, ConvTranspose2d
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from .position_encoding import PositionEmbeddingSine
from ..utils.utils import _get_clones, _get_activation_fn
from .ops.modules import MSDeformAttn


def build_pixel_decoder(cfg, input_shape):
    """
    Build a pixel decoder from `cfg.MODEL.MaskDINO.PIXEL_DECODER_NAME`.
    """
    name = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME
    model = SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)
    forward_features = getattr(model, "forward_features", None)
    if not callable(forward_features):
        raise ValueError(
            "Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. "
            f"Please implement forward_features for {name} to only return mask features."
        )
    return model


class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, enc_n_points=4,):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds):

        enable_mask = 0
        if masks is not None:
            for src in srcs:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
            
        #input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        return memory, spatial_shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

      
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # parallel adapter
        self.adapter = nn.Linear(d_model, d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)

        # parallel adapter
        adapter_output = self.adapter(src)
        src = src + adapter_output

        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
      
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src = self.forward_ffn(src)

        return src


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


@SEM_SEG_HEADS_REGISTRY.register()
class MaskDINOEncoder(nn.Module):
    """
    This is the multi-scale encoder in detection models, also named as pixel decoder in segmentation models.
    """
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        # deformable transformer encoder args
        transformer_in_features: List[str],
        common_stride: int,
        num_feature_levels: int,
        total_num_feature_levels: int,
        feature_order: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout in transformer
            transformer_nheads: number of heads in multi-headed attention
            transformer_dim_feedforward: feature dimension in feedforward network
            transformer_enc_layers: number of encoder layers
            conv_dim: the output dimension of the convolution layers
            mask_dim: the dimension of the mask features
            norm: normalization for convolution layers
            transformer_in_features: name of input features to be passed to transformer
            common_stride: the common stride of input features
            num_feature_levels: the first num_feature_levels feature levels will be taken as the input of deformable DETR encoder
            total_num_feature_levels: total number of feature levels
            feature_order: order of features
        """
        super().__init__()

        self.transformer_in_features = transformer_in_features

        transformer_in_channels = [input_shape[f].channels for f in transformer_in_features]
        transformer_in_strides = [input_shape[f].stride for f in transformer_in_features]

        self.num_feature_levels = num_feature_levels
        self.total_num_feature_levels = total_num_feature_levels
        self.feature_order = feature_order

        if feature_order == "high2low":
            self.transformer_input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )
                for channels in transformer_in_channels
            ])
        elif feature_order == "low2high":
            self.transformer_input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(
                        transformer_in_channels[-i-1], conv_dim, kernel_size=1
                    ),
                    nn.GroupNorm(32, conv_dim),
                )
                for i in range(len(transformer_in_channels))
            ])

        self.transformer_input_proj.extend([
            nn.Sequential(
                nn.Conv2d(
                    transformer_in_channels[-1], conv_dim, kernel_size=3, stride=2, padding=1
                ),
                nn.GroupNorm(32, conv_dim),
            )
            for _ in range(num_feature_levels - len(transformer_in_channels))
        ])

        self.pe_layer = PositionEmbeddingSine(128, normalize=True)
        self.mask_features = Conv2d(conv_dim, mask_dim, kernel_size=1, stride=1, padding=0)
        self.mask_features.apply(weight_init.c2_xavier_fill)

        self.encoder = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            nhead=transformer_nheads,
            num_encoder_layers=transformer_enc_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            num_feature_levels=self.total_num_feature_levels,
        )

        self.encoder_output_dropout = nn.Dropout(0.1)

        self.maskformer_num_feature_levels = num_feature_levels

        self.common_stride = common_stride

    @classmethod
    def from_config(cls, cfg, input_shape):
        enc_cfg = {
            "transformer_dropout": cfg.MODEL.MaskDINO.ENCODER_DROPOUT,
            "transformer_nheads": cfg.MODEL.MaskDINO.ENCODER_NHEADS,
            "transformer_dim_feedforward": cfg.MODEL.MaskDINO.ENCODER_DIM_FEEDFORWARD,
            "transformer_enc_layers": cfg.MODEL.MaskDINO.ENCODER_LAYERS,
            "conv_dim": cfg.MODEL.MaskDINO.CONVS_DIM,
            "mask_dim": cfg.MODEL.MaskDINO.MASK_DIM,
            "norm": cfg.MODEL.MaskDINO.NORM,
            "transformer_in_features": cfg.MODEL.MaskDINO.TRANSFORMER_IN_FEATURES,
            "common_stride": cfg.MODEL.MaskDINO.COMMON_STRIDE,
            "num_feature_levels": cfg.MODEL.MaskDINO.NUM_FEATURE_LEVELS,
            "total_num_feature_levels": cfg.MODEL.MaskDINO.TOTAL_NUM_FEATURE_LEVELS,
            "feature_order": cfg.MODEL.MaskDINO.FEATURE_ORDER,
        }
        return enc_cfg

    def forward_features(self, features):
        srcs = []
        pos = []
        for l, feat in enumerate(self.transformer_in_features):
            x = features[feat]
            srcs.append(self.transformer_input_proj[l](x))
            pos.append(self.pe_layer(x))
        for l in range(len(self.transformer_in_features), self.num_feature_levels):
            if l == len(self.transformer_in_features):
                src = self.transformer_input_proj[l](features[self.transformer_in_features[-1]])
            else:
                src = self.transformer_input_proj[l](srcs[-1])
            mask = torch.zeros((src.shape[0], src.shape[2], src.shape[3]), device=src.device, dtype=torch.bool)
            pos_l = self.pe_layer(src)
            srcs.append(src)
            pos.append(pos_l)

        masks = [torch.zeros((src.shape[0], src.shape[2], src.shape[3]), device=src.device, dtype=torch.bool) for src in srcs]

        return srcs, masks, pos

    def forward(self, features, targets=None):
        srcs, masks, pos = self.forward_features(features)
        memory, spatial_shapes, level_start_index = self.encoder(srcs, masks, pos)
        return memory, spatial_shapes, level_start_index, masks

