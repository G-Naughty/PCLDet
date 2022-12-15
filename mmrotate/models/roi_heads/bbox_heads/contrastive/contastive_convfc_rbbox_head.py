# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_linear_layer

from ....builder import ROTATED_HEADS, build_loss
from ..rotated_bbox_head import RotatedBBoxHead
from ..convfc_rbbox_head import RotatedConvFCBBoxHead
@ROTATED_HEADS.register_module()
class ContrastRotatedConvFCBBoxHead(RotatedConvFCBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-
                                    > cls convs -> cls fcs -> cls
        shared convs -> shared fcs  |-> reg convs -> reg fcs -> reg
                                    \-> contrastive -> similarity between classes
    Args:
        num_shared_convs (int, optional): number of ``shared_convs``.
        num_shared_fcs (int, optional): number of ``shared_fcs``.
        num_cls_convs (int, optional): number of ``cls_convs``.
        num_cls_fcs (int, optional): number of ``cls_fcs``.
        num_reg_convs (int, optional): number of ``reg_convs``.
        num_reg_fcs (int, optional): number of ``reg_fcs``.
        conv_out_channels (int, optional): output channels of convolution.
        fc_out_channels (int, optional): output channels of fc.
        conv_cfg (dict, optional): Config of convolution.
        norm_cfg (dict, optional): Config of normalization.
        init_cfg (dict, optional): Config of initialization.
    """

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=2,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 contrast_out_channels=512,
                 loss_contrast=dict(
                                  type='SupConProxyAnchorLoss',
                                  class_num=37,
                                  size_contrast=512,
                                  stage=2,
                                  mrg=0,
                                  alpha=32,
                                  loss_weight=0.5),
                 *args,
                 **kwargs):
        super(ContrastRotatedConvFCBBoxHead, self).__init__(
            num_shared_convs=num_shared_convs,
            num_shared_fcs=num_shared_fcs,
            num_cls_convs=num_cls_convs,
            num_cls_fcs=num_cls_fcs,
            num_reg_convs=num_reg_convs,
            num_reg_fcs=num_reg_fcs,
            conv_out_channels=conv_out_channels,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        self.contrast_out_channels = contrast_out_channels
        self.encoder = Contrastivebranch(self.shared_out_channels, self.contrast_out_channels)  #self.mlp_head_dim 256 or 128
        self.loss_contrast = build_loss(loss_contrast)
    def init_weights(self):
        super(ContrastRotatedConvFCBBoxHead, self).init_weights()
        # conv layers are already initialized by ConvModule
        self.encoder.init_weights()

    def forward(self, x):
        """Forward function."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x
        x_contrast = x
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        contrast_feature = self.encoder(x_contrast)
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred, contrast_feature
    @force_fp32(apply_to=('cls_score', 'bbox_pred','loss_contrast'))
    def loss(self,
             cls_score,
             bbox_pred,
             contrast_feature,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """Loss function.

        Args:
            cls_score (torch.Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 5).
            rois (torch.Tensor): Boxes to be transformed. Has shape
                (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            labels (torch.Tensor): Shape (n*bs, ).
            label_weights(torch.Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
            bbox_targets(torch.Tensor):Regression target for all
                  proposals, has shape (num_proposals, 5), the
                  last dimension 5 represents [cx, cy, w, h, a].
            bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 5) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 5).
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.

        Returns:
        """
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        if contrast_feature is not None:
            if self.loss_contrast.stage == 1:
                 self.loss_contrast.init_proxies(contrast_feature, labels, cls_score)
            if self.loss_contrast.stage == 2:
                 loss_contrast= self.loss_contrast(contrast_feature, labels)  #仿ployiouloss  重新计算iou？
                 losses['loss_contrast'] = loss_contrast
        return losses

@ROTATED_HEADS.register_module()
class ContrastRotatedShared2FCBBoxHead(ContrastRotatedConvFCBBoxHead):
    """Shared2FC RBBox head."""

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(ContrastRotatedShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


class Contrastivebranch(nn.Module):
    """MLP head for contrastive representation learning, https://arxiv.org/abs/2003.04297
    Args:
        dim_in (int): dimension of the feature intended to be contrastively learned
        feat_dim (int): dim of the feature to calculated contrastive loss

    Return:
        feat_normalized (tensor): L-2 normalized encoded feature,
            so the cross-feature dot-product is cosine similarity (https://arxiv.org/abs/2004.11362)
    """
    def __init__(self, dim_in, feat_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )
    def init_weights(self):
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        #nn.init.xavier_uniform_(self.last_fc_weight)
    def forward(self, x):
        feat = self.head(x)
        return feat