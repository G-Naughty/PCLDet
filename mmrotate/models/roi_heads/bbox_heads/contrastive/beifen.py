import torch.nn as nn
from mmcv.cnn import ConvModule
from torch.nn import Parameter
from mmdet.models.builder import HEADS,build_loss
import torch
from mmdet.models.losses import accuracy
from mmdet.core import (auto_fp16, build_bbox_coder, force_fp32, multi_apply,
                        multiclass_arb_nms, get_bbox_dim, bbox2type)
import torch.nn.functional as F
#from mmdet.models.roi_heads import OBBConvFCBBoxHead
from ..obb.obb_convfc_bbox_head import OBBConvFCBBoxHead


@HEADS.register_module()
class OBBConvFCBBoxCntrastHeadV2(OBBConvFCBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs  |-> reg convs -> reg fcs -> reg
                                    \-> contrastive -> similarity between classes
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=2,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 fc_out_channels=1024,
                 contrast_out_channels=512,
                 loss_contrast=dict(
                                  type='SupConLoss',
                                  temperature=0.2,
                                  iou_threshold=0.5,
                                  loss_weight=0.5),
                 *args,
                 **kwargs):
        super(OBBConvFCBBoxCntrastHeadV2, self).__init__(
            num_shared_convs=num_shared_convs,
            num_shared_fcs=num_shared_fcs,
            num_cls_convs=num_cls_convs,
            num_cls_fcs=num_cls_fcs,
            num_reg_convs=num_reg_convs,
            num_reg_fcs=num_reg_fcs,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        #super(OBBConvFCBBoxCntrastHead, self).__init__(*args, **kwargs)
        # add contrastive branch
        self.contrast_out_channels = contrast_out_channels
        self.encoder = Contrastivebranch(self.shared_out_channels, self.contrast_out_channels)  #self.mlp_head_dim 256 or 128
        self.loss_contrast = build_loss(loss_contrast)


    def init_weights(self):
        super(OBBConvFCBBoxCntrastHeadV2, self).init_weights()
        # conv layers are already initialized by ConvModule
        self.encoder.init_weights()

    def forward(self, x):
        # shared part
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
        x_contrast = x
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:              # may + thing
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
        cls_score = self.fc_cls(x_cls)if self.with_cls else None
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
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            target_dim = self.reg_dim
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                    target_dim = get_bbox_dim(self.end_bbox_type)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), target_dim)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        target_dim)[pos_inds.type(torch.bool),
                                    labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
        if contrast_feature is not None:
            if self.loss_contrast.stage == 1:
                 self.loss_contrast.init_proxies(contrast_feature, labels, cls_score)
            if self.loss_contrast.stage == 2:
                 loss_contrast= self.loss_contrast(contrast_feature, labels)  #仿ployiouloss  重新计算iou？
                 losses['loss_contrast'] = loss_contrast
        return losses



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
