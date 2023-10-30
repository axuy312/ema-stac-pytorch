import torch.nn as nn

from nets.classifier import Resnet50RoIHead, VGG16RoIHead
from nets.resnet50 import resnet50
from nets.rpn import RegionProposalNetwork
from nets.vgg16 import decom_vgg16


class FasterRCNN(nn.Module):
    def __init__(self,  num_classes,  
                    mode = "training",
                    feat_stride = 16,
                    anchor_scales = [8, 16, 32],
                    ratios = [0.5, 1, 2],
                    backbone = 'vgg',
                    pretrained = False):
        super(FasterRCNN, self).__init__()
        self.feat_stride = feat_stride
        #---------------------------------#
        # 一共存在兩個backbone
        # vgg和resnet50
        #---------------------------------#
        if backbone == 'vgg':
            self.extractor, classifier = decom_vgg16(pretrained)
            #---------------------------------#
            # 建立Region Proposal Network(RPN)
            #---------------------------------#
            self.rpn = RegionProposalNetwork(
                512, 512,
                ratios          = ratios,
                anchor_scales   = anchor_scales,
                feat_stride     = self.feat_stride,
                mode            = mode
            )
            #---------------------------------#
            # 建構classifier網絡(VGG)
            #---------------------------------#
            self.head = VGG16RoIHead(
                n_class         = num_classes + 1,
                roi_size        = 7,
                spatial_scale   = 1,
                classifier      = classifier
            )
        elif backbone == 'resnet50':
            self.extractor, classifier = resnet50(pretrained)
            #---------------------------------#
            # 建立Region Proposal Network(RPN)
            #---------------------------------#
            self.rpn = RegionProposalNetwork(
                1024, 512,
                ratios          = ratios,
                anchor_scales   = anchor_scales,
                feat_stride     = self.feat_stride,
                mode            = mode
            )
            #---------------------------------#
            # 建構classifier網絡(resnet)
            #---------------------------------#
            self.head = Resnet50RoIHead(
                n_class         = num_classes + 1,
                roi_size        = 14,
                spatial_scale   = 1,
                classifier      = classifier
            )
            
    def forward(self, x, scale=1., mode="forward"):
        if mode == "forward":
            #---------------------------------#
            # 計算輸入圖片的大小
            #---------------------------------#
            img_size        = x.shape[2:]
            #---------------------------------#
            # 利用backbone網路提取特徵
            #---------------------------------#
            base_feature    = self.extractor.forward(x)

            #---------------------------------#
            # 獲得建議框(Proposal)
            #---------------------------------#
            _, _, rois, roi_indices, _  = self.rpn.forward(base_feature, img_size, scale)
            #---------------------------------------#
            # 獲得classifier的分類結果和迴歸結果
            #---------------------------------------#
            roi_cls_locs, roi_scores    = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores, rois, roi_indices
        elif mode == "extractor":
            #---------------------------------#
            #  利用backbone網路提取特徵
            #---------------------------------#
            base_feature    = self.extractor.forward(x)
            return base_feature
        elif mode == "rpn":
            base_feature, img_size = x
            #---------------------------------#
            # 獲得建議框(Proposal)
            #---------------------------------#
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.forward(base_feature, img_size, scale)
            return rpn_locs, rpn_scores, rois, roi_indices, anchor
        elif mode == "head":
            base_feature, rois, roi_indices, img_size = x
            #---------------------------------------#
            # 獲得classifier的分類結果和迴歸結果
            #---------------------------------------#
            roi_cls_locs, roi_scores    = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
