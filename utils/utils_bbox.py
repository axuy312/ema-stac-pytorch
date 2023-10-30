import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import nms


def loc2bbox(src_bbox, loc):
    if src_bbox.size()[0] == 0:
        return torch.zeros((0, 4), dtype=loc.dtype)

    src_width   = torch.unsqueeze(src_bbox[:, 2] - src_bbox[:, 0], -1)
    src_height  = torch.unsqueeze(src_bbox[:, 3] - src_bbox[:, 1], -1)
    src_ctr_x   = torch.unsqueeze(src_bbox[:, 0], -1) + 0.5 * src_width
    src_ctr_y   = torch.unsqueeze(src_bbox[:, 1], -1) + 0.5 * src_height

    dx          = loc[:, 0::4]
    dy          = loc[:, 1::4]
    dw          = loc[:, 2::4]
    dh          = loc[:, 3::4]

    ctr_x = dx * src_width + src_ctr_x
    ctr_y = dy * src_height + src_ctr_y
    w = torch.exp(dw) * src_width
    h = torch.exp(dh) * src_height

    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h

    return dst_bbox

class DecodeBox():
    def __init__(self, std, num_classes):
        self.std            = std
        self.num_classes    = num_classes + 1    

    def frcnn_correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        #-----------------------------------------------------------------#
        # 把y軸放前面是因為方便預測框和影像的寬高進行相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def forward(self, roi_cls_locs, roi_scores, rois, image_shape, input_shape, nms_iou = 0.3, confidence = 0.5):
        results = []
        bs      = len(roi_cls_locs)
        #--------------------------------#
        #   batch_size, num_rois, 4
        #--------------------------------#
        rois    = rois.view((bs, -1, 4))
        #----------------------------------------------------------------------------------------------------------------#
        # 對每一張圖片進行處理，由於在predict.py的時候，我們只輸入一張圖片，所以for i in range(len(mbox_loc))只進行一次
        #----------------------------------------------------------------------------------------------------------------#
        for i in range(bs):
            #----------------------------------------------------------#
            # 對迴歸參數進行reshape
            #----------------------------------------------------------#
            roi_cls_loc = roi_cls_locs[i] * self.std
            #----------------------------------------------------------#
            # 一維度是建議框的數量，第二維度是每個種類
            # 第三維度是對應種類的調整參數
            #----------------------------------------------------------#
            roi_cls_loc = roi_cls_loc.view([-1, self.num_classes, 4])

            #-------------------------------------------------------------#
            # 利用classifier網路的預測結果對建議框進行調整以獲得預測框
            # num_rois, 4 -> num_rois, 1, 4 -> num_rois, num_classes, 4
            #-------------------------------------------------------------#
            roi         = rois[i].view((-1, 1, 4)).expand_as(roi_cls_loc)
            cls_bbox    = loc2bbox(roi.contiguous().view((-1, 4)), roi_cls_loc.contiguous().view((-1, 4)))
            cls_bbox    = cls_bbox.view([-1, (self.num_classes), 4])
            #-------------------------------------------------------------#
            # 將預測框進行歸一化，調整到0-1之間
            #-------------------------------------------------------------#
            cls_bbox[..., [0, 2]] = (cls_bbox[..., [0, 2]]) / input_shape[1]
            cls_bbox[..., [1, 3]] = (cls_bbox[..., [1, 3]]) / input_shape[0]

            roi_score   = roi_scores[i]
            prob        = F.softmax(roi_score, dim=-1)

            results.append([])
            for c in range(1, self.num_classes):
                #--------------------------------#
                # 取出屬於該類別的所有框的置信度
                # 判斷是否大於門限
                #--------------------------------#
                c_confs     = prob[:, c]
                c_confs_m   = c_confs > confidence

                if len(c_confs[c_confs_m]) > 0:
                    #----------------------------出-------------#
                    # 取得分高於confidence的框
                    #-----------------------------------------#
                    boxes_to_process = cls_bbox[c_confs_m, c]
                    confs_to_process = c_confs[c_confs_m]

                    keep = nms(
                        boxes_to_process,
                        confs_to_process,
                        nms_iou
                    )
                    #-----------------------------------------#
                    # 取出在非極大抑制中效果較好的內容
                    #-----------------------------------------#
                    good_boxes  = boxes_to_process[keep]
                    confs       = confs_to_process[keep][:, None]
                    labels      = (c - 1) * torch.ones((len(keep), 1)).cuda() if confs.is_cuda else (c - 1) * torch.ones((len(keep), 1))
                    #-----------------------------------------#
                    # 將label、置信度、框的位置進行堆疊。
                    #-----------------------------------------#
                    c_pred      = torch.cat((good_boxes, confs, labels), dim=1).cpu().numpy()
                    # 加入result裡
                    results[-1].extend(c_pred)

            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4])/2, results[-1][:, 2:4] - results[-1][:, 0:2]
                results[-1][:, :4] = self.frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape)

        return results
        
