import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from nets.frcnn import FasterRCNN
from utils.utils import (cvtColor, get_classes, get_new_img_size, resize_image,
                         preprocess_input, show_config)
from utils.utils_bbox import DecodeBox


#--------------------------------------------#
# 使用自己訓練好的模型預測需要修改2個參數
# model_path和classes_path都需要修改！
# 如果出現shape不匹配
# 一定要注意訓練時的NUM_CLASSES、
# model_path和classes_path參數的修改
#--------------------------------------------#
class FRCNN(object):
    _defaults = {
        #------------------------------------------------- -------------------------#
        # 使用自己訓練好的模型進行預測一定要修改model_path和classes_path！
        # model_path指向logs資料夾下的權值文件，classes_path指向model_data下的txt
        #
        # 訓練好後logs資料夾下存在多個權值文件，選擇驗證集損失較低的即可。
        # 驗證集損失較低不代表mAP較高，僅代表該權值在驗證集上泛化表現較好。
        # 如果出現shape不匹配，同時要注意訓練時的model_path和classes_path參數的修改
        #------------------------------------------------- -------------------------#
        "model_path"    : 'model_data/voc_weights_resnet.pth',
        "classes_path"  : 'model_data/classes.txt',
        #---------------------------------------------------------------------#
        #   網路的backbone特徵提取網絡，resnet50或vgg
        #---------------------------------------------------------------------#
        "backbone"      : "resnet50",
        #---------------------------------------------------------------------#
        #   只有得分大於置信度的預測框會被保留下來
        #---------------------------------------------------------------------#
        "confidence"    : 0.8,
        #---------------------------------------------------------------------#
        #   非極大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"       : 0.1,
        #---------------------------------------------------------------------#
        #   用於指定先驗框的大小
        #---------------------------------------------------------------------#
        'anchors_size'  : [1, 2, 4],
        #-------------------------------------#
        # 是否使用Cuda
        # 沒有GPU可以設定成False
        #-------------------------------------#
        "cuda"          : True,
        #-------------------------------------#
        # show_config
        #-------------------------------------#
        "show_config"          : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
    #---------------------------------------------------#
    #   初始化faster RCNN
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
        #------------------------------------------------- --#
        # 取得種類和先驗框的數量
        #------------------------------------------------- --#
        self.class_names, self.num_classes  = get_classes(self.classes_path)

        self.std    = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes + 1)[None]
        
        if self.cuda:
            self.std    = self.std.cuda()
        self.bbox_util  = DecodeBox(self.std, self.num_classes)

        #------------------------------------------------- --#
        # 畫框設定不同的顏色
        #------------------------------------------------- --#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()
        
        if self.show_config:
            show_config(**self._defaults)

    
    #------------------------------------------------- --#
    # 載入模型
    #------------------------------------------------- --#
    def generate(self):
        #-------------------------------------#
        # 載入模型與權值
        #-------------------------------------#
        if self.model_path != "":
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.net = FasterRCNN(self.num_classes, "predict", anchor_scales = self.anchors_size, backbone = self.backbone)
            self.net.load_state_dict(torch.load(self.model_path, map_location=device))
            print('{} model, anchors, and classes loaded.'.format(self.model_path))
            self.net    = self.net.eval()
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()
        else:
            self.net = None
        
    
    #---------------------------------------------------#
    #   檢測圖片
    #---------------------------------------------------#
    def detect_image(self, image, img_name = '', xml_save_path = '',crop = False, count = False, output = "default", model = None):
        #---------------------------------------------------#
        #   計算輸入圖片的高和寬
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------#
        #   計算resize後的圖片的大小，resize後的圖片短邊為600
        #---------------------------------------------------#
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        #------------------------------------------------- --------#
        # 在這裡將影像轉換成RGB影像，防止灰階圖在預測時報錯。
        # 程式碼僅支援RGB影像的預測，所有其它類型的影像都會轉換成RGB
        #------------------------------------------------- --------#
        image       = cvtColor(image)
        #------------------------------------------------- --------#
        # 給予原始影像進行resize，resize到短邊為600的大小上
        #------------------------------------------------- --------#
        image_data  = resize_image(image, [input_shape[1], input_shape[0]])
        #------------------------------------------------- --------#
        # 新增上batch_size維度
        #------------------------------------------------- --------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #------------------------------------------------- ------------#
            # roi_cls_locs 建議框的調整參數
            # roi_scores 建議框的種類得分
            # rois 建議框的座標
            #------------------------------------------------- ------------#
            if model != None:
                roi_cls_locs, roi_scores, rois, _ = model(images)
            else:
                roi_cls_locs, roi_scores, rois, _ = self.net(images)
            #------------------------------------------------- ------------#
            # 利用classifier的預測結果對建議框進行解碼，取得預測框
            #------------------------------------------------- ------------#
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)
            #------------------------------------------------- --------#
            # 如果沒有偵測出物體，返回原圖
            #------------------------------------------------- --------#         
            if len(results[0]) <= 0:
                if output == "default":
                    return image
                elif output == "list":
                    return [], [], []
                
            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]
        
        if output == "list":
            if len(top_boxes) != 0:
                boxestmp = top_boxes.copy()
                top_boxes[:, 0] = boxestmp[:, 1]
                top_boxes[:, 1] = boxestmp[:, 0]
                top_boxes[:, 2] = boxestmp[:, 3]
                top_boxes[:, 3] = boxestmp[:, 2]
            return top_label, top_conf, top_boxes
        
        #---------------------------------------------------------#
        #   設定字體與邊框厚度
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))
        #---------------------------------------------------------#
        #   計數
        #---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        #---------------------------------------------------------#
        #   是否進行目標的裁剪
        #---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        #---------------------------------------------------------#
        #   影像繪製
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            # print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        #---------------------------------------------------------#
        #   產生XML
        #---------------------------------------------------------#
        if xml_save_path :
            xml_file = open((xml_save_path+ '/' +os.path.splitext(img_name)[0]+ '.xml'), 'w')
            xml_file.write('<annotation>\n')
            xml_file.write('    <folder>PCB</folder>\n')
            xml_file.write('    <filename>' +img_name+'</filename>\n')
            xml_file.write('    <source>\n')
            xml_file.write('        <database>'+'Unknown'+'</database>\n')
            xml_file.write('    </source>\n')
            xml_file.write('    <size>\n')
            xml_file.write('        <width>' + str(image.size[0]) + '</width>\n')
            xml_file.write('        <height>' + str(image.size[1]) + '</height>\n')
            xml_file.write('        <depth>'+'3'+'</depth>\n')
            xml_file.write('    </size>\n')
            xml_file.write('    <segmented>'+'0'+'</segmented>\n')

            for i, c in list(enumerate(top_label)):
                predicted_class = self.class_names[int(c)]
                box             = top_boxes[i]
                score           = top_conf[i]

                top, left, bottom, right = box
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))

                xml_file.write('    <object>\n')
                xml_file.write('        <name>' + predicted_class + '</name>\n')
                xml_file.write('        <pose>Unspecified</pose>\n')
                xml_file.write('        <truncated>0</truncated>\n')
                xml_file.write('        <difficult>0</difficult>\n')
                xml_file.write('        <bndbox>\n')
                xml_file.write('            <xmin>' + str(left) + '</xmin>\n')
                xml_file.write('            <ymin>' + str(top) + '</ymin>\n')
                xml_file.write('            <xmax>' + str(right) + '</xmax>\n')
                xml_file.write('            <ymax>' + str(bottom) + '</ymax>\n')
                xml_file.write('        </bndbox>\n')
                xml_file.write('    </object>\n')
            xml_file.write('</annotation>')

        return image

    def get_FPS(self, image, test_interval):
        #---------------------------------------------------#
        #   計算輸入圖片的高和寬
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        #------------------------------------------------- --------#
        # 在這裡將影像轉換成RGB影像，防止灰階圖在預測時報錯。
        # 程式碼僅支援RGB影像的預測，所有其它類型的影像都會轉換成RGB
        #------------------------------------------------- --------#
        image       = cvtColor(image)
        
        #------------------------------------------------- --------#
        # 給予原始影像進行resize，resize到短邊為600的大小上
        #------------------------------------------------- --------#
        image_data  = resize_image(image, [input_shape[1], input_shape[0]])
        #------------------------------------------------- --------#
        # 新增上batch_size維度
        #------------------------------------------------- --------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            roi_cls_locs, roi_scores, rois, _ = self.net(images)
            #------------------------------------------------- ------------#
            # 利用classifier的預測結果對建議框進行解碼，取得預測框
            #------------------------------------------------- ------------#
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                roi_cls_locs, roi_scores, rois, _ = self.net(images)
                #------------------------------------------------- ------------#
                # 利用classifier的預測結果對建議框進行解碼，取得預測框
                #------------------------------------------------- ------------#
                results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, 
                                                        nms_iou = self.nms_iou, confidence = self.confidence)
                
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    #------------------------------------------------- --#
    # 偵測圖片
    #------------------------------------------------- --#
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w")
        #------------------------------------------------- --#
        # 計算輸入圖片的高和寬
        #------------------------------------------------- --#
        image_shape = np.array(np.shape(image)[0:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        #------------------------------------------------- --------#
        # 在這裡將影像轉換成RGB影像，防止灰階圖在預測時報錯。
        # 程式碼僅支援RGB影像的預測，所有其它類型的影像都會轉換成RGB
        #------------------------------------------------- --------#
        image       = cvtColor(image)
        
        #------------------------------------------------- --------#
        # 給予原始影像進行resize，resize到短邊為600的大小上
        #------------------------------------------------- --------#
        image_data  = resize_image(image, [input_shape[1], input_shape[0]])
        #------------------------------------------------- --------#
        # 新增上batch_size維度
        #------------------------------------------------- --------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            roi_cls_locs, roi_scores, rois, _ = self.net(images)
            #------------------------------------------------- ------------#
            # 利用classifier的預測結果對建議框進行解碼，取得預測框
            #------------------------------------------------- ------------#
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)
            #----------------------------------------------------#
            # 如果沒有偵測到物體，則傳回原圖
            #----------------------------------------------------#
            if len(results[0]) <= 0:
                return 

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]
        
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
