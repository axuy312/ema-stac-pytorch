import cv2, random
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input
from frcnn import FRCNN
from tqdm import tqdm


class FRCNNDataset(Dataset):
    def __init__(self, annotation_lines, input_shape = [600, 600], train = True, cutout = False, max_num = -1):
        self.annotation_lines_all = annotation_lines
        #random.shuffle(self.annotation_lines_all)
        self.annotation_lines = None
        if max_num == -1:
            self.annotation_lines = self.annotation_lines_all
        else:
            self.annotation_lines = random.sample(self.annotation_lines_all, min(max_num, len(self.annotation_lines_all)))
        self.input_shape = input_shape
        self.train = train
        self.cutout = cutout

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        index       = index % len(self.annotation_lines)
        #------------------------------------------------- --#
        # 訓練時進行資料的隨機增強
        # 驗證時不進行資料的隨機增強
        #------------------------------------------------- --#
        image, y    = self.get_random_data(self.annotation_lines[index], self.input_shape[0:2], random = self.train)
        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box_data    = np.zeros((len(y), 5))
        if len(y) > 0:
            box_data[:len(y)] = y

        box         = box_data[:, :4]
        label       = box_data[:, -1]
        return image, box, label

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def resample_data(self, max_num = -1):
        if max_num == -1:
            self.annotation_lines = self.annotation_lines_all
        else:
            self.annotation_lines = random.sample(self.annotation_lines_all, min(max_num, len(self.annotation_lines_all)))
        
    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line = annotation_line.split()
        #------------------------------#
        # 讀取影像並轉換成RGB影像
        #------------------------------#
        image   = Image.open(line[0])
        image   = cvtColor(image)
        #------------------------------#
        # 獲得影像的高寬與目標高寬
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        # 獲得預測框
        #------------------------------#
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            # 將影像多餘的部分加上灰條
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            #---------------------------------#
            # 對真實框進行調整
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_data, box
                
        #------------------------------------------#
        # 將影像縮放並且進行長和寬的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        # 將影像多餘的部分加上灰條
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        # 翻轉影像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        # 對影像進行色域變換
        # 計算色域變換的參數
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        # 將影像轉到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        # 應用變換
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        # 對真實框進行調整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        
        #---------------------------------------------------
        #   對影像作cutout argumentation
        #---------------------------------------------------
        if self.cutout:
            square_size = 16
            num_cutout = 50
            iou_threashold = 0.2
            gray_square = np.ones((square_size, square_size, 3), dtype=np.uint8) * 128  # 128 表示灰色
            while num_cutout > 0:
                g_x = int(np.random.uniform(0, w - square_size))
                g_y =int(np.random.uniform(0, h - square_size))
                cutout_box = [g_x, g_y, g_x + square_size, g_y + square_size]
                iou_threashold_pass = True
                for b in box:
                    if (calculate_box_overlay_ratio(b, cutout_box) > iou_threashold):
                        iou_threashold_pass = False
                        break
                if iou_threashold_pass:
                    image_data[g_y:g_y+square_size, g_x:g_x+square_size] = gray_square
                    num_cutout -= 1
        
        return image_data, box
    
    def updateGT(self, model):
        #不能用self.frcnn
        frcnn = FRCNN(model_path="", confidence=0.8, nms_iou=0.1, show_config=False)
        #get teacher pseudo label
        
        box_cnt = 0
        #tmp = []
        with torch.no_grad():
            with tqdm(total=len(self.annotation_lines),desc=f'Pseudo updating...',postfix=dict,mininterval=0.3) as pbar:
                for i, line in enumerate(self.annotation_lines):
                    img_path = line.split(" ")[0]
                    
                    image = Image.open(img_path)
                    labels, confs, boxes = frcnn.detect_image(image, output="list", model=model)
                    new_line = img_path
                    for j in range(len(labels)):
                        new_line = f"{new_line} {int(boxes[j][0])},{int(boxes[j][1])},{int(boxes[j][2])},{int(boxes[j][3])},{labels[j]}"
                        box_cnt += 1
                        pbar.set_postfix(**{'#box'    : box_cnt})
                    self.annotation_lines[i] = new_line
                    #if len(labels) != 0:
                    #    tmp.append(new_line)
                    pbar.update(1)
        #self.annotation_lines = tmp

# DataLoader中collate_fn使用
def frcnn_dataset_collate(batch):
    images = []
    bboxes = []
    labels = []
    for img, box, label in batch:
        images.append(img)
        bboxes.append(box)
        labels.append(label)
    images = torch.from_numpy(np.array(images))
    return images, bboxes, labels

