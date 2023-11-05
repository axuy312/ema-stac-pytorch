import os
import random
import xml.etree.ElementTree as ET
import numpy as np
from utils.utils import get_classes

#-------------------------------------------------------------------#
# 用於產生train.txt、val.txt的目標訊息
# 與訓練和預測所用的classes_path一致即可
# 如果產生的train.txt裡面沒有目標訊息
# 那麼就是因為classes沒有設定正確
#-------------------------------------------------------------------#
classes_path = 'model_data/classes.txt'
classes, _ = get_classes(classes_path)
#-------------------------------------------------------#
# 資料集名稱
# 預設指向根目錄下的PCB資料集
#-------------------------------------------------------#
Dataset  = 'PCB'
#-------------------------------------------------------#
# 統計目標數量
#-------------------------------------------------------#
nums = np.zeros(len(classes))

def convert_annotation(image_id, path, save_file):
    in_file = open(os.path.join(path, '%s.xml' % (image_id)), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        bbox = (int(float(xmlbox.find(i).text)) for i in ['xmin', 'ymin', 'xmax', 'ymax'])
        save_file.write(f" {','.join(map(str, bbox))},{cls_id}")
        nums[classes.index(cls)] += 1

def process_data(txtfile, images_path, anno_path):
    file_list = [os.path.splitext(file)[0] for file in os.listdir(anno_path) if file.endswith('.xml')]
    with open(txtfile, 'w', encoding='utf-8') as save_file:
        for image_id in file_list:
            save_file.write(f'dataset/{Dataset}/{images_path}/{image_id}.jpg')
            convert_annotation(image_id, anno_path, save_file)
            save_file.write('\n')

def printTable(List1, List2):
    for i in range(len(List1[0])):
        print("|", end=' ')
        for j in range(len(List1)):
            print(List1[j][i].rjust(int(List2[j])), end=' ')
            print("|", end=' ')
        print()

if __name__ == "__main__":
    random.seed(0)
    if " " in os.path.abspath('dataset'):
        raise ValueError("資料集存放的資料夾路徑與圖片名稱中不可以有空格，否則會影響正常的模型訓練，請注意修改")

    directories = {
        'train': ('Annotations_labeled_train', 'JPEGImages_labeled'),
        'val':   ('Annotations_labeled_val', 'JPEGImages_labeled')
    }

    for d, (anno, img) in directories.items():
        process_data(f'{d}.txt', img, os.path.join('dataset', Dataset, anno))

    str_nums = [str(int(x)) for x in nums]
    tableData = [classes, str_nums]
    colWidths = [0]*len(tableData)
    len1 = 0
    for i in range(len(tableData)):
        for j in range(len(tableData[i])):
            if len(tableData[i][j]) > colWidths[i]:
                colWidths[i] = len(tableData[i][j])
    printTable(tableData, colWidths)
