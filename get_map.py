import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from frcnn import FRCNN

if __name__ == "__main__":
    '''
    Recall和Precision不像AP是面積的概念，因此在閘限值（Confidence）不同時，網路的Recall和Precision值是不同的。
    預設情況下，本程式碼計算的Recall和Precision代表的是當閘限值（Confidence）為0.5時，所對應的Recall和Precision值。

    受到mAP計算原理的限制，網路在計算mAP時需要得到近乎所有的預測框，這樣才可以計算不同閘限條件下的Recall和Precision值
    因此，本程式碼所獲得的map_out/detection-results/裡面的txt的框的數量一般會比直接predict多一些，目的是列出所有可能的預測框，
    '''
    #------------------------------------------------- -------------------------------------------------- ---------------#
    # map_mode用於指定該檔案執行時計算的內容
    # map_mode為0代表整個map計算流程，包含獲得預測結果、取得真實方塊、計算VOC_map。
    # map_mode為1代表僅獲得預測結果。
    # map_mode為2代表僅獲得真實框。
    # map_mode為3代表僅計算VOC_map。
    # map_mode為4代表利用COCO工具箱計算目前資料集的0.50:0.95map。需要取得預測結果、取得真實方塊後並安裝pycocotools才行
    #------------------------------------------------- -------------------------------------------------- ----------------#
    map_mode        = 0
    #------------------------------------------------- -------------------------------------#
    # 此處的classes_path用來指定需要測量VOC_map的類別
    # 一般情況下與訓練和預測所用的classes_path一致即可
    #------------------------------------------------- -------------------------------------#
    classes_path    = 'model_data/classes.txt'
    #------------------------------------------------- -------------------------------------#
    # MINOVERLAP用來指定想要獲得的mAP0.x，mAP0.x的意義是什麼請同學們百度一下。
    # 例如計算mAP0.75，可以設定MINOVERLAP = 0.75。
    #
    # 當某一預測框與真實框重合度大於MINOVERLAP時，此預測框被視為正樣本，否則為負樣本。
    # 因此MINOVERLAP的值越大，預測框要預測的越準確才能被認為是正樣本，此時算出來的mAP值越低，
    #------------------------------------------------- -------------------------------------#
    MINOVERLAP      = 0.5
    #------------------------------------------------- -------------------------------------#
    # 受到mAP計算原理的限制，網路在計算mAP時需要取得近乎所有的預測框，這樣才可以計算mAP
    # 因此，confidence的值應設定的盡量小進而獲得全部可能的預測框。
    #
    # 該值一般不調整。因為計算mAP需要取得近乎所有的預測框，這裡的confidence不能隨便改變。
    # 想要取得不同閘限值下的Recall和Precision值，請修改下方的score_threhold。
    #------------------------------------------------- -------------------------------------#
    confidence      = 0.02
    #------------------------------------------------- -------------------------------------#
    # 預測時使用到的非極大抑制值的大小，越大表示非極大抑制越不嚴格。
    #
    # 該值一般不調整。
    #------------------------------------------------- -------------------------------------#
    nms_iou         = 0.1
    #------------------------------------------------- -------------------------------------------------- ------------#
    # Recall和Precision不像AP是一個面積的概念，因此在閘限值不同時，網路的Recall和Precision值是不同的。
    #
    # 預設情況下，本程式碼計算的Recall和Precision代表的是當閘限值為0.5（此處定義為score_threhold）時所對應的Recall和Precision值。
    # 因為計算mAP需要取得近乎所有的預測框，所以上面定義的confidence不能隨便更改。
    # 這裡特別定義一個score_threhold用來代表閘限值，進而在計算mAP時找到閘限值對應的Recall和Precision值。
    #------------------------------------------------- -------------------------------------------------- ------------#
    score_threhold  = 0.5
    #------------------------------------------------- ------#
    # map_vis用於指定是否開啟VOC_map計算的可視化
    #------------------------------------------------- ------#
    map_vis         = False
    #------------------------------------------------- ------#
    # 指向VOC資料集所在的資料夾
    # 預設指向根目錄下的VOC資料集
    #------------------------------------------------- ------#
    Dataset  = 'PCB'
    #------------------------------------------------- ------#
    # 結果輸出的資料夾，預設為map_out
    #------------------------------------------------- ------#
    map_out_path    = 'map_out'

    image_ids = [os.path.splitext(fn)[0] for fn in os.listdir(os.path.join('dataset', Dataset, "Annotations_labeled_test/")) if fn.endswith('.xml')]

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        frcnn = FRCNN(model_path = "model_data/voc_weights_resnet.pth", confidence = confidence, nms_iou = nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join('dataset', Dataset, "JPEGImages_labeled/"+image_id+".jpg")
            image       = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            frcnn.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")
        
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                root = ET.parse(os.path.join('dataset', Dataset, "Annotations_labeled_test/"+image_id+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox  = obj.find('bndbox')
                    left    = bndbox.find('xmin').text
                    top     = bndbox.find('ymin').text
                    right   = bndbox.find('xmax').text
                    bottom  = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold = score_threhold, path = map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")
