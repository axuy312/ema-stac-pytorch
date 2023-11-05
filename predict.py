#------------------------------------------------- ---#
# 將單張圖片預測、相機偵測和FPS測試功能
# 整合到了一個py檔案中，透過指定mode進行模式的修改。
#------------------------------------------------- ---#
import time

import cv2
import numpy as np
from PIL import Image

from frcnn import FRCNN

if __name__ == "__main__":
    frcnn = FRCNN(model_path="model_data/voc_weights_resnet.pth", confidence=0.8, nms_iou=0.1)
    #------------------------------------------------- -------------------------------------------------- -------#
    # mode用於指定測試的模式：
    # 'predict' 表示單張圖片預測，如果想對預測過程進行修改，如儲存圖片，截取物件等，可以先看下方詳細的註釋
    # 'video' 表示視訊偵測，可呼叫相機或影片進行偵測，詳情請參閱下方註解。
    # 'fps' 表示測試fps，使用的圖片是img裡面的street.jpg，詳情查看下方註解。
    # 'dir_predict' 表示遍歷資料夾進行偵測並儲存。預設遍歷img資料夾，儲存img_out資料夾，詳情查看下方註解。
    #------------------------------------------------- -------------------------------------------------- -------#
    mode = "dir_predict"
    #------------------------------------------------- ------------------------#
    # crop 指定了是否在單張圖片預測後對目標進行截取
    # count 指定了是否進行目標的計數
    # crop、count僅在mode='predict'時有效
    #------------------------------------------------- ------------------------#
    crop            = False
    count           = False
    #------------------------------------------------- -------------------------------------------------- -------#
    # video_path 用於指定視訊的路徑，當video_path=0時表示偵測相機
    # 想要偵測視頻，則設定如video_path = "xxx.mp4"即可，代表讀取出根目錄下的xxx.mp4檔。
    # video_save_path 表示影片儲存的路徑，當video_save_path=""時表示不儲存
    # 想要儲存視頻，則設定如video_save_path = "yyy.mp4"即可，代表儲存為根目錄下的yyy.mp4檔案。
    # video_fps 用於保存的影片的fps
    #
    # video_path、video_save_path和video_fps僅在mode='video'時有效
    # 儲存影片時需要ctrl+c退出或執行到最後一幀才會完成完整的儲存步驟。
    #------------------------------------------------- -------------------------------------------------- -------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #------------------------------------------------- -------------------------------------------------- -------#
    # test_interval 用來指定測量fps的時候，圖片偵測的次數。理論上test_interval越大，fps越準確。
    # fps_image_path 用來指定測試的fps圖片
    #
    # test_interval和fps_image_path僅在mode='fps'有效
    #------------------------------------------------- -------------------------------------------------- -------#
    test_interval   = 100
    fps_image_path  = "img/street.jpg"
    #------------------------------------------------- ------------------------#
    # dir_origin_path 指定了用於偵測的圖片的資料夾路徑
    # dir_save_path 指定了偵測完圖片的儲存路徑
    #
    # dir_origin_path和dir_save_path僅在mode='dir_predict'時有效
    #------------------------------------------------- ------------------------#
    dir_origin_path = "dataset/PCB/JPEGImages_unlabeled"
    dir_save_path   = "result"
    xml_save_path   = ""

    if mode == "predict":
        '''
        1.程式碼無法直接進行批次預測，如果想要批次預測，可以利用os.listdir()遍歷資料夾，利用Image.open開啟圖片檔案進行預測。
        具體流程可以參考get_dr_txt.py，在get_dr_txt.py即實現了遍歷也實現了目標資訊的保存。
        2.如果想要進行檢測完的圖片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py裡進行修改即可。
        3.如果想要取得預測框的座標，可以進入frcnn.detect_image函數，在繪圖部分讀取top，left，bottom，right這四個值。
        4.如果想要利用預測框截取下目標，可以進入frcnn.detect_image函數，在繪圖部分利用取得到的top，left，bottom，right這四個值
        在原圖上利用矩陣的方式進行截取。
        5.如果想要在預測圖上寫額外的字，例如偵測到的特定目標的數量，可以進入frcnn.detect_image函數，在繪圖部分對predicted_class進行判斷，
        例如判斷if predicted_class == 'car': 即可判斷目前目標是否為車，然後記錄數量即可。利用draw.text即可寫字。
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = frcnn.detect_image(image, crop = crop, count = count)
                r_image.show()

    elif mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        while(True):
            t1 = time.time()
            # 讀取某一幀
            ref,frame=capture.read()
            # 格式轉變，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 轉變成Image
            frame = Image.fromarray(np.uint8(frame))
            # 進行偵測
            frame = np.array(frcnn.detect_image(frame))
            # RGBtoBGR滿足opencv顯示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = frcnn.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = frcnn.detect_image(image, img_name, xml_save_path)
                if dir_save_path:
                    if not os.path.exists(dir_save_path):
                        os.makedirs(dir_save_path)
                    r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
