#-------------------------------------#
#       對數據集進行訓練
#-------------------------------------#
import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.frcnn import FasterRCNN
from nets.frcnn_training import (FasterRCNNTrainer, get_lr_scheduler,
                                 set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import FRCNNDataset, frcnn_dataset_collate
from utils.utils import (get_classes, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch

from PIL import Image
from utils.utils import get_new_img_size, resize_image, cvtColor, preprocess_input

'''
1、訓練前仔細檢查自己的格式是否滿足要求，該庫要求數據集格式為VOC格式，需要準備好的內容有輸入圖片和標籤
輸入圖片為.jpg圖片，無需固定大小，傳入訓練前會自動進行resize。
灰度圖會自動轉成RGB圖片進行訓練，無需自己修改。
輸入圖片如果後綴非jpg，需要自己批量轉成jpg後再開始訓練。

標籤為.xml格式，文件中會有需要檢測的目標信息，標籤文件和輸入圖片文件相對應。

2、損失值的大小用於判斷是否收斂，比較重要的是有收斂的趨勢，即驗證集損失不斷下降，如果驗證集損失基本上不改變的話，模型基本上就收斂了。
損失值的具體大小並沒有什麼意義，大和小只在於損失的計算方式，並不是接近於0才好。如果想要讓損失好看點，可以直接到對應的損失函數裡面除上10000。
訓練過程中的損失值會保存在logs文件夾下的loss_%Y_%m_%d_%H_%M_%S文件夾中

3、訓練好的權值文件保存在logs文件夾中，每個訓練世代（Epoch）包含若干訓練步長（Step），每個訓練步長（Step）進行一次梯度下降。
如果只是訓練了幾個Step是不會保存的，Epoch和Step的概念要講清楚一下。
'''
if __name__ == "__main__":
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以設置成False
    #-------------------------------#
    Cuda            = True
    #----------------------------------------------#
    #   Seed    用于固定隨機種子
    #           使得每次獨立訓練都可以獲得一樣的结果
    #----------------------------------------------#
    seed            = 11
    #---------------------------------------------------------------------#
    #   train_gpu   訓練用到的GPU
    #               默認為第一張卡、雙卡為[0, 1]、三卡為[0, 1, 2]
    #               在使用多GPU時，每個卡上的batch為總batch除以卡的數量。
    #---------------------------------------------------------------------#
    train_gpu       = [0,]
    #---------------------------------------------------------------------#
    #   train_mode  
    #               Teacher
    #               Student
    #---------------------------------------------------------------------#
    train_mode       = "Teacher"
    #---------------------------------------------------------------------#
    #   fp16        是否使用混合精度訓練
    #               可减少约一半的記憶體、需要pytorch1.7.1以上
    #---------------------------------------------------------------------#
    fp16            = False
    #---------------------------------------------------------------------#
    #   classes_path    指向model_data下的txt，與自己訓練的數據集相關 
    #                   訓練前一定要修改classes_path，使其動應自己的數據集
    #---------------------------------------------------------------------#
    classes_path    = 'model_data/classes.txt'
    #----------------------------------------------------------------------------------------------------------------------------#
    #   權值文件的下載請看README，可以通過網路下载。模型的預訓練權重 對不同數據集是通用的，因為特徵是通用的。
    #   模型的 預訓練權重 比较重要的部分是 主幹特徵提取網路的權值部分，用於進行特徵提取。
    #   預訓練權重對於99%的情況都必須要使用，不使用的話主幹部分的權值過於隨機，特徵提取效果不明顯，網絡訓練的結果也不會好。    #
    #   如果訓練過程中出現中斷訓練的情況，可以將model_path設定為logs文件夾中的權值文件，以重新載入已經訓練了一部分的權值。
    #   同時修改下方的「凍結階段」或「解凍階段」的參數，以確保模型epoch的連續性。
    #   
    #   當model_path = ''的時候，不會載入整個模型的權值。
    #
    #   此處使用的是整個模型的權重，因此在train.py進行載入，下面的pretrain不會影響此處的權值載入。
    #   如果想要讓模型從主干的預訓練權值開始訓練，則設置model_path = ''，下面的pretrain = True，此時僅載入主干部分的權值。
    #   如果想要讓模型從零開始訓練，則設置model_path = ''，下面的pretrain = False，Freeze_Train = False，此時將從零開始訓練，並且不會凍結主幹部分的過程。
    #   
    #   一般來說，網絡從零開始的訓練效果會很差，因為權值太過隨機，特徵提取效果不明顯，因此強烈、強烈、強烈不建議大家從零開始訓練！
    #   如果一定要從零開始，可以了解Imagenet數據集，首先訓練分類模型，獲得網絡的主幹部分權值，分類模型的主幹部分和該模型通用，基於此進行訓練。
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = 'model_data/pcb_weights_resnet.pth'
    #------------------------------------------------------#
    #   input_shape     輸入的shape大小
    #------------------------------------------------------#
    input_shape     = [600, 600]
    #---------------------------------------------#
    #   vgg
    #   resnet50
    #---------------------------------------------#
    backbone        = "resnet50"
    #----------------------------------------------------------------------------------------------------------------------------#
    # pretrained 是否使用主干網絡的預訓練權重，此處使用的是主干的權重，因此是在模型構建的時候進行加載的。
    # 如果設置了model_path，則主干的權值無需加載，pretrained的值無意義。
    # 如果不設置model_path，pretrained = True，此時僅加載主干開始訓練。
    # 如果不設置model_path，pretrained = False，Freeze_Train = False，此時從0開始訓練，且沒有凍結主干的過程。
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = True
    #------------------------------------------------------------------------#
    #   anchors_size用於設定先驗框的大小，每個特徵點均存在9個先驗框。
    #   anchors_size每個數對應3個先驗框。
    #   當anchors_size = [8, 16, 32]的時候，生成的先驗框寬高約為：
    #   [90, 180] ; [180, 360]; [360, 720]; [128, 128]; 
    #   [256, 256]; [512, 512]; [180, 90] ; [360, 180]; 
    #   [720, 360]; 詳情查看anchors.py
    #   如果想要檢測小物體，可以減小anchors_size靠前的數。
    #   比如設置anchors_size = [4, 16, 32]
    #------------------------------------------------------------------------#
    anchors_size    = [1, 2, 4]
    #----------------------------------------------------------------------------------------------------------------------------#
    #   訓練分為兩個階段，分別是凍結階段和解凍階段。設置凍結階段是為了滿足機器性能不足的同學的訓練需求。
    #   凍結訓練需要的顯存較小，顯卡非常差的情況下，可設置Freeze_Epoch等於UnFreeze_Epoch，此時僅僅進行凍結訓練。
    #      
    #   在此提供若干參數設置建議，各位訓練者根據自己的需求進行靈活調整：
    #   （一）從整個模型的預訓練權重開始訓練： 
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-4。（凍結）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-4。（不凍結）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 150，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2。（凍結）
    #           Init_Epoch = 0，UnFreeze_Epoch = 150，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2。（不凍結）
    #       其中：UnFreeze_Epoch可以在100-300之間調整。
    #   （二）從主幹網絡的預訓練權重開始訓練：
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-4。（凍結）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-4。（不凍結）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 150，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2。（凍結）
    #           Init_Epoch = 0，UnFreeze_Epoch = 150，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2。（不凍結）
    #       其中：由於從主幹網絡的預訓練權重開始訓練，主幹的權值不一定適合目標檢測，需要更多的訓練跳出局部最優解。
    #             UnFreeze_Epoch可以在150-300之間調整，YOLOV5和YOLOX均推薦使用300。
    #             Adam相較於SGD收斂的快一些。因此UnFreeze_Epoch理論上可以小一點，但依然推薦更多的Epoch。
    #   （三）batch_size的設置：
    #       在顯卡能夠接受的範圍內，以大為好。顯存不足與數據集大小無關，提示顯存不足（OOM或者CUDA out of memory）請調小batch_size。
    #       faster rcnn的Batch BatchNormalization層已經凍結，batch_size可以為1
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   凍結階段訓練參數
    #   此時模型的主幹被凍結了，特徵提取網絡不發生改變
    #   佔用的顯存較小，僅對網絡進行微調
    #   Init_Epoch          模型當前開始的訓練世代，其值可以大於Freeze_Epoch，如設置：
    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #                       會跳過凍結階段，直接從60代開始，並調整對應的學習率。
    #                       （斷點續訓時使用）
    #   Freeze_Epoch        模型凍結訓練的
    #                       (當Freeze_Train=False時失效)
    #   Freeze_batch_size   模型凍結訓練的batch_size
    #                       (當Freeze_Train=False時失效)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 3
    Freeze_batch_size   = 4
    #------------------------------------------------------------------#
    #   解凍階段訓練參數
    #   此時模型的主幹不被凍結了，特徵提取網絡會發生改變
    #   佔用的顯存較大，網絡所有的參數都會發生改變
    #   UnFreeze_Epoch          模型總共訓練的epoch
    #                           SGD需要更長的時間收斂，因此設置較大的UnFreeze_Epoch
    #                           Adam可以使用相對較小的UnFreeze_Epoch
    #   Unfreeze_batch_size     模型在解凍後的batch_size
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 2
    Unfreeze_batch_size = 2
    #------------------------------------------------------------------#
    #   Freeze_Train    是否進行凍結訓練
    #                   默認先凍結主幹訓練後解凍訓練。
    #                   如果設置Freeze_Train=False，建議使用優化器為sgd
    #------------------------------------------------------------------#
    Freeze_Train        = True
    #------------------------------------------------------------------#
    # 其它訓練參數：學習率、優化器、學習率下降有關
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    # Init_lr 模型的最大學習率
    # 當使用Adam優化器時建議設置 Init_lr=1e-4
    # 當使用SGD優化器時建議設置 Init_lr=1e-2
    # Min_lr 模型的最小學習率，默認為最大學習率的0.01
    #------------------------------------------------------------------#
    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的優化器種類，可選的有adam、sgd
    #                   當使用Adam優化器時建議設置  Init_lr=1e-4
    #                   當使用SGD優化器時建議設置   Init_lr=1e-2
    #   momentum        優化器內部使用到的momentum參數
    #   weight_decay    權值衰減，可防止過擬合
    #                   adam會導致weight_decay錯誤，使用adam時建議設置為0。
    #------------------------------------------------------------------#
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的學習率下降方式，可選的有'step'、'cos'
    #------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    #------------------------------------------------------------------#
    #   save_period     多少個epoch保存一次權值
    #------------------------------------------------------------------#
    save_period         = 1
    #------------------------------------------------------------------#
    #   save_dir        權值與日誌文件保存的文件夾
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   eval_flag       是否在訓練時進行評估，評估對象為驗證集
    #                   安裝pycocotools庫後，評估體驗更佳。
    #   eval_period     代表多少個epoch評估一次，不建議頻繁的評估
    #                   評估需要消耗較多的時間，頻繁評估會導致訓練非常慢
    #   此處獲得的mAP會與get_map.py獲得的會有所不同，原因有二：
    #   （一）此處獲得的mAP為驗證集的mAP。
    #   （二）此處設置評估參數較為保守，目的是加快評估速度。
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 5
    #------------------------------------------------------------------#
    #   num_workers     用於設置是否使用多線程讀取數據，1代表關閉多線程
    #                   開啟後會加快數據讀取速度，但是會佔用更多內存
    #                   在IO為瓶頸的時候再開啟多線程，即GPU運算速度遠大於讀取圖片的速度。
    #------------------------------------------------------------------#
    num_workers         = 4
    #----------------------------------------------------#
    #   獲取圖片路徑和標籤
    # --------------------------------------------------#
    train_annotation_path_sup   = 'train.txt'
    train_annotation_path_unsup   = 'unsup.txt'
    val_annotation_path     = 'val.txt'
    
    #----------------------------------------------------#
    #   獲取classes和anchor
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    # --------------------------------------------------#
    #   設置要使用的GPU
    # --------------------------------------------------#
    os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in train_gpu)
    ngpus_per_node                      = len(train_gpu)
    print('Number of devices: {}'.format(ngpus_per_node))
    seed_everything(seed)
    
    model = FasterRCNN(num_classes, anchor_scales = anchors_size, backbone = backbone, pretrained = pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        
        # --------------------------------------------------#
        #   根據預訓練權重的Key和模型的Key進行加載
        # --------------------------------------------------#
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   顯示没有匹配上的Key
        #------------------------------------------------------#
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    #----------------------#
    #   紀錄Loss
    #----------------------#
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history    = LossHistory(log_dir, model, input_shape=input_shape)

    #------------------------------------------------------------------#
    #   torch 1.2不支持amp，建議使用torch 1.7.1及以上正確使用fp16
    #   因此torch1.2這裡顯示"could not be resolve"
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model_train)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    #---------------------------#
    #   讀取數據集對應的txt
    #---------------------------#
    with open(train_annotation_path_sup, encoding='utf-8') as f:
        train_lines_sup = f.readlines()
    
    if train_mode == "Teacher":
        train_lines_unsup = []
    else:
        with open(train_annotation_path_unsup, encoding='utf-8') as f:
            train_lines_unsup = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train_sup   = len(train_lines_sup)
    num_train_unsup = len(train_lines_unsup)
    num_val     = len(val_lines)
    show_config(
        classes_path = classes_path, model_path = model_path, input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train_sup, num_val = num_val
    )
    #---------------------------------------------------------#
    #   總訓練世代指的是遍歷全部數據的總次數
    #   總訓練步長指的是梯度下降的總次數 
    #   每個訓練世代包含若干訓練步長，每個訓練步長進行一次梯度下降。
    #   此處僅建議最低訓練世代，上不封頂，計算時只考慮了解凍部分
    # --------------------------------------------------#
    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    total_step  = num_train_sup // Unfreeze_batch_size * UnFreeze_Epoch
    if total_step <= wanted_step:
        if num_train_sup // Unfreeze_batch_size == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
        wanted_epoch = wanted_step // (num_train_sup // Unfreeze_batch_size) + 1
        print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"%(optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"%(num_train_sup, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
        print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))

    #-------------------------------------------------------#
    # 主backbone特徵提取網絡特徵通用，凍結訓練可以加快訓練速度
    # 也可以在訓練初期防止權值被破壞。
    # Init_Epoch為起始世代
    # Freeze_Epoch為凍結訓練的世代
    # UnFreeze_Epoch總訓練世代
    # 提示OOM或顯存不足請調小Batch_size
    #-------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #   凍結一定部分訓練
        #------------------------------------#
        if Freeze_Train:
            for param in model.extractor.parameters():
                param.requires_grad = False
        # ------------------------------------#
        #   凍結bn層
        # ------------------------------------#
        model.freeze_bn()

        #-------------------------------------------------------------------#
        #   如果不凍結訓練的話，直接設定batch_size為Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        # 判斷當前batch_size，自適應調整學習率        
        #-------------------------------------------------------------------#
        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 5e-2
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        
        
        #-------------------------------------------------#
        # 根據optimizer_type選擇優化器
        #-------------------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        #---------------------------------------#
        #   獲得學習率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        #   判斷每一個世代的長度
        #---------------------------------------#
        epoch_step      = num_train_sup // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("資料集太小，無法繼續進行訓練，請擴充資料集。")

        train_dataset_sup       = FRCNNDataset(train_lines_sup, input_shape, train = True)
        if train_mode == "Teacher":
            train_dataset_unsup = None
        else:
            train_dataset_unsup = FRCNNDataset(train_lines_unsup, input_shape, train = True)
        val_dataset             = FRCNNDataset(val_lines, input_shape, train = False)

        gen_sup                 = DataLoader(train_dataset_sup, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=frcnn_dataset_collate, 
                                            worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
        if train_mode == "Teacher":
            gen_unsup = None
        else:
            gen_unsup           = DataLoader(train_dataset_unsup, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=frcnn_dataset_collate, 
                                            worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
                                            
        gen_val                 = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last=True, collate_fn=frcnn_dataset_collate, 
                                            worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

        train_util      = FasterRCNNTrainer(model_train, optimizer)
        #----------------------#
        #   紀錄eval的map曲線
        #----------------------#
        eval_callback   = EvalCallback(model_train, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                        eval_flag=eval_flag, period=eval_period)

        #---------------------------------------#
        #   開始模型訓練
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #-------------------------------------------------#
            # 如果模型有凍結學習部分
            # 則解凍，並設定參數
            #-------------------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #------------------------------------------------- ------------------#
                # 判斷當前batch_size，自適應調整學習率
                #------------------------------------------------- ------------------#
                nbs             = 16
                lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 5e-2
                lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #-------------------------------------------------#
                # 獲得學習率下降的公式
                #-------------------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                
                for param in model.extractor.parameters():
                    param.requires_grad = True
                # ------------------------------------#
                # 凍結bn層
                # ------------------------------------#
                model.freeze_bn()

                epoch_step      = num_train_sup // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("資料集太小，無法繼續進行訓練，請擴充資料集。")
                
                '''
                gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=frcnn_dataset_collate, 
                                            worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last=True, collate_fn=frcnn_dataset_collate, 
                                            worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
                '''
                
                UnFreeze_flag = True
                
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
          
            print("Epoch:", epoch, ":", datetime.datetime.now())  # 取得現在時間
            fit_one_epoch(model, train_util, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen_sup, gen_unsup, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir)
            print("Epoch:", epoch, " finish:", datetime.datetime.now())  # 取得現在時間
          
        loss_history.writer.close()
