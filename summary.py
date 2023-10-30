#--------------------------------------------#
# 此部分程式碼用於查看網路結構
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.frcnn import FasterRCNN

if __name__ == "__main__":
    input_shape     = [600, 600]
    num_classes     = 21
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = FasterRCNN(num_classes, backbone = 'vgg').to(device)
    summary(model, (3, input_shape[0], input_shape[1]))
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    # flops * 2是因為profile沒有將波形當作兩個操作
    # 有些論文將波形算乘法、加法兩種運算。此時乘以2
    # 有些論文只考慮乘法的運算次數，忽略加法。此時不乘2
    # 本代碼選擇乘2，參考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
