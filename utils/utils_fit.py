import os, cv2
import random
import torch
import numpy as np
from tqdm import tqdm

from utils.utils import get_lr, get_classes

from utils.utils_bbox import DecodeBox
import copy

def fit_one_epoch(model, train_util, loss_history, eval_callback, optimizer, epoch, gen_sup, gen_unsup, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, ema_teacher=None, mini_batch_size=12):
    batch_size_cnt = 0
    
    total_loss = 0
    rpn_loc_loss = 0
    rpn_cls_loss = 0
    roi_loc_loss = 0
    roi_cls_loss = 0
    
    val_loss = 0
    
    with tqdm(total=len(gen_sup),desc=f'Supervise Train Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_sup):
            images, boxes, labels = batch[0], batch[1], batch[2]
            
            step = False
            batch_size_cnt += len(labels)
            if batch_size_cnt >= mini_batch_size:
                batch_size_cnt = 0
                step = True
            
            if cuda:
                images = images.cuda()

            rpn_loc, rpn_cls, roi_loc, roi_cls, total = train_util.train_step(images, boxes, labels, 1, fp16, scaler, True, step=step)
            total_loss      += total.item()
            rpn_loc_loss    += rpn_loc.item()
            rpn_cls_loss    += rpn_cls.item()
            roi_loc_loss    += roi_loc.item()
            roi_cls_loss    += roi_cls.item()
            
            pbar.set_postfix(**{'total_loss'    : total_loss / (iteration + 1), 
                                'rpn_loc'       : rpn_loc_loss / (iteration + 1),  
                                'rpn_cls'       : rpn_cls_loss / (iteration + 1), 
                                'roi_loc'       : roi_loc_loss / (iteration + 1), 
                                'roi_cls'       : roi_cls_loss / (iteration + 1), 
                                'lr'            : get_lr(optimizer)})
            pbar.update(1)
    
    if gen_unsup != None and epoch > -1:
        batch_size_cnt = 0
        
        total_loss_unsup = 0
        rpn_loc_loss_unsup = 0
        rpn_cls_loss_unsup = 0
        roi_loc_loss_unsup = 0
        roi_cls_loss_unsup = 0
        
        with tqdm(total=len(gen_unsup),desc=f'Unsupervise Train Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
            for iteration, batch in enumerate(gen_unsup):
                
                images, boxes, labels = batch[0], batch[1], batch[2]
            
                step = False
                batch_size_cnt += len(labels)
                if batch_size_cnt >= mini_batch_size:
                    batch_size_cnt = 0
                    step = True
            
                
                if cuda:
                    images = images.cuda()
                rpn_loc, rpn_cls, roi_loc, roi_cls, total = train_util.train_step(images, boxes, labels, 1, fp16, scaler, False, step=step)
                total_loss_unsup += total.item()
                rpn_loc_loss_unsup     += rpn_loc.item()
                rpn_cls_loss_unsup     += rpn_cls.item()
                roi_loc_loss_unsup     += roi_loc.item()
                roi_cls_loss_unsup     += roi_cls.item()
                
                pbar.set_postfix(**{'total_loss'    : total_loss_unsup / (iteration + 1),
                                    'rpn_loc'       : rpn_loc_loss_unsup / (iteration + 1),
                                    'rpn_cls'       : rpn_cls_loss_unsup / (iteration + 1),
                                    'roi_loc'       : roi_loc_loss_unsup / (iteration + 1),
                                    'roi_cls'       : roi_cls_loss_unsup / (iteration + 1),
                                    'lr'            : get_lr(optimizer)})
                      
                pbar.update(1)
    
    if ema_teacher != None and ema_teacher.decay < 1.0:
        ema_teacher.update(model)
    
    with tqdm(total=len(gen_val), desc=f'Validation Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            images, boxes, labels = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    images = images.cuda()

                train_util.optimizer.zero_grad()
                _, _, _, _, val_total = train_util.forward(images, boxes, labels, 1)
                val_loss += val_total.item()
                
                pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1)})
                pbar.update(1)
    
    loss_history.append_loss(epoch + 1, total_loss / len(gen_sup), val_loss / len(gen_val))
    eval_callback.on_epoch_end(epoch + 1)
    
    #-----------------------------------------------#
    # 保存權值
    #-----------------------------------------------#
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / len(gen_sup), val_loss / len(gen_val))))

    if len(loss_history.val_loss) <= 1 or (val_loss / len(gen_val)) <= min(loss_history.val_loss):
        print('Save best model to best_epoch_weights.pth')
        torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
    torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
