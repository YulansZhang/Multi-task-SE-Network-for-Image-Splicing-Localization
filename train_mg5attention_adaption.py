# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:05:07 2019

@author: HuangYu
"""

import os
import time
import torch
import Movenet7f4attention_adaption
import numpy as np
import torch.optim as optim
import extra_data_coco_dataset_move_edge_edge as dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"


def calIOU(img1, img2):
    Area1 = np.sum(img1)
    Area2 = np.sum(img2)      
    ComArea = np.sum(img1&img2)
    iou = ComArea/(Area1+Area2-ComArea+1e-8)
    return iou


def train():
    
    device_ids = [0,1]
    lr = 0.001
    epochs = 300
    batch_size = 8
    display_interval = 100
    steps = [150000,40000000]
    save_model_path = './snapshot7f4attention_adaption_all_02_08'
    if not os.path.isdir(save_model_path): os.makedirs(save_model_path)
    fid = open('./log/7f4m_snapshot7f4attention_all%s.txt'%(time.time()),'wt')

    #torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    # dataset_train = dataset.dataset('/data/zyl/cpmove/cocodataset/train/384/images', '/data/zyl/cpmove/cocodataset/train/train_ann.json','/data/zyl/cpmove/cocodataset/train/384_cpmove_mask')
    # dataset_train = dataset.dataset('/data/zyl/cpmove/cocodataset/train/384/images', '/data/zyl/cpmove/cocodataset/train/train_ann.json', '/data/zyl/cpmove/cocodataset/train/384_cpmove_mask',
    #                '/data/zyldata/RRU-Net/data/Aualldata_384', '/data/zyldata/RRU-Net/data/Aualldata_384/ data_384.txt',ext_root_folder2=None)
    dataset_train = dataset.dataset('/data/zyldata/RRU-Net/data/spliced_data_train',
                                    '/data/zyldata/RRU-Net/data/spliced_data_train/ trainall.txt')
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=False)
    

    model = Movenet7f4attention_adaption.Movenet([384,384]).to(device)
    print(model)
    #model = torch.nn.DataParallel(model, device_ids=device_ids)
    st = 0
    pretrain = torch.load('./all-50w.pkl').to(device)
    model.load_state_dict(pretrain.state_dict(), strict=False)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=0.1, last_epoch=-1)
                          
    model.train()
    idx = 0
    t0 = time.time()
    for epoch in range(0, epochs + 1):
        
        for batch_idx, (data,data_edge0, label0, label_edge) in enumerate(train_loader):
            
            data, data_edge, label, label_edge = data.to(device), data_edge0.to(device), label0.to(device).long(), label_edge.to(device).long()
            optimizer.zero_grad()
            end1,end2,end3, predict1_,predict2_,predict3_ = model(data) 
            #-------------------------- loss --------------------------#         
            loss1 = F.cross_entropy(end1, label)
            loss2 = F.cross_entropy(end2, label_edge)
            #loss3 = F.cross_entropy(end3, data_edge)
            loss3 = F.mse_loss(end3, data_edge)
            w1 = 0.2
            w2 = 0.8
            loss = loss1+w1*loss2+w2*loss3
            '''
            Temp = end.permute(0,2,3,1)
            Temp = Temp.contiguous().view(-1, 2)
            # 自定义softmax loss, label: onehot   
            shape = data.size()
            onehot_label = torch.zeros(shape[0]*shape[2]*shape[3],2).cuda().scatter_(1,label.long().view(-1,1),1)  
            logits = -F.log_softmax(Temp, dim=1)        
            loss1 = torch.sum(logits.mul(onehot_label), 1)           
            loss = torch.mean(loss1)      
            '''
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if batch_idx % display_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                label2 = label0.numpy()
                predict1 = predict1_.cpu().detach().numpy()                
                predict1 = predict1.astype(np.uint8)
                
                predict2 = predict2_.cpu().detach().numpy()                
                predict2 = predict2.astype(np.uint8)
                
                predict3 = predict3_.cpu().detach().numpy()                
                predict3 = predict3.astype(np.uint8)
                
                data_edge2 = data_edge0.numpy()
                
                ious1 = []
                ious2 = []
                ious3 = [loss3.item()]
                for i in range(batch_size):
                    ious1.append(calIOU(label2[i], predict1[i]))
                    ious2.append(calIOU(label2[i], predict2[i]))
                    #ious3.append(calIOU(data_edge2[i], predict3[i]))
                print('Epoch:[%d/%d %d]\tlr: %f\tLoss: %0.6f\tIOU: %0.4f,%0.4f,%0.4f\ttime: %0.1fs'%(
                    epoch, epochs, idx+st, lr, loss.item(), np.mean(ious1),np.mean(ious2),np.mean(ious3), time.time()-t0))
                fid.write('Epoch:[%d/%d %d]\tlr: %f\tLoss: %0.6f\tIOU: %0.4f,%0.4f,%0.4f\ttime: %0.1fs\n'%(
                    epoch, epochs, idx+st, lr, loss.item(), np.mean(ious1),np.mean(ious2),np.mean(ious3), time.time()-t0))
                fid.flush()
                t0 = time.time()
            idx += 1 
            if idx%1000==0:
                save_path = '%s/model7f4attentionadaption-%02d.pkl'%(save_model_path,idx+st)
                torch.save(model.module,save_path)
                print(save_path)
    fid.close()

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    train()

