# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:38:58 2020


"""



import os
import cv2
import time
from PIL import Image
import torch
import Movenet7f4attention_adaption
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def calIOU(img1, img2):
    Area1 = np.sum(img1)
    Area2 = np.sum(img2)      
    ComArea = np.sum(img1&img2)
    iou = ComArea/(Area1+Area2-ComArea+1e-8)
    return iou


def calF1score(img1, img2):
    img1zero = (1 - img1)
    img2zero = (1 - img2)
    sumTP = np.sum(img1 & img2)
    sumFP = np.sum(img1zero & img2)
    TPR = sumTP/np.sum(img1)
    FPR = sumFP/np.sum(img1zero)
    F1_score = 2*sumTP/(sumTP+sumFP+np.sum(img1))
    return F1_score

#
# def cv_imread(filePath, color=cv2.IMREAD_COLOR):
#     cv_img = cv2.imdecode(np.fromfile(filePath,dtype=np.uint8), color)
#     return cv_img



# 测试1张图片
def test_one():

    model_path = r'./model7f4attentionadaption-291000.pkl'
    # img = cv_imread('./tam/yu.jpg')
    img_fn = './tam/yu.jpg'
    # ori_fn = './masks/yu.jpg'
    img = Image.open(img_fn)
    reim = img.resize((384,384))
    imgs = np.array(reim)
    imgs = imgs.astype(np.float32)/255.
    imgs = imgs.transpose([2,0,1])
    data=np.expand_dims(imgs,axis=0)
    data = torch.from_numpy(data)
    # predict_ = self.model(data)
    # predict = predict_.cpu().detach().numpy().astype(np.uint8)

    # ori = Image.open(ori_fn)
    # reim = img.resize((384,384))
    # reori = ori.resize((384,384))
    # img = np.array(reim)
    # ori = np.array(reori)
    # label = img-ori
    # label_binary = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
    # ret, binary = cv2.threshold(label_binary, 0, 255, 0 | cv2.THRESH_BINARY_INV)

    # imgs = img.astype(np.float32)/255.

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 图片尺寸为384*384
    model = Movenet7f4attention_adaption.Movenet([384,384]).to(device)
    pretrain = torch.load(model_path).to(device)
    model.load_state_dict(pretrain.state_dict(), strict=True)    
    model.eval()
        
    t0 = time.time() 
    # label = binary==255
    # label = label.astype(np.uint8)
    # mask = label*255
    # cv2.imwrite('mask.png',mask)
    # imgs = np.array([img])
    # imgs = imgs.astype(np.float32)/255.
    # imgs = imgs.transpose([2,0,1])
    # data=np.expand_dims(imgs,axis=0)
    # data = torch.from_numpy(data)
    
    with torch.no_grad():
        predict_ = model(data.to(device))        
    predict = predict_.cpu().detach().numpy().astype(np.uint8)
    # iou1 = calIOU(label, predict[0]) # none

    # print('IOU: %0.4f  time: %0.5fs'%(iou1, time.time()-t0))
    cv2.imwrite('yu.jpg',predict[0]*255)



if __name__ == '__main__':
    torch.cuda.empty_cache()
    test_one()
