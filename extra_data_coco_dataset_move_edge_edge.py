# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 18:19:05 2020

@author: huangyu45
"""




from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import json



class dataset(Dataset):
    def __init__(self, root_folder, txt_path, root_folder2=None):

        self.img_list = []
        self.label_list = []
        self.root_folder = root_folder + '/'
        if root_folder2 is None:
            self.root_folder2 = self.root_folder
        else:
            self.root_folder2 = root_folder2 + '/'
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                s = line.split()
                self.img_list.append(s[0])
                self.label_list.append(s[1])
        if len(self.img_list) == 0:
            print('warning: find none images')
        else:
            print('datasize', len(self.label_list))
        self.kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


    
    # read images
    def cv_imread(self, filePath, color=cv2.IMREAD_COLOR):    
        cv_img = cv2.imdecode(np.fromfile(filePath,dtype=np.uint8), color)    
        return cv_img
    
    def __getitem__(self, index):

        # image process
        img = self.cv_imread(self.root_folder + self.img_list[index])
        label = self.cv_imread(self.root_folder2 + self.label_list[index], cv2.IMREAD_GRAYSCALE)
        label = label//255

        #img_edge = cv2.Canny(img,10,100)
        #img_edge = img_edge//255
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_edge = self.toedge(gray)
        img_edge = img_edge.astype(np.float32)/255
        
        img = img.astype(np.float32)/255.
        img = img.transpose([2,0,1])
        label_edge = self.toedge(label)

        # label = label == 255
        # label = label.astype(np.uint8)
        return img, img_edge, label, label_edge
        

    def __len__(self):
        return len(self.label_list)
    


    
    def toedge(self,gray):
        
        edge = cv2.erode(gray,self.kernel)
        edge = gray-edge
        return edge
    
    
if __name__=='__main__':
    
    epochs = 1
    data = dataset('/data/zyldata/RRU-Net/data/show', '/data/zyldata/RRU-Net/data/show/ data_384.txt')
    
    loader = DataLoader(data, batch_size=1, shuffle=False, drop_last=True, num_workers=0)

    for ep in range(epochs):
        for batch_idx, (data, data_edge, label, label_edge) in enumerate(loader):
            img = data.detach().numpy()
            data_edge = data_edge.detach().numpy()
            mask = label.detach().numpy()
            label_edge = label_edge.detach().numpy()



            img = img[0]*255
            img = img.transpose([1,2,0])

            data_edge = data_edge[0]* 255
            data_edge = data_edge.astype(np.int)
            data_edge[data_edge > 0] += 20
            data_edge[data_edge > 255] = 255
            data_edge = data_edge.astype(np.uint8)



            mask = mask[0]* 255
            mask = mask.astype(np.int)
            mask[mask > 0] += 20
            mask[mask > 255] = 255
            mask = mask.astype(np.uint8)



            label_edge = label_edge[0]* 255
            label_edge = label_edge.astype(np.int)
            label_edge[label_edge > 0] += 20
            label_edge[label_edge > 255] = 255
            label_edge = label_edge.astype(np.uint8)


            cv2.imwrite('/data/zyldata/RRU-Net/data/show/data.png', img.astype(np.uint8))
            cv2.imwrite('/data/zyldata/RRU-Net/data/show/data_edge.png', data_edge.astype(np.uint8))
            cv2.imwrite('/data/zyldata/RRU-Net/data/show/label.png', mask.astype(np.uint8))
            cv2.imwrite('/data/zyldata/RRU-Net/data/show/label_edge.png', label_edge.astype(np.uint8))
            break
    # cv2.destroyAllWindows()
            
        
        
        
        
        