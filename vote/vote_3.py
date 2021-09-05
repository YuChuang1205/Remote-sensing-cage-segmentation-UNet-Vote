#coding=gbk
'''
Created on 2021年6月28日

@author: 余创
'''
import numpy as np
import cv2
import os

root_path = os.path.abspath('.')
pic_path1 = os.path.join(root_path,'data/in/1')
pic_path2 = os.path.join(root_path,'data/in/2')
pic_path3 = os.path.join(root_path,'data/in/3')

pic_out = os.path.join(root_path,'data/out')

pic_list = os.listdir(pic_path1)
for i in range(len(pic_list)):
    print("--------------------------------------------")
    print("处理的图像为：%s" %(pic_list[i]))
    img1 = cv2.imread(os.path.join(pic_path1,pic_list[i]),cv2.IMREAD_GRAYSCALE)/255
    img2 = cv2.imread(os.path.join(pic_path2,pic_list[i]),cv2.IMREAD_GRAYSCALE)/255
    img3 = cv2.imread(os.path.join(pic_path3,pic_list[i]),cv2.IMREAD_GRAYSCALE)/255
    
    height,width = img1.shape
    img4 = np.zeros((height,width))
    img4 = img1+img2+img3
    print(np.max(img4),np.min(img4))
    img4 = np.expand_dims(img4, axis=0)
    for a,item in enumerate(img4):
        img=item[:,:]
        print(np.max(img),np.min(img))
#         print([img>=1.5])
#         print(img[img>=1.5])
#         print(len(img[img>=1.5]))
        img[img>=1.5]=255#此时1是浮点数，下面的0也是
        img[img<1.5]=0
        print(np.max(img),np.min(img))
    img4 = img.reshape((height,width))
    
    cv2.imwrite(os.path.join(pic_out,pic_list[i]),img4)
    print("图像 %s 已保存！！！" %(pic_list[i]))
    print("--------------------------------------------")
