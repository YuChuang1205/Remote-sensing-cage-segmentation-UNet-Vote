#coding=gbk
'''
Created on 2020年3月28日

@author: 17720
'''
from model import *
from data import *
import os
import skimage.io
import numpy as np 
import cv2

root_path = os.path.abspath('.')
pic_path = os.path.join(root_path,"data/mydata/test")
pic_out_path = os.path.join(root_path,"data/mydata/test_results2")
model_path = os.path.join(root_path,"unet_mydata.hdf5")
model = load_model(model_path)
stride = 512
image_size = 1024

piclist = os.listdir(pic_path)
piclist.sort(key= lambda x:int(x[:-4])) 
for n in range(len(piclist)):
#     image = skimage.io.imread(os.path.join(pic_path,piclist[i]))
#     print(image.shape)
    new_name = piclist[n].split('.')[0]+'.png'
    image = cv2.imread(os.path.join(pic_path,piclist[n]),cv2.IMREAD_GRAYSCALE)
    print("--------------------------------------------------------")
    print(np.max(image),np.min(image))
    image = image/255
    print(np.max(image),np.min(image))
    #print(image.shape)
    h,w = image.shape
    image_out = np.zeros((h,w))
    print("长为%d,宽为%d" %(h,w))
    padding_h = (h//stride +3) *stride
    padding_w = (w//stride +3) *stride
    print(padding_h,padding_w)
    padding_img = np.zeros((padding_h,padding_w))
    padding_img[stride:h+stride,stride:w+stride] = image[:,:]
    mask_whole = np.zeros((padding_h,padding_w))
    #skimage.io.imsave(os.path.join(pic_out_path,'4.png'), padding_img)
    for i in range(padding_h//stride - 1):
            for j in range(padding_w//stride - 1):
                crop = padding_img[i*stride:i*stride+image_size,j*stride:j*stride+image_size]
                ch,cw = crop.shape
                if ch != image_size or cw != image_size:
                    print("invalid size!")
                    continue
                crop = np.expand_dims(crop, axis=2)
                crop = np.expand_dims(crop, axis=0)
                pred = model.predict(crop,verbose=1)
                print(pred.shape)
                pred = pred.reshape((image_size,image_size))
#                 print(pred)
#                 for a,item in enumerate(pred):
#                     img=item[:,:,0]
#                     print(np.max(img),np.min(img))
#                     img[img>=0.5]=1#此时1是浮点数，下面的0也是
#                     img[img<0.5]=0
#                     print(np.max(img),np.min(img))
#                 pred = img.reshape((image_size,image_size))
                print(i,j)
                print(np.max(pred),np.min(pred))
                pred = pred/4
                print(np.max(pred),np.min(pred))
                #print(pred)
                mask_whole[i*stride:i*stride+image_size,j*stride:j*stride+image_size] = mask_whole[i*stride:i*stride+image_size,j*stride:j*stride+image_size]+pred[:,:]
    #skimage.io.imsave(os.path.join(pic_out_path,'4k.png'), mask_whole)
    print(np.max(mask_whole),np.min(mask_whole))
    image_out[:,:] = mask_whole[stride:h+stride,stride:w+stride]
    print(np.max(image_out),np.min(image_out))
    image_out = np.expand_dims(image_out, axis=0)
    for a,item in enumerate(image_out):
        img=item[:,:]
        print(np.max(img),np.min(img))
        img[img>=0.5]=255#此时1是浮点数，下面的0也是
        img[img<0.5]=0
        print(np.max(img),np.min(img))
    image_out = img.reshape((h,w))
    cv2.imwrite(os.path.join(pic_out_path,new_name), image_out)
    print("--------------------------------------------------------")






