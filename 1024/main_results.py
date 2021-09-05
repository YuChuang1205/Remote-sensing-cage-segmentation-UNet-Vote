
#coding=gbk
'''
Created on 2020年3月7日

@author: 17720
'''
import os
import cv2
import numpy as np
import sys
import time
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a",encoding='utf-8')    
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass
root_path = os.path.abspath('.')
time1 = time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
new_name = "logs/test_result"+time1+".txt"

sys.stdout = Logger(os.path.join(root_path,new_name))





#IOU
def cal_iou(image1,image2):
    image1 = image1/255
    image2 = image2/255
    
    n = image1*image2
    u = image1+image2-n
    
    iou = np.sum(n)/np.sum(u)
    print(np.sum(n))
    print(np.sum(u))
    return iou



#准确率
def cal_acc_acc(image1,image2):
    h,w = image1.shape
    area = h*w
    
    c = (image1 ==image2)
    
    d = np.where(c==True,1,0)
    count2 = np.sum(d)
    
    print(count2)
    print(area)
    acc =count2 / area
    return acc



#精确率
def cal_acc(image1,image2):
    image1 = image1/255
    image2 = image2/255

    
    count_1 = np.sum(image2)
    
 
    d = image1*image2
    count_2 = np.sum(d)
    

    print(count_2)
    print(count_1)
    acc = count_2 / count_1
    return acc


#召回率
def cal_recall(image1,image2):
    image1 = image1/255
    image2 = image2/255
    
    count_1 = np.sum(image1)
    
    
    d = image1*image2
    count_2 = np.sum(d)
    
    
    print(count_2)
    print(count_1)
    recall = count_2 / count_1
    return recall

#平均相对误差
def relative_loss(image1,image2):
    image1 = image1/255
    image2 = image2/255
    

    count_1 = np.sum(image1)
    
    count_2 = np.sum(image2)
    
    print(count_2)
    print(count_1)
    rel_loss = abs(count_2 - count_1)/count_1
    return rel_loss
                

root_path = os.path.abspath('.')
pic_path = os.path.join(root_path,"data/mydata/test_standard")
pic_path2 = os.path.join(root_path,"data/mydata/test_results")
list1 = os.listdir(os.path.join(pic_path))
#list1.sort(key= lambda x:int(x[:-4]))
list2 = os.listdir(os.path.join(pic_path2))
#list2.sort(key= lambda x:int(x[:-4]))

iou_list = np.zeros(len(list1))
acc_list1 = np.zeros(len(list1))
acc_list2 = np.zeros(len(list1))
recall_list = np.zeros(len(list1))
rel_loss_list = np.zeros(len(list1))
for i in range(len(list1)):
    print("-------------------------------------------------")
    print("对比的图片名：%s" %(list1[i]))
    print("测试的图片名：%s" %(list2[i]))
    image1 = cv2.imread(os.path.join(pic_path,list1[i]),cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(os.path.join(pic_path2,list2[i]),cv2.IMREAD_GRAYSCALE)
    #print(image2)
    
    iou = cal_iou(image1,image2)
    acc1 = cal_acc_acc(image1,image2)
    acc2 = cal_acc(image1,image2)
    recall = cal_recall(image1,image2)
    rel_loss = relative_loss(image1,image2)
    
    
    iou_list[i] = iou
    acc_list1[i] = acc1
    acc_list2[i] = acc2
    recall_list[i] = recall
    rel_loss_list[i] = rel_loss
    
    print("测试的图片为%s，其IOU为：%f" %(list2[i],iou))
    print("测试的图片为%s，其准确率为：%f" %(list2[i],acc1))
    print("测试的图片为%s，其精确率为：%f" %(list2[i],acc2))
    print("测试的图片为%s，其召回率为：%f" %(list2[i],recall))
    print("测试的图片为%s，其相对误差为：%f" %(list2[i],rel_loss))
    print("-------------------------------------------------")

print(iou_list)
print(acc_list1)
print(acc_list2)
print(recall_list)
print(rel_loss_list)

iou_mean = np.mean(iou_list)
acc_mean1 = np.mean(acc_list1)
acc_mean2 = np.mean(acc_list2)
recall_mean = np.mean(recall_list)
rel_loss_mean = np.mean(rel_loss_list)

print("mIOU为：%f" %(iou_mean))
print("平均准确率为：%f" %(acc_mean1))
print("平均精确率为：%f" %(acc_mean2))
print("平均召回率为：%f" %(recall_mean))
print("平均相对误差为：%f" %(rel_loss_mean))
