from model import *
from data import *

import sys
import os
import time
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a",encoding='utf-8')     #文件权限为'a'，追加模式
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass
root_path = os.path.abspath('.')
time1 = time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
new_name = "logs/log"+time1+".txt"

sys.stdout = Logger(os.path.join(root_path,new_name))


epochs = 20
bitch_size = 2
img_h = 512
img_w = 512
train_num = 40000
val_num = 10000

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# data_gen_args = dict(rotation_range=60,
                    # width_shift_range=0.05,
                    # height_shift_range=0.05,
                    # shear_range=0.05,
                    # zoom_range=0.05,
                    # fill_mode='nearest')
data_gen_args = dict(
                    width_shift_range=0, 
                    fill_mode='nearest')
#myTrain = trainGenerator(bitch_size,'data/mydata/train','image','label',data_gen_args,save_to_dir = "data/mydata/train/aug",target_size = (img_h,img_w))
myTrain = trainGenerator(bitch_size,'data/mydata/train','image','label',data_gen_args,save_to_dir = None,target_size = (img_h,img_w))
#myVal = valGenerator(bitch_size,'data/mydata/val','image','label',data_gen_args,save_to_dir = "data/mydata/val/aug",target_size = (img_h,img_w))
myVal = valGenerator(bitch_size,'data/mydata/val','image','label',data_gen_args,save_to_dir = None,target_size = (img_h,img_w))


#model = unet(pretrained_weights = 'unet_mydata.hdf5' ,input_size = (img_h,img_w,1))
model = unet(input_size = (img_h,img_w,1))
model_checkpoint = ModelCheckpoint('unet_mydata.hdf5', monitor='loss',verbose=1, save_best_only=True)

#H = model.fit_generator(myTrain,steps_per_epoch=100,epochs=epochs,validation_data=myVal,validation_steps=100,callbacks=[model_checkpoint])
model.fit_generator(myTrain,steps_per_epoch=train_num/bitch_size,epochs=epochs,validation_data=myVal,validation_steps=val_num/bitch_size,callbacks=[model_checkpoint])

