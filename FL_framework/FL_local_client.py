#from part_seg.train import train_pointnet
import numpy as np
import tensorflow as tf
import os
import sys
#import copy
from tensorflow.python import pywrap_tensorflow

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(BASE_DIR+"/..")
from part_seg.train import train_pointnet

class FL_Client(object):

    def __init__(self, name, train_file_list, val_file_list):
        self.name = name
        self.train_file_list = train_file_list
        self.val_file_list = val_file_list
        self.num_samples = self.help_read_num_samples()
        self.train_round = 0
        self.learning_rate = 0.002
    
    def train_local(self, gpu=0, batch=4, epoch=50, point_num=10000, output_dir='train_results', wd=0):
        self.train_round += epoch
        #copied = copy.deepcopy(self.learning_rate)
        self.learning_rate = train_pointnet(gpu, batch, epoch, point_num, output_dir, wd,\
                                            self.train_file_list, self.val_file_list, self.name,\
                                            self.train_round, 0, self.learning_rate)
              
    def pred(self, gpu=0, batch=4, epoch=1, point_num=10000, output_dir='train_results', wd=0):
        _ = train_pointnet(gpu, batch, epoch, point_num, output_dir, wd,\
                            self.train_file_list, self.val_file_list, self.name,\
                            self.train_round+epoch, 1, self.learning_rate)
    
    def get_parameter(self, epoch=4):
        # model_dir = "/Users/wmz/Desktop/test"
        # ckpt = tf.train.get_checkpoint_state(model_dir)
        ckpt_path = BASE_DIR+'/train_results/'+self.name+'_trained_models/epoch_'+str(self.train_round)+'/'+self.name+'_epoch_'+str(self.train_round)+'.ckpt'      
        reader = tf.train.NewCheckpointReader(ckpt_path)
        param_dict = reader.get_variable_to_shape_map()
        # t=reader.get_tensor('conv5/bn/conv5/bn/moments/Squeeze/ExponentialMovingAverage')
        parameters={}
        for key,val in param_dict.items():
            parameters[key]=reader.get_tensor(key) 
        return self.name, self.num_samples, parameters

    def update_parameter(self, param):
        tf.reset_default_graph()
        with tf.Session() as sess:
            new_var_list=[]
            for var_name, val in param.items():
                new_var_list.append(tf.Variable(val,name=var_name,dtype=tf.float32))
            saver = tf.train.Saver(var_list=new_var_list)
            sess.run(tf.global_variables_initializer())
            checkpoint_path = BASE_DIR+'/train_results/'+self.name+'_trained_models/epoch_'+str(self.train_round)+'/'+self.name+'_epoch_'+str(self.train_round)+'.ckpt'
            saver.save(sess, checkpoint_path)

    def help_read_num_samples(self):

        return len([line.rstrip() for line in open(BASE_DIR+'/../part_seg/hdf5_data/'+self.train_file_list)])
