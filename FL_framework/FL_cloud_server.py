import numpy as np
import tensorflow as tf
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import FL_local_client
import train

class FL_Server(object):
    def __init__(self, clients_list):
        self.clients_list = clients_list  #client name list 
        self.n_clients = len(clients_list)
        self.FL_round = 0
        self.model_dict = {}


    def upload_model(self, name, num_samples, parameter):
        self.model_dict[name] = (num_samples, parameter)

    def aggreation(self):

        total_samples = 0
        for num, _ in self.model_dict.values():
            total_samples += num 

        new_model_param={}
        for name in self.clients_list:
            for key,val in self.model_dict[name][1].items():
                
                weight_i = self.model_dict[name][0]*1.0/total_samples
                if key not in new_model_param.keys():
                    new_model_param[key] = val * weight_i
                else:
                    new_model_param[key] += val * weight_i

        self.new_model_param = new_model_param
        self.FL_round += 1
     
    def get_new_params(self):
        return self.new_model_param
