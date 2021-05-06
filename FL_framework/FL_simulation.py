import numpy as np
import tensorflow as tf
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import FL_local_client
import FL_cloud_server

def simulation():

    clients_name_list = ['A','B','C', 'D', 'E']
    # clients_name_list = ['A']
    num_clients = len(clients_name_list)
    clients_list = []

    #create cloud server instance
    server = FL_cloud_server.FL_Server(clients_name_list)

    #create clocal clients instances
    for i in range(num_clients):
        name = clients_name_list[i]
        clients_list.append(FL_local_client.FL_Client(name, name+'_train_hdf5_file_list.txt', name+'_val_hdf5_file_list.txt'))
    #simulate 10 rounds 
    for round in range(4):
        #locally train one round
        for i in range(num_clients):
            clients_list[i].train_local()
            # clients_list[i].pred()
        #upload client models to server
        for i in range(num_clients):
            name,num_samples,param= clients_list[i].get_parameter()
            server.upload_model(name,num_samples,param)
        server.aggreation()   
        #update local models
        param = server.get_new_params()
        for i in range(num_clients):
            clients_list[i].update_parameter(param)
    
    for i in range(num_clients):
            clients_list[i].train_local()
            # clients_list[i].pred()

if __name__=='__main__':
    simulation()
