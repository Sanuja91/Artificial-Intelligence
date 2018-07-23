import helper
import train
import torch
import predict
import os
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


flag = False
learn_rate = None
hidden_layers = []
model = None
criterion = None
optimizer = None
actions = ['predict', 'build', 'train', 'exit']

device = helper.get_device()

while (flag != True):
    action = input("\nWhat would you like to do? Actions include: \n{}\n".format(actions))
    if(action == actions[0]):
        if(model != None):
            predict.predict(model)
        else:
            predict.predict(None)

    elif(action == actions[1]):
        model = helper.build_new_network()

    elif(action == actions[2]):
        if(model != None):
            model = train.train_network(model)
        else:
            model = train.train_network(None)

    elif(action == actions[3]):
        flag = False

    else:
        print('Invalid selection. Please select from: \n{}\n'.format(actions))
    












