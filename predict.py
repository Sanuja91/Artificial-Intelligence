import helper
import torch
import os
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
import constants as Constant
from torchvision import datasets, transforms, models

# Predicts the class of an image using a trained model
def predict(model):
    if(model == None):
        model = helper.load_checkpoint()

    device = helper.get_device()
   
    image_tensor, image = helper.get_image_tensor('\nPlease enter the path of the image you want to analyse\n')
    image_tensor = image_tensor.to(device)
    topk = helper.get_int('\nPlease enter how many to the top predictions you want to see (topk)\n')

    model = model.to(device)

    print('\nPredicting\n')

    with torch.no_grad():
        output = model.forward(image_tensor)
    
    ps = torch.exp(output)
    
    topK_ps = torch.topk(ps, topk)
    
    probs = topK_ps[0].cpu().numpy().squeeze()
    sorted_ps_label_keys = topK_ps[1].cpu().numpy().squeeze()
    classes = []

    print('Sorted label keys {}'.format(sorted_ps_label_keys))

    try:
        get_label = lambda x:idx_to_class[str(x)]
        for i in sorted_ps_label_keys[0:topk]:
            classes.append(get_label(i))

    except NameError:
        print('\nCaught Key Error idx_to_class does not exist\nUsing normal keys\n')  
        for i in sorted_ps_label_keys[0:topk]:
            classes.append(i)

    print('\nFinished predicting\n')

    helper.view_classify(image,probs,classes)
    return
    

# Handles for initiating from predict.py
if __name__ == "__main__":
    predict(None)
