import torch 
import torch.nn as nn 
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, datasets
import random
import numpy as np
import argparse
import datetime
import dataset
import model_zoo
#-------------------------------------
# argument parser
parser = argparse.ArgumentParser(description='Training MLP on MNIST/synthetic dataset')
parser.add_argument('--batch_size', type=int, default=100, help='Number of samples per mini-batch')
parser.add_argument('--epochs', type=int, default=1, help='Number of epoch to train')
parser.add_argument('--depth', type=int, default=4, help='the depth (number of FC layers) of the MLP')
parser.add_argument('--width', type=int, default=8, help='the width (number of neurons per layers) of the MLP')
parser.add_argument('--num_seg', type=int, default=2, help='the number of segmentation for the synthetic dataset')
parser.add_argument('--tc', type=int, default=20, help='the number of tc')
parser.add_argument('--dataset', type=str, default='MNIST', help='the type of dataset')
parser.add_argument('--sigma_log_file', type=str, default='logs/mlp_sigma.logs', help='the name of file used to record the LDI record of MLPs')
parser.add_argument('--iter_times', type=int, default=5, help='the number of iteration times to calculate the LDI of the same architecture')

args = parser.parse_args()





### for isometry at initialization

train_loader,test_loader =dataset.mnist_dataloaders()


for iter_times in range(args.iter_times):
    model = model_zoo.Dense_MLP(args.width, args.depth, args.tc, input_dims=784, num_classses=10)    # model.init_network(func)
    sig_mean=0
    sig_std=0
    for i, (images, labels) in enumerate(test_loader):
        images=images
        sig_mean_tmp,sig_std_tmp=model.isometry(images.view([args.batch_size,784]))
        sig_mean=sig_mean+sig_mean_tmp
        sig_std=sig_std+sig_std_tmp
    sig_mean=sig_mean/(i+1)
    sig_std=sig_std/(i+1)
    with open(args.sigma_log_file,'a+') as train_logs:
        print(model.nn_mass, sig_mean.item(),sig_std.item(), 
            model.params, model.flops, args.width, args.depth, 
            args.tc, args.num_seg,file=train_logs) 


