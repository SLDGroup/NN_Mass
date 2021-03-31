import os 
import math
import random
import torch
import torchvision
import numpy as np 
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as dsets

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def mnist_dataloaders(train_batch_size=64, test_batch_size=100, num_workers=2, cutout_length=4, data_dir = './data'):
    train_dataset = dsets.MNIST(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='./data',
                            train=False,
                            transform=transforms.ToTensor())

    # Dataset Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=train_batch_size,
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=test_batch_size,
                                            shuffle=False)

    return train_loader, test_loader


def cifar10_dataloaders(train_batch_size=64, test_batch_size=100, num_workers=2, cutout_length=4, data_dir = './data'):
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # Cutout(cutout_length),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False,num_workers=2, pin_memory=True)
    return train_loader, test_loader

def cifar100_dataloaders(train_batch_size=64, test_batch_size=100, num_workers=2, cutout_length=4, data_dir = 'datasets/cifar100'):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        Cutout(cutout_length),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
    ])
    trainset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    testset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False,num_workers=2, pin_memory=True)
    return train_loader, test_loader

def circle_dataset_generation(range_min=0,range_max=100,train_num=12000,test_num=1200,max_category=120):
    train_data_temp=[]
    train_data_temp_1=[]
    train_label=np.zeros(train_num,dtype=int)
    test_data_temp=[]
    test_label=np.zeros(test_num,dtype=int)
    steps=(range_max-range_min)/(max_category)
    
    for i in range(max_category):
        counter = 1 
        mu=(range_min+i*steps+range_min+(i+1)*steps)/2
        sig=steps/2
        while(counter<=(train_num/max_category)): 
            len_temp=(np.random.normal(mu, sig, 1))
            #temp_rand=random.uniform(range_min+i*steps,range_min+(i+1)*steps)
            if len_temp < (i+1)*steps and len_temp>(i)*steps:
                theta=random.uniform(0,2*3.14159265357)
                temp_rand_1=len_temp*math.cos(theta)
                temp_rand_2=len_temp*math.sin(theta)
                temp_rand=[temp_rand_1,temp_rand_2]
                if(temp_rand not in train_data_temp):
                    train_data_temp.append(temp_rand); 
                    counter+=1
                    #print(counter)
                    

    for i in range(max_category):
        counter = 1 
        mu=(range_min+i*steps+range_min+(i+1)*steps)/2
        sig=steps/2
        while(counter<=(test_num/max_category)): 
            len_temp=(np.random.normal(mu, sig, 1))
            if len_temp < (i+1)*steps and len_temp>(i)*steps:
                theta=random.uniform(0,2*3.14159265357)
                temp_rand_1=len_temp*math.cos(theta)
                temp_rand_2=len_temp*math.sin(theta)
                temp_rand=[temp_rand_1,temp_rand_2]
                if(temp_rand not in test_data_temp):
                    if(temp_rand not in train_data_temp):
                        test_data_temp.append(temp_rand); 
                        counter+=1
                        #print(counter)
    train_data=np.squeeze(train_data_temp)
    test_data=np.squeeze(test_data_temp)
    np.savetxt("data/train_circle.csv", np.array(train_data), delimiter=',')
    np.savetxt("data/test_circle.csv", np.array(test_data), delimiter=',')

def linear_dataset_generation(range_min=0,range_max=100,train_num=12000,test_num=1200,max_category=120):
    train_data_temp=[]
    train_label=np.zeros(train_num,dtype=int)
    test_data_temp=[]
    test_label=np.zeros(test_num,dtype=int)
    steps=(range_max-range_min)/(max_category)
    
    for i in range(max_category):
        counter = 1 
        while(counter<=(train_num/max_category)): 
            temp_rand=random.uniform(range_min+i*steps,range_min+(i+1)*steps)
            if(temp_rand not in train_data_temp):
                train_data_temp.append(temp_rand); 
                counter+=1
    for i in range(max_category):
        counter = 1 
        while(counter<=(test_num/max_category)): 
            temp_rand=random.uniform(range_min+i*steps,range_min+(i+1)*steps)
            if(temp_rand not in test_data_temp):
                if(temp_rand not in train_data_temp):
                    test_data_temp.append(temp_rand); 
                    counter+=1
    train_data=np.zeros((train_num,2))
    train_data[:,0]=np.array(train_data_temp)
    train_data[:,1]=np.array(train_data_temp)

    test_data=np.zeros((test_num,2))
    test_data[:,0]=np.array(test_data_temp)
    test_data[:,1]=np.array(test_data_temp)


    np.savetxt("data/train_linear.csv", np.array(train_data), delimiter=',')
    np.savetxt("data/test_linear.csv", np.array(test_data), delimiter=',')

