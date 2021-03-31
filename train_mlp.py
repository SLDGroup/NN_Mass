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
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--depth', type=int, default=4, help='the depth (number of FC layers) of the MLP')
parser.add_argument('--width', type=int, default=8, help='the width (number of neurons per layers) of the MLP')
parser.add_argument('--num_seg', type=int, default=2, help='the number of segmentation for the synthetic dataset')
parser.add_argument('--tc', type=int, default=20, help='the number of tc')
parser.add_argument('--dataset', type=str, default='MNIST', help='the type of dataset')
parser.add_argument('--make_dataset', action='store_true', help='generate/regenerate the synthetic dataset or not.')
parser.add_argument('--train_log_file', type=str, default='logs/mlp_train.logs', help='the name of file used to record the training/test record of MLPs')
parser.add_argument('--res_log_file', type=str, default='logs/mlp_res.logs', help='the name of file used to record the training/test record of MLPs')

parser.add_argument('--iter_times', type=int, default=5, help='the number of iteration times to train the same architecture')
args = parser.parse_args()


def make_dataset(args,):
    if args.make_dataset:
        if 'lin' in args.dataset:
            dataset.linear_dataset_generation()
            train_file='data/train_linear.csv'
            test_file='data/test_linear.csv'
        if 'cir' in args.dataset:
            dataset.circle_dataset_generation()
            train_file='data/train_circle.csv'
            test_file='data/test_circle.csv'

    if 'lin' in args.dataset:
        train_file='data/train_linear.csv'
        test_file='data/test_linear.csv'
    if 'cir' in args.dataset:
        train_file='data/train_circle.csv'
        test_file='data/test_circle.csv'

    train_data = np.loadtxt(open(train_file,"rb"), delimiter=",",dtype=float)
    test_data = np.loadtxt(open(test_file,"rb"), delimiter=",",dtype=float)
    train_num = len(train_data)
    test_num = len(test_data)

    train_label=np.zeros(train_num,dtype=int)
    test_label=np.zeros(test_num,dtype=int)
    for i in range(train_num):
        idx=int(i/(train_num/args.num_seg))
        train_label[i]=int(idx % 2)
    for i in range(test_num):
        idx=int(i/(test_num/args.num_seg))
        test_label[i]=int(idx % 2)
    print(train_data.shape,test_data.shape,train_label.shape,test_label.shape)
    return train_data,test_data,train_label,test_label,train_num,test_num

def accuracy(output, target):
    """Computes the precision"""
    new_output=output.detach().cpu()
    new_target=target.detach().cpu()
    _, predicted = new_output.max(1)
   
    correct = predicted.eq(new_target).sum().item()
    return correct
 

def train(images, labels, optimizer, model, criterion, device='cpu'):
    input_size = images.size()
    images = Variable(images).view(input_size[0],input_size[2]*input_size[2]).to(device)
    labels = Variable(labels).to(device)

    optimizer.zero_grad()
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    # print(loss)
    return loss,output

def test(epoch,test_loader,model, criterion, device='cpu'):
    correct = 0.0
    total = 0.0
    total_loss = 0.0
    i=0
    # predicted_label=torch.zeros(test_num)
    top1_acc=0
    model.eval()
    for j,(images, labels) in enumerate(test_loader):        
        with torch.no_grad():
            #print(j)
            input_size = images.size()
            images = Variable(images).view(input_size[0],input_size[2]*input_size[2]).to(device)
            outputs = model(images)
            # _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            total += labels.size(0)
            correct+=accuracy(output=outputs, target=labels)

            #temp=(predicted .detach().cpu().numpy()== labels.detach().cpu().numpy()).sum()
            #correct=correct+temp
            total_loss=total_loss+loss.item()
            i=i+1
    print('Test  epoch:{}  top1_acc:{:.4f}  loss:{:.4f}'.format(epoch,100.0*correct/total,total_loss/i))
  
    return 100.0*correct/total,total_loss/i


def main(args):
    if args.dataset == 'MNIST':
        train_loader,test_loader =dataset.mnist_dataloaders()
        for iter_times in range(args.iter_times):
            model = model_zoo.Dense_MLP(args.width, args.depth, args.tc, input_dims=784, num_classses=10)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
            print(datetime.datetime.now())
            max_acc=-1000
            #print(model.features)

            for epoch in range(args.epochs):
                correct = 0
                total = 0
                total_loss=0  
                for i, (images, labels) in enumerate(train_loader):
                    loss,outputs = train(images, labels,optimizer=optimizer,model=model, criterion=criterion)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                    total_loss=total_loss+loss
                if max_acc<float(correct) / total:
                    max_acc=float(correct) / total
                total_loss=total_loss/float(i+1)
                with open(args.train_log_file,'a+') as train_logs:
                    print(100 * float(correct) / total,total_loss.item(),file=train_logs) 

            #print('------******------******------******------******------******------******------',file=file_logs)
            test_acc,test_loss = test(epoch,test_loader,model, criterion)
            with open(args.res_log_file,'a+') as train_logs:
                print(model.nn_mass, test_acc,test_loss, 
                    model.params, model.flops, args.width, args.depth, 
                    args.tc, args.num_seg,file=train_logs) 
            print('Accuracy of the model on the test set: % f %%' % (test_acc))


    else:
        train_data_raw,test_data,train_label_raw,test_label,train_num,test_num=make_dataset(args)
        raw_label=np.zeros([train_num,2])
        for i in range(train_num):
            raw_label[i,train_label_raw[i]]=1
        train_data_raw=torch.tensor(train_data_raw,dtype=torch.float)
        test_data=torch.tensor(test_data,dtype=torch.float)
        train_label_raw=Variable(torch.from_numpy(train_label_raw))
        test_label=Variable(torch.from_numpy(test_label))



        for iter_times in range(args.iter_times):
            model = model_zoo.Dense_MLP(args.width, args.depth, args.tc, input_dims=2, num_classses=2)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
            print(datetime.datetime.now())
            max_acc=-1000
            for epoch in range(args.epochs):
                perm_idx=torch.randperm(train_num)
                train_data=train_data_raw[perm_idx]
                train_label=train_label_raw[perm_idx]
                steps=int(train_num/args.batch_size)
                train_data=train_data.view([steps,args.batch_size,-1])
                train_label=train_label.view([steps,args.batch_size])
                correct = 0
                total = 0  
                total_loss = 0    
                for i in range(steps):
                    image = Variable(train_data[i])
                    label = Variable(train_label[i])
                    # Forward + Backward + Optimize
                    optimizer.zero_grad()
                    outputs = model(image)
                    loss = criterion(outputs, label)
                    loss.backward()
                    optimizer.step()
                    _, predicted = torch.max(outputs.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum()
                    total_loss=total_loss+loss
                total_loss = total_loss/(i+1)
  
                with open(args.train_log_file,'a+') as train_logs:
                    print(100 * float(correct) / total,total_loss.item(),file=train_logs)                

            # Test the Model
            correct = 0
            total = 0
            predicted_label=torch.zeros(test_num)
            test_data=test_data.view([-1,args.batch_size,2])
            test_label=test_label.view([-1,args.batch_size])
            steps= int(test_num/args.batch_size)
            for i in range(steps):
                image = Variable(test_data[i])
                label = Variable(test_label[i])
                outputs = model(image)
                _, predicted = torch.max(outputs.data, 1)
                predicted_label[i*args.batch_size:(i+1)*args.batch_size]=predicted
                total += label.size(0)
                correct += (predicted == label).sum()
            #print('------******------******------******------******------******------******------',file=file_logs)
            if max_acc<float(correct) / total:
                    max_acc=float(correct) / total
            with open(args.res_log_file,'a+') as train_logs:
                print(model.nn_mass, (100 * float(correct) / total),100*max_acc, 
                    model.params, model.flops, args.width, args.depth, 
                    args.tc, args.num_seg,file=train_logs) 
            print('Accuracy of the model on the test set: % f %%' % (100 * float(correct) / total))
if __name__ == '__main__':
    main(args)
