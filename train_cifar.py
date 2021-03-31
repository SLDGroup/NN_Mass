import argparse
import pdb
import sys 
import pickle
import matplotlib.pyplot as plt
import os
import datetime
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.models as models
from torch.autograd import Variable

from dataset import *
import utils
import model_zoo
parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')

########################## base setting ##########################
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--arch', type=str, default='dense_depth', help='network architecture')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default='ckpt', type=str)
parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--width_factor', default=10, type=int, help='width-factor of wideresnet')
parser.add_argument('--result_file', help='file name for result', default='result.pkl', type=str)
parser.add_argument('--save_all', action='store_true', help='whether to save all checkpoint')
parser.add_argument('--res_log_file', type=str, default='logs/cifar_cnn_res.logs', help='the name of file used to record the training/test record of MLPs')

########################## training setting ##########################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--decreasing_lr', default='50,150', help='decreasing strategy')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
parser.add_argument('--save_everyepoch', action="store_true", help="save the checkpoint for each epoch")

########################## densenet setting ##########################
parser.add_argument('--wm', type=int, default=2, help='width multipler of CNN cells')
parser.add_argument('--num_cells', type=int, default=3, help='number of cells')
parser.add_argument('--cell_depth', type=int, default=13, help='number of layers for each cell')
parser.add_argument('--tc1', type=int, default=40,  help='tc for the first cell')
parser.add_argument('--tc2', type=int, default=80,  help='tc for the second cell')
parser.add_argument('--tc3', type=int, default=125,  help='tc for the thrid cell')
parser.add_argument('--tc4', type=int, default=200, help='tc for the fourth cell')


best_prec1 = 0

def accuracy(output, target):
    """Computes the precision"""
    new_output=output.detach().cpu()
    new_target=target.detach().cpu()
    _, predicted = new_output.max(1)
   
    correct = predicted.eq(new_target).sum().item()
    return correct

def train(images, labels, optimizer, model, criterion, device):


    images = Variable(images).to(device)
    labels = Variable(labels).to(device)

    optimizer.zero_grad()
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    # print(loss)
    return loss,output


def test(epoch,test_loader,model,file_name, criterion, device):
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
            images = Variable(images).to(device)
            labels = labels.to(device)
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
  
    with open(file_name,'a+') as logs:
        #print(top1_acc/float(i),top5_acc/float(i), total_loss/float(i),file=logs)
        print('MN-Mass:{}  Test  epoch:{}  top1_acc:{:.4f}  loss:{:.4f}'.format(model.nn_mass,epoch,100.0*correct/total,total_loss/i),file=logs)
    return 100.0*correct/total,total_loss/i

def save_checkpoint(dir, epoch, sa_best=False, swa=False, inplace=True, **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)

    if swa:
        sa_best_file = 'SWA_SA_best.pt'
    else:
        sa_best_file = 'SA_best.pt'

    if sa_best:
        filepath = os.path.join(dir, sa_best_file)
    elif inplace:
        filepath = os.path.join(dir, 'checkpoint.pt')
    else:
        filepath = os.path.join(dir, 'checkpoint-%d.pt' % epoch)

    torch.save(state, filepath)
    
    # if args.save_all:
    #     filepath = os.path.join(dir, 'checkpoint-%d.pt' % epoch)
    #     torch.save(state, filepath)


def main():
    global args, best_prec1
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ########################## prepare dataset ##########################
    if args.dataset == 'cifar10':
        print('training on cifar10 dataset')
        if 'dense_depth' in args.arch:
            model = model_zoo.depth_wise_dense_cnn(cell_depth=args.cell_depth,wm=2,tc_array=[args.tc1,args.tc2,args.tc3,args.tc4],num_cells=args.num_cells,num_classes=10)
        elif 'regular_dense' in args.arch:
            model = model_zoo.regular_dense_cnn(cell_depth=args.cell_depth,wm=2,tc_array=[args.tc1,args.tc2,args.tc3,args.tc4],num_cells=args.num_cells,num_classes=10)

        train_loader, test_loader = cifar10_dataloaders(train_batch_size= args.batch_size, test_batch_size=args.batch_size, data_dir =args.data)

    elif args.dataset == 'cifar100':
        print('training on cifar100 dataset')
        if 'dense_depth' in args.arch:
            model = model_zoo.depth_wise_dense_cnn(cell_depth=args.cell_depth,wm=2,tc_array=[args.tc1,args.tc2,args.tc3,args.tc4],num_cells=args.num_cells,num_classes=100)
        elif 'regular_dense' in args.arch:
            model = model_zoo.regular_dense_cnn(cell_depth=args.cell_depth,wm=2,tc_array=[args.tc1,args.tc2,args.tc3,args.tc4],num_cells=args.num_cells,num_classes=100)
        
        train_loader, test_loader = cifar100_dataloaders(train_batch_size= args.batch_size, test_batch_size=args.batch_size, data_dir =args.data)
    else:
        print('dataset not support')

    ########################## optimizer and scheduler ##########################
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, float(args.epochs), eta_min=0.0)


    ########################## resume ##########################
    start_epoch = 0
    if args.resume:
        print('resume from checkpoint')

        if args.resume_specific<=0:
            checkpoint = torch.load(os.path.join(args.save_dir, 'checkpoint.pt'))
        else:
            checkpoint = torch.load(os.path.join(args.save_dir, 'checkpoint-{}.pt'.format(args.resume_specific)))

        best_prec1 = checkpoint['best_prec1']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])


    print(model.nn_mass,'number of parameters: {}'.format( sum(p.numel() for p in model.parameters() if p.requires_grad)))
    

    print('nnmass:{} nndensity:{} cell_depth:{} width:{}'.format(model.nn_mass,model.density,args.cell_depth,model.width_base))
    print('number of parameters: {}'.format( sum(p.numel() for p in model.parameters() if p.requires_grad)))
    # PATH='ckpt/num_cell{}_cell_depth{}_wm_{}_tc_{}_{}_{}'.format(3,depth_array[cell_depth],args.wm,tc_array[tc_group][0],tc_array[tc_group][1],tc_array[tc_group][2])
    # torch.save(model, PATH)
    with open(args.res_log_file,'a+') as logs:
        print('NN-Density:',model.density,'NN-Mass:',model.nn_mass,'NN-width:',model.width_base,'NN-cell_depth:',args.cell_depth,
                'TC:',[args.tc1,args.tc2,args.tc3,args.tc4],
                'number of parameters: {}'.format( sum(p.numel() for p in model.parameters() if p.requires_grad)),file=logs)

    model = model.to(device) 
    criterion = nn.CrossEntropyLoss()
    max_acc=-1000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(args.epochs), eta_min=0.0)
    
    for epoch in range(args.epochs):
        print('epoch:{}'.format(epoch))
        correct = 0.0
        total = 0.0
        total_loss=0.0  
        i=0

        temp_top1=0
        top1_acc=0
        temp_loss=0
        temp_total=0
        for j,(images, labels) in enumerate(train_loader):

            loss,outputs = train(images, labels,optimizer=optimizer,model=model, criterion=criterion, device=device)
            _, predicted = torch.max(outputs.data, 1)
            temp_loss=temp_loss+loss

            correct+=accuracy(output=outputs, target=labels)
            total_loss=total_loss+loss.item()
            temp_top1+=accuracy(output=outputs, target=labels)
            total += labels.size(0)

            total_loss=total_loss+loss
            i=i+1
            # print(datetime.datetime.now())


        print('Time:{} Train  epoch:{}  top1_acc:{:.4f}  loss:{:.4f}'.format(datetime.datetime.now(),epoch,100.0*correct/total,total_loss/i))
        with open(args.res_log_file,'a+') as logs:
            print('MN-Mass:{}  Train  epoch:{}  Train top1_acc:{:.4f}  Train loss:{:.4f}'.format(model.nn_mass,epoch,100.0*correct/total,total_loss/i),file=logs)
        test_sa,test_loss=test(epoch,test_loader,model,args.res_log_file,criterion,device)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print(lr)
        is_sa_best = test_sa  > best_prec1

        if is_sa_best:                
            save_checkpoint(
                args.save_dir,
                epoch + 1,
                sa_best=True,
                state_dict=model.state_dict(),
                best_prec1 = best_prec1,
                optimizer = optimizer.state_dict(),
                scheduler =  scheduler.state_dict(),
            )
        elif args.save_all:
            save_checkpoint(
                args.save_dir,
                epoch + 1,
                inplace=False,
                sa_best=False,
                state_dict=model.state_dict(),
                best_prec1 = best_prec1,
                optimizer = optimizer.state_dict(),
                scheduler =  scheduler.state_dict(),
        )

if __name__ == '__main__':
    main()


