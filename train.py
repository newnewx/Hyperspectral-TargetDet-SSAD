from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torch.autograd import Variable
import time
from numpy import *
from data_loader.dataset import train_dataset
import models.model as network
from metrics import StreamSegMetrics
import torch.nn.functional as F
# torch.backends.cudnn.enabled = False
# CUDA_LAUNCH_BLOCKING=1

os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

parser = argparse.ArgumentParser(description='Training a Scattnet model')
parser.add_argument('--batch_size', type=int, default=2, help='equivalent to instance normalization with batch_size=1')
parser.add_argument('--input_nc', type=int, default=3)
parser.add_argument('--output_nc', type=int, default=2)  
parser.add_argument('--num_classes', type=int, default=2)                                                                       
parser.add_argument('--pretrain', type=bool, default=False, help='whether to load pre-trained model weights')
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda',type=bool,default=True, help='enables cuda')
parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--num_workers', type=int, default=4, help='how many threads of cpu to use while loading data')
parser.add_argument('--size_w', type=int, default=128, help='scale image to this size')
parser.add_argument('--size_h', type=int, default=128, help='scale image to this size')
parser.add_argument('--flip', type=int, default=1, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--net', type=str, default='', help='path to pre-trained network')
parser.add_argument('--train_path', default='./dataset/data/train', help='path to training images')
parser.add_argument('--test_path', default='./dataset/data/val', help='path to testing images')
parser.add_argument('--outf', default='./checkpoint', help='folder to output images and model checkpoints')
parser.add_argument('--save_epoch', default=10, help='path to val images')
parser.add_argument('--test_step', default=300, help='path to val images')
parser.add_argument('--log_step', default=1, help='path to val images')
parser.add_argument('--num_GPU', default=1, help='number of GPU')
opt = parser.parse_args()
# print(opt)

try:
    os.makedirs(opt.outf)
    #os.makedirs(opt.outf + '/model/')
except OSError:
    pass

if opt.manual_seed is None:
    opt.manual_seed = random.randint(1, 10000)
# print("Random Seed: ", opt.manual_seed)
random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)
cudnn.benchmark = False 

def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        #m.bias.data.fill_(0)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        #m.bias.data.fill_(0)
def main():
    
    train_datatset_ = train_dataset(opt.train_path, opt.size_w, opt.size_h, opt.flip)
    train_loader = torch.utils.data.DataLoader(dataset=train_datatset_, batch_size=opt.batch_size, shuffle=True,
                                            num_workers=opt.num_workers)
    test_datatset_ = train_dataset(opt.test_path, opt.size_w, opt.size_h, opt.flip)
    test_loader = torch.utils.data.DataLoader(dataset=test_datatset_, batch_size=opt.batch_size, shuffle=False,    
                                            num_workers=opt.num_workers)
    print('train set:{} val set:{}'.format(len(train_datatset_), len(test_datatset_)))

    net = network.network(opt.input_nc,opt.output_nc)  # 3,2

    if opt.net != '':
        net.load_state_dict(torch.load(opt.netG))
    else:
        net.apply(weights_init)

    initial_image = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size_w, opt.size_h)             
    semantic_image = torch.FloatTensor(opt.batch_size, 2, opt.size_w, opt.size_h)  
    initial_image = Variable(initial_image)
    semantic_image = Variable(semantic_image)
    if opt.cuda:
        net.cuda()
        ###########   GLOBAL VARIABLES   ###########
        initial_image = initial_image.cuda()
        semantic_image = semantic_image.cuda()

    if opt.num_GPU > 1:
        net=nn.DataParallel(net)


    ###########   LOSS & OPTIMIZER   ##########
    # criterion = nn.BCELoss() 
    from loss.Binary_CE import BCE_Loss
    criterion = BCE_Loss(bcekwargs={'reduction':'mean'})     
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=1e-8)  
          

    metrics = StreamSegMetrics(opt.num_classes)

    log = open('./checkpoint/200/HSN_net.txt', 'w')
    start = time.time()
    
    train_acc = 0
    test_acc = 0
    best_acc = 0

    for epoch in range(1, opt.niter+1):
        net.train()
        train_batch_loader = iter(train_loader)
        test_batch_loader = iter(test_loader)
        train_acc_batch = 0
        test_acc_batch = 0
        metrics.reset()

        for i in range(0, train_datatset_.__len__(), opt.batch_size):
            initial_image_, semantic_image_, name = train_batch_loader.next()
            initial_image.resize_(initial_image_.size()).copy_(initial_image_)
            semantic_image.resize_(semantic_image_.size()).copy_(semantic_image_)
            semantic_image_pred= net(initial_image) 
            
            ### loss ###
            loss = criterion(semantic_image_pred, semantic_image)
            optimizer.zero_grad()#zero the parameter gradients
            loss.backward()#backward and solve the gradients 
            optimizer.step()#update the weight parameters

            ### metric ###
            target = semantic_image.cpu().numpy()
            pred = torch.argmax(semantic_image_pred, 1).cpu().numpy()   
            metrics.update(target, pred)
            score = metrics.get_results()
                                         
        iou_unchange,iou_change = score['Class IoU']          
        iou = iou_change
        miou =  score['Mean IoU']
        mrecall = score['M_recall']
        train_acc = mrecall


        print('epoch = %d, Train Loss = %.4f, train acc = %.4f, mIoU = %.4f, iou = %.4f' % (epoch, loss.item(), train_acc, miou, iou))
        log.write('epoch = %d, Train Loss = %.4f, train acc = %.4f, mIoU = %.4f, iou = %.4f\n' % (epoch, loss.item(), train_acc, miou, iou))

        if epoch % 10 == 0:
            metrics.reset()
            for i in range(0, test_datatset_.__len__(), opt.batch_size):
                initial_image_, semantic_image_, name = test_batch_loader.next()
                initial_image.resize_(initial_image_.size()).copy_(initial_image_)
                semantic_image.resize_(semantic_image_.size()).copy_(semantic_image_)
                semantic_image_pred= net(initial_image) 

                ### metric ###
                target = semantic_image.cpu().numpy()
                pred = torch.argmax(semantic_image_pred, 1).cpu().numpy() 
                metrics.update(target, pred)
                score = metrics.get_results()

            iou_unchange, iou_change = score['Class IoU']
            miou =  score['Mean IoU']
            mrecall = score['M_recall']
            iou = iou_change
            test_acc = mrecall
            print('epoch = %d, test acc = %.4f, mIoU = %.4f, iou = %.4f' % (epoch, test_acc, miou, iou))
            log.write('epoch = :%d, test acc = %.4f, mIoU = %.4f, iou = %.4f\n' % (epoch, test_acc, miou, iou))
            print()

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch =  epoch
                torch.save(net.state_dict(), '%s/200_3/HSN.pth' % (opt.outf))

        if epoch % 10 == 0:
            torch.save(net.state_dict(), '%s/200_3/HSN_%s.pth' % (opt.outf, epoch))
        '''
        if test_acc >= best_acc and test_acc > 0.8:
            best_acc = test_acc
            torch.save(net.state_dict(), '%s/800/seg_net.pth' % (opt.outf, epoch, test_acc))
        '''
    


    end = time.time()
    print('best_acc: {} best_epoch:{} '.format(best_acc, best_epoch))
    print('Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')
    log.close()



if __name__ == '__main__':
    main()