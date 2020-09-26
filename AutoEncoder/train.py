import argparse
import os
import random
import progressbar
import time
import logging
import pdb
import sys
from tqdm import tqdm
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

from models.network import Capsule_handnet
from models.capsulenet.pointcapsnet import PointCapsNet
from data.dataset import HandPointDataset
from data.dataset import subject_names

from data.dataset import gesture_names
import scipy.io as io


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


import warnings
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, 'models')))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, 'data')))


def main():
    torch.backends.cudnn.enabled = False
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf);
    print(torch.cuda.is_available())
    # Load MSRA data
    train_data = HandPointDataset(root_path='../data/preprocess', opt=opt, train=True)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize,
                                                   shuffle=True, num_workers=int(opt.workers), pin_memory=False,drop_last=True)

    test_data = HandPointDataset(root_path='../data/preprocess', opt=opt, train=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
                                                   shuffle=False, num_workers=int(opt.workers), pin_memory=False,drop_last=True)
    # (train_data[0][0].size())  (1024,6)



    USE_CUDA = True
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    capsule_net = PointCapsNet(opt.prim_caps_size, opt.prim_vec_size, opt.latent_caps_size, opt.latent_vec_size,
                               opt.num_points) #1024 16 64 64 1024


    if USE_CUDA:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        capsule_net = capsule_net.cuda()
        #capsule_net = torch.nn.DataParallel(capsule_net)
        #capsule_net.to(device)

    if opt.model != '':
        capsule_net.load_state_dict(torch.load(os.path.join(save_dir, opt.model)))

    # for name, param in capsule_net.named_parameters():
    #     print(name, param)
    optimizer = optim.Adam(capsule_net.parameters(), lr=0.00001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    capsule_net.train()
    #print('# Model parameters:', sum(param.numel() for param in capsule_net.parameters()))

    train_Loss_list = []

    for epoch in range(0,opt.nepoch):
        scheduler.step(epoch)
        train_loss_sum = 0
        print('======>>>>> Online epoch: #%d, lr=%f,Test: %s <<<<<======' % (
            epoch, scheduler.get_lr()[0], subject_names[opt.test_index]))
        for batch_id, data in enumerate(tqdm(train_dataloader, 0)):
            capsule_net.train()
            points, volume_length, gt_pca, gt_xyz = data
            #points, input_sn, gt_xyz, volume_length, volume_offset, volume_rotate = data
            # print(points.size()) (B,1024,6)
            if (points.size(0) < opt.batchSize):
                break
            points = Variable(points)
            points = points.transpose(2, 1)  #(B,6,1024)
            if USE_CUDA:
                points = points.cuda()

            optimizer.zero_grad()
            latent_caps, reconstructions,Group = capsule_net(points)
            point, p = points.split(3, 1)
            # print(reconstructions.size()) (B,3,1024)
            train_loss = capsule_net.loss(point, reconstructions)
            #print("res",reconstructions)
            #print(reconstructions[0].size())
            #print(reconstructions.squeeze(0).shape)
            #reconstructions1 = np.array(reconstructions.cpu().detach().numpy())
            #point1 = np.array(point.cpu().detach().numpy())
            #io.savemat('save.mat', {'reconstructions1': reconstructions1})
            #np.savetxt('reconstructions1.txt', np.swapaxes(reconstructions1[0],0,1))
            #np.savetxt('point1.txt', np.swapaxes(point1[3], 0, 1))
            train_loss.backward()
            optimizer.step()
            train_loss_sum += train_loss.item()
            # info = {'train_loss': train_loss.item()}

        print('Average train loss of epoch %d : %f' %
              (epoch, (train_loss_sum / len(train_dataloader))))
        train_Loss_list.append(train_loss_sum / (len(train_dataloader)))


        test_loss_sum = 0
        capsule_net.eval()
        for batch_id, data in enumerate(tqdm(test_dataloader, 0)):
            capsule_net.eval()
            points, volume_length, gt_pca, gt_xyz = data
            # print(points.size()) (B,1024,6)

            points, volume_length, gt_xyz = points.cuda(), volume_length.cuda(), gt_xyz.cuda()

            points = Variable(points)
            points = points.transpose(2, 1)  # (B,6,1024)
            if USE_CUDA:
                points = points.cuda()

            optimizer.zero_grad()
            latent_caps, reconstructions, Group = capsule_net(points)
            point, p = points.split(3, 1)
            # print(reconstructions.size()) (B,3,1024)
            test_loss = capsule_net.loss(point, reconstructions)

            test_loss_sum += test_loss.item()
            # info = {'train_loss': train_loss.item()}

        print('Average test loss of epoch %d : %f' %
              (epoch, (test_loss_sum / len(test_dataloader))))
        #torch.save(capsule_net.module.state_dict(), save_dir)
        torch.save(capsule_net.state_dict(), '%s/netR_%d.pth' % (save_dir, epoch))

    np.savetxt("MSRA_loss.txt", train_Loss_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prim_caps_size', type=int, default=1024, help='number of primary point caps')
    parser.add_argument('--prim_vec_size', type=int, default=16, help='scale of primary point caps')
    parser.add_argument('--latent_caps_size', type=int, default=64, help='number of latent caps')
    parser.add_argument('--latent_vec_size', type=int, default=64, help='scale of latent caps')

    parser.add_argument('--num_points', type=int, default=1024, help='input point set size')
    parser.add_argument('--outf', type=str, default='tmp_checkpoints', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model name for training resume')

    parser.add_argument('--test_index', type=int, default=0, help='test index for cross validation, range: 0~8')
    parser.add_argument('--save_root_dir', type=str, default='results', help='output folder')
    parser.add_argument('--optimizer', type=str, default='', help='optimizer name for training resume')

    parser.add_argument('--size', type=str, default='full', help='how many samples do we load: small | full')
    parser.add_argument('--batchSize', type=int, default=8, help='input batch size')#
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--nepoch', type=int, default=120, help='number of epochs to train for')
    parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
    parser.add_argument('--main_gpu', type=int, default=1, help='main GPU id')

    parser.add_argument('--Data_name', type=str, default='', help='name of data')
    parser.add_argument('--SAMPLE_NUM', type=int, default=1024, help='number of sample points')
    parser.add_argument('--JOINT_NUM', type=int, default=21, help='number of joints')
    parser.add_argument('--INPUT_FEATURE_NUM', type=int, default=6, help='number of input point features')
    parser.add_argument('--PCA_SZ', type=int, default=42, help='number of PCA components')
    opt = parser.parse_args()
    save_dir = os.path.join(opt.save_root_dir, subject_names[opt.test_index]) #results/P0
    print(opt)


    main()
