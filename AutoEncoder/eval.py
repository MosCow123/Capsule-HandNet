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
    # 1. Load data
    test_data = HandPointDataset(root_path='../data/preprocess', opt=opt, train=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
                                                   shuffle=False, num_workers=int(opt.workers), pin_memory=False,drop_last=True)
    # (train_data[0][0].size())  (1024,6)

    USE_CUDA = True
    capsule_net = PointCapsNet(opt.prim_caps_size, opt.prim_vec_size, opt.latent_caps_size, opt.latent_caps_size,
                               opt.num_points)


    if USE_CUDA:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        capsule_net = capsule_net.cuda()

    if opt.model != '':
        capsule_net.load_state_dict(torch.load(os.path.join(save_dir, opt.model)))

    for name, param in capsule_net.named_parameters():
        if param.requires_grad:
            print(name,param)

    capsule_net.eval()
    timer = time.time()
    test_loss_sum = 0
    for batch_id, data in enumerate(tqdm(test_dataloader, 0)):
        points, volume_length, gt_pca, gt_xyz = data
        # print(points.size()) (B,1024,6)
        if (points.size(0) < opt.batchSize):
            break
        points = Variable(points)
        points = points.transpose(2, 1)  # (B,6,1024)
        if USE_CUDA:
            points = points.cuda()

        latent_caps, reconstructions, Group = capsule_net(points)
        point, p = points.split(3, 1)
        # print(reconstructions.size()) (B,3,1024)
        test_loss = capsule_net.loss(point, reconstructions)
        test_loss_sum += test_loss.item()

    timer = time.time() - timer
    timer = timer / len(test_data)
    print('==> time to learn 1 sample = %f (ms)' % (timer * 1000))

    test_loss_sum = test_loss_sum / float(len(test_dataloader))
    print('test loss is : %f' % (test_loss_sum))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--prim_caps_size', type=int, default=1024, help='number of primary point caps')
    parser.add_argument('--prim_vec_size', type=int, default=16, help='scale of primary point caps')
    parser.add_argument('--latent_caps_size', type=int, default=64, help='number of latent caps')
    parser.add_argument('--latent_vec_size', type=int, default=64, help='scale of latent caps')

    parser.add_argument('--num_points', type=int, default=1024, help='input point set size')
    parser.add_argument('--outf', type=str, default='tmp_checkpoints', help='output folder')
    parser.add_argument('--model', type=str, default='encoder.pth', help='model name for training resume')

    parser.add_argument('--test_index', type=int, default=0, help='test index for cross validation, range: 0~8')
    parser.add_argument('--save_root_dir', type=str, default='results', help='output folder')
    parser.add_argument('--optimizer', type=str, default='', help='optimizer name for training resume')

    parser.add_argument('--size', type=str, default='full', help='how many samples do we load: small | full')
    parser.add_argument('--batchSize', type=int, default=8, help='input batch size')#
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--nepoch', type=int, default=60, help='number of epochs to train for')
    parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
    parser.add_argument('--main_gpu', type=int, default=1, help='main GPU id')

    parser.add_argument('--SAMPLE_NUM', type=int, default=1024, help='number of sample points')
    parser.add_argument('--JOINT_NUM', type=int, default=21, help='number of joints')
    parser.add_argument('--INPUT_FEATURE_NUM', type=int, default=6, help='number of input point features')
    parser.add_argument('--PCA_SZ', type=int, default=42, help='number of PCA components')
    opt = parser.parse_args()
    save_dir = os.path.join(opt.save_root_dir, subject_names[opt.test_index]) #results/P0
    print(opt)


    main()
