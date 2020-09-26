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
from data.dataset import HandPointDataset
from data.dataset import subject_names

from data.dataset import gesture_names
from models.capsulenet.pointcapsnet import PointCapsNet

from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, 'models')))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, 'data')))


def main():
    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    # torch.backends.cudnn.enabled = False
    # create folder to save trained models
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf);

    USE_CUDA = True
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    caphand_net = Capsule_handnet(opt)

    if USE_CUDA:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        caphand_net = caphand_net.cuda()
        #caphand_net.to(device)

    if opt.ngpu > 1:
        # caphand_net.Capsulenet1 = torch.nn.DataParallel(caphand_net.Capsulenet1, range(opt.ngpu))
        caphand_net.netR_1 = torch.nn.DataParallel(caphand_net.netR_1, range(opt.ngpu))
        caphand_net.netR_2 = torch.nn.DataParallel(caphand_net.netR_2, range(opt.ngpu))
        caphand_net.netR_3 = torch.nn.DataParallel(caphand_net.netR_3, range(opt.ngpu))

    #caphand_net = torch.nn.DataParallel(caphand_net)
    #print(caphand_net)

    train_data = HandPointDataset(root_path='data/preprocess', opt=opt, train=True)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize,
                                                   shuffle=True, num_workers=int(opt.workers), pin_memory=False,drop_last=True)

    test_data = HandPointDataset(root_path='data/preprocess', opt=opt, train=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
                                                  shuffle=False, num_workers=int(opt.workers), pin_memory=False ,drop_last=True)


    print('#Train data:', len(train_data))
    print('#Test data:', len(test_data))


    if opt.model != '':
        caphand_net.load_state_dict(torch.load(os.path.join(save_dir, opt.model)))
        print(torch.load(os.path.join(save_dir, opt.model)))
    caphand_net.cuda()

    if opt.capsule_model != '':
        new_state_dict = OrderedDict()
        pretrained_dict = torch.load(os.path.join("AutoEncoder/results/P0", opt.capsule_model))
        for k, v in pretrained_dict.items():
            name = "Capsulenet1."+ k
            # print(name)
            new_state_dict[name] = v
        model_dict = caphand_net.state_dict()
        restroe_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        model_dict.update(restroe_dict)
        #print(model_dict)

        caphand_net.load_state_dict(model_dict)
        for name, param in caphand_net.named_parameters():
            print(name,param)


    criterion = nn.MSELoss(size_average=True).cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, caphand_net.parameters()),lr=opt.learning_rate, betas=(0.5, 0.999), eps=1e-06
                           ,weight_decay=opt.weight_decay)
    if opt.optimizer != '':
        optimizer.load_state_dict(torch.load(os.path.join(save_dir, opt.optimizer)))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # for name, param in caphand_net.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    for epoch in range(opt.nepoch):
        scheduler.step(epoch)
        print('======>>>>> Online epoch: #%d, lr=%f, Test: %s <<<<<======' % (
        epoch, scheduler.get_lr()[0], subject_names[opt.test_index]))

        torch.cuda.synchronize()
        caphand_net.train()
        train_mse = 0.0
        train_mse_wld = 0.0
        timer = time.time()

        for i, data in enumerate(tqdm(train_dataloader, 0)):
            if len(data[0]) == 1:
                continue

            torch.cuda.synchronize()
            # 3.1.1 load inputs and targets
            points, volume_length, gt_pca, gt_xyz = data
            gt_pca = Variable(gt_pca, requires_grad=False).cuda()
            points, volume_length, gt_xyz = points.cuda(), volume_length.cuda(), gt_xyz.cuda()
            with torch.no_grad():
                points = Variable(points)
            # 3.1.2 compute output
            optimizer.zero_grad()

            estimation, reconstructions= caphand_net(points)
            # print(estimation)
            #print(gt_pca)
            #print(estimation)
            #print(estimation.size(),reconstructions.size())
            #point,p =  points.split(3, 2)
            loss = criterion(estimation, gt_pca) * opt.PCA_SZ
            #print(loss)


            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()


            train_mse = train_mse + loss.item() * len(points)


            outputs_xyz = train_data.PCA_mean.expand(estimation.data.size(0), train_data.PCA_mean.size(1))
            outputs_xyz = torch.addmm(outputs_xyz, estimation.data, train_data.PCA_coeff)
            diff = torch.pow(outputs_xyz - gt_xyz, 2).view(-1, opt.JOINT_NUM, 3)
            diff_sum = torch.sum(diff, 2)
            diff_sum_sqrt = torch.sqrt(diff_sum)
            diff_mean = torch.mean(diff_sum_sqrt, 1).view(-1, 1)
            diff_mean_wld = torch.mul(diff_mean, volume_length)
            train_mse_wld = train_mse_wld + diff_mean_wld.sum()


        torch.cuda.synchronize()
        timer = time.time() - timer
        timer = timer / len(train_data)
        print('==> time to learn 1 sample = %f (ms)' % (timer * 1000))


        train_mse = train_mse / len(train_data)
        train_mse_wld = train_mse_wld / len(train_data)
        print('mean-square error of 1 sample: %f, #train_data = %d' % (train_mse, len(train_data)))
        print('average estimation error in world coordinate system: %f (mm)' % (train_mse_wld))

        torch.save(caphand_net.state_dict(), '%s/netR_%d.pth' % (save_dir, epoch))
        if epoch%10==0:
            torch.save(optimizer.state_dict(), '%s/optimizer_%d.pth' % (save_dir, epoch))


        torch.cuda.synchronize()
        caphand_net.eval()
        test_mse = 0.0
        test_wld_err = 0.0
        timer = time.time()
        for i, data in enumerate(tqdm(test_dataloader, 0)):
            torch.cuda.synchronize()
            points, volume_length, gt_pca, gt_xyz = data
            with torch.no_grad():
                gt_pca = Variable(gt_pca).cuda()
            points, volume_length, gt_xyz = points.cuda(), volume_length.cuda(), gt_xyz.cuda()

            with torch.no_grad():
                points = Variable(points)
            # points: B * 1024 * 6; target: B * 42


            estimation, reconstructions = caphand_net(points)
            loss = criterion(estimation, gt_pca) * opt.PCA_SZ

            torch.cuda.synchronize()
            test_mse = test_mse + loss.item() * len(points)


            outputs_xyz = test_data.PCA_mean.expand(estimation.data.size(0), test_data.PCA_mean.size(1))
            outputs_xyz = torch.addmm(outputs_xyz, estimation.data, test_data.PCA_coeff)
            diff = torch.pow(outputs_xyz - gt_xyz, 2).view(-1, opt.JOINT_NUM, 3)
            diff_sum = torch.sum(diff, 2)
            diff_sum_sqrt = torch.sqrt(diff_sum)
            diff_mean = torch.mean(diff_sum_sqrt, 1).view(-1, 1)
            diff_mean_wld = torch.mul(diff_mean, volume_length)
            test_wld_err = test_wld_err + diff_mean_wld.sum()


        torch.cuda.synchronize()
        timer = time.time() - timer
        timer = timer / len(test_data)
        print('==> time to learn 1 sample = %f (ms)' % (timer * 1000))
        # print mse
        test_mse = test_mse / len(test_data)
        print('mean-square error of 1 sample: %f, #test_data = %d' % (test_mse, len(test_data)))
        test_wld_err = test_wld_err / len(test_data)
        print('average estimation error in world coordinate system: %f (mm)' % (test_wld_err))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--batch_size', type=int, default=32, help='input batch size')

    parser.add_argument('--prim_caps_size', type=int, default=1024, help='number of primary point caps')
    parser.add_argument('--prim_vec_size', type=int, default=16, help='scale of primary point caps')
    parser.add_argument('--latent_caps_size', type=int, default=64, help='number of latent caps')
    parser.add_argument('--latent_vec_size', type=int, default=64, help='scale of latent caps')

    parser.add_argument('--num_points', type=int, default=1024, help='input point set size')
    parser.add_argument('--outf', type=str, default='tmp_checkpoints', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model name for training resume')
    parser.add_argument('--capsule_model', type=str, default='encoder.pth', help='model name for training resume')

    parser.add_argument('--size', type=str, default='full', help='how many samples do we load: small | full')
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')#
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--nepoch', type=int, default=120, help='number of epochs to train for')
    parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
    parser.add_argument('--main_gpu', type=int, default=1, help='main GPU id')  # CUDA_VISIBLE_DEVICES=0 python train.py

    parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate at t=0')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (SGD only)')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay (SGD only)')
    parser.add_argument('--learning_rate_decay', type=float, default=1e-7, help='learning rate decay')

    # hand
    parser.add_argument('--SAMPLE_NUM', type=int, default=1024, help='number of sample points')
    parser.add_argument('--JOINT_NUM', type=int, default=21, help='number of joints')
    parser.add_argument('--INPUT_FEATURE_NUM', type=int, default=6, help='number of input point features')
    parser.add_argument('--PCA_SZ', type=int, default=42, help='number of PCA components')

    parser.add_argument('--test_index', type=int, default=0, help='test index for cross validation, range: 0~8')
    parser.add_argument('--save_root_dir', type=str, default='results', help='output folder')
    parser.add_argument('--optimizer', type=str, default='', help='optimizer name for training resume')

    opt = parser.parse_args()

    print(torch.cuda.is_available())

    #save_dir = os.path.join(opt.save_root_dir, subject_names[opt.test_index])
    save_dir = os.path.join(opt.save_root_dir, "P0")
    print(opt)

    # logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
    #                     filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
    # logging.info('======================================================')

    main()
