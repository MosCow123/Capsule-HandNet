import argparse
import os
import random
import progressbar
import time
import logging
import pdb
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

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()

parser.add_argument('--prim_caps_size', type=int, default=1024, help='number of primary point caps')
parser.add_argument('--prim_vec_size', type=int, default=16, help='scale of primary point caps')
parser.add_argument('--latent_caps_size', type=int, default=64, help='number of latent caps')
parser.add_argument('--latent_vec_size', type=int, default=64, help='scale of latent caps')

parser.add_argument('--num_points', type=int, default=1024, help='input point set size')
parser.add_argument('--outf', type=str, default='tmp_checkpoints', help='output folder')
parser.add_argument('--model', type=str, default='regression.pth', help='model name for training resume')
parser.add_argument('--capsule_model', type=str, default='', help='model name for training resume')

parser.add_argument('--size', type=str, default='full', help='how many samples do we load: small | full')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')  #
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=120, help='number of epochs to train for')
parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
parser.add_argument('--main_gpu', type=int, default=1, help='main GPU id')  # CUDA_VISIBLE_DEVICES=0 python train.py

parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate at t=0')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum (SGD only)')
parser.add_argument('--weight_decay', type=float, default=0.00005, help='weight decay (SGD only)')
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

print(opt)


opt.manualSeed = 1
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

save_dir = os.path.join(opt.save_root_dir, subject_names[opt.test_index])
# save_dir = os.path.join(opt.save_root_dir, "test5")

test_data = HandPointDataset(root_path='data/preprocess', opt=opt, train=False)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
                                              shuffle=True, num_workers=int(opt.workers), pin_memory=False,
                                              drop_last=True)

print(len(test_dataloader))
print('#Test data:', len(test_data))
print(opt)


caphand_net = Capsule_handnet(opt)

if opt.model != '':
    caphand_net.load_state_dict(torch.load(os.path.join(save_dir, opt.model)))  # 仅加载模型参数

caphand_net.cuda()


criterion = nn.MSELoss(size_average=True).cuda()


torch.cuda.synchronize()

caphand_net.eval()
test_cdf  = np.empty(shape=[32, 0])
test_finger = np.zeros(shape=[32,21])
test_mse = 0.0
test_wld_err = 0.0
timer = time.time()
for i, data in enumerate(tqdm(test_dataloader, 0)):
    torch.cuda.synchronize()

    points, volume_length, gt_pca, gt_xyz = data
    gt_pca = Variable(gt_pca, volatile=True).cuda()
    points, volume_length, gt_xyz = points.cuda(), volume_length.cuda(), gt_xyz.cuda()



    estimation, reconstructions= caphand_net(points)
    loss = criterion(estimation, gt_pca) * opt.PCA_SZ
    torch.cuda.synchronize()
    test_mse = test_mse + loss.item() * len(points)  # change to data[0]

    #
    # hand_xyz_MSRA = np.array(gt_xyz.cpu().detach().numpy())
    # np.savetxt('hand_xyz_MSRA.txt', hand_xyz_MSRA[0].reshape(-1, 3))
    # msra = np.array(points.split(3, 2)[0].cpu().detach().numpy())
    # np.savetxt('msra.txt', msra[0].reshape(-1, 3))
    # msra_reconstruction = np.array(reconstructions.cpu().detach().numpy())
    # np.savetxt('msra_reconstruction.txt', msra_reconstruction[0].reshape(-1, 3))



    outputs_xyz = test_data.PCA_mean.expand(estimation.data.size(0), test_data.PCA_mean.size(1))
    outputs_xyz = torch.addmm(outputs_xyz, estimation.data, test_data.PCA_coeff)
    # np.savetxt('hand_xyz_esti_MSRA.txt', outputs_xyz[0].cpu().reshape(-1, 3))

    diff = torch.pow(outputs_xyz - gt_xyz, 2).view(-1, opt.JOINT_NUM, 3)
    diff_sum = torch.sum(diff, 2)
    diff_sum_sqrt = torch.sqrt(diff_sum)

    test_cdf = np.append(test_cdf, torch.mul(diff_sum_sqrt,volume_length).max(1)[0].cpu().numpy())

    finger = torch.mul(diff_sum_sqrt,volume_length)
    test_finger = test_finger + finger.cpu().numpy()


    diff_mean = torch.mean(diff_sum_sqrt, 1).view(-1, 1)
    diff_mean_wld = torch.mul(diff_mean, volume_length)

    test_wld_err = test_wld_err + diff_mean_wld.sum()


# np.savetxt('a.txt', test_cdf.reshape(-1,1))

torch.cuda.synchronize()
timer = time.time() - timer
timer = timer / len(test_data)
print('==> time to learn 1 sample = %f (ms)' % (timer * 1000))


test_wld_err = test_wld_err / len(test_data)
print('average estimation error in world coordinate system: %f (mm)' % (test_wld_err))

print((test_finger.mean(0)/len(test_dataloader)))