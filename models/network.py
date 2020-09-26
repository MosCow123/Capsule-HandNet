import torch
import torch.nn as nn
import argparse
import math
from models.utils import group_points_2
from models.capsulenet.pointcapsnet import PointCapsNet

from torch.autograd import Variable

from models.nndistance.modules.nnd import NNDModule
distChamfer = NNDModule()



nstates_plus_1 = [64, 512,1024]
nstates_plus_2 = [128, 256]
nstates_plus_3 = [256, 512, 1024, 512]
nstates_plus_4 = [1024, 512 , 256]

class Capsule_handnet(nn.Module):
    def __init__(self, opt):
        super(Capsule_handnet, self).__init__()
        self.batchSize = opt.batchSize
        self.latent_caps_size = opt.latent_caps_size
        self.num_outputs = opt.PCA_SZ
        # self.knn_K = opt.knn_K
        #self.ball_radius2 = opt.ball_radius2
        # self.sample_num_level1 = opt.sample_num_level1  #64
        # self.sample_num_level2 = opt.sample_num_level2  #32
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM
        #胶囊层 input:(1024,16,64,64,2048)    output:latant(B,64,64),res(B,3,2048)
        self.Capsulenet1=PointCapsNet(opt.prim_caps_size,opt.prim_vec_size,opt.latent_caps_size,
                                      opt.latent_vec_size,opt.num_points)#(1024,16,64,64,1024)
        for p in self.parameters():
            p.requires_grad = False
        self.netR_0 = nn.Sequential(
            nn.MaxPool1d(16, 1)
        )

        self.netR_1 = nn.Sequential(
            # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K B*64*64*1
            #B * 64 * 64
            torch.nn.Conv1d(nstates_plus_1[0], nstates_plus_1[1], 1),
            torch.nn.BatchNorm1d(nstates_plus_1[1]),
            nn.ReLU(inplace=True),
            #B * 128 * 64
            torch.nn.Conv1d(nstates_plus_1[1], nstates_plus_1[2], 1),
            torch.nn.BatchNorm1d(nstates_plus_1[2]),
            nn.ReLU(inplace=True),

            # torch.nn.Conv1d(nstates_plus_1[2], nstates_plus_1[3], 1),
            # torch.nn.BatchNorm1d(nstates_plus_1[3]),
            # nn.ReLU(inplace=True),

            nn.MaxPool1d(64,1)
        )

        self.netR_2 = nn.Sequential(
            torch.nn.Conv1d(nstates_plus_1[2]+64, nstates_plus_1[2], 1),
            torch.nn.BatchNorm1d(nstates_plus_1[2]),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(64, 1)  #or everage

            # torch.nn.Conv1d(nstates_plus_1[2] + 3 , nstates_plus_2[0], 1),
            # torch.nn.BatchNorm1d(nstates_plus_2[0]),
            # nn.ReLU(inplace=True),

            # torch.nn.Conv1d(nstates_plus_2[0], nstates_plus_2[1], 1),
            # torch.nn.BatchNorm1d(nstates_plus_2[1]),
            # nn.ReLU(inplace=True),

            # torch.nn.Conv1d(nstates_plus_2[0], nstates_plus_2[1], 1),
            # torch.nn.BatchNorm1d(nstates_plus_2[1]),
            # nn.ReLU(inplace=True),
        )

        self.netR_3 = nn.Sequential(


            torch.nn.Conv1d(nstates_plus_3[0], nstates_plus_3[1], 1),
            torch.nn.BatchNorm1d(nstates_plus_3[1]),
            nn.ReLU(inplace=True),

            torch.nn.Conv1d(nstates_plus_3[1], nstates_plus_3[2], 1),
            torch.nn.BatchNorm1d(nstates_plus_3[2]),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(64,1)
        )

        self.netR_FC = nn.Sequential(
            # B*1024
            # nn.Linear(nstates_plus_3[2], nstates_plus_3[3]),
            # nn.BatchNorm1d(nstates_plus_3[3]),
            # nn.ReLU(inplace=True),
            # B*512
            nn.Linear(nstates_plus_4[0] , nstates_plus_4[1]),
            nn.BatchNorm1d(nstates_plus_4[1]),
            nn.ReLU(inplace=True),

            nn.Linear(nstates_plus_4[1], nstates_plus_4[2]),
            nn.BatchNorm1d(nstates_plus_4[2]),
            nn.ReLU(inplace=True)

            # nn.Dropout(0.9)
            # nn.Linear(nstates_plus_4[1], self.num_outputs)
            # nn.Dropout(0.6)
            # B*num_outputs
        )
        self.linear_fc = nn.Linear(nstates_plus_4[2], self.num_outputs)

    def forward(self, x): # x: B * 1024 * 6
        # x: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, y: B*3*sample_num_level1*1
        # latent_capsules B*latent_caps_size*latent_vec_size B*64*64,
        # reconstructions B * 3 * num_points  B * 3 * 2048,
        # cap_Group B * 3 * latent_caps_size  B * 3 * 64
        x = x.transpose(1, 2)  # B * 6 * 1024

        x, restructions, y = self.Capsulenet1(x)
        # x [64 caps,64 vec]
        # for p in self.parameters():
        #     p.requires_grad = False
        # x = x.unsqueeze(-1).transpose(1,2).contiguous() # (B*64*64*1)
        x = x.transpose(2, 1).contiguous()  # (B*64*64)
        temp = x.detach().cuda()

        x = self.netR_1(x)

        x = x.expand(self.batchSize, nstates_plus_1[2], self.latent_caps_size)

        x = torch.cat((temp, x), 1).contiguous()

        x = self.netR_2(x)
        # print(y)
        # B*128*sample_num_level1*1

        # B*(3+128)*sample_num_level1 B*131*64*1

        #x = self.netR_2(x)

        #x = self.netR_3(x)  #B*1024*1

        # print("x",x.size())

        x = x.view(-1, nstates_plus_3[2])
        # print("x", x.size())  4,1024
        x = self.netR_FC(x)
        x = self.linear_fc(x)
        # inputs_level2, inputs_level2_center = group_points_2(x, self.sample_num_level1, self.sample_num_level2,
        #                                                      self.knn_K, self.ball_radius2)
        # # B*131*sample_num_level2*knn_K, B*3*sample_num_level2*1
        #
        # # B*131*sample_num_level2*knn_K
        # x = self.netR_2(inputs_level2)
        # # B*256*sample_num_level2*1
        # x = torch.cat((inputs_level2_center, x), 1)
        # # B*259*sample_num_level2*1
        #
        # x = self.netR_3(x)
        # # B*1024*1*1
        # x = x.view(-1, nstates_plus_3[2])
        # # B*1024
        # x = self.netR_FC(x)
        # # B*num_outputs
        return x,restructions

    def loss(self, data, reconstructions):
        return self.reconstruction_loss(data, reconstructions)

    def reconstruction_loss(self, data, reconstructions):
        data_ = data.transpose(2, 1).contiguous()
        reconstructions_ = reconstructions.transpose(2, 1).contiguous()
        dist1, dist2 = distChamfer(data_, reconstructions_)
        loss = (torch.mean(dist1)) + (torch.mean(dist2))
        return loss



