import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class vgg_enc(nn.Module):
    def __init__(self, Pre_Train):
        super().__init__()
        self.backbone = models.vgg13_bn(Pre_Train)
        self.backbone.features._modules['0'] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.share_net = self.backbone.features
        self.flatten = nn.Flatten()

    def forward(self, x_ir):
        x = self.share_net(x_ir)  # 512, 4, 4
        return x

class Shared_Network156(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)     # -> (64, 156, 156)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                    # -> (64, 78, 78)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)   # -> (128, 78, 78)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                    # -> (128, 39, 39)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # -> (256, 39, 39)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=3)                    # -> (256, 13, 13)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)  # -> (512, 13, 13)
        self.pool4 = nn.AdaptiveAvgPool2d((4, 4))                             # -> (512, 4, 4)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.pool4(self.relu(self.conv4(x)))
        return x  # (B, 512, 4, 4)

class AuxEnc(nn.Module):
    def __init__(self, ratio_num, value_fdim):
        super().__init__()
        self.ratio_num = ratio_num

        self.value_encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(1, 4), nn.ReLU(),
                          nn.Linear(4, value_fdim))
            for _ in range(ratio_num)
        ])
        self.flatten = nn.Flatten()

    def forward(self, values):
        bs = values.size(0)
        valueFs = []  # Values: [B, ratio_num, 1]
        for i in range(self.ratio_num):
            valueF = self.value_encoders[i](values[:, i].unsqueeze(dim=-1))
            valueFs.append(valueF)
        value_feat = torch.cat([i.unsqueeze(dim=1) for i in valueFs], dim=1)
        return value_feat    # [B, ratio_num, value_fdim]

class PRG_SALSTM8(nn.Module):
    def __init__(self, in_channels, aux_num):
        super(PRG_SALSTM8, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv6 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)

        self.conv7 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        self.conv8 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        self.conv9 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)

        self.conv10 = nn.Conv2d(in_channels*2+aux_num, in_channels, kernel_size=1)
        self.conv11 = nn.Conv2d(in_channels*2+aux_num, in_channels, kernel_size=1)
        self.conv12 = nn.Conv2d(in_channels*2+aux_num, in_channels, kernel_size=1)
        self.conv13 = nn.Conv2d(in_channels*2+aux_num, in_channels, kernel_size=1)

    def sa_conv_lstm(self, x, en_1d): ##[T, B, 512, 4, 4] [B,1,4,4]
        # [B, 512, 4, 4]
        H = torch.zeros_like(x[0])
        C = torch.randn_like(x[0]) * 1e-6
        for i in range(x.size(0)):
            a_xh = torch.sigmoid(self.conv10(torch.cat((H, x[i], en_1d), dim=1))) 
            ca_xh = C*a_xh
            ga = torch.sigmoid(self.conv11(torch.cat((H, x[i], en_1d), dim=1)))
            gv = torch.tanh(self.conv12(torch.cat((H, x[i], en_1d), dim=1)))
            C = ca_xh + ga*gv
            a_xh1 = torch.sigmoid(self.conv13(torch.cat((H, x[i], en_1d), dim=1)))
            H = a_xh1*torch.tanh(C)
            memory, H = self.self_attention_memory(memory, H)  # H:torch.Size([B, 1, 16, 16])
        return H

    def self_attention_memory(self, m, h): #[B, 512, 4, 4]
        vh = self.conv1(h)
        kh = self.conv2(h)
        qh = self.conv3(h)

        qh = torch.transpose(qh, 2, 3)
        ah = F.softmax(kh*qh,dim=-1)
        zh = vh*ah

        km = self.conv4(m)
        vm = self.conv5(m)
        am = F.softmax(qh*km,dim=-1)
        zm = vm*am
        z0 = torch.cat((zh, zm), dim=1)
        z = self.conv6(z0)
        hz = torch.cat((h, z), dim=1)

        ot = torch.sigmoid(self.conv7(hz))
        gt = torch.tanh(self.conv8(hz))
        it = torch.sigmoid(self.conv9(hz))

        gi = gt*it
        mf = (1-it)*m
        mt = gi+mf
        ht = ot*mt

        return mt,ht

    def forward(self, x, en_1d):
        B,_,_,_,_ = x.size()
        x = x.permute(1, 0, 2, 3, 4)
        H = self.sa_conv_lstm(x, en_1d)

        return H

class MultiCondEnc(nn.Module):
    def __init__(self, ratio_num):
        super().__init__()
        self.enc1 = Shared_Network156()
        self.enc2 = Shared_Network156()
        self.ratio_num = ratio_num
        self.aux_fc = AuxEnc(ratio_num, 16)
        self.prg_fus = PRG_SALSTM8(in_channels=512, aux_num=ratio_num)

    def forward(self, x, pr):
        f1 = self.enc1(x[:, :4]).unsqueeze(dim=1)      # b, 1, 512, 4, 4
        f2 = self.enc1(x[:, 4:]).unsqueeze(dim=1)      # b, 1, 512, 4, 4
        tf = torch.cat([f1, f2], dim=1)          # b, 2, 512, 4, 4
        b, t, c, h, w = tf.size()

        auxFs = self.aux_fc(pr).reshape(b, self.ratio_num, h, w)   # b, ratio_num, 4*4
        prgf = self.prg_fus(tf, auxFs)   # b, 512, 4, 4

        return prgf
