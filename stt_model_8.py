import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np

class Speak2Embed(nn.Module):
    def __init__(self):
        super(Speak2Embed, self).__init__()
        #===================================================
        #time sequence to feature vector
        self.conv_1ch_2ch = nn.Conv1d(1,2,5, stride=2)
        self.conv_2ch_4ch = nn.Conv1d(2,4,3, stride=1)
        self.conv_4ch_8ch = nn.Conv1d(4,8,3, stride=1)
        self.conv_8ch_16ch = nn.Conv1d(8,16,3, stride=1)
        self.conv_16ch_32ch = nn.Conv1d(16,32,3, stride=1)
        self.conv_32ch_64ch = nn.Conv1d(32,64,3, stride=1)
        self.conv_64ch_128ch = nn.Conv1d(64,128,3, stride=1)

        # self.fc_layer = torch.nn.Linear(3,30)
        self.relu = torch.nn.ReLU()
        self.avg_pool_3 = nn.AvgPool1d(3)
        self.avg_pool_2 = nn.AvgPool1d(2)
        self.adaptive_pool = torch.nn.AdaptiveAvgPool1d(21)
        self.elu = nn.ELU()


        #==================================================
        #frequency map(melspectogram) to feature vector
        self.conv2_1ch_2ch = nn.Conv2d(1,2,3, stride=1)
        self.conv2_2ch_4ch = nn.Conv2d(2,4,3, stride=1)
        self.conv2_4ch_8ch = nn.Conv2d(4,8,3, stride=1)
        self.conv2_8ch_16ch = nn.Conv2d(8,16,3, stride=1)
        self.conv2_16ch_32ch = nn.Conv2d(16,32,3, stride=1)
        self.conv2_32ch_64ch = nn.Conv2d(32,64,3, stride=1)
        self.conv2_64ch_128ch = nn.Conv2d(64,128,3, stride=1)

        self.avg2_pool = nn.AvgPool2d(2)



        #==================================================
        #fusion time + frequency
        self.norm_layer = nn.LayerNorm([128,21])
        self.transpose_conv = nn.ConvTranspose1d(128,128,1)

    def forward(self,x, batch_size):
        for i in range(batch_size):
            D= np.abs(librosa.stft(x[i].cpu().numpy(), n_fft=2048, win_length=2048, hop_length=512))
            mfcc = torch.tensor(librosa.feature.mfcc(S=librosa.power_to_db(D), sr=16000, n_mfcc=40))
            if i == 0:
                y = torch.zeros(batch_size,1,mfcc.size(0),mfcc.size(1))
            y[i] = mfcc

        x = torch.reshape(x, (x.size(0),1,x.size(1))).float().cuda()

        x = self.conv_1ch_2ch(x)
        x = self.elu(x)
        x = self.avg_pool_3(x)
        x = self.conv_2ch_4ch(x)
        x = self.elu(x)
        x = self.avg_pool_3(x)
        x = self.conv_4ch_8ch(x)
        x = self.elu(x)
        x = self.avg_pool_3(x)
        x = self.conv_8ch_16ch(x)
        x = self.elu(x)
        x = self.avg_pool_2(x)
        x = self.conv_16ch_32ch(x)
        x = self.elu(x)
        x = self.avg_pool_2(x)
        x = self.conv_32ch_64ch(x)
        x = self.elu(x)
        x = self.avg_pool_2(x)
        x = self.conv_64ch_128ch(x)
        x = self.elu(x)
        x = self.adaptive_pool(x)

        y = self.conv2_1ch_2ch(y.float().cuda())
        y = self.conv2_2ch_4ch(y)
        y = self.conv2_4ch_8ch(y)
        y = self.conv2_8ch_16ch(y)
        y = self.conv2_16ch_32ch(y)
        y = self.elu(y)
        y = self.avg2_pool(y)
        y = self.conv2_32ch_64ch(y)
        y = self.elu(y)
        y = self.avg2_pool(y)
        y = self.conv2_64ch_128ch(y)

        y = torch.reshape(y, (batch_size,128,-1))
        y = self.adaptive_pool(y)

        x = x+y
        x = self.norm_layer(x)
        x = self.transpose_conv(x)
        return x
