# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:55:05 2020

@author: raoyu
"""
from torch import nn
import torch
import numpy as np

class AutoEncoderCNN(nn.Module):
    def __init__(self):
        super(AutoEncoderCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # (b, 16, 10, 10)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # (b, 16, 5, 5)
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # (b, 8, 3, 3)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # (b, 8, 2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # (b, 16, 5, 5)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # (b, 8, 15, 15)
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # (b, 1, 28, 28)
            nn.Tanh()
        )
    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode
    
    
class AutoEncoderCNN_STL10(nn.Module):
    def __init__(self):
        super(AutoEncoderCNN_STL10, self).__init__()
        self.Encoder = nn.Sequential(
            # param [input_c, output_c, kernel_size, stride, padding]
            nn.Conv2d(3, 64, 3, 1, 1),   # [, 64, 96, 96]
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1), # [, 64, 96, 96]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),             # [, 64, 48, 48]
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1),  # [, 64, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 1, 1), # [, 128, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, 1, 1), # [, 128, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, 1, 1), # [, 256, 48, 48]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                 # [, 256, 24, 24]
            nn.BatchNorm2d(256)   
        )
        
        # decoder
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3 ,1, 1),   # [, 128, 24, 24]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 3, 2, 1, 1),   # [, 128, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),    # [, 64, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, 1, 1),      # [, 32, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, 1, 1),      # [, 32, 48, 48]
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),  # [, 16, 96, 96]
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, 3, 1, 1),         # [, 3, 96, 96]
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoder = self.Encoder(x)
        decoder = self.Decoder(encoder)
        return encoder, decoder