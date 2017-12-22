import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np

class View(nn.Module):
    def __init__(self, size):
        super(View,self).__init__()
        self.size = size
    def forward(self, input):
        return input.view(self.size)

class CNN_VAE(nn.Module):
    def __init__(self):
        super(CNN_VAE,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1,12,5,padding=2), 
            nn.ELU(), 
            nn.MaxPool3d(kernel_size=4,stride=3),
            nn.Conv3d(12,24,5,padding=2),
            nn.ELU(),
            nn.MaxPool3d(kernel_size=4,stride=3),
            View([-1,3000]),
            nn.Linear(3000,2000),
            nn.ELU(),
            nn.Linear(2000,2000),
            )
        self.decoder = nn.Sequential(
            nn.Linear(1000,2000),
            nn.ELU(),
            nn.Linear(2000,3000),
            nn.ELU(),
            View([-1,24,5,5,5]),
            nn.Upsample(size=[16,16,16],mode='trilinear'),
            nn.Conv3d(24,12,5,padding=2),
            nn.ELU(),
            nn.Upsample(size=[50,50,50],mode='trilinear'),
            nn.Conv3d(12,1,5,padding=2),
            )
    def forward(self, input):
        output = input
        output = self.encoder(output)
        mu, log_std = output[:,:1000], output[:,1000:]
        output = Variable(torch.randn(mu.size()))*log_std.exp() + mu
        output = self.decoder(output)
        return output, mu, log_std
    def calculate_loss(self, input):
        output, mu, log_std = self.forward(input)
        #decoder loss
        loss1 = nn.functional.binary_cross_entropy_with_logits(output,input)
        #latent loss
        loss2 = -0.5*(1+log_std-mu**2-log_std.exp())
        loss2 = loss2.mean()
        return loss1,loss2
        
