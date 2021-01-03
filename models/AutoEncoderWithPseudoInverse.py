from torch import nn
import torch


def get_pseudo_inverse(a):
    u, s, v = torch.svd(a)
    for i in range(len(s)):
        if s[i] != 0:
            s[i] = 1.0 / s[i]
    pseudo_inverse = torch.mm(torch.mm(v, torch.diag(s).t()), u.t())
    return pseudo_inverse


class AutoEncoderWithPseudoInverse(nn.Module):
    def __init__(self):
        super(AutoEncoderWithPseudoInverse, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(28*28, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, 64),
                                     nn.ReLU(True),
                                     nn.Linear(64, 12),
                                     nn.ReLU(True),
                                     nn.Linear(12, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 12),
                                     nn.ReLU(True),
                                     nn.Linear(12, 64),
                                     nn.ReLU(True),
                                     nn.Linear(64, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, 28*28),
                                     nn.Tanh())

    def forward(self, x):
        encoded = self.encoder(x)
        self.decoder[6].weight.data = torch.pinverse(self.encoder[0].weight.data)
        self.decoder[4].weight.data = torch.pinverse(self.encoder[2].weight.data)
        self.decoder[2].weight.data = torch.pinverse(self.encoder[4].weight.data)
        self.decoder[0].weight.data = torch.pinverse(self.encoder[6].weight.data)
        decoded = self.decoder(encoded)
        return encoded, decoded
