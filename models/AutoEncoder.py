from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
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
        # self.decoder[6].weight.data = self.encoder[0].weight.data.transpose(0, 1)
        # self.decoder[4].weight.data = self.encoder[2].weight.data.transpose(0, 1)
        # self.decoder[2].weight.data = self.encoder[4].weight.data.transpose(0, 1)
        # self.decoder[0].weight.data = self.encoder[6].weight.data.transpose(0, 1)
        decoded = self.decoder(encoded)
        return encoded, decoded