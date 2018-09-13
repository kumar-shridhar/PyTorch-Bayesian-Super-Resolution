import torch.nn as nn
import torch.nn.init as init
from utils.BBBlayers import BBBConv2d

class BBBNet(nn.Module):
    def __init__(self, upscale_factor ):
        super(BBBNet, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = BBBConv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = BBBConv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = BBBConv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = BBBConv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        layers = [self.conv1, self.relu, self.conv2, self.relu, self.conv3,
                  self.relu, self.conv4, self.relu, self.pixel_shuffle]

        self.layers = nn.ModuleList(layers)


       # self._initialize_weights()


    def probforward(self, x):
        'Forward pass with Bayesian weights'
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        #print('logits', logits)
        return logits, kl

'''
    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)
'''