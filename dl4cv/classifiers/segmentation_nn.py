"""SegmentationNN"""
import torch
import torch.nn as nn
from torchvision import models
import numpy as np

class SegmentationNN(nn.Module):

    
    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        #model = models.alexnet(pretrained = True)
        model = models.vgg16(pretrained= True)
        feats, cls = list(model.features.children()), list(model.classifier.children())

        feats[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(100, 100));
        
        #feats(0).padding(100, 100)

        for i, f in enumerate(feats):
            f.requires_grad = False
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True
            
        self.features5 = nn.Sequential(*feats)

        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        fc6.weight.data.copy_(cls[0].weight.data.view(4096, 512, 7, 7))
        fc6.bias.data.copy_(cls[0].bias.data)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        fc7.weight.data.copy_(cls[3].weight.data.view(4096, 4096, 1, 1))
        fc7.bias.data.copy_(cls[3].bias.data)
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1, padding = 0)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()
        self.score_fr = nn.Sequential(
            fc6, 
            nn.ReLU(inplace=True), 
            nn.Dropout(), 
            fc7, 
            nn.ReLU(inplace=True), 
            nn.Dropout(), 
            score_fr
        )
        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
    
        
    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################

        x_size = x.size()
        pool5 = self.features5(x)
        score_fr = self.score_fr(pool5)
        #upscore = self.upscore(score_fr)
        #x = upscore[:, :, 19: (19 + x_size[2]), 19: (19 + x_size[3])].contiguous()
        deconv = nn.Upsample(size = x_size[2:], mode = 'bilinear')
        upscore = deconv(score_fr)
        x = upscore
        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

