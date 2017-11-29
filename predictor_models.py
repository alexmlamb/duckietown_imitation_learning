import torch
import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self, num_images_back, num_targets_forward):
        super(Predictor, self).__init__()


        self.conv_part = nn.Sequential(
            nn.Conv2d(num_images_back*3, 64, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02),
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02),
            nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02),
            nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.02))

        self.fc_part = nn.Sequential(
            nn.Linear(512*5*4, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.02),
            nn.Linear(1024,2*num_targets_forward))

    def forward(self,inp):
        conv = self.conv_part(inp).resize(inp.size(0),512*5*4)

        out = self.fc_part(conv)

        return out

