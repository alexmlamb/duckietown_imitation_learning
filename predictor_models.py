import torch
import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self, num_images_back, num_targets_forward, num_targets_back):
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
            nn.Linear(512*5*4 + 2*num_targets_back, 1024),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.5),
            nn.Linear(1024,2*num_targets_forward))

    def forward(self,image,targets_past):
        out = self.conv_part(image).resize(image.size(0),512*5*4)

        out = torch.cat((out, targets_past),1)

        out = self.fc_part(out)

        return out

