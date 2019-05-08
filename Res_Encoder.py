import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable


n_channel = 3
n_disc = 16
n_gen = 64
n_encode = 64
n_l = 10
n_z = 50
img_size = 128
batchSize = 20
use_cuda = torch.cuda.is_available()
n_age = int(n_z/n_l)
n_gender = int(n_z/2)
kernel_size = (3, 7)

def swish(x):
    return x * F.sigmoid(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

class Encoder_Plus(nn.Module):
    def __init__(self):
        super(Encoder_Plus,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels=64, kernel_size =5, stride=2, padding = 2)
        self.block1 = ResidualBlock(64)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels=128, kernel_size =5, stride=2, padding = 2)

        layers = []
        for i in range(2):
            layers.append(ResidualBlock(128))
        self.block2 = nn.Sequential(*layers)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels=256, kernel_size =5, stride=2, padding = 2)

        layers = []
        for i in range(4):
            layers.append(ResidualBlock(256))
        self.block3 = nn.Sequential(*layers)

        self.conv4 = nn.Conv2d(in_channels = 256, out_channels=512, kernel_size =(3, 7), stride=2, padding = 2)

        layers = []
        for i in range(3):
            layers.append(ResidualBlock(512))
        self.block4 = nn.Sequential(*layers)

        self.fc = nn.Linear(512*9*7, 50)

    def forward(self,x):
        output = self.conv1(input)
        output = self.block1(output)
        output = self.conv2(output)
        output = self.block2(output)
        output = self.conv3(output)
        output = self.block3(output)
        output = self.conv4(output)
        output = self.block4(output)
        output = output.view(-1, 512*9*7)
        output = self.fc(output)
        return output

enc = Encoder_Plus().cuda()
input = torch.randn(20, 3, 128, 128).cuda()
output = enc(input)
print(output.size())
# conv1 = nn.Conv2d(in_channels = 3, out_channels=64, kernel_size =5, stride=2, padding = 2)
# block1 = ResidualBlock(64)
# conv2 = nn.Conv2d(in_channels = 64, out_channels=128, kernel_size =5, stride=2, padding = 2)

# layers = []
# for i in range(2):
#     layers.append(ResidualBlock(128))
# block2 = nn.Sequential(*layers)
# conv3 = nn.Conv2d(in_channels = 128, out_channels=256, kernel_size =5, stride=2, padding = 2)

# layers = []
# for i in range(4):
#     layers.append(ResidualBlock(256))
# block3 = nn.Sequential(*layers)

# conv4 = nn.Conv2d(in_channels = 256, out_channels=512, kernel_size =(3, 7), stride=2, padding = 2)

# layers = []
# for i in range(3):
#     layers.append(ResidualBlock(512))
# block4 = nn.Sequential(*layers)

# fc = nn.Linear(512*9*7, 50)

# output = conv1(input)
# print(output.size())
# output = block1(output)
# print(output.size())
# output = conv2(output)
# print(output.size())
# output = block2(output)
# print(output.size())
# output = conv3(output)
# print(output.size())
# output = block3(output)
# print(output.size())
# output = conv4(output)
# print(output.size())
# output = block4(output)
# print(output.size())
# output = output.view(-1, 512*9*7)
# output = fc(output)
# print(output.size())