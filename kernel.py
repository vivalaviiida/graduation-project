import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torchvision.models as models
from dataloader import *


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

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.conv = nn.Sequential(
            #input: 3*128*128

            nn.Conv2d(n_channel,n_encode,kernel_size,2,2),
            nn.ReLU(),

            nn.Conv2d(n_encode,2*n_encode,kernel_size,2,2),
            nn.ReLU(),

            nn.Conv2d(2*n_encode,4*n_encode,kernel_size,2,2),
            nn.ReLU(),

            nn.Conv2d(4*n_encode,8*n_encode,kernel_size,2,2),
            nn.ReLU(),

        )
        # self.fc = nn.Linear(8*n_encode*8*8,50)
        self.fc = nn.Linear(35840,50)

    def forward(self,x):
        # conv = self.conv(x).view(-1,8*n_encode*8*8)
        conv = self.conv(x).view(-1, 35840)
        out = self.fc(conv)
        return out

enc = Encoder()
input = torch.randn(20, 3, 128, 128)

output = enc(input)
print(output.size())
# output = enc.conv(input)
# print(output.size())
# output = output.view(-1,8*n_encode*8*8)
# print(output.size())
# fc = nn.Linear(716800, 50)
# output = fc(output)
# print(output.size())
