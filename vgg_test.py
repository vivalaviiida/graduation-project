import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
from PIL import ImageFile
from dataloader import *
import torch
from torch import nn
import os
from torch.autograd import Variable
from models_mine import *
from misc import *
from loss import *

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


class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.feature_map = nn.Sequential(*list(model.features)[:37]).eval()

    def forward(self, x):
        x = self.feature_map(x)
        return x

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
dataloader = loadImgs()
data_iter = iter(dataloader)
img, label = data_iter.next()
age = label/2
gender = label%2*2-1
vgg19 = models.vgg19(pretrained=True).cuda()

for j in range(batchSize):
    img[j] = normalize(img[j])

G_N = GeneratorLoss().cuda()
net = G_N.loss_network
print(net)
img = img.cuda()
img_v = Variable(img).cuda()
vutils.save_image(img_v.data,
                'pths/img_v.png',
                normalize=True)

img_fm = Variable(net(img_v))
print(img_fm.size())
net_E = Encoder().cuda()
net_E.load_state_dict(torch.load('/home/annalei/WJX/new-CAAE/pths/netE_041.pth'))
z = net_E(img_v)


age_ohe = one_hot(age,batchSize,n_l,use_cuda)
gender_v = Variable(gender.float()).cuda()
net_G = Generator().cuda()
net_G.load_state_dict(torch.load('/home/annalei/WJX/Face-Aging-CAAE-Pytorch-master/result_tv_gender/netG_041.pth'))
reconst = net_G(z, age_ohe, gender_v)
vutils.save_image(reconst.data,
                'pths/reconst.png',
                normalize=True)
reconst_fm = Variable(net(reconst))
print(reconst_fm.size())

L1 = nn.L1Loss().cuda()
loss = L1(img_fm, reconst_fm)
print(loss.data[0])