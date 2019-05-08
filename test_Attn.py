import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from dataloader import *
from misc import *
from models_Attn import *
import pickle
from makeLabel import *
import os
import datetime
import torchvision
from loss import *

use_cuda = torch.cuda.is_available()

makeDir()
moveFiles()

if use_cuda:
    # netE = Encoder_advanced().cuda()
    netE = Encoder_Plus().cuda()
    netD_img = Dimg().cuda()
    netD_z  = Dz().cuda()
    netG = Generator().cuda()
    feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True)).cuda()
else:
    # nete = Encoder_advanced()
    netE = Encoder_Plus()
    netD_img = Dimg()
    netD_z  = Dz()
    netG = Generator()
    feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))

netE.apply(weights_init)
netD_img.apply(weights_init)
netD_z.apply(weights_init)
netG.apply(weights_init)

fixed_l = -torch.ones(80*10).view(80,10)
for i,l in enumerate(fixed_l):
    l[i//8] = 1

fixed_l_v = Variable(fixed_l)

if use_cuda:
    fixed_l_v = fixed_l_v.cuda()

dataloader = loadImgs()
data_iter = iter(dataloader)
img_data, img_label = data_iter.next()
img_age = img_label/2
img_gender = img_label%2*2-1

fixed_noise = img_data[:8].repeat(10,1,1,1)
fixed_img_v = Variable(fixed_noise)

if use_cuda:
    fixed_img_v = fixed_img_v.cuda()

fixed_l = -torch.ones(80*10).view(80,10)
for i,l in enumerate(fixed_l):
    l[i//8] = 1

fixed_l_v = Variable(fixed_l)

if use_cuda:
    fixed_l_v = fixed_l_v.cuda()

fixed_g = img_gender[:8].view(-1,1).repeat(10,1)
fixed_g_v = Variable(fixed_g)
if use_cuda:
    fixed_g_v = fixed_g_v.cuda()

z = netE(fixed_img_v)
reconst_Attn = netG(z,fixed_l_v,fixed_g_v)
print(reconst_Attn.size())