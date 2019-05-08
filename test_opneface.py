from dataloader import *
from models import *
import torch
from torch import nn
import os
import pickle
from scipy.misc import imread
from skimage import io, transform
import torch._utils
from misc import *
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

outf='./result_openface'
if not os.path.exists(outf):
    os.mkdir(outf)
origin_E_path = '/home/annalei/WJX/new-CAAE/result_tv_gender/netE_041.pth'
origin_G_path = '/home/annalei/WJX/new-CAAE/result_tv_gender/netG_041.pth'
advanced_E_path = '/home/annalei/WJX/new-CAAE/result_tv_gender_overall/netE_041.pth'
advanced_G_path = '/home/annalei/WJX/new-CAAE/result_tv_gender_overall/netG_041.pth'

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

dataloader = loadImgs(batchSize=1)
data_iter = iter(dataloader)
img_data, img_label = data_iter.next()
img_age = img_label/2
img_gender = img_label%2*2-1

img_data_v = Variable(img_data)
img_gender_v = Variable(img_gender.float())
if use_cuda:
    img_data_v = img_data_v.cuda()
    img_gender_v = img_gender_v.cuda()

vutils.save_image(img_data_v.data,
    '%s/input.png'%(outf),
    normalize=True)

batchSize = img_data_v.size(0)
age_ohe = one_hot(img_age,batchSize,n_l,use_cuda)

Encoder_origin = Encoder()
Encoder_Plus = Encoder_Plus()        
#Encoder_Plus()
Generator_origin = Generator()
Generator_advanced = Generator()

if use_cuda:
    Encoder_origin = Encoder_origin.cuda()
    Encoder_Plus = Encoder_Plus.cuda()
    Generator_origin = Generator_origin.cuda()
    Generator_advanced = Generator_advanced.cuda()

Encoder_origin.load_state_dict(torch.load(origin_E_path))
Generator_origin.load_state_dict(torch.load(origin_G_path))
Encoder_Plus.load_state_dict(torch.load(advanced_E_path))
Generator_advanced.load_state_dict(torch.load(advanced_G_path))

z1 = Encoder_origin(img_data_v)
g1 = Generator_origin(z1, age_ohe, img_gender_v)
vutils.save_image(g1.data,
                '%s/one_pic.png' % (outf),
                normalize=True)
z2 = Encoder_Plus(img_data_v)
g2 = Generator_advanced(z2, age_ohe, img_gender_v)
vutils.save_image(g2.data,
                '%s/two_pic.png' % (outf),
                normalize=True)