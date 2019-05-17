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
import models_origin
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

origin_E_path = '/home/annalei/WJX/new-CAAE/result_tv_gender/netE_041.pth'
VGG_E_path = '/home/annalei/WJX/new-CAAE/result_tv_gender_VGG/netE_041.pth'
encoder_E_path = '/home/annalei/WJX/new-CAAE/result_tv_gender_encoder/netE_041.pth'
encoderP_E_path = '/home/annalei/WJX/new-CAAE/result_tv_gender_encoder_plus/netE_041.pth'
overall_E_path = '/home/annalei/WJX/new-CAAE/result_tv_gender_overall/netE_041.pth'
# Attn_E_path = '/home/annalei/WJX/new-CAAE/result_lowAttn/netE_041.pth'

origin_G_path = '/home/annalei/WJX/new-CAAE/result_tv_gender/netG_041.pth'
VGG_G_path = '/home/annalei/WJX/new-CAAE/result_tv_gender_VGG/netG_041.pth'
encoder_G_path = '/home/annalei/WJX/new-CAAE/result_tv_gender_encoder/netG_041.pth'
encoderP_G_path = '/home/annalei/WJX/new-CAAE/result_tv_gender_encoder_plus/netG_041.pth'
overall_G_path = '/home/annalei/WJX/new-CAAE/result_tv_gender_overall/netG_041.pth'
# Attn_G_path = '/home/annalei/WJX/new-CAAE/result_lowAttn/netG_041.pth'

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

dataloader = loadImgs(batchSize=batchSize)
data_iter = iter(dataloader)


Encoder_origin = Encoder()
Encoder_VGG = Encoder()
Encoder_advanced = Encoder_advanced()
Encoder_P = Encoder_Plus()
Encoder_overall = models_origin.Encoder_Plus()
# Encoder_Plus()
#Encoder_P()
Generator_origin = Generator()
Generator_VGG = Generator()
Generator_advanced = Generator()
Generator_Plus = Generator()
Generator_overall = Generator()

if use_cuda:
    Encoder_origin = Encoder_origin.cuda()
    Encoder_VGG = Encoder_VGG.cuda()
    Encoder_advanced = Encoder_advanced.cuda()
    Encoder_P = Encoder_P.cuda()
    Encoder_overall = Encoder_overall.cuda()
    
    Generator_origin = Generator_origin.cuda()
    Generator_VGG = Generator_VGG.cuda()  
    Generator_advanced = Generator_advanced.cuda()  
    Generator_Plus = Generator_Plus.cuda()
    Generator_overall = Generator_overall.cuda()

Encoder_origin.load_state_dict(torch.load(origin_E_path))
Encoder_VGG.load_state_dict(torch.load(VGG_E_path))
Encoder_advanced.load_state_dict(torch.load(encoder_E_path))
Encoder_P.load_state_dict(torch.load(encoderP_E_path))
Encoder_overall.load_state_dict(torch.load(overall_E_path))

Generator_origin.load_state_dict(torch.load(origin_G_path))
Generator_VGG.load_state_dict(torch.load(VGG_G_path))
Generator_advanced.load_state_dict(torch.load(encoder_G_path))
Generator_Plus.load_state_dict(torch.load(encoderP_G_path))
Generator_overall.load_state_dict(torch.load(overall_G_path))

for i in range(20):
    img_data, img_label = data_iter.next()
    img_age = img_label/2
    img_gender = img_label%2*2-1

    print(img_age)

    # fixed_l = -torch.ones(80*10).view(80,10)
    # for i,l in enumerate(fixed_l):
    #     l[i//8] = 1

    # fixed_l_v = Variable(fixed_l)

    # if use_cuda:
    #     fixed_l_v = fixed_l_v.cuda()

    outf='./result_openface/imgs'

    if not os.path.exists(outf):
        os.mkdir(outf)

    img_data_v = Variable(img_data)
    img_age_v = Variable(img_age).view(-1,1)
    img_gender_v = Variable(img_gender.float())
    if use_cuda:
        img_data_v = img_data_v.cuda()
        img_age_v = img_age_v.cuda()
        img_gender_v = img_gender_v.cuda()
    batchSize = img_data_v.size(0)
    age_ohe = one_hot(img_age,batchSize,n_l,use_cuda)
    age_v = Variable(age_ohe)
    if use_cuda:
        age_v = age_v.cuda()
   
    for j in range(batchSize):
        vutils.save_image(img_data_v[j].data, '%s/input%03d.png'%(outf, i*batchSize+j), normalize=True)

    z_origin = Encoder_origin(img_data_v)
    z_VGG = Encoder_VGG(img_data_v)
    z_encoder = Encoder_advanced(img_data_v)
    z_encoderP = Encoder_P(img_data_v)
    z_overall = Encoder_overall(img_data_v)
    fake_origin = Generator_origin(z_origin,age_v,img_gender_v)
    fake_VGG = Generator_VGG(z_VGG,age_v,img_gender_v)
    fake_encoder = Generator_advanced(z_encoder,age_v,img_gender_v)
    fake_encoderP = Generator_Plus(z_encoderP,age_v,img_gender_v)
    fake_overall = Generator_overall(z_overall,age_v,img_gender_v)
    for j in range(batchSize):
        vutils.save_image(fake_origin[j].data, '%s/origin%03d.png'%(outf, i*batchSize+j), normalize=True)
        vutils.save_image(fake_VGG[j].data, '%s/VGG%03d.png'%(outf, i*batchSize+j), normalize=True)
        vutils.save_image(fake_encoder[j].data, '%s/encoder%03d.png'%(outf, i*batchSize+j), normalize=True)
        vutils.save_image(fake_encoderP[j].data, '%s/encoder_plus%03d.png'%(outf, i*batchSize+j), normalize=True)
        vutils.save_image(fake_overall[j].data, '%s/overall%03d.png'%(outf, i*batchSize+j), normalize=True)
