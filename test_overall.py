from dataloader import *
from models import *
import torch
from torch import nn
import os
import pickle
from scipy.misc import imread
from skimage import io, transform
import torch._utils
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
origin_G_path = '/home/annalei/WJX/new-CAAE/result_tv_gender/netG_041.pth'
advanced_E_path = '/home/annalei/WJX/new-CAAE/result_tv_gender_label/netE_041.pth'
advanced_G_path = '/home/annalei/WJX/new-CAAE/result_tv_gender_label/netG_041.pth'

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

dataloader = loadImgs()
data_iter = iter(dataloader)
img_data, img_label = data_iter.next()
img_age = img_label/2
img_gender = img_label%2*2-1

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


fixed_l = -torch.ones(80*10).view(80,10)
for i,l in enumerate(fixed_l):
    l[i//8] = 1

fixed_l_v = Variable(fixed_l)

if use_cuda:
    fixed_l_v = fixed_l_v.cuda()

outf='./result_tv_gender_kkk'

if not os.path.exists(outf):
    os.mkdir(outf)

img_data_v = Variable(img_data)
img_age_v = Variable(img_age).view(-1,1)
img_gender_v = Variable(img_gender.float())

fixed_noise = img_data[:8].repeat(10,1,1,1)
print(img_data.type())
print(img_data.size())
print(fixed_noise.type())
print(fixed_noise.data.size())
fixed_g = img_gender[:8].view(-1,1).repeat(10,1)


fixed_img_v = Variable(fixed_noise)
fixed_g_v = Variable(fixed_g)

pickle.dump(fixed_noise,open("fixed_noise.p","wb"))

if use_cuda:
    fixed_img_v = fixed_img_v.cuda()
    fixed_g_v = fixed_g_v.cuda()

vutils.save_image(fixed_img_v.data,
    '%s/input.png'%(outf),
    normalize=True)
z = io.imread('%s/input.png'%(outf))
loader = transforms.Compose([transforms.ToTensor()])
tensor = loader(z)
print(tensor.size())
lan = torch.randn(80, 3, 129, 129)
lan[1] = tensor[:, 0:129, 0:129]
print(lan[1].size())
vutils.save_image(lan[1], 'aaa.png', normalize=True)
if use_cuda:
    img_data_v = img_data_v.cuda()
    img_age_v = img_age_v.cuda()

z_origin = Encoder_origin(fixed_img_v)
z_advanced = Encoder_Plus(fixed_img_v)
fake_origin = Generator_origin(z_origin,fixed_l_v,fixed_g_v)
fake_advanced = Generator_advanced(z_advanced,fixed_l_v,fixed_g_v)
vutils.save_image(fake_origin.data, '%s/origin.png'%(outf), normalize=True)
vutils.save_image(fake_advanced.data, '%s/advanced_label.png'%(outf), normalize=True)

