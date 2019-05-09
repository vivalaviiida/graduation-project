## import models

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torchvision.models as models


n_channel = 3
n_disc = 16
n_gen = 64
n_encode = 64
n_l = 10
n_z = 50
img_size = 128
# batchSize = 20
use_cuda = torch.cuda.is_available()
n_age = int(n_z/n_l)
n_gender = int(n_z/2)
kernel_size = (3, 7)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

class Encoder_Plus(nn.Module):
    def __init__(self):
        super(Encoder_Plus,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels=64, kernel_size =5, stride=2, padding = 2),
            nn.ReLU())
        self.block1 = ResidualBlock(64)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels = 64, out_channels=128, kernel_size =5, stride=2, padding = 2),
            nn.ReLU())

        layers = []
        for i in range(2):
            layers.append(ResidualBlock(128))
        self.block2 = nn.Sequential(*layers)
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels=256, kernel_size =5, stride=2, padding = 2),
            nn.ReLU())

        layers = []
        for i in range(4):
            layers.append(ResidualBlock(256))
        self.block3 = nn.Sequential(*layers)

        self.conv4 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels=512, kernel_size =5, stride=2, padding = 2),
            nn.ReLU())
        # self.conv4 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels=512, kernel_size =(3, 7), stride=2, padding = 2),
        #     nn.ReLU())

        layers = []
        for i in range(3):
            layers.append(ResidualBlock(512))
        self.block4 = nn.Sequential(*layers)

        # self.fc = nn.Linear(512*9*7, 50)
        self.fc = nn.Linear(512*8*8, 50)

    def forward(self,input):
        output = self.conv1(input)
        output = self.block1(output)
        output = self.conv2(output)
        output = self.block2(output)
        output = self.conv3(output)
        output = self.block3(output)
        output = self.conv4(output)
        output = self.block4(output)
        # output = output.view(-1, 512*9*7)
        output = output.view(-1, 512*8*8)
        output = self.fc(output)
        return output

class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer = 11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)

class Encoder_advanced(nn.Module):
    def __init__(self):
        super(Encoder_advanced,self).__init__()
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

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.conv = nn.Sequential(
            #input: 3*128*128
            nn.Conv2d(n_channel,n_encode,5,2,2),
            nn.ReLU(),

            nn.Conv2d(n_encode,2*n_encode,5,2,2),
            nn.ReLU(),

            nn.Conv2d(2*n_encode,4*n_encode,5,2,2),
            nn.ReLU(),

            nn.Conv2d(4*n_encode,8*n_encode,5,2,2),
            nn.ReLU(),

        )
        self.fc = nn.Linear(8*n_encode*8*8,50)

    def forward(self,x):
        conv = self.conv(x).view(-1,8*n_encode*8*8)
        out = self.fc(conv)
        return out

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x)
        #print(proj_query.size())
        proj_query = proj_query.view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        #print(proj_query.size())
        proj_key =  self.key_conv(x)
        #print(proj_key.size())
        proj_key = proj_key.view(m_batchsize,-1,width*height) # B X C x (*W*H)
        #print(proj_key.size())
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.fc = nn.Sequential(nn.Linear(n_z+n_l*n_age+n_gender,
                                          8*8*n_gen*16),
                                nn.ReLU())
        self.upconv1= nn.Sequential(
            nn.ConvTranspose2d(16*n_gen,8*n_gen,4,2,1),
            nn.ReLU(),

            nn.ConvTranspose2d(8*n_gen,4*n_gen,4,2,1),
            nn.ReLU(),

            nn.ConvTranspose2d(4*n_gen,2*n_gen,4,2,1),
            nn.ReLU(),

            # nn.ConvTranspose2d(2*n_gen,n_gen,4,2,1),
            # nn.ReLU(),

            # nn.ConvTranspose2d(n_gen,n_channel,3,1,1),
            # nn.Tanh(),

        )

        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(2*n_gen,n_gen,4,2,1),
            nn.ReLU(),)
        
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(n_gen,n_channel,3,1,1),
            nn.Tanh(),
        )

        self.Attn1 = Self_Attn(128, 'relu')
        self.Attn2 = Self_Attn(64, 'relu')
        

    def forward(self,z,age,gender):
        ## duplicate age & gender conditions as descripted in https://github.com/ZZUTK/Face-Aging-CAAE
        l = age.repeat(1,n_age).float()
        k = gender.view(-1,1).repeat(1,n_gender).float()

        x = torch.cat([z,l,k],dim=1)
        fc = self.fc(x).view(-1,16*n_gen,8,8)
        out = self.upconv1(fc)
        out,_ = self.Attn1(out)
        out = self.upconv2(out)
        out,_ =self.Attn2(out)
        out = self.upconv3(out)
        return out


class Dimg(nn.Module):
    def __init__(self):
        super(Dimg,self).__init__()
        self.conv_img = nn.Sequential(
            nn.Conv2d(n_channel,n_disc,4,2,1),
        )
        self.conv_l = nn.Sequential(
            nn.ConvTranspose2d(n_l*n_age+n_gender, n_l*n_age+n_gender, 64, 1, 0),
            nn.ReLU()
        )
        self.total_conv = nn.Sequential(
            nn.Conv2d(n_disc+n_l*n_age+n_gender,n_disc*2,4,2,1),
            nn.ReLU(),

            nn.Conv2d(n_disc*2,n_disc*4,4,2,1),
            nn.ReLU(),

            nn.Conv2d(n_disc*4,n_disc*8,4,2,1),
            nn.ReLU()
        )

        self.fc_common = nn.Sequential(
            nn.Linear(8*8*img_size,1024),
            nn.ReLU()
        )
        self.fc_head1 = nn.Sequential(
            nn.Linear(1024,1),
            nn.Sigmoid()
        )
        self.fc_head2 = nn.Sequential(
            nn.Linear(1024,n_l),
            nn.Softmax()
        )

    def forward(self,img,age,gender):
        ## duplicate age & gender conditions as descripted in https://github.com/ZZUTK/Face-Aging-CAAE
        l = age.repeat(1,n_age,1,1,)
        k = gender.repeat(1,n_gender,1,1,)
        conv_img = self.conv_img(img)
        conv_l   = self.conv_l(torch.cat([l,k],dim=1))
        catted   = torch.cat((conv_img,conv_l),dim=1)
        total_conv = self.total_conv(catted).view(-1,8*8*img_size)
        body = self.fc_common(total_conv)

        head1 = self.fc_head1(body)
        head2 = self.fc_head2(body)

        return head1,head2


class Dz(nn.Module):
    def __init__(self):
        super(Dz,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_z,n_disc*4),
            nn.ReLU(),

            nn.Linear(n_disc*4,n_disc*2),
            nn.ReLU(),

            nn.Linear(n_disc*2,n_disc),
            nn.ReLU(),

            nn.Linear(n_disc,1),
            nn.Sigmoid()
        )
    def forward(self,z):
        return self.model(z)
