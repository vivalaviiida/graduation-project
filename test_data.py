import os
from dataloader import *
from misc import *
from models import *
import pickle
from makeLabel import *
import os
import datetime
import torchvision
from loss import *

# dataloader = loadImgs('./UTKFace', batchSize=1)
# for (img_data, img_label) in dataloader:
#     vutils.save_image(img_data, "./result_tv_gender_kkk/a.png", normalize=True)
#     print(img_label)
#     print(img_label/2)
#     age_ohe = one_hot(img_label, 1, 10, False)
#     print(age_ohe)
#     input("press Enter to continue...")

imgFiles = [file for file in os.listdir("./UTKFace")]

def encodeAge(n):
    if n<=5:
        return 0
    elif n<=10:
        return 1
    elif n<=15:
        return 2
    elif n<=20:
        return 3
    elif n<=30:
        return 4
    elif n<=40:
        return 5
    elif n<=50:
        return 6
    elif n<=60:
        return 7
    elif n<=70:
        return 8
    else:
        return 9

for i in range(20):
    print(format(i, ">02"))

for file in imgFiles:
    lst = file.split("_")
    print(lst)
    age = int(lst[0])
    gender = int(lst[1])
    folder = format(encodeAge(age)*2 + gender,">02")
    print(encodeAge(age)*2+gender)
    print(folder)
    input("Press Enter to continue...")