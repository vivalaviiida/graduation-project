import os,random, shutil

# count=0
# for root, dirs, files in os.walk('/home/annalei/WJX/CACD'):
#     count+=1
#     print(root)

# print(count)
fileDir = '/home/annalei/UTKFace'
pathDir = os.listdir(fileDir)
filenumber = len(pathDir)
rate = 0.1
picknumber = int(filenumber*rate)
sample = random.sample(pathDir, picknumber)
print(len(sample))