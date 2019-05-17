import time
start = time.time()
# import torch
import cv2
import itertools
import os
import numpy as np
import openface
import argparse
from log import logger

dlib_model_dir = '/home/annalei/openface/models/dlib'
openface_model_dir = '/home/annalei/openface/models/openface'
img_dir = '/home/annalei/WJX/new-CAAE/evaluation/Attn_imgs'

parser = argparse.ArgumentParser()
# parser.add_argument('imgs', type=str, nargs='+', help='Input images.')
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.", default=os.path.join(dlib_model_dir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.", default=os.path.join(openface_model_dir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int, help="Default image dimension.", default=96)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

if args.verbose:
    logger.info("Argument parsing and loading libraries took {} seconds.".format(time.time() - start))

start = time.time()
align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, args.imgDim)
if args.verbose:
    logger.info("Loading the dlib and OpenFace models took {} seconds.".format(
        time.time() - start))

def getRep(imgPath):
    if args.verbose:
        logger.info("Processing {}.".format(imgPath))
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
            raise Exception("Unable to load image: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
            logger.info("Original size: {}".format(rgbImg.shape))   

    start = time.time()
    faceBoundingBox = align.getLargestFaceBoundingBox(rgbImg)
    if faceBoundingBox is None:
        raise Exception("Unable to find a face: {}".format(imgPath))
    if args.verbose:
        logger.info("Face detection took {} seconds.".format(time.time() - start))

    start = time.time()
    alignedFace = align.align(args.imgDim, rgbImg, faceBoundingBox, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)     
    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))
    if args.verbose:
        logger.info("Face alignment took {} seconds.".format(time.time() - start))

    start = time.time()
    rep = net.forward(alignedFace)  
    if args.verbose:
        logger.info("OpenFace forward pass took {} seconds.".format(time.time()-start))
        logger.info("Representation:")
        logger.info(rep)

    return rep

# for (img1, img2) in itertools.combinations(args.imgs, 2):
#     distance = getRep(img1) - getRep(img2)
#     logger.info("Comparing {} with {}.".format(img1, img2))
#     logger.info("Squared l2 distance between representations: {:0.3f}".format(np.dot(distance, distance)))
print('aaa')
fail = [0 for i in range(2)]
diff = [0 for i in range(2)]
dist_avg = [0 for i in range(2)]
# logger.info("dist_avg[{:03d}]: {:03d}".format(i, int(dist_avg[i])))
# logger.info("    | distance_origin | distance_VGG| distance_encoder | distance_encoderP| distance_overall |")   
logger.info("    |distance_overall | distance_Attn")   

for i in range(400):
    img_input='%s/input%03d.png'%(img_dir,i)
    # img_origin='result_openface/imgs/origin%03d.png'%(i)
    # img_VGG='result_openface/imgs/VGG%03d.png'%(i)
    # img_encoder='result_openface/imgs/encoder%03d.png'%(i)
    # img_encoder_plus='result_openface/imgs/encoder_plus%03d.png'%(i)
    img_overall='%s/overall%03d.png'%(img_dir,i)
    img_Attn = '%s/Attn%03d.png'%(img_dir, i)

    # distance_overall = getRep(img_input) - getRep(img_overall)
    # distance_Attn = getRep(img_input) - getRep(img_Attn)
    
    try:
        # distance_origin = getRep(img_input) - getRep(img_origin)
        # distance_VGG = getRep(img_input) - getRep(img_VGG)
        # distance_encoder = getRep(img_input) - getRep(img_encoder)
        # distance_encoder_plus = getRep(img_input) - getRep(img_encoder_plus)
        distance_overall = getRep(img_input) - getRep(img_overall)
        distance_Attn = getRep(img_input) - getRep(img_Attn)
    except:
        # try:
        #     getRep(img_origin)
        # except:
        #     fail[0]+=1
        # try:
        #     getRep(img_VGG)
        # except:
        #     fail[1]+=1
        # try:
        #     getRep(img_encoder)
        # except:
        #     fail[2]+=1
        # try:
        #     getRep(img_encoder_plus)
        # except:
        #     fail[3]+=1
        try:
            getRep(img_overall)
        except:
            fail[0]+=1
        # print("round %d fail."%(i))
        try:
            getRep(img_Attn)
        except:
            fail[1]+=1
        print("round %d fail."%(i))
        continue
    # l2_origin=np.dot(distance_origin, distance_origin)
    # dist_avg[0]+=l2_origin
    # if l2_origin>1.5:
    #     diff[0]+=1
    # l2_VGG=np.dot(distance_VGG, distance_VGG)
    # dist_avg[1]+=l2_VGG
    # if l2_VGG>1.5:
    #     diff[1]+=1
    # l2_encoder=np.dot(distance_encoder, distance_encoder)
    # dist_avg[2]+=l2_encoder
    # if l2_encoder>1.5:
    #     diff[2]+=1
    # l2_encoder_plus=np.dot(distance_encoder_plus, distance_encoder_plus)
    # dist_avg[3]+=l2_encoder_plus
    # if l2_encoder_plus>1.5:
    #     diff[3]+=1
    l2_overall=np.dot(distance_overall, distance_overall)
    dist_avg[0]+=l2_overall
    if l2_overall>1.5:
        diff[0]+=1
    l2_Attn=np.dot(distance_Attn, distance_Attn)
    dist_avg[1]+=l2_Attn
    if l2_Attn>1.5:
        diff[1]+=1
    print("round %d."%(i))
        # logger.info("   | distance_origin | distance_VGG | distance_encoder | distance_encoderP | distance_overall |".format(i, l2_origin))   
    # logger.info("#{:03d}| {:0.5f}         | {:0.5f}      | {:0.5f}          | {:0.5f}           | {:0.5f}          |".format(i, l2_origin, l2_VGG, l2_encoder, l2_encoder_plus, l2_overall))   
    print('%0.5f, %0.5f, %5f, %5f'%(l2_overall, l2_Attn, dist_avg[0], dist_avg[1]))
    logger.info('#{:03d}| {:0.5f}         | {:0.5f}      |'.format(i, l2_overall, l2_Attn))

    # logger.info("#{:03d} distance_origin: {:0.3f}".format(i, l2_origin))   
    # logger.info("#{:03d} distance_VGG: {:0.3f}".format(i, l2_VGG))
    # logger.info("#{:03d} distance_encoder: {:0.3f}".format(i, l2_encoder))   
    # logger.info("#{:03d} distance_encoder_plus: {:0.3f}".format(i, l2_encoder_plus))   
    # logger.info("#{:03d} distance_overall: {:0.3f}".format(i,l2_overall))   

logger.info('----------------------------------------------')
for i in range(2):
    dist_avg[i]/=400
    logger.info("dist_avg[{:03d}]: {:03d}".format(i, int(dist_avg[i])))
    logger.info("diff[{:03d}]: {:03d}".format(i, int(diff[i])))
    logger.info("fail[{:03d}]: {:03d}".format(i, int(fail[i])))
    print(dist_avg[i], diff[i])
logger.info('----------------------------------------------')
    
