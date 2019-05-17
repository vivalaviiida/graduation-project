# *_*coding:utf-8 *_*
# author: 许鸿斌
# 邮箱:2775751197@qq.com

import time
start = time.time()

import cv2
import itertools
import os
import numpy as np
import openface
import argparse
from log import logger

dlib_model_dir = '/home/annalei/openface/models/dlib'
openface_model_dir = '/home/annalei/openface/models/openface'

parser = argparse.ArgumentParser()
parser.add_argument('imgs', type=str, nargs='+', help='Input images.')
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
        logger.info("Face detection took {} seconds.".format(tim.time() - start))

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

for (img1, img2) in itertools.combinations(args.imgs, 2):
    distance = getRep(img1) - getRep(img2)
    logger.info("Comparing {} with {}.".format(img1, img2))
    logger.info("Squared l2 distance between representations: {:0.3f}".format(np.dot(distance, distance)))
