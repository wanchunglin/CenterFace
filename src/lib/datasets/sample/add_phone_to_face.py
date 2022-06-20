#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:58:30 2020

@author: yang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os


import math
import matplotlib.pyplot as plt
import random
from PIL import Image, ExifTags, ImageEnhance, ImageOps, ImageFile


import pycocotools.coco as coco



import torch.utils.data as data

def _coco_box_to_bbox(box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
            dtype=np.float32)
#print(box)
    return bbox

def _get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

split = 'train'
data_dir = os.path.join('/home/yang/CenterFace/data', 'mix')
img_dir = os.path.join(data_dir, 'image_mix')   
_ann_name = {'train': 'train', 'val': 'val'}
annot_path = os.path.join(
data_dir, 'annotations', 
  '{}_wider_face_mixed.json').format(_ann_name[split])


print('==> initializing pascal {} data.'.format(_ann_name[split]))
coco = coco.COCO(annot_path)
images = sorted(coco.getImgIds())
num_samples = len(images)

phone0 = cv2.imread('/home/yang/CenterFace/phone00.png')
phone1 = cv2.imread('/home/yang/CenterFace/phone11.png')
#phone2 = cv2.imread('/home/yang/CenterFace/phone2.png')
phone3 = cv2.imread('/home/yang/CenterFace/phone33.png')




for index in range(len(num_samples)):
    index = 0
    img_id = images[index]
    file_name = coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(img_dir, file_name)
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ids=ann_ids)
    
    
    img = cv2.imread(img_path)
    
    height, width = img.shape[0], img.shape[1]
    for box_num in range(len(anns)):
        ann = anns[box_num]
        bbox = _coco_box_to_bbox(ann['bbox'])
        a = random.random()
        if a>0:
            select = random.randint(0,3)
            if select ==0:
                phone = phone0
            elif select ==1:
                phone = phone1
            #elif select ==2:
            #    phone = phone2
            elif select ==3:
                phone = phone3
            h,w = phone.shape[0],phone.shape[1]

            
            bh = bbox[3]-bbox[1]
            bw = bbox[2]-bbox[0]
            centerw = int((bbox[2]+bbox[0])/2)
            centerh = int((bbox[3]+bbox[1])/2)
            neww = int(bw*0.5)
            newh = int(bh*0.5)
            newh = 2*int(newh/2)
            neww = 2*int(neww/2)
            
            phone = cv2.resize(phone,(neww,newh),interpolation=cv2.INTER_CUBIC)
            img[centerh-int(newh/2):centerh+int(newh/2),centerw-int(neww/2):centerw+int(neww/2),:] = phone
        
    cv2.imwrite(('/home/yang/CenterFace/data/mix/image_mix_with_phone/'+file_name),img)
        
