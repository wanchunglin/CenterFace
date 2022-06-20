#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 21:55:41 2020

@author: yang
"""
#%%
txt_path_t ="/home/yang/Pytorch_Retinaface/data/widerface/train/label.txt"
f_t = open(txt_path_t,'r')
linest = f_t.readlines()

txt_path_v ="/home/yang/Pytorch_Retinaface/data/widerface/val/wider_val.txt"
f_v = open(txt_path_v,'r')
linesv = f_v.readlines()

wordst = []
imgs_patht=[]
        
isFirst = True
labelst = []
for line in linest:
    line = line.rstrip()
    if line.startswith('#'):
        if isFirst is True:
            isFirst = False
        else:
            labels_copy = labelst.copy()
            wordst.append(labels_copy)
            labelst.clear()
        path = line[2:]
        path=path.split('/')[-1]
        imgs_patht.append(path)
    else:
        line = line.split(' ')
        label = [float(x) for x in line]
        labelst.append(label)

wordst.append(labelst)
"""
wordsv = []
imgs_pathv=[]
        
isFirst = True
labelsv = []
for line in linesv:
    line = line.rstrip()
    if line.startswith('#'):
        if isFirst is True:
            isFirst = False
        else:
            labels_copy = labelsv.copy()
            wordsv.append(labels_copy)
            labelsv.clear()
        path = line[2:]
        path=path.split('/')[-1]
        imgs_pathv.append(path)
    else:
        line = line.split(' ')
        label = [float(x) for x in line]
        labelsv.append(label)

wordsv.append(labelsv)
"""

#%%
import sys
import os
import json
import xml.etree.ElementTree as ET
import glob
import cv2

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {'face':1,'face_mask':2}
# If necessary, pre-define category and its id
#  PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
                         #  "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
                         #  "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
                         #  "motorbike": 14, "person": 15, "pottedplant": 16,
                         #  "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return int(filename)
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.'%(filename))

json_file = '/home/yang/CenterFace/data/mix/hp/train_wider_face_mixed.json'
filedir='/home/yang/datasets/aizoo/train/*.xml'
xml_list = glob.glob(filedir)

json_dict = {"images":[], "type": "instances", "annotations": [],
             "categories": []}

categories = PRE_DEFINE_CATEGORIES
bnd_id = START_BOUNDING_BOX_ID
image_id=-1
for line in xml_list:
    line = line.strip()
    print("Processing %s"%(line))
    xml_f = os.path.join(filedir, line)
    tree = ET.parse(xml_f)
    root = tree.getroot()
    path = get(root, 'path')
    """
    if len(path) == 1:
        filename = os.path.basename(path[0].text)
    elif len(path) == 0:
        filename = get_and_check(root, 'filename', 1).text
    else:
        raise NotImplementedError('%d paths found in %s'%(len(path), line))
    """
    filename = line.split('/')[-1][0:-4]+'.jpg'
    ## The filename must be a number
    #image_id = get_filename_as_int(filename)
    image_id +=1
    #size = get_and_check(root, 'size', 1)
    #width = int(get_and_check(size, 'width', 1).text)
    #height = int(get_and_check(size, 'height', 1).text)
    img=cv2.imread('/home/yang/datasets/aizoo/train/'+line.split('/')[-1][0:-4]+'.jpg')
    height = img.shape[0]
    width = img.shape[1]
    image = {'file_name': filename, 'height': height, 'width': width,
             'id':image_id}
    json_dict['images'].append(image)
    ## Cruuently we do not support segmentation
    #  segmented = get_and_check(root, 'segmented', 1).text
    #  assert segmented == '0'
    if line.split('/')[-1][0:-4]+'.jpg' in imgs_patht:
        print('with landmarks')
        ind = imgs_patht.index(line.split('/')[-1][0:-4]+'.jpg')
        labels = wordst[ind]
        for label in labels:
            
            category_id = categories['face']
            ann = {'area': label[2]*label[3], 'iscrowd': 0, 'image_id':
                   image_id, 'bbox':[label[0], label[1], label[2],label[3]],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': [],'keypoints':[label[4],label[5],1,label[7],label[8],1,label[10],label[11]\
                                    ,1,label[13],label[14],1,label[16],label[17],1],'num_keypoints':5}
            json_dict['annotations'].append(ann)
            
            bnd_id = bnd_id + 1
        
    else:
        
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category not in categories:
                print('new')
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text)# - 1
            ymin = int(get_and_check(bndbox, 'ymin', 1).text)# - 1
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert(xmax > xmin)
            assert(ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id':
                   image_id, 'bbox':[xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': [],'keypoints':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],'num_keypoints':5}
            json_dict['annotations'].append(ann)
            
            bnd_id = bnd_id + 1

for cate, cid in categories.items():
    cat = {'supercategory': 'none', 'id': cid, 'name': cate}
    json_dict['categories'].append(cat)
json_fp = open(json_file, 'w')
json_str = json.dumps(json_dict)
json_fp.write(json_str)
json_fp.close()
#list_fp.close()

