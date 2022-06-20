#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 21:11:44 2020

@author: yang-aimm
"""

#%%
from evaluation import *
import cv2
import os

gt_path='./ground_truth'
pred_path='/home/yang/CenterFace/output/widerface/'
facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
pred = get_preds(pred_path)
norm_score(pred)
for i in range(61):
    
    event=str(event_list[i][0])
    event=event[2:(-2)]
    gt_bbx_list = facebox_list[i][0]
    hd_gt_list=hard_gt_list[i][0]
    img_list= file_list[i][0]
    pr=pred[event]
    os.makedirs('/home/yang/CenterFace/output/gt_pred/widerface/'+event)
    for img_ind in range(gt_bbx_list.shape[0]):
        
        hd=hd_gt_list[img_ind][0]
        img=img_list[img_ind][0]
        
        img=str(img)
        img=img[2:(-2)]
        I=cv2.imread('/home/yang/datasets/WIDERFACE/WIDER_val/images/'+event+'/'+img+'.jpg')
        
        img_gt_boxes=gt_bbx_list[img_ind][0].astype('float')
        pr_boxes=pr[img]
        #for box in img_gt_boxes:
        #    x,y,w,h=box[0],box[1],box[2],box[3]
        #    cv2.rectangle(I, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
        for box in pr_boxes:
            
            x,y,w,h,s=box[0],box[1],box[2],box[3],box[4]
            if s>=0.4:
                
                cv2.rectangle(I, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 2)
        cv2.imwrite('/home/yang/CenterFace/output/gt_pred/widerface/'+event+'/'+img+'.jpg',I)    
        #cv2.imshow('img',I)
        #cv2.waitKey(0)
#%%
from evaluation import *
import cv2
import os

gt_path='./ground_truth'
face_pred_path='/home/yang/CenterFace/output/widerface/face/'
mask_pred_path='/home/yang/CenterFace/output/widerface/mask/'
facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
pred_face = get_preds(face_pred_path)
pred_mask = get_preds(mask_pred_path)
norm_score(pred_face)
norm_score(pred_mask)
for i in range(61):
    
    event=str(event_list[i][0])
    event=event[2:(-2)]
    gt_bbx_list = facebox_list[i][0]
    hd_gt_list=hard_gt_list[i][0]
    img_list= file_list[i][0]
    prf=pred_face[event]
    prm=pred_mask[event]
    os.makedirs('/home/yang/CenterFace/output/gt_pred/widerface/'+event)
    for img_ind in range(gt_bbx_list.shape[0]):
        
        hd=hd_gt_list[img_ind][0]
        img=img_list[img_ind][0]
        
        img=str(img)
        img=img[2:(-2)]
        I=cv2.imread('/home/yang/datasets/WIDERFACE/WIDER_val/images/'+event+'/'+img+'.jpg')
        
        img_gt_boxes=gt_bbx_list[img_ind][0].astype('float')
        prf_boxes=prf[img]
        prm_boxes=prm[img]
       
        for box in prf_boxes:
            
            x,y,w,h,s=box[0],box[1],box[2],box[3],box[4]
            if s>=0.4:
                
                cv2.rectangle(I, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 2)
        for box in prm_boxes:
            
            x,y,w,h,s=box[0],box[1],box[2],box[3],box[4]
            if s>=0.4:
                print(event+'/'+img)
                
                cv2.rectangle(I, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
        cv2.imwrite('/home/yang/CenterFace/output/gt_pred/widerface/'+event+'/'+img+'.jpg',I)    
        #cv2.imshow('img',I)
        #cv2.waitKey(0)
#%%
from evaluation import *
import cv2
import os

gt_path='/home/yang-aimm/centerface/data/MAFA_VAL/Val_txt/'
pred_path='/home/yang-aimm/centerface/output/MAFA/'
gt = get_preds(gt_path)
gt = gt['MAFA']
pred = get_preds(pred_path)
norm_score(pred)
pred=pred['MAFA']
img_files=os.listdir('/home/yang-aimm/centerface/data/MAFA_VAL/MAFA_val/')
os.makedirs('/home/yang-aimm/centerface/output/gt_pred/MAFA')
for img in img_files:
    
    I=cv2.imread('/home/yang-aimm/centerface/data/MAFA_VAL/MAFA_val/'+img)
    img_dict_ind=img.split('.')[0]
    gt_bbx_list = gt[img_dict_ind]
    pr_boxes=pred[img_dict_ind]

    for box in gt_bbx_list:
        x,y,w,h=box[0],box[1],box[2],box[3]
        cv2.rectangle(I, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
    for box in pr_boxes:
        
        x,y,w,h,s=box[0],box[1],box[2],box[3],box[4]
        if s>=0.2:
            
            cv2.rectangle(I, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 2)
    cv2.imwrite('/home/yang-aimm/centerface/output/gt_pred/MAFA/'+img+'.jpg',I)    
    #cv2.imshow('img',I)
    #cv2.waitKey(0)
        
#%%
from evaluation import get_preds,norm_score
import cv2
import os


pred_path_face='/home/yang/CenterFace/output/NCTU_face/'

pred_face = get_preds(pred_path_face)
norm_score(pred_face)
pred_face=pred_face['MAFA']

pred_path_mask='/home/yang/CenterFace/output/NCTU_mask/'

pred_mask = get_preds(pred_path_mask)
norm_score(pred_mask)
pred_mask=pred_mask['MAFA']
img_files=os.listdir('/home/yang/CenterFace/NCTU_data_new/')
os.makedirs('/home/yang/CenterFace/output/NCTU_img/')
for img in img_files:
    
    I=cv2.imread('/home/yang/CenterFace/NCTU_data_new/'+img)
    img_dict_ind=img.split('.')[0]
    pr_boxes_face=pred_face[img_dict_ind]
    pr_boxes_mask=pred_mask[img_dict_ind]

    
    for box in pr_boxes_face:
        
        x,y,w,h,s=box[0],box[1],box[2],box[3],box[4]
        if s>=0.6:
            
            cv2.rectangle(I, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 2)
    for box in pr_boxes_mask:
        
        x,y,w,h,s=box[0],box[1],box[2],box[3],box[4]
        if s>=0.44:
            
            cv2.rectangle(I, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
    cv2.imwrite('/home/yang/CenterFace/output/NCTU_img/'+img+'.jpg',I)    
    #cv2.imshow('img',I)
    #cv2.waitKey(0)   
    
#%%
import cv2
import glob
import numpy as np
Path = '/home/yang/CenterFace/NCTU_data/'
newPath = '/home/yang/CenterFace/NCTU_data_new/'
Path=Path+'*.jpg'
num=0
Paths = glob.glob(Path)
Paths = [Paths[0]]
def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans
def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result
def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

for i in Paths:
    I=cv2.imread(i)
    I=cv2.resize(I,(800,800))
    bbox = [300,500,350,600]
    x1=int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])
    cv2.rectangle(I, (x1, y1), (x2, y2), (0, 255, 0), 2)
    c = np.array([I.shape[1] / 2., I.shape[0] / 2.], dtype=np.float32)
    trans = get_affine_transform(c,800,180,[I.shape[1],I.shape[0]])
    bbox[:2] = affine_transform(bbox[:2], trans)
    bbox[2:] = affine_transform(bbox[2:], trans)
    x1=int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])
    Irot = (np.rot90(I,2))
    Irot=np.ascontiguousarray(Irot)
    cv2.rectangle(Irot, (x1, y1), (x2, y2), (0, 0,255), 2)
    
    
    cv2.imshow('img',I)
    
    cv2.imshow('rot',Irot)
    cv2.waitKey(0)
    