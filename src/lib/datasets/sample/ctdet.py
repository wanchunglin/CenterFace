from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math
import matplotlib.pyplot as plt
import random
from PIL import Image, ExifTags, ImageEnhance, ImageOps, ImageFile

def randomColor(image):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度
    
class CTDetDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    #print(box)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    rotate=0
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)
    """
    #phone_img_dir= '/home/yang/CenterFace/data/mix/image_mix_with_phone/'
    phone_img_dir= '/home/yang/CenterFace/data/mix/image_mix_with_phone_expand/'
    if os.path.exists(phone_img_dir+file_name) and random.random()>0.50:
        img = cv2.imread(phone_img_dir+file_name)
        #cv2.imwrite('/home/yang/CenterFace/1_phone.png',img)
        #print('phone')
    else:
        img = cv2.imread(img_path)
    """
    img = cv2.imread(img_path)
    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    if self.opt.keep_res:
      input_h = (height | self.opt.pad) + 1
      input_w = (width | self.opt.pad) + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:
      s = max(img.shape[0], img.shape[1]) * 1.0
      input_h, input_w = self.opt.input_h, self.opt.input_w
    
    flipped = False
    if self.split == 'train':
      
      if np.random.random() < self.opt.random_crop_rate:
        s = s * np.random.choice(np.arange(0.4, 1.6, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      
      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        c[0] =  width - c[0] - 1
      if np.random.random() < self.opt.random_rotate:
          
          rotate=np.random.randint(low=0,high=360,size=1)
          if np.random.random() < 0.5:
              rotate = (-rotate)

    trans_input = get_affine_transform(
      c, s, int(rotate), [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input, 
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)
    
    
   
    if self.opt.consis==1:
        inp1 = inp.copy()
        # h w channel
        inp1 = inp1[:, ::-1, :].copy()
        inp1 = (inp1.astype(np.float32) / 255.)
        inp1 = (inp1 - self.mean) / self.std
        inp1 = inp1.transpose(2, 0, 1)
    if self.opt.ss ==1:
        
        angle = random.randint(1,3)
        #inpp = cv2.resize(img,(input_w,input_h),interpolation = cv2.INTER_LINEAR)
        inp1=np.rot90(inp,angle).copy()
        if self.opt.img2_type ==2:
            ss_c = np.array([inp.shape[1] / 2., inp.shape[0] / 2.], dtype=np.float32)
            ss_s = s * np.random.choice(np.arange(0.1,1, 0.1))
            ss_w_border = self._get_border(128, inp.shape[1])
            ss_h_border = self._get_border(128, inp.shape[0])
            ss_c[0] = np.random.randint(low=ss_w_border, high=inp.shape[1] - ss_w_border)
            ss_c[1] = np.random.randint(low=ss_h_border, high=inp.shape[0] - ss_h_border)
            inp2 = inp.copy()
            inp2 = inp2[max(int(ss_c[1]-ss_s),0):min(int(ss_c[1]+ss_s),inp2.shape[1]),max(int(ss_c[0]-ss_s),0):min(int(ss_c[0]+ss_s),inp2.shape[0]),:]
            inp2 = cv2.resize(inp2,(input_w,input_h),interpolation = cv2.INTER_LINEAR)
            inp2 = Image.fromarray(inp2)
            inp2=randomColor(inp2)
            inp2 = np.array(inp2)
        elif self.opt.img2_type ==1:
            inp2= cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY).copy()
            inp2=np.fliplr(inp2).copy()
            inp2 = np.stack((inp2,inp2,inp2)).copy()
            inp2 = inp2.transpose(1, 2, 0)
        
        #img2=np.rot90(img2,-90).copy()
        #print(img2.shape)
        if self.opt.img2_type ==2:
            inp3= cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY).copy()
            inp3=np.fliplr(inp3).copy()
            inp3 = np.stack((inp3,inp3,inp3)).copy()
            inp3 = inp3.transpose(1, 2, 0)
        else:
            inp3 = inp.copy()
            inp3 = Image.fromarray(inp3)
            inp3=randomColor(inp3)
            inp3 = np.array(inp3)
        
        inp1 = (inp1.astype(np.float32) / 255.)
        inp2 = (inp2.astype(np.float32) / 255.)
        inp3 = (inp3.astype(np.float32) / 255.)
        inp1 = (inp1 - self.mean) / self.std
        inp2 = (inp2 - self.mean) / self.std
        inp3 = (inp3 - self.mean) / self.std
        
        inp1 = inp1.transpose(2, 0, 1)
        inp2 = inp2.transpose(2, 0, 1)
        inp3 = inp3.transpose(2, 0, 1)
    if self.opt.ss ==2:
        
        inp0 = cv2.resize(img,(input_w, input_h),interpolation=cv2.INTER_LINEAR)
        inp1=np.rot90(inp0,1).copy()
        inp2=np.rot90(inp0,2).copy()
        inp3=np.rot90(inp0,3).copy()
        
        inp0 = (inp0.astype(np.float32) / 255.)
        inp1 = (inp1.astype(np.float32) / 255.)
        inp2 = (inp2.astype(np.float32) / 255.)
        inp3 = (inp3.astype(np.float32) / 255.)
        inp0 = (inp0 - self.mean) / self.std
        inp1 = (inp1 - self.mean) / self.std
        inp2 = (inp2 - self.mean) / self.std
        inp3 = (inp3 - self.mean) / self.std
        
        inp0 = inp0.transpose(2, 0, 1)
        inp1 = inp1.transpose(2, 0, 1)
        inp2 = inp2.transpose(2, 0, 1)
        inp3 = inp3.transpose(2, 0, 1)
        
    
    inp = (inp.astype(np.float32) / 255.)
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio
    num_classes = self.num_classes
    
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])
    
   

    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    mask = np.zeros((num_classes,output_h, output_w), dtype=np.uint8)
    mask1d = np.zeros((output_h, output_w), dtype=np.uint8)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    Ct_f = np.zeros((self.max_objs, 2), dtype=np.float32)
    Ct_m = np.zeros((self.max_objs, 2), dtype=np.float32)
    BBox_f = np.zeros((self.max_objs, 4), dtype=np.float32)
    BBox_m = np.zeros((self.max_objs, 4), dtype=np.float32)
    BBox = np.zeros((self.max_objs, 4), dtype=np.float32)
    dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
    cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
    
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

    gt_det = []
    nonzero_ind_f = 0
    nonzero_ind_m = 0
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])
      #print(bbox)
      cls_id = int(self.cat_ids[ann['category_id']])
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
      #########rotate bounding box###################
      
      if rotate !=0:
          x_ur=bbox[2]
          y_ur=bbox[1]
          x_ll=bbox[0]
          y_ll=bbox[3]
          x_ur,y_ur = affine_transform((x_ur,y_ur), trans_output)
          x_ll,y_ll = affine_transform((x_ll,y_ll), trans_output)
          x_ur,y_ur=np.clip([x_ur,y_ur], 0, output_w - 1)
          x_ll,y_ll=np.clip([x_ll,y_ll], 0, output_h - 1)
     
      #########rotate bounding box###################
      ###############################################
      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
      ###############################################
      #########rotate bounding box###################
      
      if rotate !=0:
          bbox[0]=min(bbox[0],bbox[2],x_ur,x_ll)
          bbox[2]=max(bbox[0],bbox[2],x_ur,x_ll)
          bbox[1]=min(bbox[1],bbox[3],y_ur,y_ll)
          bbox[3]=max(bbox[1],bbox[3],y_ur,y_ll)
      
      #########rotate bounding box###################
      
      
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      
      if h > 0 and w > 0:
        add_w = int(w*0.25)
        add_h = int(h*0.25)
        #mask1d[int(bbox[1])-5:int(bbox[3])+5,int(bbox[0])-5:int(bbox[2])+5] =1 
        mask1d[max(0,int(bbox[1])-add_h):min(int(bbox[3])+add_h,output_h),max(0,int(bbox[0])-add_w):min(int(bbox[2])+add_w,output_w)] =1 
        mask1d[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])] = 2
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))                # 高斯半径
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)       # 求中点
        ct_int = ct.astype(np.int32)
        #print(cls_id)
        if cls_id ==0:
            Ct_f[nonzero_ind_f,0] = ct[1]
            Ct_f[nonzero_ind_f,1] = ct[0]
            BBox_f[nonzero_ind_f] = bbox[0],bbox[1],bbox[2],bbox[3]
            #print('0',ct[0])
            mask[0,int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])] = 1
            nonzero_ind_f +=1
        elif cls_id ==1:
            #print('1',ct[0])
            Ct_m[nonzero_ind_m,0] = ct[1]
            Ct_m[nonzero_ind_m,1] = ct[0]
            BBox_m[nonzero_ind_m] = bbox[0],bbox[1],bbox[2],bbox[3]
            mask[1,int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])] = 1
            nonzero_ind_m +=1
        draw_gaussian(hm[cls_id], ct_int, radius)                             # 每个类别一个channel
        wh[k] = 1. * w, 1. * h                                        # 目标的wh
        #wh[k] = np.log(1. * w / 4), np.log(1. * h / 4)
        
        ind[k] = ct_int[1] * output_w + ct_int[0]                     # 索引，y*w + x
        reg[k] = ct - ct_int                                          # 整数化的误差
        reg_mask[k] = 1
        cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
        cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
        if self.opt.dense_wh:
          draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                       ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
        #meta_simi = {'c': c, 's': s}
        
    
    if self.opt.consis:
        ret = {'input': inp,'inp1':inp1, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'ctf':Ct_f,'ctm':Ct_m,'bboxf':BBox_f,'bboxm':BBox_m,'mask':mask,'mask1d':mask1d}
        
    elif self.opt.simi and not self.opt.ss:
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'ctf':Ct_f,'ctm':Ct_m,'bboxf':BBox_f,'bboxm':BBox_m,'mask':mask,'mask1d':mask1d}
    
    elif self.opt.ss ==1:
        ret = {'input': inp,'inp1':inp1,'inp2':inp2,'inp3':inp3, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'ctf':Ct_f,'ctm':Ct_m,'bboxf':BBox_f,'bboxm':BBox_m,'mask':mask,'mask1d':mask1d}
    elif self.opt.ss == 2:
        ret = {'input': inp,'inp0':inp0,'inp1':inp1,'inp2':inp2,'inp3':inp3, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'ctf':Ct_f,'ctm':Ct_m,'bboxf':BBox_f,'bboxm':BBox_m,'mask':mask,'mask1d':mask1d}
    
    else:
        
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
    
    if self.opt.dense_wh:
      hm_a = hm.max(axis=0, keepdims=True)
      dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
      ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
      del ret['wh']
    elif self.opt.cat_spec_wh:
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 6), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta
    return ret