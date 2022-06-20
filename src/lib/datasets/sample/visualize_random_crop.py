#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 20:58:26 2020

@author: yang
"""

#%%
import cv2
import numpy as np

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
def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result
def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)
def _get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i
img=cv2.imread('/home/yang/CenterFace/NCTU_data/照片_200416_0022.jpg')
c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
s = max(img.shape[0], img.shape[1]) * 1.0
input_h, input_w = 512,512
s = s * 1#np.random.choice(np.arange(0.6, 1.4, 0.1))
w_border = _get_border(128, img.shape[1])
h_border = _get_border(128, img.shape[0])
c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)

trans_input = get_affine_transform(
      c, s, 0, [input_w, input_h])
inp = cv2.warpAffine(img, trans_input, 
                     (input_w, input_h),
                     flags=cv2.INTER_LINEAR)
import matplotlib.pyplot as plt
#%%
def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]
trans_input2 = get_affine_transform(
      c, s, 0, [input_w, input_w])
inp2 = cv2.warpAffine(img, trans_input2, 
                     (800, 800),
                     flags=cv2.INTER_LINEAR)
trans_output = get_affine_transform(c, s, 0, (input_w, input_h))
trans_output_rot = get_affine_transform(c, s, -180, (input_w, input_h))
x1=300
x2=500
y1=600
y2=800
x_ur=x2
y_ur=y1
x_ll=x1
y_ll=y2

cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
x11,y11 = affine_transform((x1,y1), trans_output)
x22,y22 = affine_transform((x2,y2), trans_output)
x11,y11=np.clip([x11,y11], 0, input_w - 1)
x22,y22=np.clip([x22,y22], 0, input_h - 1)
cv2.rectangle(inp, (int(x11), int(y11)), (int(x22), int(y22)), (0, 255, 0), 2)


x11,y11 = affine_transform((x1,y1), trans_output_rot)
x22,y22 = affine_transform((x2,y2), trans_output_rot)
x11,y11=np.clip([x11,y11], 0, input_w - 1)
x22,y22=np.clip([x22,y22], 0, input_h - 1)
cv2.rectangle(inp2, (int(x11), int(y11)), (int(x22), int(y22)), (0, 255, 0), 2)

#x_ur,y_ur = affine_transform((x_ur,y_ur), trans_output)
#x_ll,y_ll = affine_transform((x_ll,y_ll), trans_output)
#x_ur,y_ur=np.clip([x_ur,y_ur], 0, input_w - 1)
#x_ll,y_ll=np.clip([x_ll,y_ll], 0, input_h - 1)
#cv2.rectangle(inp2, (int(x_ll), int(y_ll)), (int(x_ur), int(y_ur)), (255, 0, 0), 2)

x_ur,y_ur = affine_transform((x_ur,y_ur), trans_output_rot)
x_ll,y_ll = affine_transform((x_ll,y_ll), trans_output_rot)
x_ur,y_ur=np.clip([x_ur,y_ur], 0, input_w - 1)
x_ll,y_ll=np.clip([x_ll,y_ll], 0, input_h - 1)
x11=min(x11,x_ur,x22,x_ll)
x22=max(x11,x_ur,x22,x_ll)
y11=min(y11,y_ur,y22,y_ll)
y22=max(y11,y_ur,y22,y_ll)
x11,y11=np.clip([x11,y11], 0, input_w - 1)
x22,y22=np.clip([x22,y22], 0, input_h - 1)
cv2.rectangle(inp2, (int(x11), int(y11)), (int(x22), int(y22)), (255, 0, 0), 2)



imgplot = plt.imshow(img)
plt.show()
imgplot = plt.imshow(inp)
plt.show()
imgplot = plt.imshow(inp2)
plt.show()
#cv2.rectangle(inp2, (int(x11), int(y11)), (int(x22), int(y22)), (0, 0, 255), 2)
imgplot = plt.imshow(inp2)
plt.show()

