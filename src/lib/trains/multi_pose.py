from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss, RegL1Loss, RegLoss, RegWeightedL1Loss
from models.decode import multi_pose_decode
from models.utils import _sigmoid, flip_tensor, flip_lr_off, flip_lr
from utils.debugger import Debugger
from utils.post_process import multi_pose_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer

import random
from torch.nn.modules.distance import PairwiseDistance
from torch.autograd import Function

way = 3
margint = 0.1 #margin for pos_dist and neg_dist 
margin = 0.1  # if neg dist - pos dist < margin choose this sample
use_mask_anchor = 1

class TripletLoss(Function):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)

    def forward(self, anchor, positive, negative):
        pos_dist = self.pdist.forward(anchor, positive)
        neg_dist = self.pdist.forward(anchor, negative)

        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)
        loss = torch.mean(hinge_dist)

        return loss

class MultiPoseLoss(torch.nn.Module):
  def __init__(self, opt):
    super(MultiPoseLoss, self).__init__()
    #self.crit = FocalLoss()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_hm_hp = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_kp = RegWeightedL1Loss() if not opt.dense_hp else \
                   torch.nn.L1Loss(reduction='sum')
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
                    RegLoss() if opt.reg_loss == 'sl1' else None
    
    self.opt = opt

  def forward(self, outputs, batch):
    opt = self.opt
    hm_loss, wh_loss, off_loss = 0, 0, 0
    lm_loss, off_loss, hm_hp_loss, hp_offset_loss = 0, 0, 0, 0
    triplet_loss = torch.zeros(1).cuda()
    triplet_loss_m = torch.zeros(1).cuda()
    for s in range(opt.num_stacks):
      output = outputs[s]
      output['hm'] = output['hm']
      # if opt.hm_hp and not opt.mse_loss:
      #   output['hm_hp'] = _sigmoid(output['hm_hp'])
      if not opt.mse_loss:
        output['hm'] = _sigmoid(output['hm'])
      if opt.eval_oracle_hmhp:
        output['hm_hp'] = batch['hm_hp']
      if opt.eval_oracle_hm:
        output['hm'] = batch['hm']
      if opt.eval_oracle_kps:
        if opt.dense_hp:
          output['hps'] = batch['dense_hps']
        else:
          output['hps'] = torch.from_numpy(gen_oracle_map(
            batch['hps'].detach().cpu().numpy(), 
            batch['ind'].detach().cpu().numpy(), 
            opt.output_res, opt.output_res)).to(opt.device)
      if opt.eval_oracle_hp_offset:
        output['hp_offset'] = torch.from_numpy(gen_oracle_map(
          batch['hp_offset'].detach().cpu().numpy(), 
          batch['hp_ind'].detach().cpu().numpy(), 
          opt.output_res, opt.output_res)).to(opt.device)

      if way ==3 and opt.simi !=0:
          #print('use way3')
          channel=256
          norm = torch.nn.BatchNorm2d(channel).cuda()
          simi_hm = norm(output['simi'])
          face_ct = batch['bboxf']
          mask_ct = batch['bboxm']
          mask = batch['mask1d']
          #print(mask_ct)
          

          #simi_size = 5
          simi_size = int(66/8) # width size
          simi_size_h = int(88/8)
          iter_size = simi_hm.size()[0]

          l2_distance = PairwiseDistance(2).cuda()
          triplet_loss = torch.zeros(1).cuda()
          triplet_loss = torch.zeros(1).cuda()
          triplet_loss_m = torch.zeros(1).cuda()
          
          
          total_f = 0
          total_m = 0
          FFace_hm=torch.zeros(1000,simi_size*simi_size_h*channel)
          #FFace_hm_half=torch.zeros(1000,simi_size*simi_size_h*256)
          MMask_hm=torch.zeros(1000,simi_size*simi_size_h*channel)
          #MMask_hm_half=torch.zeros(1000,simi_size*simi_size_h*256)
          mask_num_num = 0
          face_num_num = 0
          for batch_ind in range(iter_size):

              numf = int(len(face_ct[batch_ind].nonzero())/4)
              numm = int(len(mask_ct[batch_ind].nonzero())/4)
              total_f += numf
              total_m += numm
              

              for face_num in range(numf):
                  
                  #print(batch_ind,numf,face_ct[batch_ind,:,:])
                  #face_hm_place = (face_ct[batch_ind,face_num,:]/4)
                  face_hm_place = (face_ct[batch_ind,face_num,:])
                  #print(face_max_hm_place)
                  face_hm_place[0] = int(face_hm_place[0])
                  face_hm_place[1] = int(face_hm_place[1])
                  face_hm_place[2] = int(face_hm_place[2])
                  face_hm_place[3] = int(face_hm_place[3])
                  if face_hm_place[2]-face_hm_place[0]==0:
                     face_hm_place[2]+=1
                  if face_hm_place[3]-face_hm_place[1]==0:
                     face_hm_place[3]+=1  
                  face_hm = \
                  simi_hm[batch_ind,:,int(max(0,face_hm_place[1])):int(min(127,face_hm_place[3]))\
                                        ,int(max(0,face_hm_place[0])):int(min(127,face_hm_place[2]))]

                  face_hm = face_hm.view(1,channel,face_hm.size()[1],face_hm.size()[2])
                  
                  face_hm = torch.nn.functional.interpolate(face_hm, [simi_size_h,simi_size],mode ='bilinear')
                  face_hm = torch.flatten(face_hm)
                  FFace_hm[face_num_num] = face_hm
                  """
                  start = (face_hm_place[3] - face_hm_place[1])/2
                  face_hm = \
                  simi_hm[batch_ind,:,int(max(0,face_hm_place[1]+start)):int(min(127,face_hm_place[3]))\
                                        ,int(max(0,face_hm_place[0])):int(min(127,face_hm_place[2]))]
                  
                  face_hm = face_hm.view(1,256,face_hm.size()[1],face_hm.size()[2])
                  face_hm = torch.nn.functional.interpolate(face_hm, [simi_size,simi_size],mode ='bilinear')
                  
                  face_hm = torch.flatten(face_hm)
                  FFace_hm_half[face_num_num] = face_hm
                  """
                  face_num_num +=1
                  
              for mask_num in range(numm):
                  
                  #mask_hm_place = mask_ct[batch_ind,mask_num,:]/4
                  mask_hm_place = mask_ct[batch_ind,mask_num,:]
                  mask_hm_place[0] = int(mask_hm_place[0])
                  mask_hm_place[1] = int(mask_hm_place[1])
                  mask_hm_place[2] = int(mask_hm_place[2])
                  mask_hm_place[3] = int(mask_hm_place[3])
                  if mask_hm_place[2]-mask_hm_place[0]==0:
                     mask_hm_place[2]+=1
                  if mask_hm_place[3]-mask_hm_place[1]==0:
                     mask_hm_place[3]+=1  

                  mask_hm = \
                  simi_hm[batch_ind,:,int(max(0,mask_hm_place[1])):int(min(127,mask_hm_place[3]))\
                                        ,int(max(0,mask_hm_place[0])):int(min(127,mask_hm_place[2]))]
                  
                  mask_hm = mask_hm.view(1,channel,mask_hm.size()[1],mask_hm.size()[2])
                  mask_hm = torch.nn.functional.interpolate(mask_hm, [simi_size_h,simi_size],mode ='bilinear')
                  
                  mask_hm = torch.flatten(mask_hm)
                  MMask_hm[mask_num_num] = mask_hm
                  """
                  start = (mask_hm_place[3] - mask_hm_place[1])/2
                  mask_hm = \
                  simi_hm[batch_ind,:,int(max(0,mask_hm_place[1]+start)):int(min(127,mask_hm_place[3]))\
                                        ,int(max(0,mask_hm_place[0])):int(min(127,mask_hm_place[2]))]
                  
                  mask_hm = mask_hm.view(1,256,mask_hm.size()[1],mask_hm.size()[2])
                  mask_hm = torch.nn.functional.interpolate(mask_hm, [simi_size,simi_size],mode ='bilinear')
                  
                  mask_hm = torch.flatten(mask_hm)
                  MMask_hm_half[mask_num_num] = mask_hm
                  """
                  
                  mask_num_num +=1 
                  
                  #print(Mask_hm)
          if total_f%2 ==0:
              
              match_size = max(int(total_f/2),total_m)
          else:
              match_size = max(int(total_f/2)+1,total_m)
          #print('match_size',match_size)
          
          if use_mask_anchor:
              if total_m%2 ==0:
                  
                  match_size_m = max(int(total_m/2),total_f)
              else:
                  match_size_m = max(int(total_m/2)+1,total_f)
                  
              Anchor_hm_m = torch.zeros(match_size_m,simi_size*simi_size_h*channel)
              Face_hm_m=torch.zeros(match_size_m,simi_size*simi_size_h*channel)
              Mask_hm_m=torch.zeros(match_size_m,simi_size*simi_size_h*channel)
              
              Anchor_hm_m[0:int(total_m/2)] = MMask_hm[0:int(total_m/2)]
              Face_hm_m[0:total_f] = FFace_hm[0:total_f]
              Mask_hm_m[0:(total_m-int(total_m/2))] = MMask_hm[int(total_m/2):total_m]
              """
              Anchor_hm_m[0:int(total_m/2)] = MMask_hm_half[0:int(total_m/2)]
              Face_hm_m[0:total_f] = FFace_hm_half[0:total_f]
              Mask_hm_m[0:(total_m-int(total_m/2))] = MMask_hm_half[int(total_m/2):total_m]
              """
          
          Anchor_hm = torch.zeros(match_size,simi_size*simi_size_h*channel)
          Face_hm=torch.zeros(match_size,simi_size*simi_size_h*channel)
          Mask_hm=torch.zeros(match_size,simi_size*simi_size_h*channel)
          
          
          
          #print(total_f)
          #print(FFace_hm[int(total_f/2):total_f].size())
          #print(Face_hm[0:(total_f-int(total_f/2))].size())
          
          Anchor_hm[0:int(total_f/2)] = FFace_hm[0:int(total_f/2)]
          Face_hm[0:(total_f-int(total_f/2))] = FFace_hm[int(total_f/2):total_f]
          Mask_hm[0:total_m] = MMask_hm[0:total_m]
          
          
          
          del FFace_hm,MMask_hm#,FFace_hm_half,MMask_hm_half
          #print(total_f,total_m)
          if total_f>1 and total_m !=0:
              #print('start')
              if match_size == total_m:
                  for number in range(total_m-int(total_f/2)):
                          rnd = random.randint(0,int(total_f/2)-1)
                          Anchor_hm[int(total_f/2)+number] = Anchor_hm[rnd]
                  for number in range(total_m-(total_f-int(total_f/2))):
                          rnd = random.randint(0,(total_f-int(total_f/2))-1)
                          Face_hm[(total_f-int(total_f/2))+number] = Face_hm[rnd]
                    
              else:
                  if total_f%2 ==1:
                      rnd = random.randint(0,int(total_f/2)-1)
                      Anchor_hm[-1] = Anchor_hm[rnd]
                      if total_m<int(total_f/2)+1:
                          for number in range(int(total_f/2)+1-total_m):
                              #rnd = random.randint(0,total_m-1)
                              #Mask_hm[total_m+number] = Mask_hm[rnd]
                              
                              
                              rnd_batch = random.randint(0,iter_size-1)
                              maskm = mask[rnd_batch].cpu().detach().numpy()
                              
                              background= np.where(maskm  == 1)
                              itera = 0
                              while len(background[0])==0 and itera <iter_size:
                                  rnd_batch = random.randint(0,iter_size-1)
                                  maskm = mask[rnd_batch].cpu().detach().numpy()
                                  
                                  background= np.where(maskm  == 1)
                                  itera +=1
                              #print(mask.size(),'\n',maskm)
                              #print(background,len(background[0]),background[0][0])
                              if len(background[0])>0:
                                  rnd_batch = random.randint(0,len(background[0])-1)
                                  rnd_center = [background[0][rnd_batch],background[1][rnd_batch]]
                                  rnd_simi_size = random.randint(1,simi_size)
                                  bg = \
                                  simi_hm[batch_ind,:,int(max(0,rnd_center[0]-(rnd_simi_size-1)/2)):int(min(opt.input_h/4,rnd_center[0]+(rnd_simi_size-1)/2+1))\
                                            ,int(max(0,rnd_center[1]-(rnd_simi_size-1)/2)):int(min(opt.input_h/4,rnd_center[1]+(rnd_simi_size-1)/2+1))]
                                  
                                  bg = bg.view(1,channel,bg.size()[1],bg.size()[2])
                                  bg = torch.nn.functional.interpolate(bg, [simi_size_h,simi_size],mode ='bilinear')
                                  
                                  bg = torch.flatten(bg)
                                  
                                  Mask_hm[total_m+number] = bg
                              else:
                                  rnd = random.randint(0,total_m-1)
                                  Mask_hm[total_m+number] = Mask_hm[rnd]
                  else:
                      if total_m<int(total_f/2):
                          for number in range(int(total_f/2)-total_m):
                              #rnd = random.randint(0,total_m-1)
                              #Mask_hm[total_m+number] = Mask_hm[rnd]
                              rnd_batch = random.randint(0,iter_size-1)
                              maskm = mask[rnd_batch].cpu().detach().numpy()
                              
                              background= np.where(maskm  == 1)
                              itera = 0
                              while len(background[0])==0 and itera <iter_size:
                                  rnd_batch = random.randint(0,iter_size-1)
                                  maskm = mask[rnd_batch].cpu().detach().numpy()
                                  
                                  background= np.where(maskm  == 1)
                                  itera +=1
                              #print(mask.size(),'\n',maskm)
                              #print(background,len(background[0]),background[0][0])
                              if len(background[0])>0:
                                  rnd_batch = random.randint(0,len(background[0])-1)
                                  rnd_center = [background[0][rnd_batch],background[1][rnd_batch]]
                                  rnd_simi_size = random.randint(1,simi_size)
                                  bg = \
                                  simi_hm[batch_ind,:,int(max(0,rnd_center[0]-(rnd_simi_size-1)/2)):int(min(opt.input_h/4,rnd_center[0]+(rnd_simi_size-1)/2+1))\
                                            ,int(max(0,rnd_center[1]-(rnd_simi_size-1)/2)):int(min(opt.input_h/4,rnd_center[1]+(rnd_simi_size-1)/2+1))]
                                  
                                  bg = bg.view(1,channel,bg.size()[1],bg.size()[2])
                                  bg = torch.nn.functional.interpolate(bg, [simi_size_h,simi_size],mode ='bilinear')
                                  
                                  bg = torch.flatten(bg)
                                  
                                  Mask_hm[total_m+number] = bg
                              else:
                                  rnd = random.randint(0,total_m-1)
                                  Mask_hm[total_m+number] = Mask_hm[rnd]
              
              
              #print(total_f)
              #print(total_m)
              #print(Face_hm.size())
              #print(Mask_hm.size()) 
          
              pos_dist = l2_distance.forward(Anchor_hm, Face_hm)
              neg_dist = l2_distance.forward(Anchor_hm, Mask_hm)
              
              #print('pos_dist',pos_dist,pos_dist.size())
              #print('neg_dist',neg_dist,neg_dist.size())
              all_tri = (neg_dist - pos_dist < margin).cpu().numpy().flatten()
              hard_triplets = np.where(all_tri == 1)
              #print(hard_triplets)
              if len(hard_triplets[0]) == 0:
                  triplet_loss = triplet_loss
              else:
                  anc_hard_embedding = Anchor_hm[hard_triplets].cuda()
                  pos_hard_embedding = Face_hm[hard_triplets].cuda()
                  neg_hard_embedding = Mask_hm[hard_triplets].cuda()
                  triplet_loss = TripletLoss(margin=margint).forward(
                        anchor=anc_hard_embedding,
                        positive=pos_hard_embedding,
                        negative=neg_hard_embedding
                    ).cuda()
                  #print('triplet_loss',triplet_loss)
                
          #num_valid_training_triplets += len(anc_hard_embedding)
          #avg_triplet_loss = 0 if (num_valid_training_triplets == 0) else triplet_loss_sum / num_valid_training_triplets
          if use_mask_anchor:
              if total_f !=0 and total_m >1:
                  if match_size == total_f:
                      for number in range(total_f-int(total_m/2)):
                              rnd = random.randint(0,int(total_m/2)-1)
                              Anchor_hm_m[int(total_m/2)+number] = Anchor_hm_m[rnd]
                      for number in range(total_f-(total_m-int(total_m/2))):
                              rnd = random.randint(0,(total_m-int(total_m/2))-1)
                              Mask_hm_m[(total_m-int(total_m/2))+number] = Mask_hm_m[rnd]
                        
                  else:
                      if total_m%2 ==1:
                          rnd = random.randint(0,int(total_m/2)-1)
                          Anchor_hm_m[-1] = Anchor_hm_m[rnd]
                          if total_f<int(total_m/2)+1:
                              for number in range(int(total_m/2)+1-total_f):
                                  #rnd = random.randint(0,total_f-1)
                                  #Face_hm_m[total_f+number] = Face_hm_m[rnd]
                                  rnd_batch = random.randint(0,iter_size-1)
                                  maskm = mask[rnd_batch].cpu().detach().numpy()
                                  
                                  background= np.where(maskm  == 1)
                                  itera = 0
                                  while len(background[0])==0 and itera <iter_size:
                                      rnd_batch = random.randint(0,iter_size-1)
                                      maskm = mask[rnd_batch].cpu().detach().numpy()
                                      
                                      background= np.where(maskm  == 1)
                                      itera +=1
                                  #print(mask.size(),'\n',maskm)
                                  #print(background,len(background[0]),background[0][0])
                                  if len(background[0])>0:
                                      rnd_batch = random.randint(0,len(background[0])-1)
                                      rnd_center = [background[0][rnd_batch],background[1][rnd_batch]]
                                      rnd_simi_size = random.randint(1,simi_size)
                                      bg = \
                                      simi_hm[batch_ind,:,int(max(0,rnd_center[0]-(rnd_simi_size-1)/2)):int(min(opt.input_h/4,rnd_center[0]+(rnd_simi_size-1)/2+1))\
                                                ,int(max(0,rnd_center[1]-(rnd_simi_size-1)/2)):int(min(opt.input_h/4,rnd_center[1]+(rnd_simi_size-1)/2+1))]
                                      
                                      bg = bg.view(1,channel,bg.size()[1],bg.size()[2])
                                      bg = torch.nn.functional.interpolate(bg, [simi_size_h,simi_size],mode ='bilinear')
                                      
                                      bg = torch.flatten(bg)
                                      
                                      Face_hm_m[total_f+number] = bg
                                  else:
                                      rnd = random.randint(0,total_f-1)
                                      Face_hm_m[total_f+number] = Face_hm_m[rnd]
                      else:
                          if total_f<int(total_m/2):
                              for number in range(int(total_m/2)-total_f):
                                  #rnd = random.randint(0,total_f-1)
                                  #Face_hm_m[total_f+number] = Face_hm_m[rnd]
                                  
                                  rnd_batch = random.randint(0,iter_size-1)
                                  maskm = mask[rnd_batch].cpu().detach().numpy()
                                  
                                  background= np.where(maskm  == 1)
                                  itera = 0
                                  while len(background[0])==0 and itera <iter_size:
                                      rnd_batch = random.randint(0,iter_size-1)
                                      maskm = mask[rnd_batch].cpu().detach().numpy()
                                      
                                      background= np.where(maskm  == 1)
                                      itera +=1
                                  #print(mask.size(),'\n',maskm)
                                  #print(background,len(background[0]),background[0][0])
                                  if len(background[0])>0:
                                      rnd_batch = random.randint(0,len(background[0])-1)
                                      rnd_center = [background[0][rnd_batch],background[1][rnd_batch]]
                                      rnd_simi_size = random.randint(1,simi_size)
                                      bg = \
                                      simi_hm[batch_ind,:,int(max(0,rnd_center[0]-(rnd_simi_size-1)/2)):int(min(opt.input_h/4,rnd_center[0]+(rnd_simi_size-1)/2+1))\
                                                ,int(max(0,rnd_center[1]-(rnd_simi_size-1)/2)):int(min(opt.input_h/4,rnd_center[1]+(rnd_simi_size-1)/2+1))]
                                      
                                      bg = bg.view(1,channel,bg.size()[1],bg.size()[2])
                                      bg = torch.nn.functional.interpolate(bg, [simi_size_h,simi_size],mode ='bilinear')
                                      
                                      bg = torch.flatten(bg)
                                      
                                      Face_hm_m[total_f+number] = bg
                                  else:
                                      rnd = random.randint(0,total_f-1)
                                      Face_hm_m[total_f+number] = Face_hm_m[rnd]
                  
                  
                  #print(total_f)
                  #print(total_m)
                  #print(Face_hm.size())
                  #print(Mask_hm.size()) 
              
                  neg_dist = l2_distance.forward(Anchor_hm_m, Face_hm_m)
                  pos_dist = l2_distance.forward(Anchor_hm_m, Mask_hm_m)
                  
                  #print('pos_dist',pos_dist,pos_dist.size())
                  #print('neg_dist',neg_dist,neg_dist.size())
                  all_tri = (neg_dist - pos_dist < margin).cpu().numpy().flatten()
                  hard_triplets = np.where(all_tri == 1)
                  #print(hard_triplets)
                  if len(hard_triplets[0]) == 0:
                      triplet_loss_m = triplet_loss_m
                  else:
                      anc_hard_embedding = Anchor_hm_m[hard_triplets].cuda()
                      neg_hard_embedding = Face_hm_m[hard_triplets].cuda()
                      pos_hard_embedding = Mask_hm_m[hard_triplets].cuda()
                      triplet_loss_m = TripletLoss(margin=margint).forward(
                            anchor=anc_hard_embedding,
                            positive=pos_hard_embedding,
                            negative=neg_hard_embedding
                        ).cuda()
                      #print(triplet_loss_m)
      hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks          # 1. focal loss,求目标的中心，
      if opt.wh_weight > 0:
        wh_loss += self.crit_reg(output['wh'], batch['reg_mask'],batch['ind'], batch['wh'], batch['wight_mask']) / opt.num_stacks # 2. 人脸bbox高度和宽度的loss
        #wh_loss += self.crit_reg(output['wh'], batch['reg_mask'],batch['ind'], batch['wh']) / opt.num_stacks
      if opt.reg_offset and opt.off_weight > 0:
        off_loss += self.crit_reg(output['hm_offset'], batch['reg_mask'],             
                                  batch['ind'], batch['hm_offset'], batch['wight_mask']) / opt.num_stacks  # 3. 人脸bbox中心点下采样，所需要的偏差补偿
        #off_loss += self.crit_reg(output['hm_offset'], batch['reg_mask'],             
        #                          batch['ind'], batch['hm_offset']) / opt.num_stacks
      if opt.dense_hp:
        mask_weight = batch['dense_hps_mask'].sum() + 1e-4
        lm_loss += (self.crit_kp(output['hps'] * batch['dense_hps_mask'],
                                 batch['dense_hps'] * batch['dense_hps_mask']) / 
                                 mask_weight) / opt.num_stacks
      else:
        lm_loss += self.crit_kp(output['landmarks'], batch['hps_mask'],               # 4. 关节点的偏移
                                batch['ind'], batch['landmarks']) / opt.num_stacks

      # if opt.reg_hp_offset and opt.off_weight > 0:                              # 关节点的中心偏移
      #   hp_offset_loss += self.crit_reg(
      #     output['hp_offset'], batch['hp_mask'],
      #     batch['hp_ind'], batch['hp_offset']) / opt.num_stacks
      # if opt.hm_hp and opt.hm_hp_weight > 0:                                    # 关节点的热力图
      #   hm_hp_loss += self.crit_hm_hp(
      #     output['hm_hp'], batch['hm_hp']) / opt.num_stacks

    loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
           opt.off_weight * off_loss + opt.lm_weight * lm_loss+opt.simi_weight*triplet_loss+opt.simi_weight*triplet_loss_m
    if opt.simi:
        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'lm_loss': lm_loss,
                  'wh_loss': wh_loss, 'off_loss': off_loss,'triplet_loss':triplet_loss,'triplet_loss_m':triplet_loss_m}
    else:
        
        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'lm_loss': lm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss}
    return loss, loss_stats

class MultiPoseTrainer(BaseTrainer):
  def __init__(self, opt, model,model_t, optimizer=None):
    super(MultiPoseTrainer, self).__init__(opt, model,model_t, optimizer=optimizer)
  
  def _get_losses(self, opt):
    if opt.simi:
        loss_states = ['loss', 'hm_loss', 'lm_loss', 'wh_loss', 'off_loss','triplet_loss','triplet_loss_m']
    else:
        
        loss_states = ['loss', 'hm_loss', 'lm_loss', 'wh_loss', 'off_loss']
    loss = MultiPoseLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    hm_hp = output['hm_hp'] if opt.hm_hp else None
    hp_offset = output['hp_offset'] if opt.reg_hp_offset else None
    dets = multi_pose_decode(
      output['hm'], output['wh'], output['hps'], 
      reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

    dets[:, :, :4] *= opt.input_res / opt.output_res
    dets[:, :, 5:39] *= opt.input_res / opt.output_res
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :4] *= opt.input_res / opt.output_res
    dets_gt[:, :, 5:39] *= opt.input_res / opt.output_res
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')

      debugger.add_img(img, img_id='out_pred')
      for k in range(len(dets[i])):
        if dets[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                 dets[i, k, 4], img_id='out_pred')
          debugger.add_coco_hp(dets[i, k, 5:39], img_id='out_pred')

      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                 dets_gt[i, k, 4], img_id='out_gt')
          debugger.add_coco_hp(dets_gt[i, k, 5:39], img_id='out_gt')

      if opt.hm_hp:
        pred = debugger.gen_colormap_hp(output['hm_hp'][i].detach().cpu().numpy())
        gt = debugger.gen_colormap_hp(batch['hm_hp'][i].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hmhp')
        debugger.add_blend_img(img, gt, 'gt_hmhp')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    reg = output['reg'] if self.opt.reg_offset else None
    hm_hp = output['hm_hp'] if self.opt.hm_hp else None
    hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
    dets = multi_pose_decode(
      output['hm'], output['wh'], output['hps'], 
      reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    
    dets_out = multi_pose_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]