from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss,distillation_loss2
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer
import random
from torch.nn.modules.distance import PairwiseDistance
from torch.autograd import Function

way = 3
margint = 0.1 #margin for pos_dist and neg_dist 
margin = 0.01  # if neg dist - pos dist < margin choose this sample
use_mask_anchor = 1

def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
             else torch.arange(x.size(i)-1, -1, -1).long()
             for i in range(x.dim()))]
    
def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = torch.nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

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
def iou_numpy(outputs, labels):
    SMOOTH = 1e-6
    outputs = outputs.squeeze(1)
    
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    #thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return iou  # Or thresholded.mean()
class CtdetLoss(torch.nn.Module):
  def __init__(self, opt):
    super(CtdetLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
              NormRegL1Loss() if opt.norm_wh else \
              RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
    self.opt = opt
    self.kd = distillation_loss2()
    

  def forward(self, outputs, batch,outputs1 = None,outputs2 = None,outputs3 = None,outputs0 = None,outputs_t=None):
    opt = self.opt
    hm_loss, wh_loss, off_loss = 0, 0, 0
    consis_loss = 0
    ss_loss = torch.zeros(1).cuda()
    batch_acc = 0
    triplet_loss = torch.zeros(1).cuda()
    triplet_loss_m = torch.zeros(1).cuda()
    for s in range(opt.num_stacks):
      output = outputs[s]
      if opt.ss==1:

          output1 = outputs1[s]
          output2 = outputs2[s]
          output3 = outputs3[s]
          #print(output1['ss'].size())
          batch_size = output['ss'].size()[0]
          
          target = torch.arange(batch_size).unsqueeze(1).contiguous().view(-1).long().cuda()
          output_ss = output['ss']
          outputn_ss = output1['ss']
          
          simi = torch.nn.functional.cosine_similarity(outputn_ss.unsqueeze(2).expand(-1,-1,1*batch_size), output_ss.unsqueeze(2).expand(-1,-1,1*batch_size).transpose(0,2), dim=1)
          ss_loss += torch.nn.functional.cross_entropy(simi, target).cuda()
          batch_acc += accuracy(simi, target, topk=(1,))[0]
          
          outputn_ss = output2['ss']
          simi = torch.nn.functional.cosine_similarity(outputn_ss.unsqueeze(2).expand(-1,-1,1*batch_size), output_ss.unsqueeze(2).expand(-1,-1,1*batch_size).transpose(0,2), dim=1)
          ss_loss += torch.nn.functional.cross_entropy(simi, target).cuda()
          batch_acc += accuracy(simi, target, topk=(1,))[0]
          
          outputn_ss = output3['ss']
          simi = torch.nn.functional.cosine_similarity(outputn_ss.unsqueeze(2).expand(-1,-1,1*batch_size), output_ss.unsqueeze(2).expand(-1,-1,1*batch_size).transpose(0,2), dim=1)
          ss_loss += torch.nn.functional.cross_entropy(simi, target).cuda()
          batch_acc += accuracy(simi, target, topk=(1,))[0]
          
          batch_acc /=3
          
      elif opt.ss ==2:
          output0 = outputs0[s]
          output1 = outputs1[s]
          output2 = outputs2[s]
          output3 = outputs3[s]
          #print(output1['ss'].size())

          batch_size = output['ss'].size()[0]
          target = torch.arange(4).unsqueeze(1).expand(-1,batch_size).contiguous().view(-1).long().cuda()
          #target = torch.zeros(batch_size).unsqueeze(1).contiguous().view(-1).long().cuda()
          out = torch.cat((output0['ss'],output1['ss'],output2['ss'],output3['ss']),dim = 0)
          #print(target)
          #print(out[0])
          
          ss_loss += torch.nn.functional.cross_entropy(torch.nn.functional.softmax(out,dim = 1), target).cuda()
          batch_acc += accuracy(torch.nn.functional.softmax(out,dim = 1), target, topk=(1,))[0]


      if not opt.mse_loss:
        output['hm'] = _sigmoid(output['hm'])#.clone() ###
        

      if opt.eval_oracle_hm:
        output['hm'] = batch['hm']#.clone() ####
      if opt.eval_oracle_wh:
        output['wh'] = torch.from_numpy(gen_oracle_map(
          batch['wh'].detach().cpu().numpy(), 
          batch['ind'].detach().cpu().numpy(), 
          output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
      if opt.eval_oracle_offset:
        output['reg'] = torch.from_numpy(gen_oracle_map(
          batch['reg'].detach().cpu().numpy(), 
          batch['ind'].detach().cpu().numpy(), 
          output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

      if way ==3 and opt.simi !=0:
          channel=256
          norm = torch.nn.BatchNorm2d(channel).cuda()
          simi_hm = norm(output['simi'])
          face_ct = batch['bboxf']
          mask_ct = batch['bboxm']
          mask = batch['mask1d']

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
          MMask_hm=torch.zeros(1000,simi_size*simi_size_h*channel)
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
              
          Anchor_hm = torch.zeros(match_size,simi_size*simi_size_h*channel)
          Face_hm=torch.zeros(match_size,simi_size*simi_size_h*channel)
          Mask_hm=torch.zeros(match_size,simi_size*simi_size_h*channel)
          
        
          Anchor_hm[0:int(total_f/2)] = FFace_hm[0:int(total_f/2)]
          Face_hm[0:(total_f-int(total_f/2))] = FFace_hm[int(total_f/2):total_f]
          Mask_hm[0:total_m] = MMask_hm[0:total_m]
          
          
          
          del FFace_hm,MMask_hm#,FFace_hm_half,MMask_hm_half
          
          if total_f>1 and total_m !=0:
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
                              
                              rnd_batch = random.randint(0,iter_size-1)
                              maskm = mask[rnd_batch].cpu().detach().numpy()
                              
                              background= np.where(maskm  == 1)
                              itera = 0
                              while len(background[0])==0 and itera <iter_size:
                                  rnd_batch = random.randint(0,iter_size-1)
                                  maskm = mask[rnd_batch].cpu().detach().numpy()
                                  
                                  background= np.where(maskm  == 1)
                                  itera +=1
                              
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
                              
                              rnd_batch = random.randint(0,iter_size-1)
                              maskm = mask[rnd_batch].cpu().detach().numpy()
                              
                              background= np.where(maskm  == 1)
                              itera = 0
                              while len(background[0])==0 and itera <iter_size:
                                  rnd_batch = random.randint(0,iter_size-1)
                                  maskm = mask[rnd_batch].cpu().detach().numpy()
                                  
                                  background= np.where(maskm  == 1)
                                  itera +=1
                              
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
            
              pos_dist = l2_distance.forward(Anchor_hm, Face_hm)
              neg_dist = l2_distance.forward(Anchor_hm, Mask_hm)
              
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
                                  
                                  rnd_batch = random.randint(0,iter_size-1)
                                  maskm = mask[rnd_batch].cpu().detach().numpy()
                                  
                                  background= np.where(maskm  == 1)
                                  itera = 0
                                  while len(background[0])==0 and itera <iter_size:
                                      rnd_batch = random.randint(0,iter_size-1)
                                      maskm = mask[rnd_batch].cpu().detach().numpy()
                                      
                                      background= np.where(maskm  == 1)
                                      itera +=1
                                  
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
                                  
                                  rnd_batch = random.randint(0,iter_size-1)
                                  maskm = mask[rnd_batch].cpu().detach().numpy()
                                  
                                  background= np.where(maskm  == 1)
                                  itera = 0
                                  while len(background[0])==0 and itera <iter_size:
                                      rnd_batch = random.randint(0,iter_size-1)
                                      maskm = mask[rnd_batch].cpu().detach().numpy()
                                      
                                      background= np.where(maskm  == 1)
                                      itera +=1
                                  
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
               
                  neg_dist = l2_distance.forward(Anchor_hm_m, Face_hm_m)
                  pos_dist = l2_distance.forward(Anchor_hm_m, Mask_hm_m)
                  
                  
                  all_tri = (neg_dist - pos_dist < margin).cpu().numpy().flatten()
                  hard_triplets = np.where(all_tri == 1)
                  
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

      hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks            # 热力图损失
      #hm_loss = hm_loss+self.crit(output['hm'], batch['hm']) / opt.num_stacks
      if opt.wh_weight > 0:
        if opt.dense_wh:
          mask_weight = batch['dense_wh_mask'].sum() + 1e-4
          wh_loss += (
            self.crit_wh(output['wh'] * batch['dense_wh_mask'],
            batch['dense_wh'] * batch['dense_wh_mask']) / 
            mask_weight) / opt.num_stacks
          
        elif opt.cat_spec_wh:
          
          wh_loss = wh_loss+self.crit_wh(
            output['wh'], batch['cat_spec_mask'],
            batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
        else:
          wh_loss += self.crit_reg(
            output['wh'], batch['reg_mask'],
            batch['ind'], batch['wh']) / opt.num_stacks
          
      if opt.reg_offset and opt.off_weight > 0:
        off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                             batch['ind'], batch['reg']) / opt.num_stacks
        
      if opt.consis:
        output1 = outputs1[s]
        
        mask1d = batch['mask1d']>0
        
        mask1d = torch.unsqueeze(mask1d,1)
        
        output1['hm'] = flip(output1['hm'],3)
        
        if not opt.mse_loss:
            output1['hm'] = _sigmoid(output1['hm'])#.clone() ###
        norm  = mask1d.sum()*2
        batch_consis_loss = (torch.pow((output1['hm']-output['hm']),2)*mask1d).sum()/norm
        consis_loss += batch_consis_loss / opt.num_stacks
      
    
    loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
           opt.off_weight * off_loss +(triplet_loss+triplet_loss_m)*opt.simi_weight+ss_loss*opt.ss_weight+consis_loss*opt.consis_weight
    loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                  'wh_loss': wh_loss, 'off_loss': off_loss,'triplet_loss':triplet_loss,'triplet_loss_m':triplet_loss_m,'ss_loss':ss_loss,'batch_acc':batch_acc,'consis_loss':consis_loss}
    
    return loss, loss_stats

class CtdetTrainer(BaseTrainer):
  def __init__(self, opt, model,model_t, optimizer=None):
    super(CtdetTrainer, self).__init__(opt, model,model_t, optimizer=optimizer)
  
  def _get_losses(self, opt):
    if opt.consis:
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss','triplet_loss','triplet_loss_m','ss_loss','consis_loss']
    elif opt.ss:
        #loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss','simi_face','simi_mask','simi']
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss','triplet_loss','triplet_loss_m','ss_loss','batch_acc']
    elif opt.simi:
        #loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss','simi_face','simi_mask','simi']
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss','triplet_loss','triplet_loss_m']    
        
    else:
            
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']

    loss = CtdetLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=opt.cat_spec_wh, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets[:, :, :4] *= opt.down_ratio
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :4] *= opt.down_ratio
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

      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                 dets_gt[i, k, 4], img_id='out_gt')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    reg = output['reg'] if self.opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets_out = ctdet_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]