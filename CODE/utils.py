import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import utils as nn_utils

import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import utils as nn_utils
from torch.nn.parameter import Parameter

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from GCN import GCN
import datetime
from Focal_Loss import FocalLoss

import torch.backends.cudnn as cudnn
import random
import higher

L_EPS = 0.05
H_EPS = 666666

def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


def get_weights_h_const(model, opt, batch_data, meta_data, loss_function_type, \
                      loss_function_entity, loss_function_attribute, config, vnet_e, vnet_a, vopt):
    lr = config.OUTER_LR
    goal_type_seq, goal_type_len, goal_type_lbl,\
    goal_entity_seq, goal_entity_len, goal_entity_lbl, final_goal_type,\
    final_goal_entity, candidate_attribute, attribute_lbl = batch_data

    goal_type_seq_m, goal_type_len_m, goal_type_lbl_m,\
    goal_entity_seq_m, goal_entity_len_m, goal_entity_lbl_m, final_goal_type_m,\
    final_goal_entity_m, candidate_attribute_m, attribute_lbl_m = meta_data

    with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
         weights = torch.ones(goal_type_seq.shape[0], 3).to(config.DEVICE)

         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq.size(0))
         with torch.backends.cudnn.flags(enabled=False):
              type_pred, entity_pred, attribute_pred, consistent_loss = fmodel(
              goal_type_seq, goal_type_len, final_goal_type,
              goal_entity_seq, goal_entity_len, final_goal_entity,
              candidate_attribute)
         type_loss0 = loss_function_type(type_pred, goal_type_lbl)
         weights[:, 1] = 4*torch.sigmoid(-type_loss0)
         entity_loss0 = loss_function_type(entity_pred, goal_entity_lbl)
         weights[:, 2] = 4*torch.sigmoid(-entity_loss0)
    return weights.data

def get_weights_h_mlp(model, opt, batch_data, meta_data, loss_function_type, \
                      loss_function_entity, loss_function_attribute, config, vnet_e, vnet_a, vopt):
    lr = config.OUTER_LR
    goal_type_seq, goal_type_len, goal_type_lbl,\
    goal_entity_seq, goal_entity_len, goal_entity_lbl, final_goal_type,\
    final_goal_entity, candidate_attribute, attribute_lbl = batch_data

    goal_type_seq_m, goal_type_len_m, goal_type_lbl_m,\
    goal_entity_seq_m, goal_entity_len_m, goal_entity_lbl_m, final_goal_type_m,\
    final_goal_entity_m, candidate_attribute_m, attribute_lbl_m = meta_data

    with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
         weights = torch.ones(goal_type_seq.shape[0], 3).to(config.DEVICE)

         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq.size(0))
         with torch.backends.cudnn.flags(enabled=False):
              type_pred, entity_pred, attribute_pred, consistent_loss = fmodel(
              goal_type_seq, goal_type_len, final_goal_type,
              goal_entity_seq, goal_entity_len, final_goal_entity,
              candidate_attribute)
         type_loss0 = loss_function_type(type_pred, goal_type_lbl)
         weights[:, 1] = vnet_e(type_loss0.view(-1,1).data).squeeze()
         entity_loss0 = loss_function_type(entity_pred, goal_entity_lbl)
         weights[:, 2] = vnet_a(entity_loss0.view(-1,1).data).squeeze()

         
         type_loss = torch.mean(weights[:,0]*loss_function_type(type_pred, goal_type_lbl))
         entity_loss = torch.mean(weights[:,1]*loss_function_entity(entity_pred, goal_entity_lbl))
         attribute_loss = torch.mean(weights[:,2]*loss_function_attribute(attribute_pred, attribute_lbl))
         loss = type_loss + entity_loss + attribute_loss + consistent_loss
         fmodel.zero_grad()
         diffopt.step(loss)

         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq_m.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq_m.size(0))
         type_pred_m, entity_pred_m, attribute_pred_m, consistent_loss_m = fmodel(
              goal_type_seq_m, goal_type_len_m, final_goal_type_m,
              goal_entity_seq_m, goal_entity_len_m, final_goal_entity_m,
              candidate_attribute_m)
         type_loss_m = torch.mean(loss_function_type(type_pred_m, goal_type_lbl_m))
         entity_loss_m = torch.mean(loss_function_entity(entity_pred_m, goal_entity_lbl_m))
         attribute_loss_m = torch.mean(loss_function_attribute(attribute_pred_m, attribute_lbl_m))
         loss_m = type_loss_m + entity_loss_m + attribute_loss_m + consistent_loss_m

         vopt.zero_grad()
         loss_m.backward()
         vopt.step()
         weights[:, 1] = vnet_e(type_loss0.view(-1,1).data).squeeze()
         weights[:, 2] = vnet_a(entity_loss0.view(-1,1).data).squeeze()
         return weights.data   
    
def get_weights_m3(model, opt, batch_data, meta_data, loss_function_type, \
                      loss_function_entity, loss_function_attribute, config):
    lr = config.OUTER_LR
    goal_type_seq, goal_type_len, goal_type_lbl,\
    goal_entity_seq, goal_entity_len, goal_entity_lbl, final_goal_type,\
    final_goal_entity, candidate_attribute, attribute_lbl = batch_data

    goal_type_seq_m, goal_type_len_m, goal_type_lbl_m,\
    goal_entity_seq_m, goal_entity_len_m, goal_entity_lbl_m, final_goal_type_m,\
    final_goal_entity_m, candidate_attribute_m, attribute_lbl_m = meta_data

    with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
         weights = torch.ones(goal_type_seq.shape[0], 3).to(config.DEVICE)
         weights.requires_grad = True
         weights_in = torch.ones(goal_type_seq.shape[0], 3).to(config.DEVICE)
         weights_in[:,0] = weights[:,0]
         weights_in[:,1] = weights[:,0]*weights[:,1]
         weights_in[:,2] = weights[:,0]*weights[:,1]*weights[:,2]

         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq.size(0))
         with torch.backends.cudnn.flags(enabled=False):
              type_pred, entity_pred, attribute_pred = fmodel(
              goal_type_seq, goal_type_len, final_goal_type,
              goal_entity_seq, goal_entity_len, final_goal_entity,
              candidate_attribute)
         type_loss = torch.mean(weights_in[:,0]*loss_function_type(type_pred, goal_type_lbl))
         entity_loss = torch.mean(weights_in[:,1]*loss_function_entity(entity_pred, goal_entity_lbl))
         attribute_loss = torch.mean(weights_in[:,2]*loss_function_attribute(attribute_pred, attribute_lbl))
         loss = type_loss + entity_loss + attribute_loss
         fmodel.zero_grad()
         diffopt.step(loss)

         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq_m.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq_m.size(0))
         type_pred_m, entity_pred_m, attribute_pred_m = fmodel(
              goal_type_seq_m, goal_type_len_m, final_goal_type_m,
              goal_entity_seq_m, goal_entity_len_m, final_goal_entity_m,
              candidate_attribute_m)
         type_loss_m = torch.mean(loss_function_type(type_pred_m, goal_type_lbl_m))
         entity_loss_m = torch.mean(loss_function_entity(entity_pred_m, goal_entity_lbl_m))
         attribute_loss_m = torch.mean(loss_function_attribute(attribute_pred_m, attribute_lbl_m))
         loss_m = type_loss_m + entity_loss_m + attribute_loss_m

         grad = torch.autograd.grad(loss_m, weights)[0].data
         deno = torch.sum(torch.abs(grad))
         update = torch.clamp(lr*grad/deno, -config.CLIP, config.CLIP)
         weights.data = weights.data - update
         weights_in[:,0] = weights[:,0]
         weights_in[:,1] = weights[:,0]*weights[:,1]
         weights_in[:,2] = weights[:,0]*weights[:,1]*weights[:,2]
         #print(weights_in[0:10,:])
         return weights_in.data
 
    
def get_weights_m1(model, opt, batch_data, meta_data, loss_function_type, \
                      loss_function_entity, loss_function_attribute, config):
    lr = config.OUTER_LR
    goal_type_seq, goal_type_len, goal_type_lbl,\
    goal_entity_seq, goal_entity_len, goal_entity_lbl, final_goal_type,\
    final_goal_entity, candidate_attribute, attribute_lbl = batch_data

    goal_type_seq_m, goal_type_len_m, goal_type_lbl_m,\
    goal_entity_seq_m, goal_entity_len_m, goal_entity_lbl_m, final_goal_type_m,\
    final_goal_entity_m, candidate_attribute_m, attribute_lbl_m = meta_data

    with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
         weights = torch.ones(goal_type_seq.shape[0], 3).to(config.DEVICE)
         weights.requires_grad = True
  
         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq.size(0))
         with torch.backends.cudnn.flags(enabled=False):
              type_pred, entity_pred, attribute_pred = fmodel(
              goal_type_seq, goal_type_len, final_goal_type,
              goal_entity_seq, goal_entity_len, final_goal_entity,
              candidate_attribute)
         type_loss = torch.mean(weights[:,0]*loss_function_type(type_pred, goal_type_lbl))
         entity_loss = torch.mean(weights[:,1]*loss_function_entity(entity_pred, goal_entity_lbl))
         attribute_loss = torch.mean(weights[:,2]*loss_function_attribute(attribute_pred, attribute_lbl))
         loss = type_loss + entity_loss + attribute_loss

         fmodel.zero_grad() 
         diffopt.step(loss)

         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq_m.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq_m.size(0))
         type_pred_m, entity_pred_m, attribute_pred_m = fmodel(
              goal_type_seq_m, goal_type_len_m, final_goal_type_m,
              goal_entity_seq_m, goal_entity_len_m, final_goal_entity_m,
              candidate_attribute_m)
         type_loss_m = torch.mean(loss_function_type(type_pred_m, goal_type_lbl_m))
         entity_loss_m = torch.mean(loss_function_entity(entity_pred_m, goal_entity_lbl_m))
         attribute_loss_m = torch.mean(loss_function_attribute(attribute_pred_m, attribute_lbl_m))
         loss_m = type_loss_m + entity_loss_m + attribute_loss_m

         grad = torch.autograd.grad(loss_m, weights)[0].data
         deno = torch.sum(torch.abs(grad))
         update = torch.clamp(lr*grad/deno, -config.CLIP, config.CLIP)
         weights.data = weights.data - update
         #print(weights[:,0:10])
         return weights.data

def get_weights_1(model, opt, batch_data, meta_data, loss_function_type, \
                      loss_function_entity, loss_function_attribute, config):
    lr = config.OUTER_LR
    goal_type_seq, goal_type_len, goal_type_lbl,\
    goal_entity_seq, goal_entity_len, goal_entity_lbl, final_goal_type,\
    final_goal_entity, candidate_attribute, attribute_lbl = batch_data

    goal_type_seq_m, goal_type_len_m, goal_type_lbl_m,\
    goal_entity_seq_m, goal_entity_len_m, goal_entity_lbl_m, final_goal_type_m,\
    final_goal_entity_m, candidate_attribute_m, attribute_lbl_m = meta_data

    weights = torch.ones(goal_type_seq.shape[0], 3).to(config.DEVICE)
    weights.requires_grad = True

    with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq.size(0))
         with torch.backends.cudnn.flags(enabled=False):
              type_pred, entity_pred, attribute_pred = fmodel(
              goal_type_seq, goal_type_len, final_goal_type,
              goal_entity_seq, goal_entity_len, final_goal_entity,
              candidate_attribute)
         type_loss = torch.mean(weights[:,0]*loss_function_type(type_pred, goal_type_lbl))
         fmodel.zero_grad()
         diffopt.step(type_loss)

         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq_m.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq_m.size(0))
         type_pred_m, entity_pred_m, attribute_pred_m = fmodel(
              goal_type_seq_m, goal_type_len_m, final_goal_type_m,
              goal_entity_seq_m, goal_entity_len_m, final_goal_entity_m,
              candidate_attribute_m)
         type_loss_m = torch.mean(loss_function_type(type_pred_m, goal_type_lbl_m))
         grad = torch.autograd.grad(type_loss_m, weights)[0].data
         deno = torch.sum(torch.abs(grad))
         update = torch.clamp(lr*grad/deno, -config.CLIP, config.CLIP)
         weights.data = weights.data - update
    
    with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq.size(0))
         with torch.backends.cudnn.flags(enabled=False):
              type_pred, entity_pred, attribute_pred = fmodel(
              goal_type_seq, goal_type_len, final_goal_type,
              goal_entity_seq, goal_entity_len, final_goal_entity,
              candidate_attribute)
         entity_loss = torch.mean(weights[:,1]*loss_function_entity(entity_pred, goal_entity_lbl))
         fmodel.zero_grad()
         diffopt.step(entity_loss)

         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq_m.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq_m.size(0))
         type_pred_m, entity_pred_m, attribute_pred_m = fmodel(
              goal_type_seq_m, goal_type_len_m, final_goal_type_m,
              goal_entity_seq_m, goal_entity_len_m, final_goal_entity_m,
              candidate_attribute_m)
         entity_loss_m = torch.mean(loss_function_entity(entity_pred_m, goal_entity_lbl_m))
         grad = torch.autograd.grad(entity_loss_m, weights)[0].data
         deno = torch.sum(torch.abs(grad))
         update = torch.clamp(lr*grad/deno, -config.CLIP, config.CLIP)
         weights.data = weights.data - update
    
    with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq.size(0))
         with torch.backends.cudnn.flags(enabled=False):
              type_pred, entity_pred, attribute_pred = fmodel(
              goal_type_seq, goal_type_len, final_goal_type,
              goal_entity_seq, goal_entity_len, final_goal_entity,
              candidate_attribute)
         attribute_loss = torch.mean(weights[:,2]*loss_function_attribute(attribute_pred, attribute_lbl))
         fmodel.zero_grad()
         diffopt.step(attribute_loss)

         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq_m.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq_m.size(0))
         type_pred_m, entity_pred_m, attribute_pred_m = fmodel(
              goal_type_seq_m, goal_type_len_m, final_goal_type_m,
              goal_entity_seq_m, goal_entity_len_m, final_goal_entity_m,
              candidate_attribute_m) 
         attribute_loss_m = torch.mean(loss_function_attribute(attribute_pred_m, attribute_lbl_m))
         grad = torch.autograd.grad(attribute_loss_m, weights)[0].data
         deno = torch.sum(torch.abs(grad))
         update = torch.clamp(lr*grad/deno, -config.CLIP, config.CLIP)
         weights.data = weights.data - update
    return weights.data

def get_weights_3_mlp(model, opt, batch_data, meta_data, loss_function_type, \
                      loss_function_entity, loss_function_attribute, config, vnet_t, vopt_t, \
                     vnet_e, vopt_e, vnet_a, vopt_a):
    lr = config.OUTER_LR
    goal_type_seq, goal_type_len, goal_type_lbl,\
    goal_entity_seq, goal_entity_len, goal_entity_lbl, final_goal_type,\
    final_goal_entity, candidate_attribute, attribute_lbl = batch_data

    goal_type_seq_m, goal_type_len_m, goal_type_lbl_m,\
    goal_entity_seq_m, goal_entity_len_m, goal_entity_lbl_m, final_goal_type_m,\
    final_goal_entity_m, candidate_attribute_m, attribute_lbl_m = meta_data

    weights = torch.ones(goal_type_seq.shape[0], 3).to(config.DEVICE)

    with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq.size(0))
         with torch.backends.cudnn.flags(enabled=False):
              type_pred, entity_pred, attribute_pred = fmodel(
              goal_type_seq, goal_type_len, final_goal_type,
              goal_entity_seq, goal_entity_len, final_goal_entity,
              candidate_attribute)
         type_loss0 = loss_function_type(type_pred, goal_type_lbl)
         first_weights = vnet_t(type_loss0.view(-1,1).data).squeeze()
         type_loss = torch.mean(first_weights*type_loss0)
         fmodel.zero_grad()
         diffopt.step(type_loss)

         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq_m.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq_m.size(0))
         type_pred_m, entity_pred_m, attribute_pred_m = fmodel(
              goal_type_seq_m, goal_type_len_m, final_goal_type_m,
              goal_entity_seq_m, goal_entity_len_m, final_goal_entity_m,
              candidate_attribute_m)
         type_loss_m = torch.mean(loss_function_type(type_pred_m, goal_type_lbl_m))
         vopt_t.zero_grad()
         type_loss_m.backward()
         vopt_t.step()
         second_weights = vnet_t(type_loss0.view(-1,1).data).squeeze()
         weights[:, 0] = second_weights.data

    with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq.size(0))
         with torch.backends.cudnn.flags(enabled=False):
              type_pred, entity_pred, attribute_pred = fmodel(
              goal_type_seq, goal_type_len, final_goal_type,
              goal_entity_seq, goal_entity_len, final_goal_entity,
              candidate_attribute)
         entity_loss0 = loss_function_entity(entity_pred, goal_entity_lbl)
         first_weights = vnet_e(entity_loss0.view(-1,1).data).squeeze()*weights[:,0]
         #print("shape", first_weights.shape)
         entity_loss = torch.mean(first_weights*entity_loss0)
         fmodel.zero_grad()
         diffopt.step(entity_loss)

         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq_m.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq_m.size(0))
         type_pred_m, entity_pred_m, attribute_pred_m = fmodel(
              goal_type_seq_m, goal_type_len_m, final_goal_type_m,
              goal_entity_seq_m, goal_entity_len_m, final_goal_entity_m,
              candidate_attribute_m)
         entity_loss_m = torch.mean(loss_function_entity(entity_pred_m, goal_entity_lbl_m))
         vopt_e.zero_grad()
         entity_loss_m.backward()
         vopt_e.step()
         second_weights = vnet_e(entity_loss0.view(-1,1).data).squeeze()*weights[:,0]
         weights[:, 1] = second_weights.data

    with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq.size(0))
         with torch.backends.cudnn.flags(enabled=False):
              type_pred, entity_pred, attribute_pred = fmodel(
              goal_type_seq, goal_type_len, final_goal_type,
              goal_entity_seq, goal_entity_len, final_goal_entity,
              candidate_attribute)
         attribute_loss0 = loss_function_attribute(attribute_pred, attribute_lbl)
         first_weights = vnet_a(attribute_loss0.view(-1,1).data).squeeze()*weights[:,1]
         attribute_loss = torch.mean(first_weights*attribute_loss0)
         fmodel.zero_grad()
         diffopt.step(attribute_loss)

         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq_m.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq_m.size(0))
         type_pred_m, entity_pred_m, attribute_pred_m = fmodel(
              goal_type_seq_m, goal_type_len_m, final_goal_type_m,
              goal_entity_seq_m, goal_entity_len_m, final_goal_entity_m,
              candidate_attribute_m)
         attribute_loss_m = torch.mean(loss_function_attribute(attribute_pred_m, attribute_lbl_m))
         vopt_a.zero_grad()
         attribute_loss_m.backward()
         vopt_a.step()
         second_weights = vnet_a(attribute_loss0.view(-1,1).data).squeeze()*weights[:,1]
         weights[:, 2] = second_weights.data
    
    print("weights", weights[0:10,:])
    return weights.data


def get_weights_1_mlp(model, opt, batch_data, meta_data, loss_function_type, \
                      loss_function_entity, loss_function_attribute, config, vnet_t, vopt_t, \
                     vnet_e, vopt_e, vnet_a, vopt_a):
    lr = config.OUTER_LR
    goal_type_seq, goal_type_len, goal_type_lbl,\
    goal_entity_seq, goal_entity_len, goal_entity_lbl, final_goal_type,\
    final_goal_entity, candidate_attribute, attribute_lbl = batch_data

    goal_type_seq_m, goal_type_len_m, goal_type_lbl_m,\
    goal_entity_seq_m, goal_entity_len_m, goal_entity_lbl_m, final_goal_type_m,\
    final_goal_entity_m, candidate_attribute_m, attribute_lbl_m = meta_data
    
    weights = torch.ones(goal_type_seq.shape[0], 3).to(config.DEVICE)
    
    with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq.size(0))
         with torch.backends.cudnn.flags(enabled=False):
              type_pred, entity_pred, attribute_pred = fmodel(
              goal_type_seq, goal_type_len, final_goal_type,
              goal_entity_seq, goal_entity_len, final_goal_entity,
              candidate_attribute)
         type_loss0 = loss_function_type(type_pred, goal_type_lbl)
         first_weights = vnet_t(type_loss0.view(-1,1).data).squeeze()
         type_loss = torch.mean(first_weights*type_loss0)
         fmodel.zero_grad()
         diffopt.step(type_loss)

         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq_m.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq_m.size(0))
         type_pred_m, entity_pred_m, attribute_pred_m = fmodel(
              goal_type_seq_m, goal_type_len_m, final_goal_type_m,
              goal_entity_seq_m, goal_entity_len_m, final_goal_entity_m,
              candidate_attribute_m)
         type_loss_m = torch.mean(loss_function_type(type_pred_m, goal_type_lbl_m))
         vopt_t.zero_grad()
         type_loss_m.backward()
         vopt_t.step()
         second_weights = vnet_t(type_loss0.view(-1,1).data).squeeze()
         weights[:, 0] = second_weights.data

    with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq.size(0))
         with torch.backends.cudnn.flags(enabled=False):
              type_pred, entity_pred, attribute_pred = fmodel(
              goal_type_seq, goal_type_len, final_goal_type,
              goal_entity_seq, goal_entity_len, final_goal_entity,
              candidate_attribute)
         entity_loss0 = loss_function_entity(entity_pred, goal_entity_lbl)
         first_weights = vnet_e(entity_loss0.view(-1,1).data).squeeze()
         entity_loss = torch.mean(first_weights*entity_loss0)
         fmodel.zero_grad()
         diffopt.step(entity_loss)

         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq_m.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq_m.size(0))
         type_pred_m, entity_pred_m, attribute_pred_m = fmodel(
              goal_type_seq_m, goal_type_len_m, final_goal_type_m,
              goal_entity_seq_m, goal_entity_len_m, final_goal_entity_m,
              candidate_attribute_m)
         entity_loss_m = torch.mean(loss_function_entity(entity_pred_m, goal_entity_lbl_m))
         vopt_e.zero_grad()
         entity_loss_m.backward()
         vopt_e.step()
         second_weights = vnet_e(entity_loss0.view(-1,1).data).squeeze()
         weights[:, 1] = second_weights.data

    with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq.size(0))
         with torch.backends.cudnn.flags(enabled=False):
              type_pred, entity_pred, attribute_pred = fmodel(
              goal_type_seq, goal_type_len, final_goal_type,
              goal_entity_seq, goal_entity_len, final_goal_entity,
              candidate_attribute)
         attribute_loss0 = loss_function_attribute(attribute_pred, attribute_lbl)
         first_weights = vnet_a(attribute_loss0.view(-1,1).data).squeeze()
         attribute_loss = torch.mean(first_weights*attribute_loss0)
         fmodel.zero_grad()
         diffopt.step(attribute_loss)

         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq_m.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq_m.size(0))
         type_pred_m, entity_pred_m, attribute_pred_m = fmodel(
              goal_type_seq_m, goal_type_len_m, final_goal_type_m,
              goal_entity_seq_m, goal_entity_len_m, final_goal_entity_m,
              candidate_attribute_m)
         attribute_loss_m = torch.mean(loss_function_attribute(attribute_pred_m, attribute_lbl_m))
         vopt_a.zero_grad()
         attribute_loss_m.backward()
         vopt_a.step()
         second_weights = vnet_a(attribute_loss0.view(-1,1).data).squeeze()
         weights[:, 2] = second_weights.data
    #print("weights", weights[0:10,:])
    return weights.data

def get_weights_3(model, opt, batch_data, meta_data, loss_function_type, \
                      loss_function_entity, loss_function_attribute, config):
    lr = config.OUTER_LR
    goal_type_seq, goal_type_len, goal_type_lbl,\
    goal_entity_seq, goal_entity_len, goal_entity_lbl, final_goal_type,\
    final_goal_entity, candidate_attribute, attribute_lbl = batch_data

    goal_type_seq_m, goal_type_len_m, goal_type_lbl_m,\
    goal_entity_seq_m, goal_entity_len_m, goal_entity_lbl_m, final_goal_type_m,\
    final_goal_entity_m, candidate_attribute_m, attribute_lbl_m = meta_data

    weights = torch.ones(goal_type_seq.shape[0], 3).to(config.DEVICE)
    weights_t = torch.ones(goal_type_seq.shape[0]).to(config.DEVICE)
    weights_t.requires_grad = True
    with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq.size(0))
         with torch.backends.cudnn.flags(enabled=False):
              type_pred, entity_pred, attribute_pred = fmodel(
              goal_type_seq, goal_type_len, final_goal_type,
              goal_entity_seq, goal_entity_len, final_goal_entity,
              candidate_attribute)
         type_loss = torch.mean(weights_t*loss_function_type(type_pred, goal_type_lbl))
         fmodel.zero_grad()
         diffopt.step(type_loss)

         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq_m.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq_m.size(0))
         type_pred_m, entity_pred_m, attribute_pred_m = fmodel(
              goal_type_seq_m, goal_type_len_m, final_goal_type_m,
              goal_entity_seq_m, goal_entity_len_m, final_goal_entity_m,
              candidate_attribute_m)
         type_loss_m = torch.mean(loss_function_type(type_pred_m, goal_type_lbl_m))
         grad = torch.autograd.grad(type_loss_m, weights_t)[0].data
         deno = torch.sum(torch.abs(grad))
         update = torch.clamp(lr*grad/deno, -config.CLIP, config.CLIP)
         weights_t.data = torch.clamp(weights_t.data - update, L_EPS, H_EPS)
         weights[:,0] = torch.clone(weights_t.data)
    
    with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq.size(0))
         with torch.backends.cudnn.flags(enabled=False):
              type_pred, entity_pred, attribute_pred = fmodel(
              goal_type_seq, goal_type_len, final_goal_type,
              goal_entity_seq, goal_entity_len, final_goal_entity,
              candidate_attribute)
         entity_loss = torch.mean(weights_t*loss_function_entity(entity_pred, goal_entity_lbl))
         fmodel.zero_grad()
         diffopt.step(entity_loss)

         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq_m.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq_m.size(0))
         type_pred_m, entity_pred_m, attribute_pred_m = fmodel(
              goal_type_seq_m, goal_type_len_m, final_goal_type_m,
              goal_entity_seq_m, goal_entity_len_m, final_goal_entity_m,
              candidate_attribute_m)
         entity_loss_m = torch.mean(loss_function_entity(entity_pred_m, goal_entity_lbl_m))
         grad = torch.autograd.grad(entity_loss_m, weights_t)[0].data
         deno = torch.sum(torch.abs(grad))
   
         update = torch.clamp(lr*grad/deno, -config.CLIP, config.CLIP)
         weights_t.data = torch.clamp(weights_t.data - update, L_EPS, H_EPS)
         weights[:,1] = torch.clone(weights_t.data)
    
    with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq.size(0))
         with torch.backends.cudnn.flags(enabled=False):
              type_pred, entity_pred, attribute_pred = fmodel(
              goal_type_seq, goal_type_len, final_goal_type,
              goal_entity_seq, goal_entity_len, final_goal_entity,
              candidate_attribute)
         attribute_loss = torch.mean(weights_t*loss_function_attribute(attribute_pred, attribute_lbl))
         fmodel.zero_grad()
         diffopt.step(attribute_loss)

         fmodel.hidden_type = fmodel.init_hidden(goal_type_seq_m.size(0))
         fmodel.hidden_entity = fmodel.init_hidden(goal_entity_seq_m.size(0))
         type_pred_m, entity_pred_m, attribute_pred_m = fmodel(
              goal_type_seq_m, goal_type_len_m, final_goal_type_m,
              goal_entity_seq_m, goal_entity_len_m, final_goal_entity_m,
              candidate_attribute_m) 
         attribute_loss_m = torch.mean(loss_function_attribute(attribute_pred_m, attribute_lbl_m))
         grad = torch.autograd.grad(attribute_loss_m, weights_t)[0].data
         deno = torch.sum(torch.abs(grad))

         update = torch.clamp(lr*grad/deno, -config.CLIP, config.CLIP)
         weights_t.data = torch.clamp(weights_t.data - update, L_EPS, H_EPS)
         weights[:,2] = torch.clone(weights_t.data)
    return weights.data

def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
def cross_entropy_with_soft_label(logits, targets):
    return -torch.sum(F.log_softmax(logits, dim=1)*targets, dim=1)#/logits.shape[0]

def normalizeAdjacency(W):
    """
    NormalizeAdjacency: Computes the degree-normalized adjacency matrix

    Input:

        W (np.array): adjacency matrix

    Output:

        A (np.array): degree-normalized adjacency matrix
    """
    # Check that the matrix is square
    assert W.shape[0] == W.shape[1]
    # Compute the degree vector
    d = np.sum(W, axis = 1)
    # Invert the square root of the degree
    d = 1/(np.sqrt(d)+1e-6)
    # And build the square root inverse degree matrix
    D = np.diag(d)
    # Return the Normalized Adjacency
    return D @ W @ D

def pair_distance(a, b):
    sq_a = a**2
    sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)
    sq_b = b**2
    sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)
    bt = b.t()
    return torch.sqrt(sum_sq_a + sum_sq_b -2*a.mm(bt))

class VNet(nn.Module):
    def __init__(self, input, hidden, output):
        super(VNet, self).__init__()
        self.linear1 = nn.Linear(input, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden, output)
    def forward(self, loss):
        x = self.linear1(loss)
        x = self.relu(x)
        out = self.linear2(x)
        return 2*torch.sigmoid(out)


if __name__ == '__main__':
    a = -torch.rand(3, 2)
    b = torch.rand(4, 2)
    print(a, "\n", b, "\n", pair_distance(a, b)) 
