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
from gcn import GCN, GCN_S
import datetime
from focalloss import FocalLoss
import time
import torch.backends.cudnn as cudnn
import random
import higher
from utils import *
import argparse
parser = argparse.ArgumentParser(description='mw goal')
parser.add_argument('--mode', type=str, default='mw-h-mlp', help='base or mw or mw-1 or mw-3 or mw-3-mlp')
parser.add_argument('--attention', type=str, default='cross', help='self or cross or both')
parser.add_argument('--meta_num', type=int, default=100)
parser.add_argument('--outer_lr', type=float, default=1e-5)
parser.add_argument('--inner_lr', type=float, default=0.001)
parser.add_argument('--attention_lr', type=float, default=0.001)
parser.add_argument('--wd', type=float, default=1e-1)
parser.add_argument('--clip', type=float, default=.5)
parser.add_argument('--itself', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--consistent', type=float, default=0.000000002)
parser.add_argument('--soft', type=float, default=0.00)
parser.add_argument('--L', type=float, default=10.00)
parser.add_argument('--vnetsize', type=int, default=100)

L_EPS = 0.05
H_EPS = 666666
softmax = nn.Softmax(dim=1)
tanh = nn.Tanh()
pe = torch.load("pe.pt")
class Config(object):
    def __init__(self):
        self.EMBED_SIZE = 256
        self.HIDDEN_SIZE = 256#64
        self.STACKED_NUM = 3
        self.GOAL_TYPE_SIZE = 22
        #self.GOAL_ENTITY_SIZE = 1355
        self.GOAL_ENTITY_SIZE = 1385
        #self.ATTRIBUTE_SIZE = 20637
        self.ATTRIBUTE_SIZE = 8569
        #self.WORD_DICT_SIZE = 7068
        self.WORD_DICT_SIZE = 7551
        self.TRAIN_RATE = 0.7
        self.VAL_RATE = 0.15
        self.BATCH_SIZE = 128
        self.LEARNING_RATE = 0.001
        self.NUM_EPOCH =  30#100
        self.GCN_DROPOUT = 0.5
        self.GCN_HIDDEN = 128
        self.GCN_OUT = 256#64
        self.GAMMA = 2.0
        self.ALPHA = None
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.DATA_ROOT_PATH = "../../data/train_data/"
        self.PADDING_DATA_INDEX = [0]
        self.GRAPH_TYPE = "FUSE" # FUSE, OC, KG
        self.SAVE_PATH = "../../model_save/hierachical_next_goal.mdl"
        self.INFERENCE_SAVE_PATH = "../../data/model_output/"
    
        #
        self.META_NUM = 100
        self.MODE = 'mw-3-g'
        self.OUTER_LR = 0.01#300.0
        self.CLIP = 0.50
        self.ITSELF = 1

class Dataset(object):
    def __init__(self, data_tag, config):
        self.data_tag = data_tag
        self.config = config
        self.batch_size = config.BATCH_SIZE

        self.goal_type_label = np.array(self.file_reader(self.config.DATA_ROOT_PATH + self.data_tag + "_goal_type_label.txt"))
        goal_type_sequence = self.file_reader(self.config.DATA_ROOT_PATH + self.data_tag + "_goal_type_sequence.txt")
        print(goal_type_sequence[0], goal_type_sequence[1], goal_type_sequence[2])
        self.goal_type_sequence, self.goal_type_length = self.padding(goal_type_sequence)
        self.final_goal_type = self.file_reader(self.config.DATA_ROOT_PATH + self.data_tag + "_final_goal_type.txt")

        goal_entity_sequence = self.file_reader(self.config.DATA_ROOT_PATH + self.data_tag + "_goal_entity_sequence.txt")
        self.goal_entity_sequence, self.goal_entity_length = self.padding(goal_entity_sequence)
        self.goal_entity_label = self.file_reader(self.config.DATA_ROOT_PATH + self.data_tag + "_goal_entity_label.txt")
        self.final_goal_entity = self.file_reader(self.config.DATA_ROOT_PATH + self.data_tag + "_final_goal_entity.txt")

        self.candidate_attribute = np.load(self.config.DATA_ROOT_PATH + self.data_tag + "_candidate_attribute.npy")
       
        self.attribute_label = self.file_reader(self.config.DATA_ROOT_PATH + self.data_tag + "_all_attribute.txt")

        for idx in range(self.candidate_attribute.shape[0]):
            if self.attribute_label[idx] != 1:
                self.candidate_attribute[idx][1] = 0

    def get_all_data(self):
        data = TensorDataset(
            torch.LongTensor(self.goal_type_sequence).to(self.config.DEVICE),
            torch.LongTensor(self.goal_type_length).to(self.config.DEVICE),
            torch.LongTensor(self.goal_type_label).to(self.config.DEVICE),
            torch.LongTensor(self.goal_entity_sequence).to(self.config.DEVICE),
            torch.LongTensor(self.goal_entity_length).to(self.config.DEVICE),
            torch.LongTensor(self.goal_entity_label).to(self.config.DEVICE),
            torch.LongTensor(self.final_goal_type).to(self.config.DEVICE),
            torch.LongTensor(self.final_goal_entity).to(self.config.DEVICE),
            torch.Tensor(self.candidate_attribute).to(self.config.DEVICE),
            torch.LongTensor(self.attribute_label).to(self.config.DEVICE)
        )
        all_data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)#, num_workers=4)
        return all_data_loader
        # return data
    def get_meta_data(self):
        indexs = np.arange(len(self.goal_type_sequence))
        np.random.shuffle(indexs)
        selected = indexs[0:self.config.META_NUM]
        data = [torch.LongTensor(np.array(self.goal_type_sequence)[selected]).to(self.config.DEVICE),
            torch.LongTensor(np.array(self.goal_type_length)[selected]).to(self.config.DEVICE),
            torch.LongTensor(np.array(self.goal_type_label)[selected]).to(self.config.DEVICE),
            torch.LongTensor(np.array(self.goal_entity_sequence)[selected]).to(self.config.DEVICE),
            torch.LongTensor(np.array(self.goal_entity_length)[selected]).to(self.config.DEVICE),
            torch.LongTensor(np.array(self.goal_entity_label)[selected]).to(self.config.DEVICE),
            torch.LongTensor(np.array(self.final_goal_type)[selected]).to(self.config.DEVICE),
            torch.LongTensor(np.array(self.final_goal_entity)[selected]).to(self.config.DEVICE),
            torch.Tensor(np.array(self.candidate_attribute)[selected]).to(self.config.DEVICE),
            torch.LongTensor(np.array(self.attribute_label)[selected]).to(self.config.DEVICE)]
        return data
         
   
    def file_reader(self, file_path):
        data = None
        with open(file_path, "r") as f:
            data = eval(f.read())
            f.close()
        return data

    def padding(self, sequence):
        sequence = [item if isinstance(item, list) else [item] for item in sequence]

        sequence_length = [(idx, len(data)) for idx, data in enumerate(sequence)]
        max_length = max([item[1] for item in sequence_length])

        new_sequence = list()
        new_sequence_length = list()

        for idx, item in enumerate(sequence_length):
            seq = [u for u in sequence[item[0]]] + [0] * (max_length - len(sequence[item[0]]))
            new_sequence.append(seq)
            new_sequence_length.append(len(sequence[item[0]]))
        return new_sequence, new_sequence_length

    def cmp(self, a, b):
        if a[1] < b[1]:
            return 1
        elif a[1] > b[1]:
            return -1
        else:
            return 0


class NextGoal(nn.Module):
    def __init__(self, config):
        super(NextGoal, self).__init__()
        self.config = config
        self.hidden_type = self.init_hidden(self.config.BATCH_SIZE)
        self.hidden_entity = self.init_hidden(self.config.BATCH_SIZE)
        self.type_embedding = nn.Embedding(self.config.GOAL_TYPE_SIZE, self.config.EMBED_SIZE)
        self.entity_embedding = nn.Embedding(self.config.GOAL_ENTITY_SIZE, self.config.EMBED_SIZE)
        self.attribute_embedding = nn.Embedding(self.config.ATTRIBUTE_SIZE, self.config.EMBED_SIZE)

        # mask
        entity_attribute_mask = np.load(self.config.DATA_ROOT_PATH + "goal_entity_attribute_mask.npy")
        type_entity_mask = np.load(self.config.DATA_ROOT_PATH + "goal_type_entity_mask.npy")
        type_attribute_mask = np.load(self.config.DATA_ROOT_PATH + "goal_type_attribute_mask.npy")
 
        self.entity_attribute_mask_p = Variable(torch.Tensor(entity_attribute_mask), requires_grad=True).to(self.config.DEVICE)
        self.type_entity_mask_p = Variable(torch.Tensor(type_entity_mask), requires_grad=True).to(self.config.DEVICE)
        self.type_attribute_mask_p = Variable(torch.Tensor(type_attribute_mask), requires_grad=True).to(self.config.DEVICE)

        entity_attribute_mask[entity_attribute_mask == 0] = 0.1
        type_entity_mask[type_entity_mask == 0] = 0.1
        type_attribute_mask[type_attribute_mask == 0] = 0.1

        self.entity_attribute_mask = Variable(torch.Tensor(entity_attribute_mask), requires_grad=True).to(self.config.DEVICE)
        self.type_entity_mask = Variable(torch.Tensor(type_entity_mask), requires_grad=True).to(self.config.DEVICE)
        self.type_attribute_mask = Variable(torch.Tensor(type_attribute_mask), requires_grad=True).to(self.config.DEVICE)
        bianry_attribute_need_mask = np.zeros((2, self.config.ATTRIBUTE_SIZE))
        bianry_attribute_need_mask[1][1] = 1
        for idx in range(len(bianry_attribute_need_mask[0])):
            bianry_attribute_need_mask[0][idx] = 1
        bianry_attribute_need_mask[0][1] = 0
        self.bianry_attribute_need_mask = torch.Tensor(bianry_attribute_need_mask).to(self.config.DEVICE)

        # graph
        oc_type_graph = np.load(self.config.DATA_ROOT_PATH + "oc_goal_type_graph.npy")
        oc_entity_graph = np.load(self.config.DATA_ROOT_PATH + "oc_goal_entity_graph.npy")
        kg_type_graph = np.load(self.config.DATA_ROOT_PATH + "kg_goal_type_graph.npy")
        kg_entity_graph = np.load(self.config.DATA_ROOT_PATH + "kg_goal_entity_graph.npy")

        oc_type_graph = normalizeAdjacency(oc_type_graph)
        oc_entity_graph = normalizeAdjacency(oc_entity_graph)
        kg_type_graph = normalizeAdjacency(kg_type_graph)
        kg_entity_graph = normalizeAdjacency(kg_entity_graph)

        type_graph = self.graph_process(0.5*oc_type_graph + 0.5*kg_type_graph)
        entity_graph = self.graph_process(0.5*oc_entity_graph + 0.5*kg_entity_graph, flag=1)

        attribute_graph = np.load(self.config.DATA_ROOT_PATH + "goal_entity_attribute_graph.npy")
        attribute_graph = self.graph_process(attribute_graph, flag=1)

        self.type_graph = torch.Tensor(type_graph).to(self.config.DEVICE)
        self.entity_graph = torch.Tensor(entity_graph).to(self.config.DEVICE)
        self.attribute_graph = torch.Tensor(attribute_graph).to(self.config.DEVICE)
        # self.attribute_graph = self.graph_process(torch.Tensor(attribute_graph).to(self.config.DEVICE))

        # graph idx
        all_goal_type_idx = [idx for idx in range(0, self.config.GOAL_TYPE_SIZE)]
        all_goal_entity_idx = [idx for idx in range(0, self.config.GOAL_ENTITY_SIZE)]
        all_attribute_idx = [idx for idx in range(0, self.config.ATTRIBUTE_SIZE)]

        self.type_idx = torch.LongTensor(all_goal_type_idx).to(self.config.DEVICE)
        self.entity_idx = torch.LongTensor(all_goal_entity_idx).to(self.config.DEVICE)
        self.attribute_idx = torch.LongTensor(all_attribute_idx).to(self.config.DEVICE)

        #attention
        if (args.attention == 'self') or (args.attention == 'both'):
            self.type2type_attention = nn.Sequential(
                                       nn.Linear(self.config.EMBED_SIZE, self.config.EMBED_SIZE)
                                       )
            self.entity2entity_attention = nn.Sequential(
                                       nn.Linear(self.config.EMBED_SIZE, self.config.EMBED_SIZE)
                                       )
        if (args.attention == 'cross') or (args.attention == 'both'):
            self.type2entity_attention = nn.Sequential(
                                       nn.Linear(self.config.EMBED_SIZE, self.config.EMBED_SIZE)
                                       )
            self.entity2type_attention = nn.Sequential(
                                       nn.Linear(self.config.EMBED_SIZE, self.config.EMBED_SIZE)
                                       ) 

        # representation learning
        self.lstm_type = nn.LSTM(
            input_size=self.config.GCN_OUT,
            hidden_size=self.config.GCN_OUT,
            num_layers=self.config.STACKED_NUM,
            batch_first=True
        )
        self.lstm_entity = nn.LSTM(
            input_size=self.config.GCN_OUT,
            hidden_size=self.config.GCN_OUT,
            num_layers=self.config.STACKED_NUM,
            batch_first=True
        )
        self.mlp_type = nn.Sequential(
            nn.Linear(2 * self.config.GCN_OUT, 16),
            nn.ReLU(),
            nn.Linear(16, self.config.GOAL_TYPE_SIZE)
        )
        self.mlp_entity = nn.Sequential(
            nn.Linear(2 * self.config.GCN_OUT, 16),
            nn.ReLU(),
            nn.Linear(16, self.config.GOAL_ENTITY_SIZE)
        )
        self.mlp_attribute = nn.Sequential(
            nn.Linear(self.config.GCN_OUT, 32),
            nn.ReLU(),
            nn.Linear(32, self.config.ATTRIBUTE_SIZE)
        )


    def my_transformer(self, goal_seq_emb, goal_seq_len, query):
        attention = torch.sum(goal_seq_emb*query.unsqueeze(1), 2)
        mask = length_to_mask(goal_seq_len, goal_seq_emb.shape[1])
        #print("goal_seq_emb shape {} goal_seq_len shap {} query shape {} mask shape {} attention shape {}".format(goal_seq_emb.shape, goal_seq_len.shape, query.shape, mask.shape, attention.shape))
        attention[mask == False] = -999999999
        #print("attention shape {}".format(attention.shape))
        attention = softmax(attention/16).unsqueeze(2)
        #print("attention {}".format(attention[0,:]))
        attention_input = torch.sum(attention*goal_seq_emb, 1)
        return attention_input
              
    def forward(self, goal_type_seq, goal_type_len, final_goal_type, 
                goal_entity_seq, goal_entity_len, final_goal_entity, candidate_attribute):
        #hete
        type_embed_hete = self.type_embedding(self.type_idx)
        entity_embed_hete = self.entity_embedding(self.entity_idx)
        attribute_embed_hete = self.attribute_embedding(self.attribute_idx)

        after_gcn_type_embed = type_embed_hete
        after_gcn_entity_embed = entity_embed_hete
        after_gcn_attribute_embed = attribute_embed_hete

        # goal type query embedding
        type_input = nn_utils.rnn.pack_padded_sequence(
            after_gcn_type_embed[goal_type_seq], goal_type_len.cpu(), batch_first=True, enforce_sorted=False)
        lstm_type_output, self.hidden_type = self.lstm_type(type_input, self.hidden_type)
        type_output, _ = nn_utils.rnn.pad_packed_sequence(lstm_type_output, batch_first=True)
        last_timestep_type = self.last_timestep(type_output, goal_type_len)
        # goal entity query embedding
        entity_input = nn_utils.rnn.pack_padded_sequence(
            after_gcn_entity_embed[goal_entity_seq], goal_entity_len.cpu(), batch_first=True, enforce_sorted=False)
        lstm_entity_output, self.hidden_entity = self.lstm_entity(entity_input, self.hidden_entity)
        entity_output, _ = nn_utils.rnn.pad_packed_sequence(lstm_entity_output, batch_first=True)
        last_timestep_entity = self.last_timestep(entity_output, goal_entity_len)
        
        pos_embedding = pe[0:goal_type_seq.shape[1],:].unsqueeze(0)
        #goal attention
        #print("goal seq shape {} goal type len shape {} data {}".format(after_gcn_type_embed[goal_type_seq].shape, goal_type_len.shape, goal_type_len))
        type_cross_attention_input = self.my_transformer(self.entity2type_attention(after_gcn_type_embed[goal_type_seq] + pos_embedding), goal_type_len, last_timestep_entity)
        entity_cross_attention_input = self.my_transformer(self.type2entity_attention(after_gcn_entity_embed[goal_entity_seq] + pos_embedding), goal_entity_len, last_timestep_type)
        #goal type prediction
        mlp_type_input = torch.cat([last_timestep_type, type_cross_attention_input], dim=1)
        type_prediction = self.mlp_type(mlp_type_input)
        #goal entity prediction
        mlp_entity_input = torch.cat([last_timestep_entity, entity_cross_attention_input], dim=1)
        entity_prediction = self.mlp_entity(mlp_entity_input)
        entity_prediction = entity_prediction + 5.0*tanh(torch.mm(type_prediction, self.type_entity_mask))
        #goal attribute prediction
        mlp_attribute_input = last_timestep_entity
        attribute_prediction = self.mlp_attribute(mlp_attribute_input)
        attribute_prediction = attribute_prediction + 5.0*tanh(torch.mm(entity_prediction, self.entity_attribute_mask))
        attribute_prediction[candidate_attribute == 0] = -99999999999999999
        #print("sum", torch.sum(mlp_entity_input))
        return type_prediction, entity_prediction, attribute_prediction, 0

    def last_timestep(self, unpacked, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(unpacked.size(0),
                                               unpacked.size(2)).unsqueeze(1)
        return unpacked.gather(1, idx).squeeze()

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.config.STACKED_NUM, batch_size, self.config.HIDDEN_SIZE).to(self.config.DEVICE),
            torch.zeros(self.config.STACKED_NUM, batch_size, self.config.HIDDEN_SIZE).to(self.config.DEVICE))

    def graph_process(self, graph, flag=0):
        if flag == 0:
            graph[graph > 1] = 1
        elif flag == 1:
            graph = graph / (np.max(graph, axis=0) + 1e-6)
        return graph


def evaluation(y_pred, y_true, flag="macro"):
    return accuracy_score(y_true, y_pred), recall_score(y_true, y_pred, average=flag), precision_score(y_true, y_pred, average=flag), f1_score(y_true, y_pred, average=flag)


def get_prediction(y_pred):
    prediction = np.argmax(y_pred, axis=1)
    return prediction


def get_all_evaluation(a_pred, a_true, b_pred, b_true):
    ans = 0.
    for idx in range(len(a_pred)):
        if a_pred[idx] == a_true[idx] and b_pred[idx] == b_true[idx]:
            ans += 1
    return ans / len(a_pred)


def file_saver(path, obj):
    with open(path, "w") as f:
        f.write(str(obj))
        f.close()

def train(train_data, val_data, config):
    print("Model Training...")
    print("Model Saved at " + config.SAVE_PATH)

    model = NextGoal(config)
    if torch.cuda.is_available():
        model = model.to(config.DEVICE)
    model = model.cuda()
    #print(model)
    attention_params = list(map(id, model.type2entity_attention.parameters())) + list(map(id, model.entity2type_attention.parameters()))
    base_params = filter(lambda p: id(p) not in attention_params, model.parameters())
    my_params = [
              {'params': base_params, 'lr': config.LEARNING_RATE},
              {'params': model.type2entity_attention.parameters(), 'lr': args.attention_lr},
              {'params': model.entity2type_attention.parameters(), 'lr': args.attention_lr},
        ]
    #optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE) # weight_decay=0.0001
    optimizer = optim.Adam(my_params)
    #optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE) # weight_decay=0.0001
    loss_function_type = torch.nn.CrossEntropyLoss(reduction='none')
    loss_function_entity = torch.nn.CrossEntropyLoss(reduction='none')
    loss_function_attribute = FocalLoss(config.GAMMA, config.ALPHA, size_average=False)
    # loss_function_need_attr = torch.nn.CrossEntropyLoss()
    vnet_entity = VNet(1, args.vnetsize, 1).cuda()
    vnet_attr = VNet(1, args.vnetsize, 1).cuda()
    vnet_optimizer = optim.Adam([{'params': vnet_entity.parameters()}, {'params': vnet_attr.parameters()}], lr=config.OUTER_LR, weight_decay=args.wd)

    max_type_acc = -1
    max_entity_acc = -1
    max_epoch = 0
    for epoch in range(config.NUM_EPOCH):
        #if epoch == 5:
        #   exit(0)
        start_time = datetime.datetime.now()
        lr = config.LEARNING_RATE*(np.cos(epoch*1.0/(config.NUM_EPOCH*1.0)*np.pi)+1)/2  
        for p in optimizer.param_groups:
            p['lr'] = lr
        print(epoch, lr)
        epoch_train_loss = list()
        for idx, batch_data in enumerate(train_data):
            if config.ITSELF:
               meta_data = batch_data
            else:
               meta_data = Dataset(data_tag="train", config=config).get_meta_data()
            goal_type_seq, goal_type_len, goal_type_lbl,\
            goal_entity_seq, goal_entity_len, goal_entity_lbl, final_goal_type,\
            final_goal_entity, candidate_attribute, attribute_lbl = batch_data
            
            if config.MODE == 'base':
                weights = torch.ones(goal_type_seq.shape[0], 3).to(config.DEVICE)
            elif config.MODE == 'mw-h-const':
                with torch.backends.cudnn.flags(enabled=False):
                    weights = get_weights_h_const(model, optimizer, batch_data, meta_data, loss_function_type, \
                              loss_function_entity, loss_function_attribute, config, vnet_entity, vnet_attr, vnet_optimizer)
            elif config.MODE == 'mw-h-mlp':
                with torch.backends.cudnn.flags(enabled=False):
                    weights = get_weights_h_mlp(model, optimizer, batch_data, meta_data, loss_function_type, \
                              loss_function_entity, loss_function_attribute, config, vnet_entity, vnet_attr, vnet_optimizer)
            if idx == 0:
               print(weights[0:10])
            model.hidden_type = model.init_hidden(goal_type_seq.size(0))
            model.hidden_entity = model.init_hidden(goal_entity_seq.size(0))
            type_pred, entity_pred, attribute_pred, consistent_loss = model(
                goal_type_seq, goal_type_len, final_goal_type,
                goal_entity_seq, goal_entity_len, final_goal_entity,
                candidate_attribute)

            indexs = torch.arange(goal_type_lbl.shape[0])
            soft = args.soft*torch.clamp(goal_type_len.float()/args.L, 0, 1)
            type_one_hot = torch.zeros(goal_type_lbl.shape[0], type_pred.shape[1]).cuda().scatter_(1, goal_type_lbl.reshape(-1,1), 1).cuda()
            type_one_hot[indexs, goal_type_lbl] = 1-soft
            type_one_hot[indexs, final_goal_type] = type_one_hot[indexs, final_goal_type] + soft
            type_loss = torch.mean(weights[:,0]*cross_entropy_with_soft_label(type_pred, type_one_hot))
            soft = args.soft*torch.clamp(goal_entity_len.float()/args.L, 0, 1)
            entity_one_hot = torch.zeros(goal_entity_lbl.shape[0], entity_pred.shape[1]).cuda().scatter_(1, goal_entity_lbl.reshape(-1,1), 1).cuda()
            entity_one_hot[indexs, goal_entity_lbl] = 1-soft
            entity_one_hot[indexs, final_goal_entity] = entity_one_hot[indexs, final_goal_entity] + soft
            entity_loss = torch.mean(weights[:,1]*cross_entropy_with_soft_label(entity_pred, entity_one_hot))
            attribute_loss = torch.mean(weights[:,2]*loss_function_attribute(attribute_pred, attribute_lbl))
            loss = type_loss + entity_loss + attribute_loss + consistent_loss
               
            epoch_train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # evaluation
        with torch.no_grad():
            val_pred, val_true = list(), list()
            val_entity_pred, val_entity_true = list(), list()
            val_attr_pred, val_attr_true = list(), list()
            # val_need_attr_pred, val_need_attr_true = list(), list()

            for idx, (goal_type_seq, goal_type_len, goal_type_lbl,\
             goal_entity_seq, goal_entity_len, goal_entity_lbl, final_goal_type,\
              final_goal_entity, candidate_attribute, attribute_lbl) in enumerate(val_data):
                model.hidden_type = model.init_hidden(goal_type_seq.size(0))
                model.hidden_entity = model.init_hidden(goal_entity_seq.size(0))
                type_pred, entity_pred, attr_pred, _ = model(
                    goal_type_seq, goal_type_len, final_goal_type,
                    goal_entity_seq, goal_entity_len, final_goal_entity,
                    candidate_attribute)
                val_pred += torch.argmax(type_pred, dim=1).cpu().numpy().tolist()
                val_true += goal_type_lbl.cpu().numpy().tolist()

                val_entity_pred += torch.argmax(entity_pred, dim=1).cpu().numpy().tolist()
                val_entity_true += goal_entity_lbl.cpu().numpy().tolist()

                val_attr_pred += torch.argmax(attr_pred, dim=1).cpu().numpy().tolist()
                val_attr_true += attribute_lbl.cpu().numpy().tolist()

            val_acc, val_rec, val_pre, val_f1 = evaluation(val_pred, val_true)
            val_entity_acc, val_entity_rec, val_entity_pre, val_entity_f1 = evaluation(val_entity_pred, val_entity_true)
            val_attr_acc, val_attr_rec, val_attr_pre, val_attr_f1 = evaluation(val_attr_pred, val_attr_true)

            print("----------------------------------------------------------------------------------")
            print("EPOCH-%d, TRAIN_LOSS: %.4f" % (epoch, np.mean(epoch_train_loss)))
            print("     GOAL TYPE | VAL_ACC: %.4f, VAL_REC: %.4f, VAL_PRE: %.4f, VAL_F1: %.4f" % (val_acc, val_rec, val_pre, val_f1))
            print("   GOAL ENTITY | VAL_ACC: %.4f, VAL_REC: %.4f, VAL_PRE: %.4f, VAL_F1: %.4f" % (val_entity_acc, val_entity_rec, val_entity_pre, val_entity_f1))
            print("     ATTRIBUTE | VAL_ACC: %.4f, VAL_REC: %.4f, VAL_PRE: %.4f, VAL_F1: %.4f" % (val_attr_acc, val_attr_rec, val_attr_pre, val_attr_f1))
            print("----------------------------------------------------------------------------------")
            
            if val_entity_acc >= max_entity_acc:
                max_type_acc = val_acc
                max_entity_acc = val_entity_acc
                max_epoch = epoch
                torch.save(model.state_dict(), config.SAVE_PATH)
    print("max_type_acc {} max_entity_acc {} max_epoch {}".format(max_type_acc, max_entity_acc, max_epoch))


def test(test_data, config, data_tag, save=False):
    model = NextGoal(config)
    model.load_state_dict(torch.load(config.SAVE_PATH))
    model.eval()
    #if torch.cuda.is_available():
    model = model.to(config.DEVICE)
    
    with torch.no_grad():
        test_pred, test_true = list(), list()
        test_entity_pred, test_entity_true = list(), list()
        test_attr_pred, test_attr_true = list(), list()
        # test_need_attr_pred, test_need_attr_true = list(), list()

        cnt = 0
        # start_time = datetime.datetime.now()
        for idx, (goal_type_seq, goal_type_len, goal_type_lbl,\
            goal_entity_seq, goal_entity_len, goal_entity_lbl, final_goal_type,\
             final_goal_entity, candidate_attribute, attribute_lbl) in enumerate(test_data):
            # if goal_type_len.item() != 1:
            #     continue
            # if cnt == 10:
            #     break
            # cnt += 2
            model.hidden_type = model.init_hidden(goal_type_seq.size(0))
            model.hidden_entity = model.init_hidden(goal_entity_seq.size(0))
            type_pred, entity_pred, attr_pred, _ = model(
                goal_type_seq, goal_type_len, final_goal_type, 
                goal_entity_seq, goal_entity_len, final_goal_entity, 
                candidate_attribute)
            test_pred += torch.argmax(type_pred, dim=1).cpu().numpy().tolist()
            test_true += goal_type_lbl.cpu().numpy().tolist()

            test_entity_pred += torch.argmax(entity_pred, dim=1).cpu().numpy().tolist()
            test_entity_true += goal_entity_lbl.cpu().numpy().tolist()

            test_attr_pred += torch.argmax(attr_pred, dim=1).cpu().numpy().tolist()
            test_attr_true += attribute_lbl.cpu().numpy().tolist()
            # test_need_attr_pred += torch.argmax(need_attr_pred, dim=1).cpu().numpy().tolist()
            # test_need_attr_true += need_attribute.cpu().numpy().tolist()
            # break

        # end_time = datetime.datetime.now()
        # print("Latency: %f s\n" % ((end_time - start_time).total_seconds()))
        # return (end_time - start_time).total_seconds()
        #print("test_pred {} test_pred perc {}".format(test_pred[0:100], np.sum(np.array(test_pred[0:100]) == 19)))
        test_acc, test_rec, test_pre, test_f1 = evaluation(test_pred, test_true)
        test_entity_acc, test_entity_rec, test_entity_pre, test_entity_f1 = evaluation(test_entity_pred, test_entity_true)
        test_attr_acc, test_attr_rec, test_attr_pre, test_attr_f1 = evaluation(test_attr_pred, test_attr_true)
        # test_need_acc, test_need_rec, test_need_pre, test_need_f1 = evaluation(test_need_attr_pred, test_need_attr_true)
        print("TEST      GOAL TYPE | TEST_ACC: %.4f, TEST_REC: %.4f, TEST_PRE: %.4f, TEST_F1: %.4f" % (
            test_acc, test_rec, test_pre, test_f1
        ))
        print("TEST    GOAL ENTITY | TEST_ACC: %.4f, TEST_REC: %.4f, TEST_PRE: %.4f, TEST_F1: %.4f" % (
            test_entity_acc, test_entity_rec, test_entity_pre, test_entity_f1
        ))
        print("TEST      ATTRIBUTE | TEST_ACC: %.4f, TEST_REC: %.4f, TEST_PRE: %.4f, TEST_F1: %.4f" % (
            test_attr_acc, test_attr_rec, test_attr_pre, test_attr_f1
        ))
        HR_T = np.sum(np.array(test_pred) == 19)/len(test_pred)
        HR_E = np.sum(np.array(test_entity_pred) == 1123)/len(test_entity_pred)
        print("Type hit ratio {} Entity hit ratio {}".format(HR_T, HR_E))

        # # print("TEST NEED ATTRIBUTE | TEST_ACC: %.4f, TEST_REC: %.4f, TEST_PRE: %.4f, TEST_F1: %.4f" % (
        # #     test_need_acc, test_need_rec, test_need_pre, test_need_f1
        # # ))
        
        if save is True:
            file_saver(config.INFERENCE_SAVE_PATH + data_tag + "_next_goal_type_pred.out", test_pred)
            file_saver(config.INFERENCE_SAVE_PATH + data_tag + "_next_goal_type_true.out", test_true)
            file_saver(config.INFERENCE_SAVE_PATH + data_tag + "_next_goal_entity_pred.out", test_entity_pred)
            file_saver(config.INFERENCE_SAVE_PATH + data_tag + "_next_goal_entity_true.out", test_entity_true)
            file_saver(config.INFERENCE_SAVE_PATH + data_tag + "_next_attribute_pred.out", test_attr_pred)
            file_saver(config.INFERENCE_SAVE_PATH + data_tag + "_next_attribute_true.out", test_attr_true)
            # file_saver(config.INFERENCE_SAVE_PATH + "binary_need_attribute_pred.out", test_need_attr_pred)
            # file_saver(config.INFERENCE_SAVE_PATH + "binary_need_attribute_true.out", test_need_attr_true)

if __name__ == "__main__":
    config = Config()
    args = parser.parse_args()
    set_seed(args.seed)
    config.MODE = args.mode
    config.META_NUM = args.meta_num
    config.OUTER_LR = args.outer_lr
    config.LEARNING_RATE = args.inner_lr
    config.ITSELF = args.itself
    config.CLIP = args.clip
    print(config.__dict__)

    train_data = Dataset(data_tag="train", config=config).get_all_data()
    val_data = Dataset(data_tag="val", config=config).get_all_data()
    test_data = Dataset(data_tag="test", config=config).get_all_data()
    print("Train Data: (%d, %d), Val Data: (%d, %d), Test Data: (%d, %d)" % (
        len(train_data), config.BATCH_SIZE, 
        len(val_data), config.BATCH_SIZE, 
        len(test_data), config.BATCH_SIZE))
    
    #train(train_data, val_data, config)
    start = time.time()
    test(test_data, config, "test", save=True)
    print("inference time", time.time() - start)
    

