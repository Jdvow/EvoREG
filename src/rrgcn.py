import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer, NewRGCNLayer
from src.model import BaseRGCN, EvolveRGCN_O, EvolveRGCN_H
from src.decoder import ConvTransE, ConvTransR, TimeConvTransE, TimeConvTransR
from src.segnn import SE_GNN

import sys
import scipy.sparse as sp
sys.path.append("..")

class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx): # 实现了BaseRGCN中的build_hidden_layer
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn": # 2层的UnionRGCNLayer
            # num_rels*2
            # return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases, activation=act, 
            #                       dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
            return NewRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases, need_neighbor_weight=True, 
                                need_loop_weight=True, need_skip_weight=True, activation=act, 
                                dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        # g: 当前历史子图; self.h: node嵌入 (num_ents, h_dim); [self.h_0, self.h_0]: 边的嵌入
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze() # node id
            g.ndata['h'] = init_ent_emb[node_id] # node embedding
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers): # n_layers = 2 两层的UnionRGCNLayer
                layer(g, [], r[i]) # g: 当前历史子图; r[i]: self.h_0 (num_rels*2, h_dim) 更新了两轮的g.ndata['h']
            return g.ndata.pop('h') # 返回了图中更新的node embedding
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')



class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, h_dim, opn, time_interval,
                 sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', use_cuda=False, gpu = 0, analysis=False,
                 segnn=False, dataset='ICEWS14', kg_layer=2, bn=False, comp_op='mul', ent_drop=0.2, rel_drop=0.1,
                 num_words=0, num_static_rels=0, weight=1, discount=0, angle=0, use_static=False):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name # convtranse
        self.encoder_name = encoder_name # uvrgcn
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.time_interval = time_interval
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.emb_rel = None
        self.gpu = gpu
        self.sin = torch.sin

        # static parameters
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.p_rel = torch.nn.Parameter(torch.Tensor(4*2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.p_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float() # 所有实体的进化嵌入
        torch.nn.init.normal_(self.dynamic_emb)

        # for time-gate
        self.rel_time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_normal_(self.rel_time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.rel_time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.rel_time_gate_bias)

        self.node_time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_normal_(self.node_time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.node_time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.node_time_gate_bias)

        # for time encoder
        self.weight_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_e = torch.nn.CrossEntropyLoss()
        self.loss_r = torch.nn.CrossEntropyLoss()

        # self.rgcn = RGCNCell(num_ents, h_dim, h_dim, num_rels * 2, num_bases, num_basis, num_hidden_layers, dropout, 
        #                      self_loop, skip_connect, encoder_name, self.opn, self.emb_rel, use_cuda, analysis)
        # self.super_rgcn = RGCNCell(num_ents, h_dim, h_dim, num_rels * 2, num_bases, num_basis, num_hidden_layers, dropout, 
        #                            self_loop, skip_connect, encoder_name, self.opn, self.emb_rel, use_cuda, analysis)

        self.rgcn = EvolveRGCN_O(num_ents, num_rels*2, h_dim, h_dim, num_bases, num_basis, num_hidden_layers, dropout,
                                    self_loop, skip_connect, encoder_name, self.opn, self.emb_rel, use_cuda, analysis)
        self.super_rgcn = EvolveRGCN_O(num_ents, num_rels*2, h_dim, h_dim, num_bases, num_basis, num_hidden_layers, dropout,
                                        self_loop, skip_connect, encoder_name, self.opn, self.emb_rel, use_cuda, analysis)

        # self.rgcn = EvolveRGCN_H(num_ents, num_rels*2, h_dim, h_dim, num_bases, num_basis, num_hidden_layers, dropout,
        #                          self_loop, skip_connect, encoder_name, self.opn, self.emb_rel, use_cuda, analysis)
        # self.super_rgcn = EvolveRGCN_H(num_ents, num_rels*2, h_dim, h_dim, num_bases, num_basis, num_hidden_layers, dropout,
        #                                self_loop, skip_connect, encoder_name, self.opn, self.emb_rel, use_cuda, analysis)

        # GRU cell for relation evolving
        # self.relation_cell_1 = nn.LSTMCell(self.h_dim*2, self.h_dim)
        # self.relation_cell_2 = nn.LSTMCell(self.h_dim*2, self.h_dim)
        # self.relation_cell_3 = nn.GRUCell(self.h_dim, self.h_dim)
        # self.entity_cell_1 = nn.GRUCell(self.h_dim, self.h_dim)

        # decoder
        if decoder_name == "convtranse":
            # self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout, sequence_len=self.sequence_len)
            # self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout, sequence_len=self.sequence_len)
            self.decoder_ob = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout, sequence_len=self.sequence_len)
            self.rdecoder = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout, sequence_len=self.sequence_len)
        else:
            raise NotImplementedError 

    def forward_my(self, g_list, super_g_list, static_graph, use_cuda):
        gate_list = []
        degree_list = []


        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0) # 演化得到的表示，和wordemb满足静态图约束
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.nodes = static_emb
        else:
            self.nodes = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        self.rels = F.normalize(self.emb_rel) if self.layer_norm else self.emb_rel

        history_embs = []
        rel_embs = []

        for i,g in enumerate(g_list):
            g = g.to(self.gpu)
            super_g = super_g_list[i]
            super_g = super_g.to(self.gpu)

            current_rels = self.super_rgcn.forward(super_g, self.rels, [self.p_rel, self.p_rel])
            current_rels = F.normalize(current_rels) if self.layer_norm else current_rels
            rel_time_gate = F.sigmoid(torch.mm(self.rels, self.rel_time_gate_weight) + self.rel_time_gate_bias)
            self.rels = rel_time_gate * current_rels + (1 - rel_time_gate) * self.rels
            self.rels = F.normalize(self.rels)
            rel_embs.append(self.rels)

            current_nodes = self.rgcn.forward(g, self.nodes, [self.rels, self.rels])
            current_nodes = F.normalize(current_nodes) if self.layer_norm else current_nodes
            node_time_gate = F.sigmoid(torch.mm(self.nodes, self.node_time_gate_weight) + self.node_time_gate_bias)
            self.nodes = node_time_gate * current_nodes + (1 - node_time_gate) * self.nodes
            self.nodes = F.normalize(self.nodes)
            
            history_embs.append(self.nodes)
        return history_embs, rel_embs[-1], static_emb, gate_list, degree_list


    def predict(self, test_graph, test_super_graph, num_rels, static_graph, test_triplets, use_cuda):
        '''
        :param test_graph: 原始时序子图
        :param test_super_graph: 时序关系子超图
        :param num_rels: 原始关系数目
        :param static_graph: 静态图
        :param test_triplets: 一个时间戳内的所有事实 [[s, r, o], [], ...] (num_triples_time, 3)
        :param use_cuda:
        :return:
        '''
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0, 3]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels  # 将逆关系换成逆关系的id
            all_triples = torch.cat((test_triplets, inverse_test_triplets)) # (batch_size, 3)

            evolve_embeddings = []
            rel_embeddings = []
            for idx in range(len(test_graph)):
                # evolve_embs, r_emb, _, _, _ = self.forward(test_graph[idx:], test_super_graph[idx:], static_graph, use_cuda)
                evolve_embs, r_emb, _, _, _ = self.forward_my(test_graph[idx:], test_super_graph[idx:], static_graph, use_cuda)
                # evolve_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
                evolve_emb = evolve_embs[-1]
                evolve_embeddings.append(evolve_emb)
                rel_embeddings.append(r_emb)
            evolve_embeddings.reverse()
            rel_embeddings.reverse()

            time_embs = self.get_init_time(all_triples)

            score_list = self.decoder_ob.forward(evolve_embeddings, rel_embeddings, time_embs, all_triples, mode="test") # all_triples: 包含反关系的三元组二维张量
            score_rel_list = self.rdecoder.forward(evolve_embeddings, rel_embeddings, time_embs, all_triples, mode="test") # (batch_size, num_rel*2)

            score_list = [_.unsqueeze(2) for _ in score_list]
            score_rel_list = [_.unsqueeze(2) for _ in score_rel_list]
            scores = torch.cat(score_list, dim=2)
            scores = torch.softmax(scores, dim=1)
            scores_rel = torch.cat(score_rel_list, dim=2)
            scores_rel = torch.softmax(scores_rel, dim=1)

            scores = torch.sum(scores, dim=-1)
            scores_rel = torch.sum(scores_rel, dim=-1)

            return all_triples, scores, scores_rel # (batch_size, 3) (batch_size, num_ents)

    def get_ft_loss(self, glist, super_glist, triple_list, static_graph, use_cuda):
        glist = [g.to(self.gpu) for g in glist]
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        inverse_triples = triple_list[-1][:, [2, 1, 0, 3]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triple_list[-1], inverse_triples])
        all_triples = all_triples.to(self.gpu)

        # for step, triples in enumerate(triple_list):
        evolve_embeddings = []
        rel_embeddings = []
        for idx in range(len(glist)):
            # evolve_embs, r_emb, _, _, _ = self.forward(glist[idx:], super_glist[idx:], static_graph, use_cuda)
            evolve_embs, r_emb, _, _, _ = self.forward_my(glist[idx:], super_glist[idx:], static_graph, use_cuda)
            # evolve_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
            evolve_emb = evolve_embs[-1]
            evolve_embeddings.append(evolve_emb)
            rel_embeddings.append(r_emb)
        evolve_embeddings.reverse()
        rel_embeddings.reverse()

        time_embs = self.get_init_time(all_triples)

        scores_ob = self.decoder_ob.forward(evolve_embeddings, rel_embeddings, time_embs, all_triples) #.view(-1, self.num_ents)
        for idx in range(len(glist)):
            loss_ent += self.loss_e(scores_ob[idx], all_triples[:, 2])
        scores_rel = self.rdecoder.forward(evolve_embeddings, rel_embeddings, time_embs, all_triples)
        for idx in range(len(glist)):
            loss_rel += self.loss_r(scores_rel[idx], all_triples[:, 1])

        evolve_embs, r_emb, static_emb, _, _ = self.forward_my(glist, super_glist, static_graph, use_cuda)
        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    angle = 90 // len(evolve_embs)
                    # step = (self.angle * math.pi / 180) * (time_step + 1)
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
        return loss_ent, loss_rel, loss_static

    def get_loss(self, glist, super_glist, static_graph, triples, use_cuda):
        """
        还需传入当前时间戳下的所有事实在各个历史子图中的历史重复事实列表
        :param glist: 历史子图列表
        :param super_glist: 历史超图列表
        :param static_graph: 静态资源
        :param triplets: 当前时间戳下的所有事实，一个时间戳内的所有事实三元组
        :param use_cuda:
        :return:
        """
        # 进行关系预测和实体预测的损失统计
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        inverse_triples = triples[:, [2, 1, 0, 3]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)

        evolve_embeddings = []
        rel_embeddings = []
        for idx in range(len(glist)):
            # evolve_embs, r_emb, _, _, _ = self.forward(glist[idx:], super_glist[idx:], static_graph, use_cuda) # evolve_embs, static_emb, r_emb在gpu上
            evolve_embs, r_emb, _, _, _ = self.forward_my(glist[idx:], super_glist[idx:], static_graph, use_cuda)
            # evolve_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
            evolve_emb = evolve_embs[-1]
            evolve_embeddings.append(evolve_emb)
            rel_embeddings.append(r_emb)
        evolve_embeddings.reverse()
        rel_embeddings.reverse()

        time_embs = self.get_init_time(all_triples)

        scores_ob = self.decoder_ob.forward(evolve_embeddings, rel_embeddings, time_embs, all_triples)
        for idx in range(len(glist)):
            loss_ent += self.loss_e(scores_ob[idx], all_triples[:, 2])
        scores_rel = self.rdecoder.forward(evolve_embeddings, rel_embeddings, time_embs, all_triples)
        for idx in range(len(glist)):
            loss_rel += self.loss_r(scores_rel[idx], all_triples[:, 1])

        # evolve_embs, r_emb, static_emb, _, _ = self.forward(glist, super_glist, static_graph, use_cuda)
        evolve_embs, r_emb, static_emb, _, _ = self.forward_my(glist, super_glist, static_graph, use_cuda)
        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    angle = 90 // len(evolve_embs)
                    # step = (self.angle * math.pi / 180) * (time_step + 1)
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
        return loss_ent, loss_rel, loss_static
    
    def get_init_time(self, quadrupleList):
        T_idx = quadrupleList[:, 3] // self.time_interval
        T_idx = T_idx.unsqueeze(1).float()
        t1 = self.weight_t1 * T_idx + self.bias_t1
        t2 = self.sin(self.weight_t2 * T_idx + self.bias_t2)
        return t1, t2
