import torch.nn as nn
import torch

from rgcn.layers import NewRGCNLayer
import torch.nn.functional as F

from torch.nn.parameter import Parameter

class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, encoder_name="", opn="sub", rel_emb=None, use_cuda=False, analysis=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_basis = num_basis
        self.num_hidden_layers = num_hidden_layers # n_layers = 2
        self.dropout = dropout
        self.skip_connect = skip_connect
        self.self_loop = self_loop
        self.encoder_name = encoder_name
        self.use_cuda = use_cuda
        self.run_analysis = analysis
        self.skip_connect = skip_connect
        print("use layer :{}".format(encoder_name))
        self.rel_emb = rel_emb
        self.opn = opn
        # create rgcn layers
        self.build_model()
        # create initial features
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):

            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        return None

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g):
        if self.features is not None:
            g.ndata['id'] = self.features
        print("h before GCN message passing")
        print(g.ndata['h'])
        print("h behind GCN message passing")
        for layer in self.layers:
            layer(g)
        print(g.ndata['h'])
        return g.ndata.pop('h')


class MatGRUCell(nn.Module):
    """
    GRU cell for matrix, similar to the official code.
    Please refer to section 3.4 of the paper for the formula.
    """

    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.update = MatGRUGate(in_feats,
                                 out_feats,
                                 torch.nn.Sigmoid())

        self.reset = MatGRUGate(in_feats,
                                out_feats,
                                torch.nn.Sigmoid())

        self.htilda = MatGRUGate(in_feats,
                                 out_feats,
                                 torch.nn.Tanh())

    def forward(self, prev_Q, z_topk=None):
        if z_topk is None:
            z_topk = prev_Q

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q


class MatGRUGate(torch.nn.Module):
    """
    GRU gate for matrix, similar to the official code.
    Please refer to section 3.4 of the paper for the formula.
    """

    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        self.W = Parameter(torch.Tensor(rows, rows))
        self.U = Parameter(torch.Tensor(rows, rows))
        self.bias = Parameter(torch.Tensor(rows, cols))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.U)
        nn.init.zeros_(self.bias)

    def forward(self, x, hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out


class TopK(torch.nn.Module):
    """
    Similar to the official `egcn_h.py`. We only consider the node in a timestamp based subgraph,
    so we need to pay attention to `K` should be less than the min node numbers in all subgraph.
    Please refer to section 3.4 of the paper for the formula.
    """

    def __init__(self, feats, k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats, 1))
        self.reset_parameters()

        self.k = k

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.scorer)

    def forward(self, node_embs):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        vals, topk_indices = scores.view(-1).topk(self.k)
        out = node_embs[topk_indices] * torch.tanh(scores[topk_indices].view(-1, 1))
        # we need to transpose the output
        return out.t()


class EvolveRGCN_O(nn.Module):
    def __init__(
            self,
            num_nodes,
            num_rels,
            h_dim,
            out_dim,
            num_bases=-1,
            num_basis=-1,
            num_layers=1,
            dropout=0,
            self_loop=False,
            skip_connect=False,
            encoder_name="",
            opn="sub",
            rel_emb=None,
            use_cuda=False,
            analysis=False,
    ):
        super(EvolveRGCN_O, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_basis = num_basis
        self.num_layers = num_layers # n_layers = 2
        self.dropout = dropout
        self.skip_connect = skip_connect
        self.self_loop = self_loop
        self.encoder_name = encoder_name
        self.use_cuda = use_cuda
        self.run_analysis = analysis
        self.skip_connect = skip_connect
        self.rel_emb = rel_emb
        self.opn = opn
        print("use layer :{}".format(encoder_name))

        self.rgcn_convs = nn.ModuleList()
        self.neighbor_weight_recurrent_layers = nn.ModuleList()
        self.rgcn_neighbor_weights_list = nn.ParameterList()
        
        # if self_loop:
        #     self.loop_weight_recurrent_layers = nn.ModuleList()
        #     self.evolve_loop_weight_recurrent_layers = nn.ModuleList()

        #     self.rgcn_loop_weights_list = nn.ParameterList()
        #     self.rgcn_evolve_loop_weights_list = nn.ParameterList()
        
        # if self.skip_connect:
        #     self.skip_connect_recurrent_layers = nn.ModuleList()
        #     self.skip_connect_bias_recurrent_layers = nn.ModuleList()
            
        #     self.rgcn_skip_connect_weights_list = nn.ParameterList()
        #     self.rgcn_skip_connect_bias_weights_list = nn.ParameterList()


        for i in range(num_layers):
            self.rgcn_convs.append(self.build_layer(i))
            self.neighbor_weight_recurrent_layers.append(MatGRUCell(in_feats=h_dim, out_feats=h_dim))
            self.rgcn_neighbor_weights_list.append(Parameter(torch.Tensor(h_dim, h_dim)))

            # if self_loop:
            #     self.loop_weight_recurrent_layers.append(MatGRUCell(in_feats=h_dim, out_feats=h_dim))
            #     self.evolve_loop_weight_recurrent_layers.append(MatGRUCell(in_feats=h_dim, out_feats=h_dim))

            #     self.rgcn_loop_weights_list.append(Parameter(torch.Tensor(h_dim, h_dim)))
            #     self.rgcn_evolve_loop_weights_list.append(Parameter(torch.Tensor(h_dim, h_dim)))
            
            # if skip_connect:
            #     self.skip_connect_recurrent_layers.append(MatGRUCell(in_feats=h_dim, out_feats=h_dim))
            #     self.skip_connect_bias_recurrent_layers.append(MatGRUCell(in_feats=h_dim, out_feats=h_dim))

            #     self.rgcn_skip_connect_weights_list.append(Parameter(torch.Tensor(h_dim, h_dim)))
            #     self.rgcn_skip_connect_bias_weights_list.append(Parameter(torch.Tensor(h_dim)))
        
        self.reset_parameters()

    def reset_parameters(self):
        for neighbor_weights in self.rgcn_neighbor_weights_list:
            nn.init.xavier_normal_(neighbor_weights)
        # if self.self_loop:
        #     for loop_weights in self.rgcn_loop_weights_list:
        #         nn.init.xavier_normal_(loop_weights)
        #     for evolve_loop_weights in self.rgcn_evolve_loop_weights_list:
        #         nn.init.xavier_normal_(evolve_loop_weights)
        # if self.skip_connect:
        #     for skip_connect_weights in self.rgcn_skip_connect_weights_list:
        #         nn.init.xavier_normal_(skip_connect_weights)
        #     for skip_connect_bias_weights in self.rgcn_skip_connect_bias_weights_list:
        #         nn.init.xavier_normal_(skip_connect_bias_weights)

    def build_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn": # 2层的UnionRGCNLayer
            # return NewRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases, need_neighbor_weight=False, 
            #                     need_loop_weight=False, need_skip_weight=False, activation=act, 
            #                     dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
            return NewRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases, need_neighbor_weight=False, 
                                need_loop_weight=True, need_skip_weight=True, activation=act, 
                                dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError

    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze() # node id
            g.ndata['h'] = init_ent_emb[node_id] # node embedding
            x, r = init_ent_emb, init_rel_emb
            for i in range(self.num_layers):
                nei_W = self.rgcn_neighbor_weights_list[i]
                nei_W = self.neighbor_weight_recurrent_layers[i](nei_W)

                # loop_W = self.rgcn_loop_weights_list[i]
                # loop_W = self.loop_weight_recurrent_layers[i](loop_W)

                # evo_loop_W = self.rgcn_evolve_loop_weights_list[i]
                # evo_loop_W = self.evolve_loop_weight_recurrent_layers[i](evo_loop_W)
                # self.rgcn_convs[i](g, [], r[i], weight_neighbor=nei_W, loop_weight=loop_W, evolve_loop_weight=evo_loop_W)
                self.rgcn_convs[i](g, [], r[i], weight_neighbor=nei_W)
            return g.ndata.pop('h') # 返回了图中更新的node embedding
        else:
            raise NotImplementedError
    

class EvolveRGCN_H(nn.Module):
    def __init__(
            self,
            num_nodes,
            num_rels,
            h_dim,
            out_dim,
            num_bases=-1,
            num_basis=-1,
            num_layers=1,
            dropout=0,
            self_loop=False,
            skip_connect=False,
            encoder_name="",
            opn="sub",
            rel_emb=None,
            use_cuda=False,
            analysis=False,
            ):
        super(EvolveRGCN_H, self).__init__()

        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_basis = num_basis
        self.num_layers = num_layers # n_layers = 2
        self.dropout = dropout
        self.skip_connect = skip_connect
        self.self_loop = self_loop
        self.encoder_name = encoder_name
        self.use_cuda = use_cuda
        self.run_analysis = analysis
        self.skip_connect = skip_connect
        self.rel_emb = rel_emb
        self.opn = opn
        print("use layer :{}".format(encoder_name))

        self.rgcn_convs = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        self.neighbor_weight_recurrent_layers = nn.ModuleList()
        self.rgcn_neighbor_weights_list = nn.ParameterList()

        # if self_loop:
        #     self.loop_weight_recurrent_layers = nn.ModuleList()
        #     self.evolve_loop_weight_recurrent_layers = nn.ModuleList()

        #     self.rgcn_loop_weights_list = nn.ParameterList()
        #     self.rgcn_evolve_loop_weights_list = nn.ParameterList()
        
        # if self.skip_connect:
        #     self.skip_connect_recurrent_layers = nn.ModuleList()
        #     self.skip_connect_bias_recurrent_layers = nn.ModuleList()
            
        #     self.rgcn_skip_connect_weights_list = nn.ParameterList()
        #     self.rgcn_skip_connect_bias_weights_list = nn.ParameterList()
        
        for i in range(num_layers):
            self.pooling_layers.append(TopK(h_dim, h_dim))
            self.rgcn_convs.append(self.build_layer(i))
            self.neighbor_weight_recurrent_layers.append(MatGRUCell(in_feats=h_dim, out_feats=h_dim))
            self.rgcn_neighbor_weights_list.append(Parameter(torch.Tensor(h_dim, h_dim)))

            # if self_loop:
            #     self.loop_weight_recurrent_layers.append(MatGRUCell(in_feats=h_dim, out_feats=h_dim))
            #     self.evolve_loop_weight_recurrent_layers.append(MatGRUCell(in_feats=h_dim, out_feats=h_dim))

            #     self.rgcn_loop_weights_list.append(Parameter(torch.Tensor(h_dim, h_dim)))
            #     self.rgcn_evolve_loop_weights_list.append(Parameter(torch.Tensor(h_dim, h_dim)))
            
            # if skip_connect:
            #     self.skip_connect_recurrent_layers.append(MatGRUCell(in_feats=h_dim, out_feats=h_dim))
            #     self.skip_connect_bias_recurrent_layers.append(MatGRUCell(in_feats=h_dim, out_feats=h_dim))

            #     self.rgcn_skip_connect_weights_list.append(Parameter(torch.Tensor(h_dim, h_dim)))
            #     self.rgcn_skip_connect_bias_weights_list.append(Parameter(torch.Tensor(h_dim)))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for neighbor_weights in self.rgcn_neighbor_weights_list:
            nn.init.xavier_normal_(neighbor_weights)
        # if self.self_loop:
        #     for loop_weights in self.rgcn_loop_weights_list:
        #         nn.init.xavier_normal_(loop_weights)
        #     for evolve_loop_weights in self.rgcn_evolve_loop_weights_list:
        #         nn.init.xavier_normal_(evolve_loop_weights)
        # if self.skip_connect:
        #     for skip_connect_weights in self.rgcn_skip_connect_weights_list:
        #         nn.init.xavier_normal_(skip_connect_weights)
        #     for skip_connect_bias_weights in self.rgcn_skip_connect_bias_weights_list:
        #         nn.init.xavier_normal_(skip_connect_bias_weights)
    
    def build_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn": # 2层的UnionRGCNLayer
            # return NewRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases, weight=False, activation=act, 
            #                     dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
            return NewRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases, need_neighbor_weight=False, 
                                need_loop_weight=True, need_skip_weight=True, activation=act, 
                                dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError
    
    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze() # node id
            g.ndata['h'] = init_ent_emb[node_id] # node embedding
            x, r = init_ent_emb, init_rel_emb
            for i in range(self.num_layers):
                nei_W = self.rgcn_neighbor_weights_list[i]
                X_title = self.pooling_layers[i](g.ndata['h'])
                nei_W = self.neighbor_weight_recurrent_layers[i](nei_W, X_title)

                self.rgcn_convs[i](g, [], r[i], weight_neighbor=nei_W)
            return g.ndata.pop('h') # 返回了图中更新的node embedding
        else:
            raise NotImplementedError

