import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=None, activation=None,
                 self_loop=False, skip_connect=False, dropout=0.0, layer_norm=False):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.skip_connect = skip_connect
        self.layer_norm = layer_norm

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            # self.loop_weight = nn.Parameter(torch.eye(out_feat), requires_grad=False)

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))   # 和self-loop不一样，是跨层的计算
            nn.init.xavier_uniform_(self.skip_connect_weight,
                                    gain=nn.init.calculate_gain('relu'))

            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if self.layer_norm:
            self.normalization_layer = nn.LayerNorm(out_feat, elementwise_affine=False)

    # define how propagation is done in subclass
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g, prev_h=[]):
        if self.self_loop:
            #print(self.loop_weight)
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)
        # self.skip_connect_weight.register_hook(lambda g: print("grad of skip connect weight: {}".format(g)))
        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)     # 使用sigmoid，让值在0~1
            # print("skip_ weight")
            # print(skip_weight)
            # print("skip connect weight")
            # print(self.skip_connect_weight)
            # print(torch.mm(prev_h, self.skip_connect_weight))

        self.propagate(g)  # 这里是在计算从周围节点传来的信息

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        # print(len(prev_h))
        if len(prev_h) != 0 and self.skip_connect:   # 两次计算loop_message的方式不一样，前者激活后再加权
            previous_node_repr = (1 - skip_weight) * prev_h
            if self.activation:
                node_repr = self.activation(node_repr)
            if self.self_loop:
                if self.activation:
                    loop_message = skip_weight * self.activation(loop_message)
                else:
                    loop_message = skip_weight * loop_message
                node_repr = node_repr + loop_message
            node_repr = node_repr + previous_node_repr
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message
            if self.layer_norm:
                node_repr = self.normalization_layer(node_repr)
            if self.activation:
                node_repr = self.activation(node_repr)
            # print("node_repr")
            # print(node_repr)
        g.ndata['h'] = node_repr
        return node_repr


class RGCNBasisLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNBasisLayer, self).__init__(in_feat, out_feat, bias, activation)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # add basis weights
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))
        if self.num_bases < self.num_rels:
            # linear combination coefficients
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels,
                                                    self.num_bases))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))

    def propagate(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.num_bases,
                                      self.in_feat * self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(
                self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight

        if self.is_input_layer:
            def msg_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                index = edges.data['type'] * self.in_feat + edges.src['id']
                return {'msg': embed.index_select(0, index)}
        else:
            def msg_func(edges):
                w = weight.index_select(0, edges.data['type'])
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                return {'msg': msg}

        def apply_func(nodes):
            return {'h': nodes.data['h'] * nodes.data['norm']}

        g.update_all(msg_func, fn.sum(msg='msg', out='h'), apply_func)


class RGCNBlockLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, layer_norm=False):
        super(RGCNBlockLayer, self).__init__(in_feat, out_feat, bias,
                                             activation, self_loop=self_loop, skip_connect=skip_connect,
                                             dropout=dropout)
        self.num_rels = num_rels
        self.num_bases = num_bases

        assert self.num_bases > 0

        self.out_feat = out_feat
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        # assuming in_feat and out_feat are both divisible by num_bases
        self.weight = nn.Parameter(torch.Tensor(
            self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges):
        weight = self.weight.index_select(0, edges.data['type']).view(
                    -1, self.submat_in, self.submat_out)    # [edge_num, submat_in, submat_out]
        node = edges.src['h'].view(-1, 1, self.submat_in)   # [edge_num * num_bases, 1, submat_in]->
        msg = torch.bmm(node, weight).view(-1, self.out_feat)   # [edge_num, out_feat]
        return {'msg': msg}

    def propagate(self, g):
        g.update_all(self.msg_func, fn.sum(msg='msg', out='h'), self.apply_func)
        # g.updata_all ({'msg': msg} , fn.sum(msg='msg', out='h'), {'h': nodes.data['h'] * nodes.data[''norm]})

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


class UnionRGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1,  bias=None, # h_dim, h_dim, num_rels*2
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, rel_emb=None):
        super(UnionRGCNLayer, self).__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.rel_emb = None
        self.skip_connect = skip_connect
        self.ob = None
        self.sub = None

        # WL
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))   # 和self-loop不一样，是跨层的计算
            nn.init.xavier_uniform_(self.skip_connect_weight,gain=nn.init.calculate_gain('relu'))
            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def propagate(self, g):
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    def forward(self, g, prev_h, emb_rel): # g: 当前历史子图; []; self.h_0 边的嵌入 (num_rels*2, h_dim)
        self.rel_emb = emb_rel
        # self.sub = sub
        # self.ob = ob
        if self.self_loop:
            #loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            # masked_index = torch.masked_select(torch.arange(0, g.number_of_nodes(), dtype=torch.long), (g.in_degrees(range(g.number_of_nodes())) > 0))
            masked_index = torch.masked_select(
                torch.arange(0, g.number_of_nodes(), dtype=torch.long).cuda(), # 全体node编号中筛选
                (g.in_degrees(range(g.number_of_nodes())) > 0)) # 筛选当前历史子图中入度不为0的所有node节点编号，返回一维张量
            loop_message = torch.mm(g.ndata['h'], self.evolve_loop_weight) # g.ndata['h']: node embedding (g_num_nodes, h_dim) (h_dim. h_dim)
            loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :] # 更新loop_message中入度不为0的node节点
        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)     # 使用sigmoid，让值在0~1

        # calculate the neighbor message with weight_neighbor
        self.propagate(g)
        node_repr = g.ndata['h'] # node embedding

        # print(len(prev_h))
        if len(prev_h) != 0 and self.skip_connect:  # 两次计算loop_message的方式不一样，前者激活后再加权
            if self.self_loop:
                node_repr = node_repr + loop_message
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message

        if self.activation: # 激活函数
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        g.ndata['h'] = node_repr
        return node_repr # 返回的是更新的节点表示g.ndata['h']

    def msg_func(self, edges):
        # if reverse:
        #     relation = self.rel_emb.index_select(0, edges.data['type_o']).view(-1, self.out_feat)
        # else:
        #     relation = self.rel_emb.index_select(0, edges.data['type_s']).view(-1, self.out_feat)
        relation = self.rel_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)
        edge_type = edges.data['type']
        edge_num = edge_type.shape[0]
        node = edges.src['h'].view(-1, self.out_feat)
        # node = torch.cat([torch.matmul(node[:edge_num // 2, :], self.sub),
        #                  torch.matmul(node[edge_num // 2:, :], self.ob)])
        # node = torch.matmul(node, self.sub)

        # after add inverse edges, we only use message pass when h as tail entity
        # 这里计算的是每个节点发出的消息，节点发出消息时其作为头实体
        # msg = torch.cat((node, relation), dim=1)
        msg = node + relation
        # calculate the neighbor message with weight_neighbor
        msg = torch.mm(msg, self.weight_neighbor)
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


class NewRGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, need_neighbor_weight=True, 
                 need_loop_weight=True, need_skip_weight=True, bias=None, activation=None, 
                 self_loop=False, dropout=0.0, skip_connect=False, rel_emb=None):
        super(NewRGCNLayer, self).__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.need_neighbor_weight = need_neighbor_weight
        self.need_loop_weight = need_loop_weight
        self.need_skip_weight = need_skip_weight
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.skip_connect = skip_connect
        self.rel_emb = None

        if need_neighbor_weight:
            self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        else:
            self.register_parameter("weight_neighbor", None)

        if self_loop and need_loop_weight:  # 有 self_loop 并且 需要权重
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
        elif self_loop and need_loop_weight is False:  # 有 self_loop 但是 会自己传权重进来
            self.register_parameter("loop_weight", None)
            self.register_parameter("evolve_loop_weight", None)
        else:
            pass

        if skip_connect and need_skip_weight:  # 有 skip_connect 并且需要权重
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))   # 和self-loop不一样，是跨层的计算
            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
        elif skip_connect and need_skip_weight is False:  # 有 skip_connect 但是会自己传权重进来
            self.register_parameter("skip_connect_weight", None)
            self.register_parameter("skip_connect_bias", None)
        else:
            pass

        self.reset_parameters()

        if dropout:
            self.dropout = nn.Dropout(dropout)  # 没有可训练的参数
        else:
            self.dropout = None

    def reset_parameters(self):
        r"""
        Reinitilize learnable parameters
        """
        if self.weight_neighbor is not None:
            nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))
        if self.self_loop and self.loop_weight is not None:
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))
        if self.skip_connect and self.skip_connect_weight is not None:
            nn.init.xavier_uniform_(self.skip_connect_weight,gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0


    def propagate(self, g, weight_neighbor):
        g.update_all(lambda x: self.msg_func(x, weight_neighbor), fn.sum(msg='msg', out='h'), self.apply_func)

    def forward(self, g, prev_h, emb_rel, weight_neighbor=None, loop_weight=None, evolve_loop_weight=None, 
                skip_connect_weight=None, skip_connect_bias=None): # g: 当前历史子图; []; self.h_0 边的嵌入 (num_rels*2, h_dim)
        if self.need_neighbor_weight:  # 模型初始化了参数
            if weight_neighbor is not None:
                raise NotImplementedError
            else:
                weight_neighbor = self.weight_neighbor
        else:  # 模型未初始化参数，需要自己传入
            if weight_neighbor is None:
                raise NotImplementedError
        
        if self.self_loop:
            if self.need_loop_weight:  # 模型初始化了参数
                if loop_weight is not None or evolve_loop_weight is not None:
                    raise NotImplementedError
                else:
                    loop_weight = self.loop_weight
                    evolve_loop_weight = self.evolve_loop_weight
            else:
                if loop_weight is None or evolve_loop_weight is None:
                    raise NotImplementedError

        if self.skip_connect:
            if self.need_skip_weight:  # 模型初始化了参数
                if skip_connect_weight is not None or skip_connect_bias is not None:
                    raise NotImplementedError
                else:
                    skip_connect_weight = self.skip_connect_weight
                    skip_connect_bias = self.skip_connect_bias
            else:
                if skip_connect_weight is None or skip_connect_bias is None:
                    raise NotImplementedError
        
        self.rel_emb = emb_rel
        if self.self_loop:
            masked_index = torch.masked_select(
                torch.arange(0, g.number_of_nodes(), dtype=torch.long).cuda(), # 全体node编号中筛选
                (g.in_degrees(range(g.number_of_nodes())) > 0)) # 筛选当前历史子图中入度不为0的所有node节点编号，返回一维张量
            loop_message = torch.mm(g.ndata['h'], evolve_loop_weight) # g.ndata['h']: node embedding (g_num_nodes, h_dim) (h_dim. h_dim)
            loop_message[masked_index, :] = torch.mm(g.ndata['h'], loop_weight)[masked_index, :] # 更新loop_message中入度不为0的node节点
        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(torch.mm(prev_h, skip_connect_weight) + skip_connect_bias)     # 使用sigmoid，让值在0~1

        # calculate the neighbor message with weight_neighbor
        self.propagate(g, weight_neighbor)
        node_repr = g.ndata['h'] # node embedding

        # print(len(prev_h))
        if len(prev_h) != 0 and self.skip_connect:  # 两次计算loop_message的方式不一样，前者激活后再加权
            if self.self_loop:
                node_repr = node_repr + loop_message
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message

        if self.activation: # 激活函数
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        g.ndata['h'] = node_repr
        return node_repr # 返回的是更新的节点表示g.ndata['h']

    def msg_func(self, edges, weight_neighbor):
        relation = self.rel_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)
        edge_type = edges.data['type']
        edge_num = edge_type.shape[0]
        node = edges.src['h'].view(-1, self.out_feat)

        msg = node + relation
        msg = torch.mm(msg, weight_neighbor)
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}
    