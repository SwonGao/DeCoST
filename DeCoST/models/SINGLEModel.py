import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ['SINGLEModel']
# from layers import GraphAttentionLayer


class SINGLEModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.eval_type = self.model_params['eval_type']
        self.problem = self.model_params['problem']

        self.encoder = SINGLE_Encoder(**model_params)
        self.decoder = SINGLE_Decoder(**model_params)
        self.servicetime_decoder = SINGLE_Decoder(**model_params)
        self.encoded_nodes = None

        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in model_params.keys() else model_params['device']

    def pre_forward(self, reset_state):
        if not (self.problem.startswith('TSP') or self.problem.startswith('OP') or self.problem.startswith('TOP')) :
            depot_xy = reset_state.depot_xy
            # shape: (batch, 1, 2)
            node_demand = reset_state.node_demand
        else:
            depot_xy = None

        node_xy = reset_state.node_xy
        distance = reset_state.distance

        if self.problem in ["CVRP", "OVRP", "VRPB", "VRPL", "VRPBL", "OVRPB", "OVRPL", "OVRPBL"]:
            feature = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
            # shape: (batch, problem, 3)
        elif self.problem in ["TOPTW", "OPTW"]:
            node_tw_start = reset_state.node_tw_start
            node_tw_end = reset_state.node_tw_end
            # shape: (batch, problem)
            tw_start = node_tw_start[:, :, None]
            tw_end = node_tw_end[:, :, None]
            profit = reset_state.profit[:, :, None]
            max_travel_distance = reset_state.max_travel_distance[:, :, None]
            # _, problem_size = node_tw_end.size()
            if self.model_params["tw_normalize"]:
                tw_end_max = node_tw_end[:, :1, None]
                tw_start = tw_start / tw_end_max
                tw_end = tw_end / tw_end_max
            feature =  torch.cat((node_xy, tw_start, tw_end, profit, max_travel_distance), dim=2)
            # feature =  torch.cat((node_xy, tw_start, tw_end, profit, max_travel_distance), dim=2)
            # shape: (batch, problem, 5)
        elif self.problem in ["OPTWVP"]:
            node_tw_start = reset_state.node_tw_start
            node_tw_end = reset_state.node_tw_end
            # shape: (batch, problem)
            tw_start = node_tw_start[:, :, None]
            tw_end = node_tw_end[:, :, None]
            profit = reset_state.profit[:, :, None]
            max_travel_distance = reset_state.max_travel_distance[:, :, None]
            # _, problem_size = node_tw_end.size()
            if self.model_params["tw_normalize"]:
                tw_end_max = node_tw_end[:, :1, None]
                tw_start = tw_start / tw_end_max
                tw_end = tw_end / tw_end_max
            # feature =  torch.cat((node_xy, tw_start, tw_end, profit, max_travel_distance), dim=2)           
            # feature =  torch.cat((node_xy, tw_start, tw_end, max_travel_distance), dim=2)           
            feature = torch.cat((node_xy, tw_start, tw_end, profit), dim=2)           
            # feature =  torch.cat((node_xy, tw_start, tw_end), dim=2)           
        elif self.problem in ["TOPTWVP"]:
            node_tw_start = reset_state.node_tw_start
            node_tw_end = reset_state.node_tw_end
            # shape: (batch, problem)
            tw_start = node_tw_start[:, :, None]
            tw_end = node_tw_end[:, :, None]
            profit = reset_state.profit[:, :, None]
            # max_travel_distance = reset_state.max_travel_distance[:, :, None]
            # _, problem_size = node_tw_end.size()
            if self.model_params["tw_normalize"]:
                tw_end_max = node_tw_end[:, :1, None]
                tw_start = tw_start / tw_end_max
                tw_end = tw_end / tw_end_max
            # feature =  torch.cat((node_xy, tw_start, tw_end, profit, max_travel_distance), dim=2)           
            # feature =  torch.cat((node_xy, tw_start, tw_end, max_travel_distance), dim=2)           
            feature =  torch.cat((node_xy, tw_start, tw_end, profit), dim=2)           
            # feature =  torch.cat((node_xy, tw_start, tw_end), dim=2)           
        elif self.problem in ["TSPTW"]:
            node_tw_start = reset_state.node_tw_start
            node_tw_end = reset_state.node_tw_end
            # shape: (batch, problem)
            tw_start = node_tw_start[:, :, None]
            tw_end = node_tw_end[:, :, None]
            # _, problem_size = node_tw_end.size()
            if self.model_params["tw_normalize"]:
                tw_end_max = node_tw_end[:, :1, None]
                tw_start = tw_start / tw_end_max
                tw_end = tw_end / tw_end_max
            feature =  torch.cat((node_xy, tw_start, tw_end), dim=2)
            # shape: (batch, problem, 4)
        elif self.problem in ['TSPDL']:
            node_demand = reset_state.node_demand
            node_draft_limit = reset_state.node_draft_limit
            feature = torch.cat((node_xy, node_demand[:, :, None], node_draft_limit[:, :, None]), dim=2)
        elif self.problem in ["VRPTW", "OVRPTW", "VRPBTW", "VRPLTW", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]:
            node_tw_start = reset_state.node_tw_start
            node_tw_end = reset_state.node_tw_end
            # shape: (batch, problem)
            feature = torch.cat((node_xy, node_demand[:, :, None], node_tw_start[:, :, None], node_tw_end[:, :, None]), dim=2)
            # shape: (batch, problem, 5)

        else:
            raise NotImplementedError

        self.encoded_nodes = self.encoder(depot_xy, feature, distance)
        # shape: (batch, problem(+1), embedding)

        self.decoder.set_kv(self.encoded_nodes)
        self.servicetime_decoder.set_kv(self.encoded_nodes)

        return self.encoded_nodes, feature

    def set_eval_type(self, eval_type):
        self.eval_type = eval_type

    def forward(self, state, selected=None, pomo = False, use_predicted_PI_mask=False, no_select_prob=False, no_sigmoid=False, tw_end=None):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long).to(self.device)
            prob = torch.ones(size=(batch_size, pomo_size))
            service_time_normed = torch.ones(size=(batch_size, pomo_size))
            # shape: (batch, pomo, problem_size+1)
        elif pomo and state.selected_count == 1 and pomo_size > 1:  # Second Move, POMO
            selected = state.START_NODE
            prob = torch.ones(size=(batch_size, pomo_size))
            service_time_normed = torch.ones(size=(batch_size, pomo_size))
        else: # Sample from the action distribution
            encoded_last_node = self._get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            attr = self.get_context(state, tw_end)
            ninf_mask = state.ninf_mask
            probs = self.decoder(encoded_last_node, attr, ninf_mask=ninf_mask)
            service_times_normed = self.servicetime_decoder(encoded_last_node, attr, ninf_mask=ninf_mask)

            if selected is None:
                while True:
                    if self.training or self.eval_type == 'softmax':
                        try:
                            selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
                        except Exception as exception:
                            # torch.save(probs,"prob.pt")
                            print(">> Catch Exception: {}, on the instances of {}".format(exception, state.PROBLEM))
                            exit(0)
                    else:
                        selected = probs.argmax(dim=2)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    service_time_normed = service_times_normed[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    if (prob != 0).all():
                        break
            else:
                # never been there
                selected = selected
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                service_time_normed = service_times_normed[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)

        return selected, prob, service_time_normed

    def get_context(self, state, tw_end):
        # 这里引入了额外的特征例如现在的时间，考虑可以加入额外的 maxdistance 特征
        if self.problem in ["CVRP"]:
            attr = state.load[:, :, None]
        elif self.problem in ["VRPB", 'TSPDL']:
            attr = state.load[:, :, None]  # shape: (batch, pomo, 1)
        elif self.problem in ["TOPTW", "OPTW", "TSPTW", "OPTWVP", "TOPTWVP"]:
            attr = state.current_time[:, :, None]  # shape: (batch, pomo, 1)
            if self.model_params["tw_normalize"]:
                attr = attr / 10. # tw_end[:, 0][:, None, None]
        elif self.problem in ["OVRP", "OVRPB"]:
            attr = torch.cat((state.load[:, :, None], state.open[:, :, None]), dim=2)  # shape: (batch, pomo, 2)
        elif self.problem in ["VRPTW", "VRPBTW"]:
            attr = torch.cat((state.load[:, :, None], state.current_time[:, :, None]), dim=2)  # shape: (batch, pomo, 2)
        elif self.problem in ["VRPL", "VRPBL"]:
            attr = torch.cat((state.load[:, :, None], state.length[:, :, None]), dim=2)  # shape: (batch, pomo, 2)
        elif self.problem in ["VRPLTW", "VRPBLTW"]:
            attr = torch.cat((state.load[:, :, None], state.current_time[:, :, None], state.length[:, :, None]), dim=2)  # shape: (batch, pomo, 3)
        elif self.problem in ["OVRPL", "OVRPBL"]:
            attr = torch.cat((state.load[:, :, None], state.length[:, :, None], state.open[:, :, None]),  dim=2)  # shape: (batch, pomo, 3)
        elif self.problem in ["OVRPTW", "OVRPBTW"]:
            attr = torch.cat((state.load[:, :, None], state.current_time[:, :, None], state.open[:, :, None]), dim=2)  # shape: (batch, pomo, 3)
        elif self.problem in ["OVRPLTW", "OVRPBLTW"]:
            attr = torch.cat((state.load[:, :, None], state.current_time[:, :, None], state.length[:, :, None],
                              state.open[:, :, None]), dim=2)  # shape: (batch, pomo, 4)
        else:
            raise NotImplementedError

        return attr

    def _get_encoding(self, encoded_nodes, node_index_to_pick):
        # encoded_nodes.shape: (batch, problem, embedding)
        # node_index_to_pick.shape: (batch, pomo)

        batch_size = node_index_to_pick.size(0)
        pomo_size = node_index_to_pick.size(1)
        embedding_dim = encoded_nodes.size(2)

        gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
        # shape: (batch, pomo, embedding)

        picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
        # shape: (batch, pomo, embedding)
    # expanded_encoded_nodes = torch.repeat_interleave(encoded_nodes, env.num_vehicles, dim=0)
    # picked_nodes = expanded_encoded_nodes.gather(dim=1, index=gathering_index)
        return picked_nodes


########################################
# ENCODER
########################################

class SINGLE_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.problem = self.model_params['problem']
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']
        # self.method = 'GAT' # GAT or Transformer
        self.method = 'Transformer' # GAT or Transformer
        if not self.problem.startswith("TSP"):
            self.embedding_depot = nn.Linear(2, embedding_dim)
        if self.problem in ["CVRP", "OVRP", "VRPB", "VRPL", "VRPBL", "OVRPB", "OVRPL", "OVRPBL"]:
            self.embedding_node = nn.Linear(3, embedding_dim)
        elif self.problem in ["TSPTW", "TSPDL"]:
            self.embedding_node = nn.Linear(4, embedding_dim)
        elif self.problem in ["VRPTW", "OVRPTW", "VRPBTW", "VRPLTW", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]:
            self.embedding_node = nn.Linear(5, embedding_dim)
        elif self.problem in ["TOPTW", "OPTW"]:
            self.embedding_node = nn.Linear(6, embedding_dim)
        elif self.problem in ["OPTWVP"]:
            in_features = 5
            self.embedding_node = nn.Linear(in_features, embedding_dim)
            if self.method == 'GAT':
                self.embedding_edge = nn.Linear(1, embedding_dim)
                
        elif self.problem in ["TOPTWVP"]:
            in_features = 5 
            self.embedding_node = nn.Linear(in_features, embedding_dim)        
        else:
            raise NotImplementedError
        if self.method == 'GAT':
        # if self.problem in ["OPTWVP"]:
            # self.layers = nn.ModuleList([GAT(embedding_dim, 64, 8, embedding_dim) for _ in range(encoder_layer_num)])
            self.layers = nn.ModuleList([GAT(embedding_dim, 64, 8, embedding_dim)])
            # ([GAT(in_features, n_hidden, n_heads, num_classes) for _ in range(encoder_layer_num)])
        else: 
            self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, feature, edge_distance):
        if depot_xy is not None:
            embedded_depot = self.embedding_depot(depot_xy)

        embedded_node = self.embedding_node(feature)

        if depot_xy is not None:
            out = torch.cat((embedded_depot, embedded_node), dim=1)
            # shape: (batch, problem+1, embedding)
        else:
            out = embedded_node
            # shape: (batch, problem, embedding)
        # out = feature

        if self.method == 'GAT':
            for layer in self.layers:
                out = layer(out, edge_distance.unsqueeze(-1))
        else: # transformer
            for layer in self.layers:
                out = layer(out, edge_distance)
                # out = layer(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = FeedForward(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

        self.edge_proj = nn.Linear(1, head_num)

    def forward(self, input1, edge_info=None):
        """
        Two implementations:
            norm_last: the original implementation of AM/POMO: MHA -> Add & Norm -> FFN/MOE -> Add & Norm
            norm_first: the convention in NLP: Norm -> MHA -> Add -> Norm -> FFN/MOE -> Add
        """
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        if edge_info is not None:
            # edge_info shape: [batch, problem, problem]
            # project edge info into attention
            edge_bias = self.edge_proj(edge_info.unsqueeze(-1))
            edge_bias = edge_bias.permute(0, 3, 1, 2) 
            # edge_bias shape: [batch, head_num, problem, problem]
        
        if torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any():
            print("<< NaN detected in Q/K/V tensors!")
        
        if self.model_params['norm_loc'] == "norm_last":
            # out_concat = multi_head_attention(q, k, v)  # (batch, problem, HEAD_NUM*KEY_DIM)
            out_concat = multi_head_attention(q, k, v, edge_bias=edge_bias)  # (batch, problem, HEAD_NUM*KEY_DIM)
            multi_head_out = self.multi_head_combine(out_concat)  # (batch, problem, EMBEDDING_DIM)
            out1 = self.addAndNormalization1(input1, multi_head_out)
            out2 = self.feedForward(out1)
            out3 = self.addAndNormalization2(out1, out2)  # (batch, problem, EMBEDDING_DIM)
        else:
            out1 = self.addAndNormalization1(None, input1)
            multi_head_out = self.multi_head_combine(out1)
            input2 = input1 + multi_head_out
            out2 = self.addAndNormalization2(None, input2)
            out2 = self.feedForward(out2)
            out3 = input2 + out2

        return out3

########################################
# DECODER
########################################

class SINGLE_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.problem = self.model_params['problem']
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        if self.problem == "CVRP":
            self.Wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
        elif self.problem in ["VRPB", "TOPTWVP", "OPTWVP", "TOPTW",  "OPTW", "TSPTW", "TSPDL"]:
            self.Wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
        elif self.problem in ["OVRP", "OVRPB", "VRPTW", "VRPBTW", "VRPL", "VRPBL"]:
            attr_num = 3 if self.model_params["extra_feature"] else 2
            self.Wq_last = nn.Linear(embedding_dim + attr_num, head_num * qkv_dim, bias=False)
        elif self.problem in ["VRPLTW", "VRPBLTW", "OVRPL", "OVRPBL", "OVRPTW", "OVRPBTW"]:
            self.Wq_last = nn.Linear(embedding_dim + 3, head_num * qkv_dim, bias=False)
        elif self.problem in ["OVRPLTW", "OVRPBLTW"]:
            self.Wq_last = nn.Linear(embedding_dim + 4, head_num * qkv_dim, bias=False)
        else:
            raise NotImplementedError

        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head_attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head_attention
        # self.q1 = None  # saved q1, for multi-head attention
        # self.q2 = None  # saved q2, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def set_kv_sl(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        if self.model_params['detach_from_encoder']:
            self.k_sl = reshape_by_heads(self.Wk_sl(encoded_nodes.detach()), head_num=head_num)
            self.v_sl = reshape_by_heads(self.Wv_sl(encoded_nodes.detach()), head_num=head_num)
            # shape: (batch, head_num, problem+1, qkv_dim)
            self.single_head_key_sl = encoded_nodes.transpose(1, 2).detach()
            # shape: (batch, embedding, problem+1)
        else:
            self.k_sl = reshape_by_heads(self.Wk_sl(encoded_nodes), head_num=head_num)
            self.v_sl = reshape_by_heads(self.Wv_sl(encoded_nodes), head_num=head_num)
            self.single_head_key_sl = encoded_nodes.transpose(1, 2)


    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def set_q2(self, encoded_q2):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q2 = reshape_by_heads(self.Wq_2(encoded_q2), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, attr, ninf_mask, use_predicted_PI_mask=False, no_select_prob = False, no_sigmoid=False):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, 1~4)
        # ninf_mask.shape: (batch, pomo, problem)

        ############# 处理所有节点都被访问过的情况, 令最开始的节点为0
        if ninf_mask is not None:
            mask_inf = torch.isinf(ninf_mask).all(dim=2)
            batch_indices, group_indices = torch.where(mask_inf)
            ninf_mask[batch_indices, group_indices, 0] = 0

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        # information of the last step 
        input_cat = torch.cat((encoded_last_node, attr), dim=2)
        # shape = (batch, group, EMBEDDING_DIM+1)

        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # q = self.q1 + self.q2 + q_last
        # # shape: (batch, head_num, pomo, qkv_dim)
        q = q_last
        # shape: (batch, head_num, pomo, qkv_dim)
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)
        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)
        
        ################### debug use ###########################
        ######################3 final check ################################
        # 检测已结束的 batch (所有 ninf_mask 的值均为 -inf)
        # finished_mask = torch.isinf(ninf_mask).all(dim=2)  # (batch, pomo)

        # # 生成特殊的概率分布: 只在 index=0 处为 1，其余为 0
        # special_probs = torch.zeros_like(probs)  # (batch, pomo, problem)
        # special_probs[:, :, 0] = 1

        # # 替换已结束的 batch 的 probs
        # probs = torch.where(finished_mask.unsqueeze(-1), special_probs, probs)
        # print(probs[0,0,:])

        return probs

########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None, edge_bias=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)
    # edge_bias shape: (batch, head_num, n, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    if edge_bias is not None:
        score = score + edge_bias

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        # mask_inf = torch.isinf(rank3_ninf_mask).all(dim=2)
        # batch_indices, group_indices = torch.where(mask_inf)
        # rank3_ninf_mask[batch_indices, group_indices, 0] = 0 # 强制第一个元素为 0，也就是说，返回0点

        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # 如果所有 score_scaled 都是 -inf，则 softmax 结果变 NaN。
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


################################
###         GAT LAYER        ###
################################

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) as described in the paper `"Graph Attention Networks" <https://arxiv.org/pdf/1710.10903.pdf>`.

        This operation can be mathematically described as:

            e_ij = a(W h_i, W h_j)
            α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_k(exp(e_ik))     
            h_i' = σ(Σ_j(α_ij W h_j))
            
            where h_i and h_j are the feature vectors of nodes i and j respectively, W is a learnable weight matrix,
            a is an attention mechanism that computes the attention coefficients e_ij, and σ is an activation function.

    """
    def __init__(self, in_features: int, out_features: int, n_heads: int, concat: bool = False, dropout: float = 0.4, leaky_relu_slope: float = 0.2):
        super(GraphAttentionLayer, self).__init__()

        self.n_heads = n_heads # Number of attention heads
        self.concat = concat # wether to concatenate the final attention heads
        self.dropout = dropout # Dropout rate

        if concat: # concatenating the attention heads
            self.out_features = out_features # Number of output features per node
            assert out_features % n_heads == 0 # Ensure that out_features is a multiple of n_heads
            self.n_hidden = out_features // n_heads
        else: # averaging output over the attention heads (Used in the main paper)
            self.n_hidden = out_features

        #  A shared linear transformation, parametrized by a weight matrix W is applied to every node
        #  Initialize the weight matrix W 
        self.W = nn.Parameter(torch.empty(size=(in_features, self.n_hidden * n_heads)))

        # Initialize the attention weights a
        self.a = nn.Parameter(torch.empty(size=(n_heads, 2 * self.n_hidden, 1)))

        self.leakyrelu = nn.LeakyReLU(leaky_relu_slope) # LeakyReLU activation function
        self.softmax = nn.Softmax(dim=1) # softmax activation function to the attention coefficients

        self.reset_parameters() # Reset the parameters


    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.a)
    

    def _get_attention_scores(self, h_transformed: torch.Tensor):
        """calculates the attention scores e_ij for all pairs of nodes (i, j) in the graph
        in vectorized parallel form. for each pair of source and target nodes (i, j),
        the attention score e_ij is computed as follows:

            e_ij = LeakyReLU(a^T [Wh_i || Wh_j]) 

            where || denotes the concatenation operation, and a and W are the learnable parameters.

        Args:
            h_transformed (torch.Tensor): Transformed feature matrix with shape (n_nodes, n_heads, n_hidden),
                where n_nodes is the number of nodes and out_features is the number of output features per node.

        Returns:
            torch.Tensor: Attention score matrix with shape (n_heads, n_nodes, n_nodes), where n_nodes is the number of nodes.
        """
        
        source_scores = torch.matmul(h_transformed, self.a[:, :self.n_hidden, :])
        target_scores = torch.matmul(h_transformed, self.a[:, self.n_hidden:, :])

        # broadcast add 
        # (n_heads, n_nodes, 1) + (n_heads, 1, n_nodes) = (n_heads, n_nodes, n_nodes)
        e = source_scores + target_scores.mT
        return self.leakyrelu(e)

    def forward(self,  h: torch.Tensor, adj_mat: torch.Tensor):
        """
        Performs a graph attention layer operation.

        Args:
            h (torch.Tensor): Input tensor representing node features.
            adj_mat (torch.Tensor): Adjacency matrix representing graph structure.

        Returns:
            torch.Tensor: Output tensor after the graph convolution operation.
        """
        batch_num, n_nodes = h.shape[0], h.shape[1]

        # Apply linear transformation to node feature -> W h
        # output shape (batch_num, n_nodes, n_hidden * n_heads) 64*8
        h_transformed = torch.matmul(h, self.W)
        h_transformed = F.dropout(h_transformed, self.dropout, training=self.training)

        # splitting the heads by reshaping the tensor and putting heads dim first
        # output shape (batch_num, n_heads, n_nodes, n_hidden)
        h_transformed = h_transformed.view(batch_num, n_nodes, self.n_heads, self.n_hidden).permute(0, 2, 1, 3)
        
        # getting the attention scores
        e = self._get_attention_scores(h_transformed)

        # Set the attention score for non-existent edges to -9e15 (MASKING NON-EXISTENT EDGES)
        connectivity_mask = -9e16 * torch.ones_like(e)
        e = torch.where(adj_mat > 0, e, connectivity_mask) # masked attention scores
        # attention coefficients are computed as a softmax over the rows
        # for each column j in the attention score matrix e
        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # final node embeddings are computed as a weighted average of the features of its neighbors
        h_prime = torch.matmul(attention, h_transformed)

        # concatenating/averaging the attention heads
        # output shape (n_nodes, out_features)
        # Output reshape: (B, n_heads, N, n_hidden) → (B, N, n_heads * n_hidden)
        if self.concat:
            h_prime = h_prime.permute(0, 2, 1, 3).contiguous().view(batch_num, n_nodes, self.out_features)
            # h_prime = h_prime.permute(0,2,1,3).contiguous().view(n_nodes, self.out_features)
        else:
            h_prime = h_prime.mean(dim=1) # mean over heads
 
        return h_prime

class GAT(nn.Module):
    """
    Graph Attention Network (GAT) as described in the paper `"Graph Attention Networks" <https://arxiv.org/pdf/1710.10903.pdf>`.
    Consists of a 2-layer stack of Graph Attention Layers (GATs). The fist GAT Layer is followed by an ELU activation.
    And the second (final) layer is a GAT layer with a single attention head and softmax activation function. 
    """
    def __init__(self,
        in_features,
        n_hidden,
        n_heads,
        num_classes,
        concat=False,
        dropout=0.4,
        leaky_relu_slope=0.2):
        """ Initializes the GAT model. 

        Args:
            in_features (int): number of input features per node.
            n_hidden (int): output size of the first Graph Attention Layer.
            n_heads (int): number of attention heads in the first Graph Attention Layer.
            num_classes (int): number of classes to predict for each node.
            concat (bool, optional): Wether to concatinate attention heads or take an average over them for the
                output of the first Graph Attention Layer. Defaults to False.
            dropout (float, optional): dropout rate. Defaults to 0.4.
            leaky_relu_slope (float, optional): alpha (slope) of the leaky relu activation. Defaults to 0.2.
        """

        super(GAT, self).__init__()

        # Define the Graph Attention layers
        self.gat1 = GraphAttentionLayer(
            in_features=in_features, out_features=n_hidden, n_heads=n_heads,
            concat=concat, dropout=dropout, leaky_relu_slope=leaky_relu_slope
            )
        
        self.gat2 = GraphAttentionLayer(
            in_features=n_hidden, out_features=num_classes, n_heads=1,
            concat=False, dropout=dropout, leaky_relu_slope=leaky_relu_slope
            )
        

    def forward(self, input_tensor: torch.Tensor , adj_mat: torch.Tensor):
        """
        Performs a forward pass through the network.

        Args:
            input_tensor (torch.Tensor): Input tensor representing node features.
            adj_mat (torch.Tensor): Adjacency matrix representing graph structure.

        Returns:
            torch.Tensor: Output tensor after the forward pass.
        """

        # Apply the first Graph Attention layer
        x = self.gat1(input_tensor, adj_mat)
        x = F.elu(x) # Apply ELU activation function to the output of the first layer

        # Apply the second Graph Attention layer
        x = self.gat2(x, adj_mat)

        return F.log_softmax(x, dim=1) # Apply log softmax activation function

class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.add = True if 'norm_loc' in model_params.keys() and model_params['norm_loc'] == "norm_last" else False
        if model_params["norm"] == "batch":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=True)
        elif model_params["norm"] == "batch_no_track":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "instance":
            self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "layer":
            self.norm = nn.LayerNorm(embedding_dim)
        elif model_params["norm"] == "rezero":
            self.norm = torch.nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        else:
            self.norm = None

    def forward(self, input1=None, input2=None):
        # input.shape: (batch, problem, embedding)
        if isinstance(self.norm, nn.InstanceNorm1d):
            added = input1 + input2 if self.add else input2
            transposed = added.transpose(1, 2)
            # shape: (batch, embedding, problem)
            normalized = self.norm(transposed)
            # shape: (batch, embedding, problem)
            back_trans = normalized.transpose(1, 2)
            # shape: (batch, problem, embedding)
        elif isinstance(self.norm, nn.BatchNorm1d):
            added = input1 + input2 if self.add else input2
            batch, problem, embedding = added.size()
            normalized = self.norm(added.reshape(batch * problem, embedding))
            back_trans = normalized.reshape(batch, problem, embedding)
        elif isinstance(self.norm, nn.LayerNorm):
            added = input1 + input2 if self.add else input2
            back_trans = self.norm(added)
        elif isinstance(self.norm, nn.Parameter):
            back_trans = input1 + self.norm * input2 if self.add else self.norm * input2
        else:
            back_trans = input1 + input2 if self.add else input2

        return back_trans

class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))

class FC(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.W1 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, input):
        # input.shape: (batch, problem, embedding)
        return F.relu(self.W1(input))