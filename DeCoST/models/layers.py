import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_encoding(encoded_nodes, node_index_to_pick):
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

    def forward(self, input1):
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
        
        if torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any():
            print("<< NaN detected in Q/K/V tensors!")
        
        if self.model_params['norm_loc'] == "norm_last":
            out_concat = multi_head_attention(q, k, v)  # (batch, problem, HEAD_NUM*KEY_DIM)
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
            if self.model_params["pip_decoder"]:
                self.Wq_last_sl = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
        elif self.problem in ["VRPB", "TOPTWVP", "OPTWVP", "TOPTW",  "OPTW", "TSPTW", "TSPDL"]:
            self.Wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
            if self.model_params["pip_decoder"]:
                self.Wq_last_sl = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
        elif self.problem in ["OVRP", "OVRPB", "VRPTW", "VRPBTW", "VRPL", "VRPBL"]:
            attr_num = 3 if self.model_params["extra_feature"] else 2
            self.Wq_last = nn.Linear(embedding_dim + attr_num, head_num * qkv_dim, bias=False)
            if self.model_params["pip_decoder"]:
                self.Wq_last_sl = nn.Linear(embedding_dim + attr_num, head_num * qkv_dim, bias=False)
        elif self.problem in ["VRPLTW", "VRPBLTW", "OVRPL", "OVRPBL", "OVRPTW", "OVRPBTW"]:
            self.Wq_last = nn.Linear(embedding_dim + 3, head_num * qkv_dim, bias=False)
            if self.model_params["pip_decoder"]:
                self.Wq_last_sl = nn.Linear(embedding_dim + 3, head_num * qkv_dim, bias=False)
        elif self.problem in ["OVRPLTW", "OVRPBLTW"]:
            self.Wq_last = nn.Linear(embedding_dim + 4, head_num * qkv_dim, bias=False)
            if self.model_params["pip_decoder"]:
                self.Wq_last_sl = nn.Linear(embedding_dim + 4, head_num * qkv_dim, bias=False)
        else:
            raise NotImplementedError

        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        if self.model_params["pip_decoder"] and self.model_params['W_kv_sl']:
            self.Wk_sl = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
            self.Wv_sl = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
            self.k_sl = None  # saved key, for multi-head_attention
            self.v_sl = None  # saved value, for multi-head_attention
            self.single_head_key_sl = None

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        if self.model_params["pip_decoder"] and self.model_params['W_out_sl']:
            self.multi_head_combine_sl = nn.Linear(head_num * qkv_dim, embedding_dim)

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
        input_cat = torch.cat((encoded_last_node, attr), dim=2)
        # shape = (batch, group, EMBEDDING_DIM+1)

        if self.model_params['pip_decoder']:
            if isinstance(use_predicted_PI_mask, bool):
                if self.model_params['detach_from_encoder']:
                    q_last_sl = reshape_by_heads(self.Wq_last_sl(input_cat.detach()), head_num=head_num)
                else:
                    q_last_sl = reshape_by_heads(self.Wq_last_sl(input_cat), head_num=head_num)

                ninf_mask_sl = ninf_mask if self.model_params['use_ninf_mask_in_sl_MHA'] else None
                if self.model_params['W_kv_sl']:
                    out_concat_sl = multi_head_attention(q_last_sl, self.k_sl, self.v_sl, rank3_ninf_mask=ninf_mask_sl)
                else:
                    out_concat_sl = multi_head_attention(q_last_sl, self.k, self.v, rank3_ninf_mask=ninf_mask_sl)
                if self.model_params['W_out_sl']:
                    mh_atten_out_sl = self.multi_head_combine_sl(out_concat_sl)
                else:
                    mh_atten_out_sl = self.multi_head_combine(out_concat_sl)

                if self.model_params['W_kv_sl']:
                    score_sl = torch.matmul(mh_atten_out_sl, self.single_head_key_sl)
                else:
                    score_sl = torch.matmul(mh_atten_out_sl, self.single_head_key)

                probs_sl = score_sl if no_sigmoid else torch.sigmoid(score_sl)
                if no_select_prob:
                    return probs_sl
            else:
                probs_sl = use_predicted_PI_mask

            if not isinstance(use_predicted_PI_mask, bool) or use_predicted_PI_mask:
                ninf_mask0 =  ninf_mask.clone()
                if isinstance(probs_sl, list):
                    for i in range(len(probs_sl)):
                        ninf_mask = ninf_mask +torch.where(probs_sl[i] > self.model_params["decision_boundary"], float('-inf'),ninf_mask) if not no_sigmoid \
                            else torch.where(torch.sigmoid(probs_sl[i]) > self.model_params["decision_boundary"], float('-inf'), ninf_mask)
                else:
                    ninf_mask = torch.where(probs_sl>self.model_params["decision_boundary"], float('-inf'), ninf_mask) if not no_sigmoid \
                        else torch.where(torch.sigmoid(probs_sl)>self.model_params["decision_boundary"], float('-inf'), ninf_mask)
                all_infsb = ((ninf_mask == float('-inf')).all(dim=-1)).unsqueeze(-1).expand(-1, -1, self.single_head_key.size(-1))
                ninf_mask = torch.where(all_infsb, ninf_mask0, ninf_mask)
        # shape: (batch, head_num, pomo, qkv_dim)

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
        if self.model_params['pip_decoder']:
            return probs, probs_sl

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

def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

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
        n_nodes = h.shape[0]

        # Apply linear transformation to node feature -> W h
        # output shape (n_nodes, n_hidden * n_heads)
        h_transformed = torch.mm(h, self.W)
        h_transformed = F.dropout(h_transformed, self.dropout, training=self.training)

        # splitting the heads by reshaping the tensor and putting heads dim first
        # output shape (n_heads, n_nodes, n_hidden)
        h_transformed = h_transformed.view(n_nodes, self.n_heads, self.n_hidden).permute(1, 0, 2)
        
        # getting the attention scores
        # output shape (n_heads, n_nodes, n_nodes)
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
        if self.concat:
            h_prime = h_prime.permute(1, 0, 2).contiguous().view(n_nodes, self.out_features)
        else:
            h_prime = h_prime.mean(dim=0)

        return h_prime

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