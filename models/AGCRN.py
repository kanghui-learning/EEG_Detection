import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros


class AVWGCN(nn.Module):
    r"""An implementation of the Node Adaptive Graph Convolution Layer.
    For details see: `"Adaptive Graph Convolutional Recurrent Network
    for Traffic Forecasting" <https://arxiv.org/abs/2007.02842>`_
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        embedding_dimensions (int): Number of node embedding dimensions.
    """

    def __init__(
        self, in_channels: int, out_channels: int, K: int, embedding_dimensions: int
    ):
        super(AVWGCN, self).__init__()
        self.K = K
        self.weights_pool = torch.nn.Parameter(
            torch.Tensor(embedding_dimensions, K, in_channels, out_channels)
        )
        self.bias_pool = torch.nn.Parameter(
            torch.Tensor(embedding_dimensions, out_channels)
        )
        glorot(self.weights_pool)
        zeros(self.bias_pool)

    def forward(self, X: torch.FloatTensor, E: torch.FloatTensor) -> torch.FloatTensor:
        r"""Making a forward pass.
        Arg types:
            * **X** (PyTorch Float Tensor) - Node features.
            * **E** (PyTorch Float Tensor) - Node embeddings.
        Return types:
            * **X_G** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """

        number_of_nodes = E.shape[0]
        supports = F.softmax(F.relu(torch.mm(E, E.transpose(0, 1))), dim=1)
        support_set = [torch.eye(number_of_nodes).to(supports.device), supports]
        for _ in range(2, self.K):
            support = torch.matmul(2 * supports, support_set[-1]) - support_set[-2]
            support_set.append(support)
        supports = torch.stack(support_set, dim=0)
        W = torch.einsum("nd,dkio->nkio", E, self.weights_pool)
        bias = torch.matmul(E, self.bias_pool)
        X_G = torch.einsum("knm,bmc->bknc", supports, X)
        X_G = X_G.permute(0, 2, 1, 3)
        X_G = torch.einsum("bnki,nkio->bno", X_G, W) + bias
        return X_G



class AGCRNCell(nn.Module):
    r"""An implementation of the Adaptive Graph Convolutional Recurrent Unit.
    For details see: `"Adaptive Graph Convolutional Recurrent Network
    for Traffic Forecasting" <https://arxiv.org/abs/2007.02842>`_
    Args:
        number_of_nodes (int): Number of vertices.
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        embedding_dimensions (int): Number of node embedding dimensions.
    """

    def __init__(
        self,
        number_of_nodes: int,
        in_channels: int,
        out_channels: int,
        K: int,
        embedding_dimensions: int,
    ):
        super(AGCRNCell, self).__init__()

        self.number_of_nodes = number_of_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.embedding_dimensions = embedding_dimensions
        self._setup_layers()

    def _setup_layers(self):
        self._gate = AVWGCN(
            in_channels=self.in_channels + self.out_channels,
            out_channels=2 * self.out_channels,
            K=self.K,
            embedding_dimensions=self.embedding_dimensions,
        )

        self._update = AVWGCN(
            in_channels=self.in_channels + self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            embedding_dimensions=self.embedding_dimensions,
        )

    def _set_hidden_state(self, batch_size):
        
        H = torch.zeros(batch_size, self.number_of_nodes, self.out_channels)
        return H


    def forward(
        self, X: torch.FloatTensor, E: torch.FloatTensor, H: torch.FloatTensor = None
    ) -> torch.FloatTensor:
        r"""Making a forward pass.
        Arg types:
            * **X** (PyTorch Float Tensor) - Node feature matrix.
            * **E** (PyTorch Float Tensor) - Node embedding matrix.
            * **H** (PyTorch Float Tensor) - Node hidden state matrix. Default is None.
        Return types:
            * **H** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """
        H = H.to(X.device)
        # print(X.shape, H.shape)
        # print(X.shape,H.shape)
        X_H = torch.cat((X, H), dim=-1)
        Z_R = torch.sigmoid(self._gate(X_H, E))
        Z, R = torch.split(Z_R, self.out_channels, dim=-1)
        C = torch.cat((X, Z * H), dim=-1)
        HC = torch.tanh(self._update(C, E))
        H = R * H + (1 - R) * HC
        return H
    


class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                # print(current_inputs[:, :, t, :].shape,state.shape)
                state = self.dcrnn_cells[i](current_inputs[:, t,:, :], node_embeddings,state )
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
            # print(current_inputs.shape)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i]._set_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class AGCRN(nn.Module):
    def __init__(self, num_nodes,input_dim,rnn_units,num_layers,embed_dim,cheb_k,num_classes):
        super(AGCRN, self).__init__()
        self.num_node = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units

        self.num_layers = num_layers
        self.embed_dim =  embed_dim
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)
            
        self.encoder = AVWDCRNN(num_nodes, input_dim, rnn_units,cheb_k,embed_dim, num_layers)

        #predictor
        #self.end_conv = nn.Conv2d(1, horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
    def forward(self, source, teacher_forcing_ratio=0.5):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        #  
        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        feature = output[:, -1, :, :]                                   #B, 1, N, hidden
        #print('feature_shape',feature.shape)
        # #CNN based predictor
        # output = self.end_conv((feature))                         #B, T*C, N, 1
        # output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        # output = output.permute(0, 1, 3, 2)                             #B, T, N, C
        # output = output.permute(0,2,1,3)  
        pooled_feature = feature.mean(dim=1) 
        #print('pooled_feature.shape:', pooled_feature.shape) # Aggregate over nodes: (B, hidden_dim)
        logits = self.classifier(pooled_feature)
        #print('logits.shape:', logits.shape)  # logits: (B, num_classes)
        return logits

     


# class AGCRN_FORCASTING(torch.nn.Module):

#     def __init__(self,node_number,node_features,horizon):
        
#         super(AGCRN_FORCASTING, self).__init__()
        
#         self.recurrent = AGCRN(number_of_nodes = node_number,
#                               in_channels = node_features,
#                               out_channels = 64,
#                               K = 2,
#                               embedding_dimensions = 10)
        
#         self.linear = torch.nn.Linear(64, horizon)

#         self.e = torch.empty(node_number,10)
#         torch.nn.init.xavier_uniform_(self.e)
#         self.h = None

    # def forward(self, x):
    #     self.e = self.e.to(x.device)
    #     h_0 = self.recurrent(x, self.e, self.h)
    #     y = F.relu(h_0)
    #     y = self.linear(y)
    #     return y, h_0



