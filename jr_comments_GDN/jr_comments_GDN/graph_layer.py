import torch
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing #install pyg..
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros
import time
import math

class GraphLayer(MessagePassing): #이거 자체가 aggregation을 사용하는 GNN convention.
    def __init__(self, in_channels, out_channels,
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1):
        

        super(GraphLayer, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout

        #self.__alpha__ = None

        self.lin = Linear(in_channels, out_channels, bias=False)

        self.a=Parameter(torch.Tensor(1, 4*out_channels))
        
        if bias:
            self.bias = Parameter(torch.Tensor( out_channels)) 
        else:
            self.register_parameter('bias',None) #MessagePassing의 method 인가봐.

        self.reset_parameters()

    def reset_parameters(self): #initialize Parameters~
        glorot(self.lin.weight)
        
        glorot(self.a)
        zeros(self.bias)


    
    #batch_mat: (3456,64)
    #gcn_out: (3456,64)
    def forward(self, batch_mat, topk_edge, embedding): 

        x = self.lin(batch_mat) #size out_channels
        batch_mat_tuple = (x, x)  #필요해. batch_mat_tuple[0]이 x_j이고, [1]이 x_i 일거야. 

        out = self.propagate(topk_edge, x=batch_mat_tuple, embedding=embedding, topk_edge=topk_edge) 
        #edge_index 까지는 그냥 보내는거고, 그 다음부터는 자유인듯.



        if self.bias is not None:
            out = out + self.bias  

        #alpha, self.__alpha__ = self.__alpha__, None


        

        return out #out == zit!!
        
        

    def message(self, x_i, x_j, edge_index_i, size_i, embedding, topk_edge):  
        
        '''
         edge_index_i == topk_edge[1] 이다. ji 순.
        '''
     
        '''
        (embedding.shape) (27,64)
        (x_i.shape) torch.Size([540, 64]), x_i는 Wxit의 의미.
        (topk_edge.shape) torch.Size([2, 540])     

        embedding_i: edge i 에 해당하는 i의 node embedding. 

        '''

        embedding_j=torch.empty(size=x_i.shape)
        embedding_i=torch.empty(size=x_i.shape)   #(540,64)
        for i in range(len(topk_edge[0])): 
            j_idx, i_idx=topk_edge[0][i], topk_edge[1][i]
            embedding_j[i]=embedding[j_idx]
            embedding_i[i]=embedding[i_idx]

        git = torch.cat((x_i, embedding_i), dim=-1) 
        gjt = torch.cat((x_j, embedding_j), dim=-1) 


        gijt=torch.cat((git,gjt),dim=-1) #(540,256)
        alpha= (self.a * gijt).sum(-1) #(540)
        alpha = alpha.view(-1, 1)  


        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        #self.__alpha__ = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha
        #(540,64)



    def __repr__(self): # print(class) 
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, 1)

