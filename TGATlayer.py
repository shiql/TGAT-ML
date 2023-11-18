import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import warnings
warnings.filterwarnings('ignore')

class TGATlayer(nn.Module):
    def __init__(self, in_dim=1,out_dim=24,residual_channels=2,dilation_channels=2,end_channels=2*10,layers=5,kernel_size=2):
        super().__init__()

        self.layers = layers
        self.residual_channels = residual_channels
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()

        self.bn = nn.ModuleList()
        self.gatlayer = nn.ModuleList()
        
        self.start_conv = nn.Conv2d(in_channels=in_dim,#weight.size[32,2,1,1] (outputchanels, inputchannel, kersizeH,kernelW)
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        receptive_field = 1
        new_dilation = 1
        gatinput_dim = out_dim
        for i in range(self.layers):
            self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, #when layers=1/2, weight[32,32,1,2], bias[32]
                                               out_channels=dilation_channels,
                                               kernel_size=(1,kernel_size),dilation=new_dilation))
            
            self.gate_convs.append(nn.Conv1d(in_channels=residual_channels, #when layers=1, weight[32,32,1,2], bias[32]
                                             out_channels=dilation_channels,
                                             kernel_size=(1, kernel_size), dilation=new_dilation))
            
            self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,#when layers=1, weight[32,32,1,1], bias[32]
                                                 out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                
            self.bn.append(nn.BatchNorm2d(residual_channels))
            
            receptive_field = receptive_field + new_dilation
            gatinput_dim = gatinput_dim-new_dilation
            if (i+1)%2 == 1:
                self.gatlayer.append(GATLayer(gatinput_dim, gatinput_dim))
            
            if new_dilation >=8:
                new_dilation = 8
            else:
                new_dilation *=2
            

        self.end_conv_1 = nn.Conv2d(in_channels=residual_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1,1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field
        
        
    def forward(self, adjm, node_feats):
        node_feats = torch.unsqueeze(node_feats, dim=0)
        node_feats = torch.unsqueeze(node_feats, dim=1)
        
        in_len = node_feats.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(node_feats,(self.receptive_field-in_len,0,0,0))
        else:
            x = node_feats 
        
        x = self.start_conv(x)

        for i in range(self.layers):
            residual = x 
            filters = self.filter_convs[i](residual)
            filters = torch.tanh(filters)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filters * gate
            if (i+1)%2 == 1:
                gc = x
                for j in range(self.residual_channels):
                    gc[0,j,:,:] = self.gatlayer[int(i/2)](adjm, gc[0,j,:,:].clone())
                x = gc
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            
            x = self.bn[i](x)

        x = F.relu(x)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

class GATLayer(nn.Module):
    def __init__(self, in_ndim, out_ndim, in_edim=1, out_edim=1): 
    
        super().__init__()
        self.fc0 = nn.Linear(in_edim, out_edim, bias=False)
        self.fc1 = nn.Linear(in_ndim, out_ndim, bias=False)
        self.fc2 = nn.Linear(in_ndim, out_ndim, bias=False)
        
        self.attn_fc = nn.Linear(2 * out_ndim + out_edim, 1, bias=False)
        
        self.activation = F.relu
        self.weights = nn.Parameter(torch.Tensor(2, 1)) # used to calculate convex combination weights
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))
        
    def forward(self, adjm, node_feats):
        node_featstmp = node_feats.detach().numpy()
        g = build_graph_from_matrix(adjm, node_featstmp)
        
        g.apply_edges(self.edge_feat_func)
        z = self.fc1(node_feats) 
        g.ndata['z'] = z
        z_i = self.fc2(node_feats) 
        g.ndata['z_i'] = z_i
        
        g.apply_edges(self.edge_attention)
        
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h') 
    
    def edge_feat_func(self, edges):
        '''
        deal with edge features
        '''
        return {'t': self.fc0(edges.data['d'])}

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['t']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'],'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        z_neighbor = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        z_i = nodes.data['z_i']
        lambda_ = F.softmax(self.weights, dim=0)
        h = self.activation(z_i + z_neighbor)
        return {'h': h}

def build_graph_from_matrix(adjm, node_feats):
    dst, src = adjm.nonzero()
    d = adjm[adjm.nonzero()]
    g = dgl.DGLGraph()
    g.add_nodes(adjm.shape[0])
    g.add_edges(src, dst, {'d': torch.tensor(d).float().view(-1, 1)})
    g.ndata['attr'] = torch.from_numpy(node_feats)
    norm = comp_deg_norm(g)
    g.ndata['norm'] = torch.from_numpy(norm).view(-1,1)
    return g
    
def comp_deg_norm(g):
    '''
    compute the degree normalization factor which is 1/in_degree
    '''
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm