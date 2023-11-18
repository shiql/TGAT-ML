import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import utils
from TGATlayer import TGATlayer

class TGATML(nn.Module):
    def __init__(self, adjm, node_feats,in_dim=1,out_dim=24, residual_channels=2,dilation_channels=2, end_channels=2*10, layers=5, reg_param=0):
        super().__init__()
        self.adjm = adjm
        self.node_feats = node_feats
        self.reg_param = reg_param
        
        self.TCNGAT1 = TCNGATlayer(in_dim,out_dim,residual_channels,dilation_channels, end_channels,layers, kernel_size=2)
        self.TCNGAT2 = TCNGATlayer(in_dim,out_dim,residual_channels,dilation_channels, end_channels,layers, kernel_size=2)
        
        self.in_regressor = nn.Linear(out_dim, 1)
        self.out_regressor = nn.Linear(out_dim, 1)
        
    def src_embed(self):
        x = self.TCNGAT1.forward(self.adjm, self.node_feats)
        embedding = x.transpose(1, 3)
        embedding = torch.squeeze(embedding)
        return embedding
    
    def dst_embed(self):
        x = self.TCNGAT2.forward(self.adjm, self.node_feats)
        embedding = x.transpose(1, 3)
        embedding = torch.squeeze(embedding)
        return embedding
        
    def est_inflow(self, trip_od, dst_embedding):
        in_nodes, in_flows_idx = torch.unique(trip_od[:, 1], return_inverse=True)
        return self.in_regressor(dst_embedding[in_nodes])
    
    def est_outflow(self, trip_od, src_embedding):
        out_nodes, out_flows_idx = torch.unique(trip_od[:, 0], return_inverse=True)
        return self.out_regressor(src_embedding[out_nodes])
    
    def get_loss(self, trip_od, scaled_trip_volume,inflows, outflows, edge_est, inflow_est, outflow_est, multitask_weights):
        out_nodes, out_flows_idx = torch.unique(trip_od[:, 0], return_inverse=True)
        in_nodes, in_flows_idx = torch.unique(trip_od[:, 1], return_inverse=True)
        scaled_outflows = utils.scale(outflows[out_nodes])
        scaled_inflows = utils.scale(inflows[in_nodes])
        
        edge_est_loss = MSE(edge_est, scaled_trip_volume)
        inflow_est_loss = MSE(inflow_est, scaled_inflows)
        outflow_est_loss = MSE(outflow_est, scaled_outflows)
        reg_loss = 0.5 * (self.regularization_loss(self.src_embed()) + self.regularization_loss(self.dst_embed()))
        total_loss = multitask_weights[0] * edge_est_loss + multitask_weights[1] * inflow_est_loss + multitask_weights[2] * outflow_est_loss + self.reg_param * reg_loss
        return total_loss
    
    
    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2))

class Edge_Regression(nn.Module):
    def __init__(self, regfunction):
        file = r'.\Distance.csv'
        distm = pd.read_csv(file, index_col=0)
        self.distm = distm.values
        self.edge_regressor = regfunction
    
    def edge_fit(self, trip_od, trip_volume,src_embedding, dst_embedding):
        src_emb = src_embedding[trip_od[:,0]]
        dst_emb = dst_embedding[trip_od[:,1]]
        scaled_distm = self.distm / self.distm.max() * np.max([src_emb.max().detach().numpy(), dst_emb.max().detach().numpy()])
        feat_dist = scaled_distm[trip_od[:, 0], trip_od[:, 1]].reshape(-1, 1)
        feat_distT = torch.from_numpy(feat_dist).view(-1, 1)
        edge_feat = torch.cat((src_emb, feat_distT, dst_emb), dim=1)
        edge_feat= edge_feat.detach().numpy()
 
        return(self.edge_regressor.fit(edge_feat,trip_volume))
    
    def edge_estimate(self, trip_od, src_embedding, dst_embedding):
        src_emb = src_embedding[trip_od[:,0]]
        dst_emb = dst_embedding[trip_od[:,1]]
        scaled_distm = self.distm / self.distm.max() * np.max([src_emb.max().detach().numpy(), dst_emb.max().detach().numpy()])
        feat_dist = scaled_distm[trip_od[:, 0], trip_od[:, 1]].reshape(-1, 1)
        feat_distT = torch.from_numpy(feat_dist).view(-1, 1)
        edge_feat = torch.cat((src_emb, feat_distT, dst_emb), dim=1)
        edge_feat= edge_feat.detach().numpy()
        edge_pre = self.edge_regressor.predict(edge_feat)
        return (torch.tensor(edge_pre))
    
def MSE(y_hat, y):
    '''
    Root mean square
    '''
    limit = 20000
    if y_hat.shape[0] < limit:
        return torch.mean((y_hat - y)**2)
    else:
        acc_sqe_sum = 0
        for i in range(0, y_hat.shape[0], limit):
            acc_sqe_sum = acc_sqe_sum + torch.sum((y_hat[i: i + limit] - y[i: i + limit]) ** 2)
        return acc_sqe_sum / y_hat.shape[0]