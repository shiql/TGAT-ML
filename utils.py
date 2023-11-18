import pandas as pd
import numpy as np
import torch

import numpy_indexed as npi

import warnings

def load_dataset():
    train = pd.read_csv(r'.\CommutingFlow_train.csv')
    valid = pd.read_csv(r'.\CommutingFlow_valid.csv')
    test = pd.read_csv(r'.\CommutingFlow_test.csv')
    # mapping OBJECTID to node id
    mapping_table = pd.read_csv(r'.\NodeId_GridId.csv') #OBJECTID to NodeId,two columns：NodeId,OBJECTID
    train = ct2nid(train, mapping_table) #field: 'src', 'dst', 'count'，src dst is the NodeId
    valid = ct2nid(valid, mapping_table)
    test = ct2nid(test, mapping_table)
    
    data_all = pd.concat([train, valid, test])
    inflow = pd.DataFrame(index=mapping_table['NodeId'])
    inflow_group = data_all.groupby('dst').agg({'count': 'sum'})
    inflow['count'] = inflow_group
    inflow = inflow.fillna(0)
    
    outflow = pd.DataFrame(index=mapping_table['NodeId'])
    outflow_group = data_all.groupby('src').agg({'count': 'sum'})
    outflow['count'] = outflow_group
    outflow = outflow.fillna(0)
       
    # load node feature table,
    node_feats = pd.read_csv(r'.\Grid_attributes.csv')
    node_feats['OBJECTID'] = mapping_table.set_index('OBJECTID').loc[node_feats['OBJECTID']].values # map census tract to node id
    node_feats = node_feats.rename(columns={'OBJECTID': 'nid'}).set_index('nid').sort_index()
    if node_feats.isnull().values.any(): # if there is any NaN in nodes' feature table
        node_feats.fillna(0, inplace=True)
        warnings.warn('Feature table contains NaN. 0 is used to fill these NaNs')
    
    # normalization
    node_featsT = node_feats.T              
    node_featsTnorm = (node_featsT - node_featsT.mean()) / node_featsT.std()
    node_feats = node_featsTnorm.T
    if node_feats.isnull().values.any(): # if there is any NaN in nodes' feature table
        node_feats = node_feats.fillna(0)
    
    # load adjacency matrix
    ct_adj = pd.read_csv(r'.\Timehour_weight040.csv', index_col=0)
    ct_inorder = mapping_table.sort_values(by='NodeId')['OBJECTID']
    ct_adj = ct_adj.loc[ct_inorder, ct_inorder.astype(str)]
    # min-max scale the weights
    ct_adj = ct_adj / ct_adj.max().max() # min is 0
    # fill nan with 0
    ct_adj = ct_adj.fillna(0)
    # define column order
    cols = ['src', 'dst', 'count']
    data = {}
    data['train'] = train[cols].values
    data['valid'] = valid[cols].values
    data['test'] = test[cols].values
    data['inflow'] = inflow.values
    data['outflow'] = outflow.values   
    data['num_nodes'] = ct_adj.shape[0]
    data['node_feats'] = node_feats.values
    data['ct_adjacency_withweight'] = ct_adj.values
    return data

def ct2nid(dataframe, mapping_table):
    frame = dataframe.copy() #frame has 3 columns: Oobjid,Dobjid,Flow, rename Flow as count
    frame.rename(columns={'Flow': 'count'}, inplace = True)
    mapping = mapping_table.copy() #mapping_table has 2 columns: NodeId,OBJECTID
    mapping = mapping.set_index('OBJECTID')
    frame['src'] = mapping.loc[frame['Oobjid']].values
    frame['dst'] = mapping.loc[frame['Dobjid']].values
    return frame[['src', 'dst', 'count']]


def mini_batch_gen(train_data, mini_batch_size, num_nodes, negative_sampling_rate = 0):
    '''
    generator of mini-batch samples
    '''
    # positive data
    pos_samples = train_data
    # negative sampling to get negative data
    neg_samples = negative_sampling(pos_samples, num_nodes, negative_sampling_rate)
    # binding together
    if neg_samples is not None:
        samples = torch.cat((pos_samples, neg_samples), dim=0)
    else:
        samples = pos_samples
    # shuffle
    samples = samples[torch.randperm(samples.shape[0])]
    # cut to mini-batches and wrap them by a generator
    for i in range(0, samples.shape[0], mini_batch_size):
        yield samples[i:i+mini_batch_size]

def negative_sampling(pos_samples, num_nodes, negative_sampling_rate = 0):
    '''
    perform negative sampling by perturbing the positive samples
    '''
    # if do not require negative sampling
    if negative_sampling_rate == 0:
        return None
    # else, let's do negative sampling
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_sampling_rate
    neg_samples = np.tile(pos_samples, [negative_sampling_rate, 1])
    neg_samples[:, -1] = 0 # set trip volume to be 0
    sample_nid = np.random.randint(num_nodes, size = num_to_generate) # randomly sample nodes
    pos_choices = np.random.uniform(size = num_to_generate) # randomly sample position
    subj = pos_choices > 0.5
    obj = pos_choices <= 0.5
    neg_samples[subj, 0] = sample_nid[subj]
    neg_samples[obj, 1] = sample_nid[obj]
 
    while(True):
        overlap = npi.contains(pos_samples[:, :2], neg_samples[:, :2]) # True means overlap
        if overlap.any():
            neg_samples_overlap = neg_samples[overlap]
            sample_nid = np.random.randint(num_nodes, size = overlap.sum())
            pos_choices = np.random.uniform(size = overlap.sum())
            subj = pos_choices > 0.5
            obj = pos_choices <= 0.5
            neg_samples_overlap[subj, 0] = sample_nid[subj]
            neg_samples_overlap[obj, 1] = sample_nid[obj]
            neg_samples[overlap] = neg_samples_overlap
        else: # if no overlap, just break resample loop
            break
    # return negative samples
    return torch.from_numpy(neg_samples)

def metric(trip_volume, prediction):
    '''
    evaluate trained model.
    '''
    prediction = scale_back(prediction)
    prediction = torch.floor(prediction)
    # get ground-truth label
    y = trip_volume.float()
    # get metric
    rmse = RMSE(prediction, y)
    mae = MAE(prediction, y)
    mape = MAPE(prediction, y)
    cpc = CPC(prediction, y)
    cpl = CPL(prediction, y)
    return rmse.item(), mae.item(), mape.item(), cpc.item(), cpl.item()

def scale(y):
    '''
    scale the target variable
    '''
    return torch.sqrt(y)

def scale_back(scaled_y):
    '''
    scale back the target varibale to normal scale
    '''
    return scaled_y ** 2

def RMSE(y_hat, y):
    '''
    Root Mean Square Error Metric
    '''
    return torch.sqrt(torch.mean((y_hat - y)**2))

def MAE(y_hat, y):
    '''
    Mean Absolute Error Metric
    '''
    abserror = torch.abs(y_hat - y)
    return torch.mean(abserror)

def MAPE(y_hat, y):
    '''
    Mean Absolute Error Metric
    '''
    abserror = torch.abs(y_hat - y)
    return torch.mean(abserror / y)

def CPC(y_hat, y):
    '''
    Common Part of Commuters Metric
    '''
    return 2 * torch.sum(torch.min(y_hat, y)) / (torch.sum(y_hat) + torch.sum(y))

def CPL(y_hat, y):
    '''
    Common Part of Links Metric. 
    
    Check the topology.
    '''
    yy_hat = y_hat > 0
    yy = y > 0
    return 2 * torch.sum(yy_hat * yy) / (torch.sum(yy_hat) + torch.sum(yy))