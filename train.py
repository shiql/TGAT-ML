import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
import utils
from TGATML import TGATML
from TGATML import Edge_Regression
from sklearn.ensemble import GradientBoostingRegressor

torch.manual_seed(2019)
np.random.seed(2019)

def main(args):
    device = args['device']
    # tensorboard writer
    writer = SummaryWriter(comment='#multitask{}'.format(args['multitask_weights']))
    # logger
    logger = logging.getLogger('#multitask{}'.format(args['multitask_weights'])) # experiment name
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler("log/training_log.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG) #
    
    data = utils.load_dataset()
    train_data = data['train'] #trips -- [(src, dst, cnt)]
    valid_data = data['valid']
    
    inflow = data['inflow'] # in/out flow counts -- [(count)]
    outflow = data['outflow']
    
    num_nodes = data['num_nodes'] # number of nodes
    adjm = data['ct_adjacency_withweight'] # census tract adjacency as matrix
    node_feats = data['node_feats'] # geographical features -- [[features]]
    node_feats = torch.from_numpy(node_feats.astype(np.float32))
    model =  TCNGATModel(adjm,node_feats,in_dim=args['in_dim'], out_dim=args['seq_length'], residual_channels=args['nhid'], dilation_channels=args['nhid'], end_channels=args['nhid']*10, layers=args['layers'],reg_param=args['reg_param'])
    model = model.to(device)
    
    train_data = torch.from_numpy(train_data)
    trip_od_train = train_data[:, :2].long().to(device)
    trip_volume_train = train_data[:, -1].float().to(device)
    trip_od_valid = torch.from_numpy(valid_data[:, :2]).long().to(device)
    trip_volume_valid = torch.from_numpy(valid_data[:, -1]).float().to(device)
    
    inflow = torch.from_numpy(inflow).view(-1, 1).float().to(device)
    outflow = torch.from_numpy(outflow).view(-1, 1).float().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.1)
    
    model_state_file = './models/model_state_multitask{}.pth'.format(args['multitask_weights'])
    #best_rmse = 1e6
    best_cpc = 0
    
    total_iter = 0
    for epoch in range(args['max_epochs']):
        model.train()
        mini_batch_gen = utils.mini_batch_gen(train_data, mini_batch_size = int(args['mini_batch_size']), num_nodes=num_nodes, negative_sampling_rate = 0)
        mini_iter = 0
        for mini_batch in mini_batch_gen:
            model.train()
            mini_iter = mini_iter+1
            trip_od = mini_batch[:, :2].long().to(device)
            scaled_trip_volume = utils.scale(mini_batch[:, -1].float()).to(device) # get trip volume
            # get predict
            src_embedding = model.src_embed()
            dst_embedding = model.dst_embed()
            
            edge_regression = Edge_Regression(GradientBoostingRegressor(max_depth=2, random_state=2019, n_estimators=100))
            edge_regression.edge_fit(trip_od, scaled_trip_volume, src_embedding, dst_embedding)
            
            edge_est = edge_regression.edge_estimate(trip_od_train, src_embedding, dst_embedding)
            inflow_est = model.est_inflow(trip_od_train, dst_embedding)#est_inflow(self, trip_od)
            outflow_est = model.est_outflow(trip_od_train, src_embedding)

            loss = model.get_loss(trip_od_train, utils.scale(trip_volume_train),inflow, outflow, edge_est, inflow_est, outflow_est, multitask_weights=args['multitask_weights'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['grad_norm']) # clip to make stable
            optimizer.step()
            
            print('epoch {},mini_batch in {},loss={}'.format(epoch, mini_iter, loss.item()))
            total_iter = total_iter+1
            writer.add_scalar('mini_loss_train', loss.item(), global_step = total_iter)
            
            #*************************alid dataset
            model.eval()
            with torch.no_grad():
                edge_est_valid = edge_regression.edge_estimate(trip_od_valid, src_embedding, dst_embedding)
                inflow_est_valid = model.est_inflow(trip_od_valid, dst_embedding)
                outflow_est_valid = model.est_outflow(trip_od_valid, src_embedding)
                loss = model.get_loss(trip_od_valid, utils.scale(trip_volume_valid),inflow, outflow, edge_est_valid, inflow_est_valid, outflow_est_valid, multitask_weights=args['multitask_weights'])
            
            rmse, mae, mape, cpc, cpl = utils.metric(trip_volume_valid, edge_est_valid)
            #
            logger.debug("Evaluation on valid dataset:")
            logger.debug("Epoch {:04d} | Loss = {:.4f}".format(epoch, loss))
            logger.debug("RMSE {:.4f} | MAE {:.4f} | MAPE {:.4f} | CPC {:.4f} | CPL {:.4f} |".format(rmse, mae, mape, cpc, cpl))
            writer.add_scalar('overall_loss_valid', loss.item(), global_step = total_iter)
            writer.add_scalar('RMSE', rmse, global_step = total_iter)
            writer.add_scalar('MAE', mae, global_step = total_iter)
            writer.add_scalar('CPC', cpc, global_step = total_iter)
            
            if cpc > best_cpc:
                best_cpc = cpc
                torch.save({'state_dict': model.state_dict(), 'epoch': mini_iter, 'rmse': rmse, 'mae': mae, 'mape': mape, 'cpc': cpc}, model_state_file)
                src_embedding = src_embedding.detach().cpu().numpy()
                dst_embedding = dst_embedding.detach().cpu().numpy()
                emb_fp = "./embeddings/multitask{}.npz".format(args['multitask_weights'])
                np.savez(emb_fp, src_embedding, dst_embedding)
                logger.info('Best RMSE found on epoch {}'.format(mini_iter))
            logger.info("-----------------------------------------")
    scheduler.step()


if __name__== "__main__":
    args = {'device': 'cpu',
            'reg_param': 0, 'dropout': 3,
            'max_epochs': 25, 'mini_batch_size': 10000, 'negative_sampling_rate': 0,
            'lr': 2e-2, 'grad_norm': 1.0,
            'seq_length': 24,
            'in_dim': 1,
            'nhid': 2,
            'layers': 5,
            'multitask_weights': (0.5, 0.25, 0.25)}
    main(args)
    