from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset, maskedMSETest, horiz_eval
from torch.utils.data import DataLoader
import time
import numpy as np
from pathlib import Path
import pandas as pd

#Ignore the warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#Import the argumnets
from model_args import args

# Evaluation mode
args['train_flag'] = False
pred_horiz = args['pred_horiz']

# Initialize network
# ------------------
net = highwayNet(args)

# load the trained model
net_fname = 'trained_models/' + args['pooling']
if args['intention_module']:
    if args['input_dim']==3:
        net_fname = net_fname + '_vel_int.tar'
    else:
        net_fname = net_fname + '_int.tar'
else:
    if args['input_dim'] == 3:
        net_fname = net_fname + '_vel.tar'
    else:
        net_fname = net_fname + '.tar'


if (args['use_cuda']):
    net.load_state_dict(torch.load(net_fname), strict=False)
    net = net.cuda()
else:
    net.load_state_dict(torch.load(net_fname , map_location= lambda storage, loc: storage), strict=False)

# Test Cases
test_dataset_files = ['TestSet', 'TestSet_keep', 'TestSet_merge', 'TestSet_left',   'TestSet_right']
test_cases = ['overall', 'keep', 'merge', 'left', 'right']
rmse_eval = np.zeros([pred_horiz, len(test_cases)])

if args['intention_module']:
    if args['input_dim']==3:
        eval_fname = 'evaluation/' + args['pooling'] + '_int_vel.csv'
    else:
        eval_fname = 'evaluation/' + args['pooling'] + '_int.csv'
else:
    if args['input_dim'] == 3:
        eval_fname = 'evaluation/' + args['pooling'] + '_vel.csv'
    else:
        eval_fname = 'evaluation/' + args['pooling'] + '.csv'

for ds_ctr, ds_name in enumerate(test_dataset_files):

    ## Initialize data loaders
    tstSubset = ds_name
    tsSet = ngsimDataset('data/' + tstSubset + '.mat')
    tsDataloader = DataLoader(tsSet, batch_size=128, shuffle=True, num_workers=8, collate_fn=tsSet.collate_fn)

    lossVals = torch.zeros(args['out_length'])
    counts = torch.zeros(args['out_length'])

    if args['use_cuda']:

        lossVals = lossVals.cuda()
        counts = counts.cuda()

    for i, data in enumerate(tsDataloader):

        hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, \
        ds_ids, vehicle_ids, frame_ids = data

        if args['use_cuda']:
            hist = hist.cuda()
            nbrs = nbrs.cuda()
            mask = mask.cuda()
            lat_enc = lat_enc.cuda()
            lon_enc = lon_enc.cuda()
            fut = fut.cuda()
            op_mask = op_mask.cuda()
            ds_ids = ds_ids.cuda()
            vehicle_ids = vehicle_ids.cuda()
            frame_ids = frame_ids.cuda()

        # Forward pass
        if args['intention_module']:
            fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            fut_pred_max = torch.zeros_like(fut_pred[0])
            for k in range(lat_pred.shape[0]):
                lat_man = torch.argmax(lat_pred[k, :]).detach()
                lon_man = torch.argmax(lon_pred[k, :]).detach()
                indx = lon_man * args['num_lat_classes'] + lat_man
                fut_pred_max[:, k, :] = fut_pred[indx][:, k, :]
            l, c = maskedMSETest(fut_pred_max, fut, op_mask)

        else:
            fut_pred = net(hist, nbrs, mask)
            l, c = maskedMSETest(fut_pred, fut, op_mask)

        lossVals += l.detach()
        counts += c.detach()

    print(tstSubset)
    # Calculate RMSE in meters
    pred_rmse = torch.pow(lossVals / counts, 0.5) * 0.3048
    # Prediction Horizon of 5s
    pred_rmse_horiz = horiz_eval(pred_rmse, pred_horiz)
    print(pred_rmse_horiz)
    rmse_eval[:, ds_ctr] = pred_rmse_horiz

# Saving Results to a csv file
rmse_eval = np.around(rmse_eval, decimals=2)
df = pd.DataFrame(rmse_eval)
df.to_csv(eval_fname, header=test_cases,index=False)






