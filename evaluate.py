from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset, maskedNLL, maskedMSE, maskedNLLTest, maskedMSETest
from torch.utils.data import DataLoader
import time
import numpy as np

# Network Arguments
#-------------------
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 16
args['out_length'] = 25
args['grid_size'] = (13,3)
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['num_lat_classes'] = 3
args['num_lon_classes'] = 3
args['train_flag'] = False

# Using Intention module?
args['intention_module'] = True

# Choose the pooling mechanism
# 'slstm', 'cslstm', 'sgan'
# -----------------------------
args['pooling'] = 'sgan'

if args['pooling'] == 'slstm':
    args['kernel_size'] = (4, 3)

elif args['pooling'] == 'cslstm':
    args['soc_conv_depth'] = 64
    args['conv_3x1_depth'] = 16

elif args['pooling'] == 'sgan':
    args['bottleneck_dim'] = 256
    args['sgan_batch_norm'] = False


# Initialize network
# ------------------
net = highwayNet(args)

# load the trained model
net_fname = 'trained_models/' + args['pooling']
if args['intention_module']:
    net_fname = net_fname + '_mnvr.tar'
else:
    net_fname = net_fname + '.tar'

## Initialize data loaders
tsSet = ngsimDataset('data/TestSet.mat')
tsDataloader = DataLoader(tsSet, batch_size=128, shuffle=True, num_workers=8, collate_fn=tsSet.collate_fn)

lossVals = torch.zeros(args['out_length'])
counts = torch.zeros(args['out_length'])

if args['use_cuda']:
    net = net.cuda()
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
    if args['use_intention']:
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

print(torch.pow(lossVals / counts, 0.5))  # Calculate RMSE
loss_total = torch.pow(lossVals / counts, 0.5)
fname = 'outfiles/rmse_from_code_' + str(args['ip_dim']) +'D_intention_4s_latOnly.csv'
rmse_file = open(fname, 'w')
np.savetxt(rmse_file, loss_total.cpu())






