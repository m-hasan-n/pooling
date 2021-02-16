from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset,maskedNLL,maskedMSE,maskedNLLTest
from torch.utils.data import DataLoader
import time
import math

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
args['train_flag'] = True

# Using Intention module?
args['intention_module'] = False

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
    args['sgan_batch_norm'] = True

# Initialize network
# ------------------
net = highwayNet(args)
if args['use_cuda']:
    net = net.cuda()

## Initialize optimizer
# ---------------------
pretrainEpochs = 5
trainEpochs = 3
optimizer = torch.optim.Adam(net.parameters())
batch_size = 128
crossEnt = torch.nn.BCELoss()

## Initialize data loaders
valSet = ngsimDataset('data/ValSet_mnvr_new_corrected.mat')
# trSet = ngsimDataset('data/ValSet_mnvr_new_corrected.mat')
trSet = ngsimDataset('data/TrainSet_mnvr_new_corrected_1.mat')#, 'data/TrainSet_mnvr_new_corrected_2.mat'


trDataloader = DataLoader(trSet,batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=trSet.collate_fn)
valDataloader = DataLoader(valSet,batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=valSet.collate_fn)

## Initialize Train and validation loss:
train_loss = []
val_loss = []
prev_val_loss = math.inf

# Main training
# -------------
for epoch_num in range(pretrainEpochs+trainEpochs):

    if epoch_num == 0:
        print('Pre-training with MSE loss')
    elif epoch_num == pretrainEpochs:
        print('Training with NLL loss')


    ## Train:
    #-----------------------------------------------------------------------------
    net.train_flag = True

    # Variables to track training performance:
    avg_tr_loss = 0
    avg_tr_time = 0
    avg_lat_acc = 0
    avg_lon_acc = 0

    for i, data in enumerate(trDataloader):
        st_time = time.time()

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

        #If using the intention module
        if args['intention_module']:
            # Pre-train with MSE loss to speed up training
            if epoch_num < pretrainEpochs:
                fut_pred, _, _= net(hist, nbrs, mask, lat_enc, lon_enc)
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)

                # Train with NLL loss
                l = maskedNLL(fut_pred, fut, op_mask) + crossEnt(lat_pred, lat_enc) \
                    + crossEnt(lon_pred, lon_enc)
                avg_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / \
                               lat_enc.size()[0]
                avg_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / \
                               lon_enc.size()[0]

        #Without the intention prediction
        else:
            fut_pred = net(hist, nbrs, mask)

            # Pre-train with MSE loss to speed up training
            if epoch_num < pretrainEpochs:
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                l = maskedNLL(fut_pred, fut, op_mask)

        # Backprop and update weights
        optimizer.zero_grad()
        l.backward()
        a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()

        # average train loss and average train time:
        batch_time = time.time() - st_time
        avg_tr_loss += l.item()
        avg_tr_time += batch_time

        # Printing
        if i % 100 == 99:
            eta = avg_tr_time / 100 * (len(trSet) / batch_size - i)
            print("Epoch no:", epoch_num + 1,
                  "| Epoch progress(%):", format(i / (len(trSet) / batch_size) * 100, '0.2f'),
                  "| Avg train loss:", format(avg_tr_loss / 100, '0.4f'),
                  "| Acc:", format(avg_lat_acc, '0.4f'), format(avg_lon_acc, '0.4f'),
                  "| Validation loss prev epoch", format(prev_val_loss, '0.4f'),
                  "| ETA(s):", int(eta))

            train_loss.append(avg_tr_loss / 100)
            avg_tr_loss = 0
            avg_lat_acc = 0
            avg_lon_acc = 0
            avg_tr_time = 0

    # _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

    # Validate:______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net.train_flag = False

    print("Epoch", epoch_num + 1, 'complete. Calculating validation loss...')
    avg_val_loss = 0
    avg_val_lat_acc = 0
    avg_val_lon_acc = 0
    val_batch_count = 0

    for i, data in enumerate(valDataloader):
        st_time = time.time()
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

        # If using the intention module
        if args['intention_module']:
            # Pre-train with MSE loss to speed up training
            if epoch_num < pretrainEpochs:
                # During pre-training with MSE loss, validate with MSE for true maneuver class trajectory
                net.train_flag = True
                fut_pred, _, _ = net(hist, nbrs, mask, lat_enc, lon_enc)
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                # During training with NLL loss, validate with NLL over multi-modal distribution
                fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                l = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask, avg_along_time=True)
                avg_val_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / \
                                   lat_enc.size()[0]
                avg_val_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / \
                                   lon_enc.size()[0]
        else:
            fut_pred = net(hist, nbrs, mask)
            if epoch_num < pretrainEpochs:
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                l = maskedNLL(fut_pred, fut, op_mask)

        avg_val_loss += l.item()
        val_batch_count += 1

    print(avg_val_loss / val_batch_count)

    # Print validation loss and update display variables
    print('Validation loss :', format(avg_val_loss / val_batch_count, '0.4f'), "| Val Acc:",
          format(avg_val_lat_acc / val_batch_count * 100, '0.4f'),
          format(avg_val_lon_acc / val_batch_count * 100, '0.4f'))

    val_loss.append(avg_val_loss / val_batch_count)
    prev_val_loss = avg_val_loss / val_batch_count

model_fname = 'trained_models/'+args['pooling']
if args['intention_module']:
    model_fname = model_fname + '_mnvr_bn.tar'
else:
    model_fname = model_fname + '_bn.tar'

torch.save(net.state_dict(), model_fname)




