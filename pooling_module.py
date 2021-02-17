
from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn

def nbrs_pooling(net, soc_enc, masks, nbrs, nbrs_enc):
    if net.pooling == 'slstm':
        soc_enc = s_pooling(net, soc_enc)
    elif net.pooling == 'cslstm':
        soc_enc = cs_pooling(net, soc_enc)
    elif net.pooling == 'sgan' or net.pooling == 'polar':
        soc_enc = sg_pooling(net, masks, nbrs, nbrs_enc)

    return soc_enc

#SLSTM
def s_pooling(net, soc_enc):

    # Zero padding from bottom
    bottom_pad = net.grid_size[0] % net.kernel_size[0]

    if bottom_pad != 0:
        pad_layer = nn.ZeroPad2d((0, 0, 0, net.kernel_size[0] - bottom_pad))
        soc_enc = pad_layer(soc_enc)

    # Sum pooling
    avg_pool = torch.nn.AvgPool2d((net.kernel_size[0], net.kernel_size[1]))
    soc_enc = net.kernel_size[0] * net.kernel_size[1] * avg_pool(soc_enc)
    soc_enc = soc_enc.view(-1, net.kernel_size[0] * net.encoder_size)
    soc_enc = net.leaky_relu(soc_enc)

    return soc_enc


## CS-LSTM: Apply convolutional social pooling:
def cs_pooling(net, soc_enc):

    soc_enc = net.soc_maxpool(net.leaky_relu(net.conv_3x1(net.leaky_relu(net.soc_conv(soc_enc)))))
    soc_enc = soc_enc.view(-1,net.soc_embedding_size)
    return soc_enc

def sg_pooling(net, masks, nbrs, nbrs_enc):
    sum_masks = masks.sum(dim=3)

    soc_enc = torch.zeros(masks.shape[0],net.bottleneck_dim).float()
    if net.use_cuda:
        soc_enc = soc_enc.cuda()

    cntr = 0
    for ind in range(masks.shape[0]):
        no_nbrs = sum_masks[ind].nonzero().size()[0]
        if no_nbrs > 0:
            curr_nbr_pos = nbrs[:, cntr:cntr+no_nbrs, :]
            curr_nbr_enc = nbrs_enc[cntr:cntr+no_nbrs, :]
            cntr += no_nbrs

            end_nbr_pos = curr_nbr_pos[-1]
            rel_pos_embedding = net.rel_pos_embedding(end_nbr_pos)
            mlp_h_input = torch.cat([rel_pos_embedding, curr_nbr_enc], dim=1)

            # if only 1 neighbor, BatchNormalization will not work
            # So calling model.eval() before feeding the data will change
            # the behavior of the BatchNorm layer to use the running estimates
            # instead of calculating them
            if mlp_h_input.shape[0] == 1 & net.batch_norm:
                net.mlp_pre_pool.eval()

            curr_pool_h = net.mlp_pre_pool(mlp_h_input)

            curr_pool_h = curr_pool_h.max(0)[0]
            soc_enc[ind] = curr_pool_h
    return soc_enc