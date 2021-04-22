from __future__ import division
import torch
import torch.nn as nn
from utils import outputActivation
from pooling_module import nbrs_pooling


class highwayNet(nn.Module):

    ## Initialization
    def __init__(self,args):
        super(highwayNet, self).__init__()

        ## Unpack arguments
        self.args = args

        ## Use gpu flag
        self.use_cuda = args['use_cuda']

        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = args['train_flag']

        ## Input Dimensionality
        self.input_dim = args['input_dim']
        ## Sizes of network layers
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.grid_size = args['grid_size']

        ## Sizes of network layers
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.grid_size = args['grid_size']

        self.dyn_embedding_size = args['dyn_embedding_size']
        self.input_embedding_size = args['input_embedding_size']
        self.num_lat_classes = args['num_lat_classes']
        self.num_lon_classes = args['num_lon_classes']

        # Pooling Mechanism
        # --------------------
        self.pooling = args['pooling']

        if self.pooling == 'slstm':
            self.kernel_size = args['kernel_size']
            # soc_embedding_size
            pooled_size = self.grid_size[0] // self.kernel_size[0]
            if self.grid_size[0] % self.kernel_size[0] != 0:
                pooled_size += 1
            self.soc_embedding_size = self.encoder_size * pooled_size

        elif self.pooling == 'cslstm':
            self.soc_conv_depth = args['soc_conv_depth']
            self.conv_3x1_depth = args['conv_3x1_depth']
            self.soc_embedding_size = (((args['grid_size'][0] - 4) + 1) // 2) * self.conv_3x1_depth

            # Convolutional social pooling layer and social embedding layer
            self.soc_conv = torch.nn.Conv2d(self.encoder_size, self.soc_conv_depth, 3)
            self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3, 1))
            self.soc_maxpool = torch.nn.MaxPool2d((2, 1), padding=(1, 0))

        elif self.pooling == 'sgan' or self.pooling == 'polar':
            self.bottleneck_dim = args['bottleneck_dim']
            self.mlp_pre_dim = 2 * self.encoder_size
            self.soc_embedding_size = self.bottleneck_dim
            self.rel_pos_embedding = nn.Linear(self.input_dim, self.encoder_size)
            self.batch_norm = args['sgan_batch_norm']
            self.mlp_pre_pool = self.make_mlp(self.mlp_pre_dim, self.bottleneck_dim, self.batch_norm)


        self.IA_module = args['intention_module']
        if self.IA_module:
            # Decoder LSTM
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size +
                                          self.num_lat_classes + self.num_lon_classes, self.decoder_size)
        else:
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size)


        ## Define network weights
        # Input embedding layer
        self.ip_emb = torch.nn.Linear(self.input_dim, self.input_embedding_size)

        # Encoder LSTM
        self.enc_lstm = torch.nn.LSTM(self.input_embedding_size, self.encoder_size, 1)

        # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(self.encoder_size, self.dyn_embedding_size)

        # Output layers:
        if self.input_dim == 2:
            op_gauss_dim = 5
        elif self.input_dim == 3:
            op_gauss_dim = 7
        self.op = torch.nn.Linear(self.decoder_size, op_gauss_dim)
        self.op_lat = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lat_classes)
        self.op_lon = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lon_classes)

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    ## Forward Pass
    def forward(self, hist, nbrs, masks, lat_enc=None, lon_enc=None):

        ## Forward pass hist:
        _, (hist_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))
        hist_enc = self.leaky_relu(self.dyn_emb(hist_enc.view(hist_enc.shape[1], hist_enc.shape[2])))

        ## Forward pass nbrs
        _, (nbrs_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])

        ## Social Tensor using Masked scatter
        soc_enc = torch.zeros_like(masks).float()
        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc)
        soc_enc = soc_enc.permute(0, 3, 2, 1)
        soc_enc = soc_enc.contiguous()

        # Pooling
        #---------
        soc_enc_pooled = nbrs_pooling(self, soc_enc, masks, nbrs, nbrs_enc)

        enc = torch.cat((soc_enc_pooled, hist_enc), 1)

        if self.IA_module:
            ## Maneuver recognition:
            lat_pred = self.softmax(self.op_lat(enc))
            lon_pred = self.softmax(self.op_lon(enc))

            if self.train_flag:
                ## Concatenate maneuver encoding of the true maneuver
                enc = torch.cat((enc, lat_enc, lon_enc), 1)

                fut_pred = self.decode(enc)

            else:
                fut_pred = []
                ## Predict trajectory distributions for each maneuver class
                for k in range(self.num_lon_classes):
                    for l in range(self.num_lat_classes):
                        lat_enc_tmp = torch.zeros_like(lat_enc)
                        lon_enc_tmp = torch.zeros_like(lon_enc)
                        lat_enc_tmp[:, l] = 1
                        lon_enc_tmp[:, k] = 1
                        enc_tmp = torch.cat((enc, lat_enc_tmp, lon_enc_tmp), 1)
                        fut_pred.append(self.decode(enc_tmp))

            return fut_pred, lat_pred, lon_pred

        else:
            fut_pred = self.decode(enc)
            return fut_pred

    # Decoder Module
    def decode(self, enc):
        enc = enc.repeat(self.out_length, 1, 1)
        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred)
        return fut_pred

    # MLP
    def make_mlp(self, dim_in, dim_out, batch_norm):
        if batch_norm:
            layers = [nn.Linear(dim_in, dim_out), nn.BatchNorm1d(dim_out), nn.ReLU()]
        else:
            layers = [nn.Linear(dim_in, dim_out), nn.ReLU()]
        return nn.Sequential(*layers)