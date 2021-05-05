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

# Dimensionality of the input:
# 2D (X and Y or R and Theta)
# 3D (adding velocity as a 3d dimension)
args['input_dim'] = 2

# Using Intention module?
args['intention_module'] = True

# Choose the pooling mechanism
# 'slstm', 'cslstm', 'sgan', 'polar'
# -----------------------------
args['pooling'] = 'slstm'

if args['pooling'] == 'slstm':
    args['kernel_size'] = (4, 3)

elif args['pooling'] == 'cslstm':
    args['soc_conv_depth'] = 64
    args['conv_3x1_depth'] = 16

elif args['pooling'] == 'sgan' or args['pooling'] == 'polar':
    args['bottleneck_dim'] = 256
    args['sgan_batch_norm'] = False

#ngsimDataset Class in utils.py
args['t_hist'] = 30
args['t_fut'] = 50
args['skip_factor'] = 2 #d_s

args['pretrainEpochs'] = 5
args['trainEpochs'] = 3

# Prediction horizon used in evaluation
args['pred_horiz'] = 5