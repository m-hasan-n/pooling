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
args['input_dim'] = 3

# Using Intention module?
args['intention_module'] = True

# Choose the pooling mechanism
# 'slstm', 'cslstm', 'sgan', 'polar'
# -----------------------------
args['pooling'] = 'polar'

#ngsimDataset Class in utils.py
args['t_hist'] = 30 #t_h
args['t_fut'] = 50 #t_f
args['skip_factor'] = 2 #d_s