from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch
from exp_args import args
#___________________________________________________________________________________________________________________________

### Dataset class for the NGSIM dataset
class ngsimDataset(Dataset):
    def __init__(self, mat_file, t_h=args['t_hist'], t_f=args['t_fut'], d_s=args['skip_factor'],
                 enc_size=args['encoder_size'], grid_size=args['grid_size'], n_lat=args['num_lat_classes'],
                 n_lon=args['num_lon_classes'], input_dim=args['input_dim'], polar=args['pooling'] == 'polar'):
        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.lat_int = scp.loadmat(mat_file)['lat_intention_masks']
        self.lon_int = scp.loadmat(mat_file)['lon_intention_masks']
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences
        self.enc_size = enc_size # size of encoder LSTM
        self.grid_size = grid_size # size of social context grid
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.polar = polar
        self.input_dim = input_dim

    def __len__(self):
        return len(self.D)


    def __getitem__(self, idx):
        # print('getitem is called ')
        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2]
        grid = self.D[idx,8:]
        neighbors = []

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist = self.getHistory(vehId,t,vehId,dsId)
        fut = self.getFuture(vehId,t,dsId)

        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid:
            neighbors.append(self.getHistory(i.astype(int), t, vehId, dsId))

        # Maneuvers 'lon_enc' = one-hot vector, 'lat_enc = one-hot vector
        lon_enc = np.zeros([self.n_lon])
        lon_enc[int(self.lon_int[idx] - 1)] = 1
        # lon_enc[int(self.D[idx, 7] - 1)] = 1

        lat_enc = np.zeros([self.n_lat])
        lat_enc[int(self.lat_int[idx] - 1)] = 1
        # lat_enc[int(self.D[idx, 6] - 1)] = 1

        return hist, fut, neighbors, lat_enc, lon_enc, dsId, vehId, t

    ## Helper function to get track history
    def getHistory(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            inp_size = self.input_dim + 1
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:inp_size]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:inp_size] - refPos
                polar = self.polar
                if polar:
                    hist= self.cart2polar(hist)

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    ## Helper function to get track future
    def getFuture(self, vehId, t, dsId):
        inp_size = self.input_dim + 1
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:inp_size]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s, 1:inp_size] - refPos
        polar = self.polar
        if polar:
            fut = self.cart2polar(fut)

        return fut

    def cart2polar(self, car_traj):
        np.seterr(divide='ignore', invalid='ignore')

        #trajectory segment in polar coordinates
        #Distance: r
        r_traj = np.sqrt(np.square(car_traj[:, 0]) + np.square(car_traj[:, 1]))
        #Angle: phi
        phi_traj = np.arctan2(car_traj[:, 1], car_traj[:, 0])

        #fill the output polar_traj with r and phi
        polar_traj = np.zeros_like(car_traj)
        polar_traj[:, 0] = r_traj
        polar_traj[:, 1] = phi_traj

        #Trajectory Orientation w.r.t the initial position
        car_traj_rel = car_traj - car_traj[0, :]
        traj_orient = np.arctan2(car_traj_rel[:, 1], car_traj_rel[:, 0])
        theta_total = traj_orient + phi_traj

        #Check if theta is nearly 180 degrees
        theta_indx = abs(theta_total-np.pi) < 0.001
        #True? use linear velocity
        polar_traj[theta_indx, 2] = car_traj[theta_indx, 2]  # linear velocity
        #False? use angular velocity
        polar_traj[not(theta_indx), 2] = car_traj[not(theta_indx), 2] * np.sin(theta_total[not(theta_indx)]) / r_traj[not(theta_indx)]  # angular velocity
        nan_inf_indx = np.logical_or(np.isnan(polar_traj[:, 2]), np.isinf(polar_traj[:, 2]))
        polar_traj[nan_inf_indx, 2] = 0

        # if abs(theta_total-np.pi) < 0.001:
        #     polar_traj[:, 2] = car_traj[:, 2]  # linear velocity
        # else:
        #     np.seterr(divide='ignore', invalid='ignore')
        #     polar_traj[:, 2] = car_traj[:, 2] * np.sin(theta_total) / r_traj  # angular velocity
        #     nan_inf_indx = np.logical_or(np.isnan(polar_traj[:, 2]), np.isinf(polar_traj[:, 2]))
        #     polar_traj[nan_inf_indx, 2] = 0

        # traj_orient = self.traj_orientation(car_traj)
        # theta_total = traj_orient + phi_traj
        # polar_traj[:, 2] = car_traj[:, 2] * np.sin(theta_total) / r_traj #angular velocity
        # nan_inf_indx = np.logical_or(np.isnan(polar_traj[:, 2]), np.isinf(polar_traj[:, 2]))
        # polar_traj[nan_inf_indx, 2] = 0
        #
        # polar_traj[:, 2] = car_traj[:, 2]  # linear velocity

        return  polar_traj

    def traj_orientation(self, car_traj):
        trj_len = car_traj.shape[0]
        mid_pnt = int(trj_len/2)
        mid_traj =  car_traj[mid_pnt,:]-car_traj[0,:]
        mid_orient = np.arctan2(mid_traj[1], mid_traj[0])
        end_traj = car_traj[-1,:]-car_traj[mid_pnt,:]
        end_orient = np.arctan2(end_traj[1], end_traj[0])
        trj_orient = np.zeros_like(car_traj[:,0])
        trj_orient[0:mid_pnt] = mid_orient
        trj_orient[mid_pnt:trj_len] = end_orient
        return trj_orient


    ## Collate function for dataloader
    def collate_fn(self, samples):
        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        for _, _, nbrs, _, _, _, _, _ in samples:
            nbr_batch_size += sum([len(nbrs[i]) != 0 for i in range(len(nbrs))])
        maxlen = self.t_h // self.d_s + 1
        nbrs_batch = torch.zeros(maxlen, nbr_batch_size, self.input_dim)

        # Initialize social mask batch:
        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0], self.enc_size)
        mask_batch = mask_batch.byte()

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen, len(samples), self.input_dim)
        fut_batch = torch.zeros(self.t_f // self.d_s, len(samples), self.input_dim)
        op_mask_batch = torch.zeros(self.t_f // self.d_s, len(samples), self.input_dim)
        lat_enc_batch = torch.zeros(len(samples), self.n_lat)
        lon_enc_batch = torch.zeros(len(samples), self.n_lon)
        ds_ids_batch = torch.zeros(len(samples), 1)
        vehicle_ids_batch = torch.zeros(len(samples), 1)
        frame_ids_batch = torch.zeros(len(samples), 1)

        count = 0
        for sampleId, (hist, fut, nbrs, lat_enc, lon_enc, ds_ids, vehicle_ids, frame_ids) in enumerate(samples):
            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            for k in range(self.input_dim):
                hist_batch[0:len(hist), sampleId, k] = torch.from_numpy(hist[:, k])
                fut_batch[0:len(fut), sampleId, k] = torch.from_numpy(fut[:, k])

            op_mask_batch[0:len(fut), sampleId, :] = 1
            lat_enc_batch[sampleId, :] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)
            ds_ids_batch[sampleId, :] = torch.tensor(ds_ids.astype(np.float64))
            vehicle_ids_batch[sampleId, :] = torch.tensor(vehicle_ids.astype(np.float64))
            frame_ids_batch[sampleId, :] = torch.tensor(frame_ids.astype(int).astype(np.float64))

            # Set up neighbor, neighbor sequence length, and mask batches:
            for id, nbr in enumerate(nbrs):
                if len(nbr) != 0:
                    for k in range(self.input_dim):
                        nbrs_batch[0:len(nbr), count, k] = torch.from_numpy(nbr[:, k])
                    pos[0] = id % self.grid_size[0]
                    pos[1] = id // self.grid_size[0]
                    mask_batch[sampleId, pos[1], pos[0], :] = torch.ones(self.enc_size).byte()
                    count += 1

        return hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, \
               op_mask_batch, ds_ids_batch, vehicle_ids_batch, frame_ids_batch

#________________________________________________________________________________________________________________________________________

## Custom activation for output layer (Graves, 2015)
def outputActivation(x):
    if x.shape[2] == 5:
        muX = x[:,:,0:1]
        muY = x[:,:,1:2]
        sigX = x[:,:,2:3]
        sigY = x[:,:,3:4]
        rho = x[:,:,4:5]
        sigX = torch.exp(sigX)
        sigY = torch.exp(sigY)
        rho = torch.tanh(rho)
        out = torch.cat([muX, muY, sigX, sigY, rho],dim=2)

    elif x.shape[2] == 7:
        muX = x[:, :, 0:1]
        muY = x[:, :, 1:2]
        muTh = x[:, :, 2:3]
        sigX = x[:, :, 3:4]
        sigY = x[:, :, 4:5]
        sigTh = x[:, :, 5:6]
        rho = x[:, :, 6:7]
        sigX = torch.exp(sigX)
        sigY = torch.exp(sigY)
        sigTh = torch.exp(sigTh)
        rho = torch.tanh(rho) #  0.4 * 0.4 sclaing to avoid NaN when computing the loss
        out = torch.cat([muX, muY, muTh, sigX, sigY, sigTh, rho], dim=2)

    return out


# Compute the NLL using the formula of Multivariate Gaussian distribution
#In matrix form
def compute_nll_mat_red(y_pred, y_gt):
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    muTh = y_pred[:, :, 2]
    sigX = y_pred[:, :, 3]
    sigY = y_pred[:, :, 4]
    sigTh = y_pred[:, :, 5]
    rho = y_pred[:, :, 6]

    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    th = y_gt[:, :, 2]

    # XU = ([x - muX, y - muY, th - muTh])
    # XU = torch.cat((x - muX, y - muY, th - muTh),0)
    XU = torch.zeros(x.shape[0], x.shape[1], 3, 1)
    XU[:, :, 0, 0] = x - muX
    XU[:, :, 1, 0] = y - muY
    XU[:, :, 2, 0] = th - muTh

    #sigma
    sigma_mat = torch.zeros(x.shape[0], x.shape[1], 3, 3)
    sigma_mat[:, :, 0, 0] = torch.pow(sigX, 2)
    sigma_mat[:, :, 1, 0] = rho * sigX * sigY
    sigma_mat[:, :, 2, 0] = rho * sigX * sigTh

    sigma_mat[:, :, 0, 1] = rho * sigX * sigY
    sigma_mat[:, :, 1, 1] = torch.pow(sigY, 2)
    sigma_mat[:, :, 2, 1] = rho * sigY * sigTh

    sigma_mat[:, :, 0, 2] = rho * sigX * sigTh
    sigma_mat[:, :, 1, 2] = rho * sigY * sigTh
    sigma_mat[:, :, 2, 2] = torch.pow(sigTh, 2)

    loss_1 = 0.5 * torch.matmul(torch.matmul(XU.transpose(2, 3), sigma_mat.inverse()), XU)
    loss_1 = loss_1.view(x.shape[0], x.shape[1])



    nll_loss = loss_1 + 2.7568 + 0.5*torch.log(sigma_mat.det())

    # if use_reg:
    #     # rho_reg_term = 1 - 3 * torch.pow(rho, 2) + 2 * torch.pow(rho, 3)
    #     rho_reg_term = 3 * torch.pow(rho, 2) - 2 * torch.pow(rho, 3)
    #     # nll_loss = nll_loss + torch.pow(rho_reg_term.cpu(),2)
    #     nll_loss = nll_loss + torch.abs(rho_reg_term.cpu())

    return nll_loss


## Batchwise NLL loss, uses mask for variable output lengths
def maskedNLL(y_pred, y_gt, mask):
    input_dim = y_pred.shape[2]
    if input_dim == 5:
        acc = torch.zeros_like(mask)
        muX = y_pred[:,:,0]
        muY = y_pred[:,:,1]
        sigX = y_pred[:,:,2]
        sigY = y_pred[:,:,3]
        rho = y_pred[:,:,4]
        ohr = torch.pow(1-torch.pow(rho,2),-0.5)
        x = y_gt[:,:, 0]
        y = y_gt[:,:, 1]
        # If we represent likelihood in feet^(-1):
        out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2*rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379
        # If we represent likelihood in m^(-1):
        # out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
        acc[:,:,0] = out
        acc[:,:,1] = out
        acc = acc*mask
        lossVal = torch.sum(acc)/torch.sum(mask)

    elif input_dim == 7:
        # FInd the NLL
        nll = compute_nll_mat_red(y_pred, y_gt)

        # nll_loss tensor filled with the loss value
        nll_loss = torch.zeros_like(mask)
        nll_loss[:, :, 0] = nll
        nll_loss[:, :, 1] = nll
        nll_loss[:, :, 2] = nll

        # mask the loss and find the mean value
        nll_loss = nll_loss * mask
        lossVal = torch.sum(nll_loss) / torch.sum(mask)

    return lossVal

## NLL for sequence, outputs sequence of NLL values for each time-step, uses mask for variable output lengths, used for evaluation
def maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=3, num_lon_classes = 3,use_maneuvers = True, avg_along_time = False):
    if use_maneuvers:
        acc = torch.zeros(op_mask.shape[0],op_mask.shape[1],num_lon_classes*num_lat_classes).cuda()
        # acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], num_lon_classes * num_lat_classes)

        count = 0
        for k in range(num_lon_classes):
            for l in range(num_lat_classes):
                wts = lat_pred[:,l]*lon_pred[:,k]
                wts = wts.repeat(len(fut_pred[0]),1)
                y_pred = fut_pred[k*num_lat_classes + l]
                y_gt = fut

                output_dim = y_pred.shape[2]

                if output_dim==5:
                    muX = y_pred[:, :, 0]
                    muY = y_pred[:, :, 1]
                    sigX = y_pred[:, :, 2]
                    sigY = y_pred[:, :, 3]
                    rho = y_pred[:, :, 4]
                    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
                    x = y_gt[:, :, 0]
                    y = y_gt[:, :, 1]
                    # If we represent likelihood in feet^(-1):
                    out = -(0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + 0.5*torch.pow(sigY, 2)*torch.pow(y-muY, 2) - rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379)
                elif output_dim == 7:
                    out = compute_nll_mat_red(y_pred, y_gt)

                # If we represent likelihood in m^(-1):
                # out = -(0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160)
                acc[:, :, count] = out + torch.log(wts.cpu())
                count+=1
        acc = -logsumexp(acc, dim = 2)
        acc = acc * op_mask[:,:,0]
        if avg_along_time:
            lossVal = torch.sum(acc) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc,dim=1)
            counts = torch.sum(op_mask[:,:,0],dim=1)
            return lossVal,counts
    else:
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1).cuda()
        y_pred = fut_pred
        y_gt = fut
        output_dim = y_pred.shape[2]
        if output_dim == 5:
            muX = y_pred[:, :, 0]
            muY = y_pred[:, :, 1]
            sigX = y_pred[:, :, 2]
            sigY = y_pred[:, :, 3]
            rho = y_pred[:, :, 4]
            ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
            x = y_gt[:, :, 0]
            y = y_gt[:, :, 1]
            # If we represent likelihood in feet^(-1):
            out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2 * rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379
            # If we represent likelihood in m^(-1):
            # out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
        elif output_dim == 7:
            out = compute_nll_mat_red(y_pred, y_gt)

        acc[:, :, 0] = out
        acc = acc * op_mask[:, :, 0:1]
        if avg_along_time:
            lossVal = torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc[:,:,0], dim=1)
            counts = torch.sum(op_mask[:, :, 0], dim=1)
            return lossVal,counts

## Batchwise MSE loss, uses mask for variable output lengths
def maskedMSE(y_pred, y_gt, mask):
    ip_dim = y_gt.shape[2]
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)

    if ip_dim==3:
        muVel = y_pred[:,:,2]
        Vel = y_gt[:,:, 2]
        out = out + torch.pow(Vel-muVel, 2)

    for k in range(ip_dim):
        acc[:, :, k] = out

    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal

## MSE loss for complete sequence, outputs a sequence of MSE values, uses mask for variable output lengths, used for evaluation
def maskedMSETest(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc[:,:,0],dim=1)
    counts = torch.sum(mask[:,:,0],dim=1)
    return lossVal, counts

## Helper function for log sum exp calculation:
def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs
