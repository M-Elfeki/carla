import os
import pdb
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

random.seed(1)
np.random.seed(1)

import torch
torch.manual_seed(1)
from torch import nn
from torch import optim
from torch.autograd import Variable

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from DataLoader import DataLoader, compute_locations, column, make_dir, visualize_output_KP_gif, visualize_locations_gif

class LSTM_Pred(nn.ModuleList):
    def __init__(self, state_dim=3, hidden_dim=256, type=1):
        """
        1: Independent Prediction
        2: Share hidden states
        3: Pass h_locs as input to LSTM_KP
        4: Pass output_locs as input to LSTM KP
        """
        super(LSTM_Pred, self).__init__()

        self.type = type
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        if self.type == 1 or self.type == 2:
            input_size_kp = state_dim
        elif self.type == 3:
            input_size_kp = state_dim + hidden_dim
        else: # if self.type == 4:
            input_size_kp = state_dim + 3


        self.lstm1_kp = nn.LSTMCell(input_size=input_size_kp, hidden_size=hidden_dim)
        self.lstm2_kp = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.fc_kp = nn.Linear(in_features=hidden_dim, out_features=state_dim)

        self.lstm1_locs = nn.LSTMCell(input_size=3, hidden_size=hidden_dim)
        self.lstm2_locs = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.fc_locs = nn.Linear(in_features=hidden_dim, out_features=3)

        self.dropout = nn.Dropout(p=0.5)

    def init_hidden(self):
        return (torch.zeros(1, self.hidden_dim),
                torch.zeros(1, self.hidden_dim))

    def forward(self, x, y, infer=False):
        input_seq, output_seq, gt_seq = [], [], []
        x_batch, output_batch, gt_batch = [], [], []
        for batch_idx in range(len(x)):
            cur_output = []
            cur_seq, cur_gt = x[batch_idx], y[batch_idx]
            centers_x, kp_x = cur_seq[:, :3], cur_seq[:, 3:]
            centers_y, kp_y = cur_gt[:, :3], cur_gt[:, 3:]

            h_t_locs, c_t_locs = self.init_hidden()
            h_t2_locs, c_t2_locs = self.init_hidden()

            if self.type != 2:
                h_t_kp, c_t_kp = self.init_hidden()
                h_t2_kp, c_t2_kp = self.init_hidden()

            for input_t_idx in range(len(cur_seq)):
                input_centers_t = centers_x[input_t_idx].unsqueeze(0)
                input_kp_t = kp_x[input_t_idx].unsqueeze(0)

                h_t_locs, c_t_locs = self.lstm1_locs(input_centers_t, (h_t_locs, c_t_locs))
                h_t2_locs, c_t2_locs = self.lstm2_locs(h_t_locs, (h_t2_locs, c_t2_locs))
                output_centers = self.dropout(self.fc_locs(h_t2_locs))

                if self.type == 2:
                    h_t_locs, c_t_locs = self.lstm1_kp(input_kp_t, (h_t_locs, c_t_locs))
                    h_t2_locs, c_t2_locs = self.lstm2_kp(h_t_locs, (h_t2_locs, c_t2_locs))
                    output_kp = self.dropout(self.fc_kp(h_t2_locs))
                else:
                    if self.type == 1:
                        kp_input = input_kp_t
                    elif self.type == 3:
                        kp_input = torch.cat([h_t2_locs, input_kp_t], 1)
                    else: # if self.type == 4
                        kp_input = torch.cat([output_centers, input_kp_t], 1)

                    h_t_kp, c_t_kp = self.lstm1_kp(kp_input, (h_t_kp, c_t_kp))
                    h_t2_kp, c_t2_kp = self.lstm2_kp(h_t_kp, (h_t2_kp, c_t2_kp))
                    output_kp = self.dropout(self.fc_kp(h_t2_kp))

                if input_t_idx < len(cur_seq) - 1:
                    output = torch.cat([output_centers, output_kp], 1)
                    output_seq.append(output)
                    gt_seq.append(cur_seq[input_t_idx + 1])
            for output_t_idx in range(len(cur_gt)):
                output_t = cur_gt[output_t_idx]
                input_centers_t = centers_y[output_t_idx].unsqueeze(0)
                input_kp_t = kp_y[output_t_idx].unsqueeze(0)

                h_t_locs, c_t_locs = self.lstm1_locs(input_centers_t, (h_t_locs, c_t_locs))
                h_t2_locs, c_t2_locs = self.lstm2_locs(h_t_locs, (h_t2_locs, c_t2_locs))
                output_centers = self.dropout(self.fc_locs(h_t2_locs))

                if self.type == 2:
                    h_t_locs, c_t_locs = self.lstm1_kp(input_kp_t, (h_t_locs, c_t_locs))
                    h_t2_locs, c_t2_locs = self.lstm2_kp(h_t_locs, (h_t2_locs, c_t2_locs))
                    output_kp = self.dropout(self.fc_kp(h_t2_locs))
                else:
                    if self.type == 1:
                        kp_input = input_kp_t
                    elif self.type == 3:
                        kp_input = torch.cat([h_t2_locs, input_kp_t], 1)
                    else: # if self.type == 4
                        kp_input = torch.cat([output_centers, input_kp_t], 1)

                    h_t_kp, c_t_kp = self.lstm1_kp(kp_input, (h_t_kp, c_t_kp))
                    h_t2_kp, c_t2_kp = self.lstm2_kp(h_t_kp, (h_t2_kp, c_t2_kp))
                    output_kp = self.dropout(self.fc_kp(h_t2_kp))

                output = torch.cat([output_centers, output_kp], 1)
                # pdb.set_trace()

                output_seq.append(output)
                gt_seq.append(output_t)
                cur_output.append(output)

            if infer:
                x_batch.append([cur_seq[i].detach().numpy() for i in range(len(cur_seq))])
                output_batch.append([cur_output[i].detach().numpy() for i in range(len(cur_output))])
                gt_batch.append([cur_gt[i].detach().numpy() for i in range(len(cur_gt))])

        tensor_output = torch.empty((len(output_seq), self.state_dim+3))
        tensor_gt = torch.empty((len(output_seq), self.state_dim+3))
        for idx in range(len(output_seq)):
            tensor_output[idx] = output_seq[idx]
            tensor_gt[idx] = gt_seq[idx]

        if infer:
            return tensor_output.view((len(output_seq), self.state_dim+3)), tensor_gt.view((len(gt_seq), self.state_dim+3)), x_batch, output_batch, gt_batch
        else:
            return tensor_output.view((len(output_seq), self.state_dim+3)), tensor_gt.view((len(gt_seq), self.state_dim+3))

def fixed_velocity_model(prev_locations):
    if len(prev_locations) == 0:
        return [0, 0]
    elif len(prev_locations) == 1:
        return [prev_locations[0][0], prev_locations[0][1], prev_locations[0][2]]

    mean_x = 0.0
    mean_y = 0.0
    mean_z = 0.0
    for i in range(1, len(prev_locations)):
        mean_x += (prev_locations[i][0] - prev_locations[i - 1][0])
        mean_y += (prev_locations[i][1] - prev_locations[i - 1][1])
        mean_z += (prev_locations[i][2] - prev_locations[i - 1][2])

    return [prev_locations[0][0] - float(mean_x / (len(prev_locations) - 1)),
            prev_locations[0][1] - float(mean_y / (len(prev_locations) - 1)),
            prev_locations[0][2] - float(mean_z / (len(prev_locations) - 1))]

def fixed_acceleration_model(prev_locations):
    if len(prev_locations) == 0:
        return [0, 0]
    elif len(prev_locations) == 1:
        return [prev_locations[0][0], prev_locations[0][1], prev_locations[0][2]]
    elif len(prev_locations) == 2:
        return np.array([prev_locations[0][0] - (prev_locations[1][0] - prev_locations[0][0]),\
                    prev_locations[0][1] - (prev_locations[1][1] - prev_locations[0][1]),\
                    prev_locations[0][2] - (prev_locations[1][2] - prev_locations[0][2])])

    mean_x = []
    mean_y = []
    mean_z = []
    for i in range(1, len(prev_locations)):
        mean_x.append((prev_locations[i][0] - prev_locations[i - 1][0]))
        mean_y.append((prev_locations[i][1] - prev_locations[i - 1][1]))
        mean_z.append((prev_locations[i][2] - prev_locations[i - 1][2]))
    mean_mean_x = []
    mean_mean_y = []
    mean_mean_z = []
    for i in range(1, len(mean_x)):
        mean_mean_x.append((mean_x[i] - mean_x[i-1]))
        mean_mean_y.append((mean_y[i] - mean_y[i-1]))
        mean_mean_z.append((mean_z[i] - mean_z[i-1]))
    avg_acceleration_x = sum(mean_mean_x) / float(len(mean_mean_x))
    avg_acceleration_y = sum(mean_mean_y) / float(len(mean_mean_y))
    avg_acceleration_z = sum(mean_mean_z) / float(len(mean_mean_z))

    dist_x = mean_x[0] - avg_acceleration_x
    dist_y = mean_y[0] - avg_acceleration_y
    dist_z = mean_z[0] - avg_acceleration_z
    return np.array([prev_locations[0][0] - dist_x,
            prev_locations[0][1] - dist_y,
            prev_locations[0][2] - dist_z])

def fixed_prediction(x, y, fixed_velocity=True):
    input_seq, output_seq, gt_seq = [], [], []
    x_batch, output_batch, gt_batch = [], [], []
    for batch_idx in range(len(x)):
        cur_output = []
        cur_seq = x[batch_idx]
        cur_gt = y[batch_idx]
        used_seq = []
        for input_t_idx in range(len(cur_seq)):
            used_seq.append(cur_seq[input_t_idx].detach().numpy())
            input_t = np.array(used_seq[::-1])
            output = np.zeros(len(input_t[0]))
            for dim in range(0, len(input_t[0]), 3):
                input_t_d = input_t[:, [dim, dim+1, dim+2]]
                if fixed_velocity == True:
                    output_d = fixed_velocity_model(input_t_d)
                else:
                    output_d = fixed_acceleration_model(input_t_d)
                output[dim:dim+3] = output_d
            if input_t_idx < len(cur_seq) - 1:
                output_seq.append(output)
                gt_seq.append(cur_seq[input_t_idx + 1])
        for output_t_idx in range(len(cur_gt)):
            input_t = np.array(used_seq[::-1])
            output = np.zeros(len(input_t[0]))
            for dim in range(0, len(input_t[0]), 3):
                input_t_d = input_t[:, [dim, dim+1, dim+2]]
                if fixed_velocity == True:
                    output_d = fixed_velocity_model(input_t_d)
                else:
                    output_d = fixed_acceleration_model(input_t_d)
                output[dim:dim+3] = output_d
            output_t = cur_gt[output_t_idx]
            output_seq.append(output)
            gt_seq.append(output_t)
            cur_output.append(output)
            used_seq.append(output)

        x_batch.append([cur_seq[i] for i in range(len(cur_seq))])
        output_batch.append([cur_output[i] for i in range(len(cur_output))])
        gt_batch.append([cur_gt[i] for i in range(len(cur_gt))])

    tensor_output = torch.empty((len(output_seq), len(output_seq[0])))
    tensor_gt = torch.empty((len(output_seq), len(output_seq[0])))
    for idx in range(len(output_seq)):
        tensor_output[idx] = Variable(torch.from_numpy(np.array(output_seq[idx])))
        tensor_gt[idx] = gt_seq[idx]

    return tensor_output.view((len(output_seq), len(output_seq[0]))), tensor_gt.view((len(gt_seq), len(output_seq[0]))), x_batch, output_batch, gt_batch

def train(dataset, num_epochs=500, observation_length=8, prediction_length=12, batch_size=16, save_path='models/locs_depth/', real_depth=False, lstm_type=1):
    save_path = save_path + dataset + '/' + str(real_depth) + '/' + str(lstm_type) + '/'
    make_dir(save_path)

    data_loader = DataLoader(dataset, batch_size, observation_length, real_depth=real_depth)
    net = LSTM_Pred(state_dim=data_loader.state_dim, type=lstm_type)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    tr_loss, vl_loss = [], []
    min_vl_loss = 1e10
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        x_train, y_train, _, _ = data_loader.next_batch(pred_length=prediction_length)
        y_train_output, y_train_gt = net(x_train, y_train)
        loss = criterion(y_train_output, y_train_gt) * 10000
        tr_loss.append(loss.item()/len(y_train_output))
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x_val, y_val, _, _ = data_loader.next_batch(split_type=1, pred_length=prediction_length)
            y_val_output, y_val_gt = net(x_val, y_val)
            loss = criterion(y_val_output, y_val_gt)
            vl_loss.append(loss.item()/len(y_val_output))

        print(epoch, tr_loss[-1], vl_loss[-1])

        if vl_loss[-1] < min_vl_loss:
            torch.save({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(save_path, str(epoch)+'.tar'))
            torch.save({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(save_path, 'best_model.tar'))
            min_vl_loss = vl_loss[-1]
    return tr_loss, vl_loss

def test(dataset, observation_length=8, prediction_length=12, batch_size=32, inference_iters=30, save_path='models/locs_depth/', real_depth=False, lstm_type=1):
    save_path = save_path + dataset + '/' + str(real_depth) + '/' + str(lstm_type) + '/'
    make_dir(save_path)
    data_loader = DataLoader(dataset, batch_size, observation_length=observation_length, max_depth=50, real_depth=real_depth)
    net = LSTM_Pred(state_dim=data_loader.state_dim, type=lstm_type)
    criterion = nn.MSELoss()

    checkpoint_path = os.path.join(save_path, dataset, 'best_model.tar')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['state_dict'])

    te_lstm_loss_locs, te_fv_loss_locs, te_fa_loss_locs = [], [], []
    te_lstm_loss_kp, te_fv_loss_kp, te_fa_loss_kp = [], [], []
    x, y, y_hat_lstm, y_hat_fv, y_hat_fa, all_origins, all_meta = [], [], [], [], [], [], []
    for _ in range(inference_iters):
        x_te, y_te, origin, meta = data_loader.next_batch(split_type=2, pred_length=prediction_length)

        y_te_output, y_te_gt, cur_x, cur_y_hat_lstm, cur_y = net(x_te, y_te, infer=True)
        centers_y_output, kp_y_output = y_te_output[:, :3], y_te_output[:, 3:]
        centers_y_gt, kp_y_gt = y_te_gt[:, :3], y_te_gt[:, 3:]
        lstm_loss_centers = criterion(centers_y_output, centers_y_gt)
        lstm_loss_kp = criterion(kp_y_output, kp_y_gt)

        y_te_output, y_te_gt, _, cur_y_hat_fv, _ = fixed_prediction(x_te, y_te, True)
        centers_y_output, kp_y_output = y_te_output[:, :3], y_te_output[:, 3:]
        centers_y_gt, kp_y_gt = y_te_gt[:, :3], y_te_gt[:, 3:]
        fv_loss_centers = criterion(centers_y_output, centers_y_gt)
        fv_loss_kp = criterion(kp_y_output, kp_y_gt)

        y_te_output, y_te_gt, _, cur_y_hat_fa, _ = fixed_prediction(x_te, y_te, False)
        centers_y_output, kp_y_output = y_te_output[:, :3], y_te_output[:, 3:]
        centers_y_gt, kp_y_gt = y_te_gt[:, :3], y_te_gt[:, 3:]
        fa_loss_centers = criterion(centers_y_output, centers_y_gt)
        fa_loss_kp = criterion(kp_y_output, kp_y_gt)

        x.extend(cur_x)
        y.extend(cur_y)
        all_origins.extend(origin)
        all_meta.extend(meta)

        y_hat_lstm.extend(cur_y_hat_lstm)
        te_lstm_loss_locs.append(lstm_loss_centers.item())
        te_lstm_loss_kp.append(lstm_loss_kp.item())

        y_hat_fv.extend(cur_y_hat_fv)
        te_fv_loss_locs.append(fv_loss_centers.item())
        te_fv_loss_kp.append(fv_loss_kp.item())

        y_hat_fa.extend(cur_y_hat_fa)
        te_fa_loss_locs.append(fa_loss_centers.item())
        te_fa_loss_kp.append(fa_loss_kp.item())

    lstm_mean_loss_locs, fv_mean_loss_locs, fa_mean_loss_locs = sum(te_lstm_loss_locs)/float(len(te_lstm_loss_locs)), sum(te_fv_loss_locs)/float(len(te_fv_loss_locs)), sum(te_fa_loss_locs)/float(len(te_fa_loss_locs))
    print('Center Prediction MSE: '),
    print(lstm_mean_loss_locs, fv_mean_loss_locs, fa_mean_loss_locs)
    lstm_mean_loss_kp, fv_mean_loss_kp, fa_mean_loss_kp = sum(te_lstm_loss_kp)/float(len(te_lstm_loss_kp)), sum(te_fv_loss_kp)/float(len(te_fv_loss_kp)), sum(te_fa_loss_kp)/float(len(te_fa_loss_kp))
    print('Keypoints Prediction MSE: '),
    print(lstm_mean_loss_kp, fv_mean_loss_kp, fa_mean_loss_kp)

    combined = list(zip(x, y, y_hat_lstm, y_hat_fv, y_hat_fa, all_origins, all_meta))
    random.shuffle(combined)
    x[:], y[:],  y_hat_lstm[:], y_hat_fv[:], y_hat_fa[:], all_origins[:], all_meta[:] = zip(*combined)
    x, y, y_hat_lstm, y_hat_fv, y_hat_fa = np.array(x).squeeze(), np.array(y).squeeze(), np.array(y_hat_lstm).squeeze(), np.array(y_hat_fv).squeeze(), np.array(y_hat_fa).squeeze()
    return x, y, y_hat_lstm, y_hat_fv, y_hat_fa, all_origins, all_meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer', default=False)
    parser.add_argument('--ds', default='carla_cars')
    parser.add_argument('--real_depth', default=False)
    parser.add_argument('--num_vis', default=5, type=int)
    # parser.add_argument('--lstm_type', default=1, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    args = parser.parse_args()

    for lstm_type in range(1, 5):
        print args, lstm_type
        if not args.infer:
            train(args.ds, num_epochs=args.num_epochs, batch_size=args.batch_size, prediction_length=12, real_depth=args.real_depth, lstm_type=lstm_type)
            print('*' * 30)
        x, y, y_hat_lstm, y_hat_fv, y_hat_fa, all_origins, all_meta = test(args.ds, prediction_length=12, real_depth=args.real_depth, lstm_type=lstm_type)
        save_path = 'trajectories/' + args.ds + '/' + str(args.real_depth) + '/' + str(lstm_type) + '/'
        make_dir(save_path)
        for vis_idx in tqdm(xrange(args.num_vis)):
            a, b, c, d, e = x[vis_idx], y[vis_idx], y_hat_lstm[vis_idx], y_hat_fv[vis_idx], y_hat_fa[vis_idx]
            map_num, camera_id, frame_num = all_meta[vis_idx]

            _, b = compute_locations(a, b, origin=all_origins[vis_idx])
            _, c = compute_locations(a, c, origin=all_origins[vis_idx])
            _, d = compute_locations(a, d, origin=all_origins[vis_idx])
            a, e = compute_locations(a, e, origin=all_origins[vis_idx])

            visualize_locations_gif(a[:, :3], b[:, :3], c[:, :3], d[:, :3], e[:, :3], all_meta, args.ds, vis_idx, save_path)
            visualize_output_KP_gif(a[:, 3:], b[:, 3:], c[:, 3:], d[:, 3:], e[:, 3:], all_meta, args.ds, vis_idx, save_path)

        print '#' * 30
        print '*' * 30
