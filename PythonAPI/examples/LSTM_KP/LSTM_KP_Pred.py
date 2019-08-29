import os, pdb
import random
import numpy as np
from PIL import Image
random.seed(1)
np.random.seed(1)

import torch
torch.manual_seed(1)
from torch import nn
from torch import optim

from DataLoader import DataLoader, compute_locations, column, make_dir, visualize_output_KP_gif, visualize_locations_gif
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.switch_backend('agg')

class LocationLSTM(nn.ModuleList):
    def __init__(self, state_dim=3, hidden_dim=256):
        super(LocationLSTM, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.lstm1 = nn.LSTMCell(input_size=state_dim, hidden_size=hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=state_dim)

    def init_hidden(self):
        return (torch.zeros(1, self.hidden_dim),
                torch.zeros(1, self.hidden_dim))

    def forward(self, x, y, infer=False):
        input_seq, output_seq, gt_seq = [], [], []
        x_batch, output_batch, gt_batch = [], [], []
        for batch_idx in range(len(x)):
            cur_output = []
            cur_seq = x[batch_idx]
            cur_gt = y[batch_idx]
            h_t, c_t = self.init_hidden()
            h_t2, c_t2 = self.init_hidden()
            unsqueezed_seq = cur_seq.unsqueeze(1)
            for input_t_idx in range(len(unsqueezed_seq)):
                input_t = cur_seq[input_t_idx, :self.state_dim].unsqueeze(0)
                h_t, c_t = self.lstm1(input_t, (h_t, c_t))
                h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                output = self.dropout(self.fc(h_t2))
                if input_t_idx < len(unsqueezed_seq) - 1:
                    output_seq.append(output)
                    gt_seq.append(cur_seq[input_t_idx + 1, :self.state_dim])
            for output_t in cur_gt:
                output_t = output_t[:self.state_dim]
                h_t, c_t = self.lstm1(output, (h_t, c_t))
                h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                output = self.dropout(self.fc(h_t2))
                output_seq.append(output)
                gt_seq.append(output_t)
                cur_output.append(output)

            if infer:
                x_batch.append([cur_seq[i].detach().numpy() for i in range(len(cur_seq))])
                output_batch.append([cur_output[i].detach().numpy() for i in range(len(cur_output))])
                gt_batch.append([cur_gt[i].detach().numpy() for i in range(len(cur_gt))])

        tensor_output = torch.empty((len(output_seq), self.state_dim))
        tensor_gt = torch.empty((len(output_seq), self.state_dim))
        for idx in range(len(output_seq)):
            tensor_output[idx] = output_seq[idx]
            tensor_gt[idx] = gt_seq[idx]

        if infer:
            return tensor_output.view((len(output_seq), self.state_dim)), tensor_gt.view((len(gt_seq), self.state_dim)), x_batch, output_batch, gt_batch
        else:
            return tensor_output.view((len(output_seq), self.state_dim)), tensor_gt.view((len(gt_seq), self.state_dim))

def train(dataset, num_epochs=500, observation_length=8, prediction_length=12, batch_size=16, save_path='models/locs_depth/', real_depth=False, use_kp=False):
    save_path = save_path + dataset + '/' + str(real_depth) + '/' + str(use_kp) + '/'
    make_dir(save_path)

    data_loader = DataLoader(dataset, batch_size, observation_length, real_depth=real_depth, is_keypoints=use_kp)
    net = LocationLSTM(state_dim=data_loader.state_dim)
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

def test(dataset, observation_length=8, prediction_length=12, batch_size=32, inference_iters=30, save_path='models/locs_depth/', real_depth=False, use_kp=False):
    save_path = save_path + dataset + '/' + str(real_depth) + '/' + str(use_kp) + '/'
    make_dir(save_path)
    data_loader = DataLoader(dataset, batch_size, observation_length=observation_length, max_depth=50, real_depth=real_depth, is_keypoints=use_kp)
    net = LocationLSTM(state_dim=data_loader.state_dim)
    criterion = nn.MSELoss()

    checkpoint_path = os.path.join(save_path, dataset, 'best_model.tar')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['state_dict'])

    te_lstm_loss, te_fv_loss, te_fa_loss = [], [], []
    x, y, y_hat_lstm, y_hat_fv, y_hat_fa, all_origins, all_meta = [], [], [], [], [], [], []
    for _ in range(inference_iters):
        x_te, y_te, origin, meta = data_loader.next_batch(split_type=2, pred_length=prediction_length)

        y_te_output, y_te_gt, cur_x, cur_y_hat_lstm, cur_y = net(x_te, y_te, infer=True)
        lstm_loss = criterion(y_te_output, y_te_gt)

        y_te_output, y_te_gt, _, cur_y_hat_fv, _ = fixed_prediction(x_te, y_te, True)
        fv_loss = criterion(y_te_output, y_te_gt)

        y_te_output, y_te_gt, _, cur_y_hat_fa, _ = fixed_prediction(x_te, y_te, False)
        fa_loss = criterion(y_te_output, y_te_gt)

        x.extend(cur_x)
        y.extend(cur_y)
        all_origins.extend(origin)
        all_meta.extend(meta)

        y_hat_lstm.extend(cur_y_hat_lstm)
        te_lstm_loss.append(lstm_loss.item())

        y_hat_fv.extend(cur_y_hat_fv)
        te_fv_loss.append(fv_loss.item())

        y_hat_fa.extend(cur_y_hat_fa)
        te_fa_loss.append(fa_loss.item())

    lstm_mean_loss, fv_mean_loss, fa_mean_loss = sum(te_lstm_loss)/float(len(te_lstm_loss)), sum(te_fv_loss)/float(len(te_fv_loss)), sum(te_fa_loss)/float(len(te_fa_loss))
    print(lstm_mean_loss, fv_mean_loss, fa_mean_loss)

    combined = list(zip(x, y, y_hat_lstm, y_hat_fv, y_hat_fa, all_origins, all_meta))
    random.shuffle(combined)
    x[:], y[:],  y_hat_lstm[:], y_hat_fv[:], y_hat_fa[:], all_origins[:], all_meta[:] = zip(*combined)
    x, y, y_hat_lstm, y_hat_fv, y_hat_fa = np.array(x).squeeze(), np.array(y).squeeze(), np.array(y_hat_lstm).squeeze(), np.array(y_hat_fv).squeeze(), np.array(y_hat_fa).squeeze()
    return x, y, y_hat_lstm, y_hat_fv, y_hat_fa, all_origins, all_meta


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer', default=False)
    parser.add_argument('--use_kp', default=False)
    parser.add_argument('--ds', default='carla_cars')
    parser.add_argument('--real_depth', default=False)
    parser.add_argument('--num_vis', default=5, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    args = parser.parse_args()
    print args

    if not args.infer:
        train(args.ds, num_epochs=args.num_epochs, batch_size=args.batch_size, prediction_length=12, real_depth=args.real_depth, use_kp=args.use_kp)
        print('*' * 30)
    x, y, y_hat_lstm, y_hat_fv, y_hat_fa, all_origins, all_meta = test(args.ds, prediction_length=12, real_depth=args.real_depth, use_kp=args.use_kp)
    save_path = 'trajectories/' + args.ds + '/' + str(args.real_depth) + '/' + str(args.use_kp) + '/'
    make_dir(save_path)
    for vis_idx in range(args.num_vis):
        a, b, c, d, e = x[vis_idx], y[vis_idx], y_hat_lstm[vis_idx], y_hat_fv[vis_idx], y_hat_fa[vis_idx]
        map_num, camera_id, frame_num = all_meta[vis_idx]

        _, b = compute_locations(a, b, origin=all_origins[vis_idx])
        _, c = compute_locations(a, c, origin=all_origins[vis_idx])
        _, d = compute_locations(a, d, origin=all_origins[vis_idx])
        a, e = compute_locations(a, e, origin=all_origins[vis_idx])

        if not args.use_kp:
            visualize_locations_gif(a, b, c, d, e, all_meta, args.ds, vis_idx, save_path)
        else:
            visualize_output_KP_gif(a, b, c, d, e, all_meta, args.ds, vis_idx, save_path)
