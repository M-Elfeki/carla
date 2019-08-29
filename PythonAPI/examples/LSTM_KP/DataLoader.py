import os
import pdb
import random
import cPickle
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

random.seed(1)
np.random.seed(1)

import torch
torch.manual_seed(1)
from torch.autograd import Variable

VIEW_WIDTH, VIEW_HEIGHT = 1920//2, 1080//2

def column(matrix, i):
    return [row[i] for row in matrix]

class DataLoader():
    def __init__(self, dataset, batch_size=16, observation_length=8, max_depth=10000, real_depth=True,
                 is_keypoints=False, force_preprocess=False, datasets_path='/home/demo/Desktop/KP/Locs_KP/'):
        self.max_depth = max_depth
        self.real_depth = real_depth
        self.batch_size = batch_size
        self.is_keypoints = is_keypoints
        self.observation_length = observation_length

        train_path = os.path.join(datasets_path, 'train', dataset)
        train_data = [os.path.join(train_path, cur_file) for cur_file in os.listdir(train_path)]

        val_path = os.path.join(datasets_path, 'val', dataset)
        val_data = [os.path.join(val_path, cur_file) for cur_file in os.listdir(val_path)]

        test_path = os.path.join(datasets_path, 'test', dataset)
        test_data = [os.path.join(test_path, cur_file) for cur_file in os.listdir(test_path)]
        self.data = [train_data, val_data, test_data]
        if self.is_keypoints and dataset == 'carla_cars':
            self.state_dim = 15
        elif self.is_keypoints and dataset == 'carla_peds':
            self.state_dim = 36
        elif self.is_keypoints and dataset == 'carla_bicycles':
            self.state_dim = 45
        else:
            self.state_dim = 3

        if os.path.exists(datasets_path+dataset+'_trajectories.pickle') and not force_preprocess:
            with open(datasets_path+dataset+'_trajectories.pickle', 'r') as traj_file:
                self.processed_data = cPickle.load(traj_file)
        else:
            self.processed_data = [[], [], []]
            for split_type in range(3):
                for cur_file in self.data[split_type]:
                    cur_arr = self.frame_preprocess(cur_file)
                    if len(cur_arr) > 0:
                        self.processed_data[split_type].extend(cur_arr)
            with open(datasets_path+dataset+'_trajectories.pickle', 'wb') as traj_file:
                cPickle.dump(self.processed_data, traj_file)
        print('Loaded Trajectories')

    def frame_preprocess(self, data_file):

        df = np.array(pd.read_csv(data_file, delimiter=' ', header=None))
        df[df<0]=0
        frame_numbers, ids, tp, center_traj = column(df, 0), column(df, 1), column(df, 2), df[:, 3:6]
        map_num, camera_id = int(data_file.split('/')[-1].split('_')[1]), int(data_file.split('/')[-1].split('_')[-1].replace('.txt', ''))

        trajectories, id_list = [], []
        for idx, cur_id in enumerate(ids):
            if cur_id not in id_list:
                id_list.append(cur_id)
                trajectories.append([])
            cur_item = [int(frame_numbers[idx]), int(cur_id), map_num, camera_id, int(tp[idx]), np.array(center_traj[idx])]
            if self.real_depth:
                dp_img = np.array(Image.open('/home/demo/carla_sam_newest/carla/Output/' + str(map_num) + '_2/RGB/camera-' + str(camera_id) + '-image-' + str(frame_num) + '.png'), dtype=np.float32)
            traj, all_z = [], []
            for item_idx in range(6, len(df[0]), 3):
                if self.real_depth:
                    cur_z = float(dp_img[min(int(df[idx][item_idx+1]), VIEW_HEIGHT-1), min(int(df[idx][item_idx]), VIEW_WIDTH-1)])
                else:
                    cur_z = float(df[idx][item_idx+2])
                traj.append([float(df[idx][item_idx]), float(df[idx][item_idx+1]), cur_z])
                all_z.append(cur_z)
            if np.isnan(traj).any():
                return []
            cur_item.append(np.array(traj))
            if  (np.array(all_z)<=self.max_depth).all():
                trajectories[id_list.index(cur_id)].append(cur_item)
        return trajectories

    def compute_displacement(self, cur_seq):
        # with respect to first location
        q = np.copy(cur_seq)
        q = q.reshape(len(cur_seq), -1)
        for idx in range(len(q) - 1, 0, -1):
            for dim_idx in range(len(q[0])):
                q[idx][dim_idx] -= q[idx - 1][dim_idx]
        for dim_idx in range(len(q[0])):
            q[0][dim_idx] = 0.0
        return q

    def next_batch(self, split_type=0, pred_length=1):
        # 0 for training, 1 for validation, 2 for testing
        if self.processed_data[split_type] is None:
            self.processed_data[split_type] = []
            for cur_file in self.data[split_type]:
                cur_arr = self.frame_preprocess(cur_file)
                if len(cur_arr) > 0:
                    self.processed_data[split_type].extend(cur_arr)
        batch, all_batch, frame_batches = [], [], []
        for _ in range(100):
            cur_seq = []
            while len(cur_seq) < self.observation_length + pred_length:
                cur_seq = random.choice(self.processed_data[split_type])
            frame_nums, map_num, camera_id, center_traj, kp_traj = np.array(column(cur_seq, 0), dtype=np.int32), int(cur_seq[0][2]), int(cur_seq[0][3]), np.array(column(cur_seq, 5)), np.array(column(cur_seq, 6))
            if self.is_keypoints:
                batch.append(kp_traj)
            else:
                batch.append(center_traj)
            all_batch.append([map_num, camera_id])
            frame_batches.append(frame_nums)

        # Compute displacements for all possible predictions
        cur_x_batch, cur_y_batch, cur_origin_batch, cur_meta_info = [], [], [], []
        for idx_seq in range(len(batch)):
            cur_seq, frame_nums = batch[idx_seq], frame_batches[idx_seq]
            map_num, camera_id = all_batch[idx_seq]
            for idx in range(self.observation_length, len(cur_seq)-pred_length):
                s = self.compute_displacement(cur_seq[idx-self.observation_length:idx+pred_length])
                cur_origin_batch.append(cur_seq[idx-self.observation_length])
                cur_x_batch.append(Variable(torch.from_numpy(s[:self.observation_length]).float()))
                cur_y_batch.append(Variable(torch.from_numpy(s[self.observation_length:self.observation_length+pred_length]).float()))
                cur_meta_info.append([map_num, camera_id, frame_nums[idx-self.observation_length]])

        combined = list(zip(cur_x_batch, cur_y_batch, cur_origin_batch, cur_meta_info))
        random.shuffle(combined)
        cur_x_batch, cur_y_batch, cur_origin_batch, cur_meta_info = zip(*combined)

        cur_x_batch, cur_y_batch, cur_origin_batch, cur_meta_info = cur_x_batch[:self.batch_size], cur_y_batch[:self.batch_size], cur_origin_batch[:self.batch_size], cur_meta_info[:self.batch_size]
        return cur_x_batch, cur_y_batch, cur_origin_batch, cur_meta_info

def compute_locations(input_seq, output_seq, origin=None):
    if origin is None:
        origin = [0] * len(input_seq[0])
    else:
        origin = origin.reshape(-1)
    # Compute Absolute Location from displacements
    cur_seq = np.zeros((len(input_seq)+len(output_seq)+1, len(input_seq[0])))
    for idx in range(len(input_seq)):
        if idx == 0:
            for dim_idx in range(len(cur_seq[0])):
                cur_seq[0][dim_idx] = origin[dim_idx]
        else:
            cur_seq[idx] = np.copy(input_seq[idx])
            for dim_idx in range(len(cur_seq[0])):
                cur_seq[idx][dim_idx] += cur_seq[idx - 1][dim_idx]
    cur_seq[len(input_seq)] = cur_seq[len(input_seq) - 1]   # For visualization (to connect observed with predicted points in the same line)
    for idx in range(len(input_seq) + 1, len(input_seq) + len(output_seq) + 1):
        cur_seq[idx] = np.copy(output_seq[idx - len(input_seq) - 1])
        for dim_idx in range(len(cur_seq[0])):
            cur_seq[idx][dim_idx] += cur_seq[idx - 1][dim_idx]
    return cur_seq[:len(input_seq)], cur_seq[len(input_seq):]

def column(matrix, i):
    return [row[i] for row in matrix]

def make_dir(save_path):
    if not os.path.exists(os.path.join(save_path)):
     os.makedirs(os.path.join(save_path))

def visualize_output_KP_gif(a, b, c, d, e, meta, ds, vis_idx, save_path, vis_margin=30):

    a = a.reshape(len(a), -1, 3)
    b = b.reshape(len(b), -1, 3)
    c = c.reshape(len(c), -1, 3)
    d = d.reshape(len(d), -1, 3)
    e = e.reshape(len(e), -1, 3)
    map_num, camera_id, frame_num = meta[vis_idx]
    min_x, min_y, min_z, max_x, max_y, max_z = 1000, 1000, 1000, -1, -1, -1

    for idx in range(len(a[0])):
        if min(min(a[:, idx, 0]), min(b[:, idx, 0]), min(c[:, idx, 0])) < min_x:
            min_x = min(min(a[:, idx, 0]), min(b[:, idx, 0]), min(c[:, idx, 0]))
        if min(min(a[:, idx, 1]), min(b[:, idx, 1]), min(c[:, idx, 1])) < min_y:
            min_y = min(min(a[:, idx, 1]), min(b[:, idx, 1]), min(c[:, idx, 1]))
        if min(min(a[:, idx, 2]), min(b[:, idx, 2]), min(c[:, idx, 2])) < min_z:
            min_z = min(min(a[:, idx, 2]), min(b[:, idx, 2]), min(c[:, idx, 2]))
        if max(max(a[:, idx, 0]), max(b[:, idx, 0]), max(c[:, idx, 0])) > max_x:
            max_x = max(max(a[:, idx, 0]), max(b[:, idx, 0]), max(c[:, idx, 0]))
        if max(max(a[:, idx, 1]), max(b[:, idx, 1]), max(c[:, idx, 1])) > max_y:
            max_y = max(max(a[:, idx, 1]), max(b[:, idx, 1]), max(c[:, idx, 1]))
        if max(max(a[:, idx, 2]), max(b[:, idx, 2]), max(c[:, idx, 2])) > max_z:
            max_z = max(max(a[:, idx, 2]), max(b[:, idx, 2]), max(c[:, idx, 2]))

    save_path = save_path + str(vis_idx)+'.gif'

    fig = plt.figure(figsize=(9, 9))
    ax2 = fig.add_subplot(311)
    ax2.set_axis_off()

    ax1 = fig.add_subplot(312)
    ax1.set_axis_off()
    ax1.set_xlim([min_x-vis_margin, max_x+vis_margin])
    ax1.set_ylim([max_y+vis_margin, min_y-vis_margin])

    ax = fig.add_subplot(313, projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_xlim([min_x-vis_margin, max_x+vis_margin])
    ax.set_ylim([max_y+vis_margin, min_y-vis_margin])
    ax.set_zlim([max_z+vis_margin, min_z-vis_margin])

    custom_lines = [Line2D([0], [0], marker='o', ls=':',color='r'), Line2D([0], [0], marker='o', ls=':',color='g'), Line2D([0], [0], marker='x', ls=':',color='b'), Line2D([0], [0], ls=':',color='m'), Line2D([0], [0], ls=':',color='c')]

    ax1.legend().set_visible(False)
    ax.legend(custom_lines, ['Observation', 'GT', 'LSTM Prediction', 'FV Prediction', 'FA Prediction'], loc=2)

    obs_x, obs_y, obs_z = [], [], []
    gt_x, gt_y, gt_z = [], [], []
    fv_x, fv_y, fv_z = [], [], []
    fa_x, fa_y, fa_z = [], [], []
    lstm_x, lstm_y, lstm_z = [], [], []

    skeleton_img_fa, = ax1.plot(fa_x, fa_y, ':c')
    skeleton_plt_fa, = ax.plot(fa_x, fa_y, fa_z, ':c')

    skeleton_img_fv, = ax1.plot(fv_x, fv_y, ':m')
    skeleton_plt_fv, = ax.plot(fv_x, fv_y, fv_z, ':m')

    skeleton_img_gt, = ax1.plot(gt_x, gt_y, ':go')
    skeleton_plt_gt, = ax.plot(gt_x, gt_y, gt_z, ':go')

    skeleton_img_lstm, = ax1.plot(lstm_x, lstm_y, ':bx')
    skeleton_plt_lstm, = ax.plot(lstm_x, lstm_y, lstm_z, ':bx')

    skeleton_img_obs, = ax1.plot(obs_x, obs_y, ':ro')
    skeleton_plt_obs, = ax.plot(obs_x, obs_y, obs_z, ':ro')


    def update(i):
        if i < len(a):
            f_n = frame_num+i
        else:
            f_n = frame_num+i-1
        img_path = '/home/demo/carla_sam_newest/carla/Output/' + str(map_num) + '_2/RGB/camera-' + str(camera_id) + '-image-' + str(f_n) + '.png'
        img = np.array(Image.open(img_path))
        ax2.imshow(img)
        ax1.imshow(img)

        obs_x, obs_y, obs_z = [], [], []
        gt_x, gt_y, gt_z = [], [], []
        fv_x, fv_y, fv_z = [], [], []
        fa_x, fa_y, fa_z = [], [], []
        lstm_x, lstm_y, lstm_z = [], [], []
        for idx in range(len(b[0])):
            if i < len(a):
                obs_x.append(a[i, idx, 0])
                obs_y.append(a[i, idx, 1])
                obs_z.append(a[i, idx, 2])
            else:
                gt_x.append(b[i-len(a), idx, 0])
                gt_y.append(b[i-len(a), idx, 1])
                gt_z.append(b[i-len(a), idx, 2])

                fv_x.append(d[i-len(a), idx, 0])
                fv_y.append(d[i-len(a), idx, 1])
                fv_z.append(d[i-len(a), idx, 2])

                fa_x.append(e[i-len(a), idx, 0])
                fa_y.append(e[i-len(a), idx, 1])
                fa_z.append(e[i-len(a), idx, 2])

                lstm_x.append(c[i-len(a), idx, 0])
                lstm_y.append(c[i-len(a), idx, 1])
                lstm_z.append(c[i-len(a), idx, 2])

        skeleton_img_fa.set_xdata(fa_x)
        skeleton_img_fa.set_ydata(fa_y)

        skeleton_img_fv.set_xdata(fv_x)
        skeleton_img_fv.set_ydata(fv_y)

        skeleton_img_gt.set_xdata(gt_x)
        skeleton_img_gt.set_ydata(gt_y)

        skeleton_img_lstm.set_xdata(lstm_x)
        skeleton_img_lstm.set_ydata(lstm_y)

        skeleton_img_obs.set_xdata(obs_x)
        skeleton_img_obs.set_ydata(obs_y)

        skeleton_plt_fa.set_xdata(fa_x)
        skeleton_plt_fa.set_ydata(fa_y)
        skeleton_plt_fa.set_3d_properties(fa_z)

        skeleton_plt_fv.set_xdata(fv_x)
        skeleton_plt_fv.set_ydata(fv_y)
        skeleton_plt_fv.set_3d_properties(fv_z)

        skeleton_plt_gt.set_xdata(gt_x)
        skeleton_plt_gt.set_ydata(gt_y)
        skeleton_plt_gt.set_3d_properties(gt_z)

        skeleton_plt_lstm.set_xdata(lstm_x)
        skeleton_plt_lstm.set_ydata(lstm_y)
        skeleton_plt_lstm.set_3d_properties(lstm_z)

        skeleton_plt_obs.set_xdata(obs_x)
        skeleton_plt_obs.set_ydata(obs_y)
        skeleton_plt_obs.set_3d_properties(obs_z)

        return ax2, ax1, ax

    anim = FuncAnimation(fig, update, frames=np.arange(0, len(a)+len(b)), interval=200)
    anim.save(save_path, dpi=80, writer='imagemagick', savefig_kwargs={'transparent': True, 'facecolor': 'none'})
    plt.close(fig)

def visualize_locations_gif(a, b, c, d, e, meta, ds, vis_idx, save_path):
    map_num, camera_id, frame_num = meta[vis_idx]
    x_limit = column(a, 0)+column(b, 0) # +column(c, 0)+column(d, 0)+column(e, 0)
    y_limit = column(a, 1)+column(b, 1) # +column(c, 1)+column(d, 1)+column(e, 1)
    z_limit = column(a, 2)+column(b, 2) # +column(c, 2)+column(d, 2)+column(e, 2)

    save_path = save_path + str(vis_idx)+'.gif'


    fig = plt.figure(figsize=(9, 9))
    ax2 = fig.add_subplot(311)
    ax2.set_axis_off()

    ax1 = fig.add_subplot(312)
    ax1.set_xlim([min(x_limit)-60, max(x_limit)+60])
    ax1.set_ylim([max(y_limit)+60, min(y_limit)-60])
    ax1.set_axis_off()

    ax = fig.add_subplot(313, projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_xlim([min(x_limit)-60, max(x_limit)+60])
    ax.set_ylim([max(y_limit)+60, min(y_limit)-60])
    ax.set_zlim([max(z_limit)+60, min(z_limit)-60])

    custom_lines = [Line2D([0], [0], marker='o', ls='-',color='r'), Line2D([0], [0], marker='x', ls='-',color='g'), Line2D([0], [0], marker='*', ls=':',color='b'), Line2D([0], [0], marker='o', ls=':',color='m'), Line2D([0], [0], ls=':',color='c')]

    ax1.legend().set_visible(False)
    ax.legend(custom_lines, ['Observation', 'GT', 'LSTM Prediction', 'FV Prediction', 'FA Prediction'], loc=2)

    def update(i):
        img_path = '/home/demo/carla_sam_newest/carla/Output/' + str(map_num) + '_2/RGB/camera-' + str(camera_id) + '-image-' + str(frame_num+i) + '.png'
        img = np.array(Image.open(img_path))
        ax2.imshow(img)

        ax1.imshow(img)
        if i < len(a):
            ax1.plot(a[:i, 0], a[:i, 1], '-rx', label='Observation')
            ax.plot(a[:i, 0], a[:i, 1], a[:i, 2], '-rx', label='Observation')
        else:
            if e is not None: ax1.plot(e[:i-len(a), 0],e[:i-len(a), 1], ':cx', label='FA Prediction')
            if d is not None: ax1.plot(d[:i-len(a), 0],d[:i-len(a), 1], ':mo', label='FV Prediction')
            ax1.plot(b[:i-len(a), 0],b[:i-len(a), 1], '-gx', label='GT')
            if c is not None: ax1.plot(c[:i-len(a), 0],c[:i-len(a), 1], ':b*', label='LSTM Prediction')

            if e is not None: ax.plot(e[:i-len(a), 0],e[:i-len(a), 1],e[:i-len(a), 2], ':cs', label='FA Prediction')
            if d is not None: ax.plot(d[:i-len(a), 0],d[:i-len(a), 1],d[:i-len(a), 2], ':mo', label='FV Prediction')
            ax.plot(b[:i-len(a), 0],b[:i-len(a), 1],b[:i-len(a), 2], '-go', label='GT')
            if c is not None: ax.plot(c[:i-len(a), 0],c[:i-len(a), 1],c[:i-len(a), 2], ':b*', label='LSTM Prediction')

        return ax2, ax1, ax

    anim = FuncAnimation(fig, update, frames=np.arange(0, len(a)+len(b)), interval=200)
    anim.save(save_path, dpi=80, writer='imagemagick', savefig_kwargs={'transparent': True, 'facecolor': 'none'})
    plt.close(fig)

if __name__ == "__main__":
    ds = 'carla_bicycles'
    data_loader = DataLoader(ds, batch_size=30, observation_length=8, max_depth=50)
    x, y, origin, meta = data_loader.next_batch(split_type=1, pred_length=12, is_keypoints=True)
    make_dir('trajectories/'+ds)
    for vis_idx in range(30):
        print vis_idx,
        map_num, camera_id, frame_num = meta[vis_idx]
        img_path = '/home/demo/carla_sam_newest/carla/Output/' + str(map_num) + '_2/RGB/camera-' + str(camera_id) + '-image-' + str(frame_num) + '.png'
        img = np.array(Image.open(img_path))
        a, b = compute_locations(x[vis_idx], y[vis_idx], origin=origin[vis_idx])

        visualize_output_KP_gif(a, b, b, None, None, meta, ds, vis_idx)
        # visualize_locations_gif(a, b, None, None, None, meta, ds, vis_idx)
