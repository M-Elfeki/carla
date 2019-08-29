import os
import pdb
import random
import cPickle
import numpy as np
import pandas as pd
from PIL import Image

random.seed(1)
np.random.seed(1)

import torch
torch.manual_seed(1)
from torch.autograd import Variable
from utils import column, make_dir

VIEW_WIDTH, VIEW_HEIGHT = 1920//2, 1080//2


class DataLoader():
    def __init__(self, dataset, batch_size=16, observation_length=8, max_depth=10000, real_depth=True,
                 force_preprocess=False, datasets_path='/home/demo/Desktop/KP/Locs_KP/'):
        self.max_depth = max_depth
        self.real_depth = real_depth
        self.batch_size = batch_size
        self.observation_length = observation_length

        train_path = os.path.join(datasets_path, 'train', dataset)
        train_data = [os.path.join(train_path, cur_file) for cur_file in os.listdir(train_path)]

        val_path = os.path.join(datasets_path, 'val', dataset)
        val_data = [os.path.join(val_path, cur_file) for cur_file in os.listdir(val_path)]

        test_path = os.path.join(datasets_path, 'test', dataset)
        test_data = [os.path.join(test_path, cur_file) for cur_file in os.listdir(test_path)]
        self.data = [train_data, val_data, test_data]
        if dataset == 'carla_cars':
            self.state_dim = 15
        elif dataset == 'carla_peds':
            self.state_dim = 36
        elif dataset == 'carla_bicycles':
            self.state_dim = 45
        else:
            self.state_dim = 3

        traj_file_name = datasets_path + dataset + '_' + str(real_depth) + '_trajectories.pickle'
        self.processed_data = [None, None, None]
        # if os.path.exists(traj_file_name) and not force_preprocess:
        #     with open(traj_file_name, 'r') as traj_file:
        #         self.processed_data = cPickle.load(traj_file)
        # else:
        #     self.processed_data = [[], [], []]
        #     for split_type in range(3):
        #         for cur_file in self.data[split_type]:
        #             cur_arr = self.frame_preprocess(cur_file)
        #             if len(cur_arr) > 0:
        #                 self.processed_data[split_type].extend(cur_arr)
        #     with open(traj_file_name, 'wb') as traj_file:
        #         cPickle.dump(self.processed_data, traj_file)
        #     print('Loaded Trajectories')

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
                dp_img = np.array(Image.open('/home/demo/carla_sam_newest/carla/Output/' + str(map_num) + '_2/Depth/camera-' + str(camera_id) + '-image-' + str(cur_item[0]) + '.png'), dtype=np.float32)
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
            bkgd_img = np.array(Image.open('/home/demo/carla_sam_newest/carla/Output/' + str(map_num) + '_2/Background/camera-' + str(camera_id) + '-image-' + str(cur_item[0]) + '.png'), dtype=np.float32)
            cur_item.append(bkgd_img)
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
                break
            print len(self.processed_data[split_type])
        batch, all_batch, frame_batches = [], [], []
        for _ in range(100):
            cur_seq = []
            while len(cur_seq) < self.observation_length + pred_length:
                cur_seq = random.choice(self.processed_data[split_type])
            frame_nums, map_num, camera_id, center_traj, kp_traj = np.array(column(cur_seq, 0), dtype=np.int32), int(cur_seq[0][2]), int(cur_seq[0][3]), np.array(column(cur_seq, 5)), np.array(column(cur_seq, 6))
            cur_b = np.concatenate([np.expand_dims(center_traj, 1), kp_traj], axis=1)
            batch.append(cur_b)
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
