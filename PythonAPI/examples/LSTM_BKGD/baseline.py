import os
import pdb
import random
import numpy as np

random.seed(1)
np.random.seed(1)

import torch
torch.manual_seed(1)
from torch import nn
from torch import optim
from torch.autograd import Variable

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
