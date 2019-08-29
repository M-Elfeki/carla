import os
import pdb
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

random.seed(1)
np.random.seed(1)
plt.switch_backend('agg')

from LSTM_Pred import LSTM_BKGD, train, test
from DataLoader import DataLoader
from utils import *

def test(dataset, observation_length=8, prediction_length=12, batch_size=32, inference_iters=30, save_path='models/locs_depth/', real_depth=False, lstm_type=1):
    save_path = save_path + dataset + '/' + str(real_depth) + '/' + str(lstm_type) + '/'
    make_dir(save_path)
    if real_depth:
        max_depth = 5
    else:
        max_depth = 50
    data_loader = DataLoader(dataset, batch_size, observation_length=observation_length, max_depth=max_depth, real_depth=real_depth)
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
    parser.add_argument('--num_vis', default=30, type=int)
    # parser.add_argument('--lstm_type', default=1, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    args = parser.parse_args()

    # for lstm_type in range(1, 5):
    #     print args, lstm_type
        # try:
        # if not args.infer:
        #     train(args.ds, num_epochs=args.num_epochs, batch_size=args.batch_size, prediction_length=12, real_depth=args.real_depth, lstm_type=lstm_type)
        #     print('*' * 30)
        # x, y, y_hat_lstm, y_hat_fv, y_hat_fa, all_origins, all_meta = test(args.ds, prediction_length=12, real_depth=args.real_depth, lstm_type=lstm_type)
        # save_path = 'trajectories/' + args.ds + '/' + str(args.real_depth) + '/' + str(lstm_type) + '/'
        # make_dir(save_path)
        # for vis_idx in tqdm(xrange(args.num_vis)):
        #     a, b, c, d, e = x[vis_idx], y[vis_idx], y_hat_lstm[vis_idx], y_hat_fv[vis_idx], y_hat_fa[vis_idx]
        #     map_num, camera_id, frame_num = all_meta[vis_idx]
    lstm_type=3
    data_loader = DataLoader(args.ds, batch_size=30, observation_length=8, max_depth=30)
    x, y, origin, meta = data_loader.next_batch(split_type=2, pred_length=25)
    save_path = 'trajectories/' + args.ds + '/' + str(args.real_depth) + '/' + str(lstm_type) + '/'
    make_dir(save_path)
    for vis_idx in tqdm(xrange(30)):
        map_num, camera_id, frame_num = meta[vis_idx]
        img_path = '/home/demo/carla_sam_newest/carla/Output/' + str(map_num) + '_2/RGB/camera-' + str(camera_id) + '-image-' + str(frame_num) + '.png'
        img = np.array(Image.open(img_path))
        a, b = compute_locations(x[vis_idx], y[vis_idx], origin=origin[vis_idx])
        c = d = e = b

        # _, b = compute_locations(a, b, origin=all_origins[vis_idx])
        # _, c = compute_locations(a, c, origin=all_origins[vis_idx])
        # _, d = compute_locations(a, d, origin=all_origins[vis_idx])
        # a, e = compute_locations(a, e, origin=all_origins[vis_idx])

        visualize_locations_gif(a[:, :3], b[:, :3], c[:, :3], d[:, :3], e[:, :3], meta, args.ds, vis_idx, save_path)
        visualize_output_KP_gif(a[:, 3:], b[:, 3:], c[:, 3:], d[:, 3:], e[:, 3:], meta, args.ds, vis_idx, save_path)
        # except:
        #     print '!!' * 30
        #     print lstm_type
        #     print 'DID NOT WORK'
    print '#' * 30
    print '*' * 30
