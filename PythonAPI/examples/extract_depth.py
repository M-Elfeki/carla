import os, pdb
import random
import shutil
import cPickle
import numpy as np
from PIL import Image
random.seed(1)
np.random.seed(1)

import matplotlib.pyplot as plt
plt.switch_backend('agg')


main_data_dir = '/home/demo/carla_sam_newest/carla/Output/'

def convert_data(map_num, iteration_num=2):
    data_dir = main_data_dir + str(map_num) + '_' + str(iteration_num) + '/'

    with open(data_dir + 'relative_data.pickle', 'r') as f:
        rel = cPickle.load(f)

    for frame_idx in range(len(rel)):
        for data_idx in rel[frame_idx]:
            camera_id, id, tp, rel_x, rel_y, bb, name_list, kp = data_idx
            img_path = data_dir + 'RGB/camera-' + str(camera_id) + '-image-' + str(frame_idx) + '.png'
            dp_path = data_dir + 'Depth/camera-' + str(camera_id) + '-image-' + str(frame_idx) + '.png'
            img = np.array(Image.open(img_path))
            dp = np.array(Image.open(dp_path))

            pdb.set_trace()

convert_data(1)
