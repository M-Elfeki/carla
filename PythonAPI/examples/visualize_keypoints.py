import os
import sys
import math
import glob
import time
import queue
import pygame
import random
import weakref
import cPickle
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
import pygame


VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
chosen_camera_id = 6692
chosen_frame_idx = 5

with open('/home/demo/carla_sam/Output/3_4/absolute_keypoints.pickle','rb') as fp:
    d1=cPickle.load(fp)

def filter(coordinates):
    x, y, z = coordinates
    if z < -1:
        return False
    if x < 0 or x >= VIEW_WIDTH:
        return False
    if y < 0 or y >= VIEW_HEIGHT:
        return False
    return True

VIEW_WIDTH = display_width = 1920//2
VIEW_HEIGHT = display_height = 1080//2
BLUE =      (  0,   0, 255)

gameDisplay = pygame.display.set_mode((display_width, display_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
bb_surface = pygame.Surface((display_width, display_height))
bb_surface.set_colorkey((0, 0, 0))

file_path = '/home/demo/carla_sam/Output/3_2/RGB/camera-' + str(chosen_camera_id) + '-image-'+str(chosen_frame_idx)+'.png'
gameDisplay.blit(pygame.image.load(file_path), (0, 0))


for time_step in range(len(d1)):
    cur_step_kps = d1[time_step]
    for camera_id, id, type, name_list, bb in cur_step_kps:
        if camera_id != chosen_camera_id or time_step != chosen_frame_idx:
            continue
        if all(bb[:, 2]>0) and  all(0<=bb[:, 0]) and all(bb[:, 0]< VIEW_WIDTH) and all(0<=bb[:, 1]) and all(bb[:, 1]<VIEW_HEIGHT):
            print bb[:, 2]
            for cur_bb_idx in range(len(bb)):
                pygame.draw.circle(bb_surface, BLUE, (int(bb[cur_bb_idx, 0]), int(bb[cur_bb_idx, 1])), 5)
                gameDisplay.blit(bb_surface, (0, 0))
                pygame.display.update()
                pygame.display.flip()

                pygame.event.pump()
pygame.image.save(gameDisplay, "screenshot.jpeg")
# pygame.quit()



    # [all(bb[:, 2]>0) and  all(0<=bb[:, 0]) and all(bb[:, 0]< VIEW_WIDTH) and all(0<=bb[:, 1]) and all(bb[:, 1]<VIEW_HEIGHT) for camera_id, id, type, name_list, bb in cur_step_kps]


# valid_agents = [all(bb[:, 2]>0) and  all(0<=bb[:, 0]) and all(bb[:, 0]< VIEW_WIDTH) and all(0<=bb[:, 1]) and all(bb[:, 1]<VIEW_HEIGHT) for bb in bounding_boxes]
# camera_id, id, type, name_list, camera_bbox

# print d1

# id = d1[0][0]
# for x, v1 in enumerate(d1):
#     for y, v2 in enumerate(v1):
#         if v2[0] == 4050 and v2[1] == 'walker.pedestrian.0005' and v2[2] =='crl_Head__C':
#             print(v2)
