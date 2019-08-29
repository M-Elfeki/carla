import pygame, time, math
import cPickle, pdb
import numpy as np, os


# import os, glob
# dd = '/home/demo/carla_sam_newest/carla/PythonAPI/examples/outs/'
# for f in os.listdir(dd):
#     if '.png' in f:
#         file_path = dd + f
#         # file_path = '/home/demo/carla_sam_newest/carla/PythonAPI/examples/outs/127681.png'
# with open('/home/demo/carla_sam_newest/carla/PythonAPI/examples/outs/relative_data.pickle', 'r') as f:
#     rel = cPickle.load(f)

chosen_camera_id = 218

chosen_frame_idx = 0
# data_dir = '/home/demo/carla_sam_newest/carla/Output/2_2/'
data_dir = 'outs/'
with open(data_dir + 'relative_data.pickle', 'r') as f:
    rel = cPickle.load(f)
depth_bounding_boxes = True
s = []
for cur_file in os.listdir(data_dir):
    if '.png' in cur_file:
        s.append(int(cur_file.replace('.png', '')))
for chosen_frame_idx in range(1):
    # file_path = data_dir+'RGB/camera-' + str(chosen_camera_id) + '-image-' + str(chosen_frame_idx) + '.png'
    file_path = (data_dir+'%06d.png' % max(s))
    print file_path

    VIEW_WIDTH = display_width = 1920//2
    VIEW_HEIGHT = display_height = 1080//2
    BB_COLOR = (248, 64, 24)
    BB2_COLOR = (0, 64, 24)
    gameDisplay = pygame.display.set_mode((display_width, display_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    bb_surface = pygame.Surface((display_width, display_height))
    bb_surface.set_colorkey((0, 0, 0))
    bb2_surface = pygame.Surface((display_width, display_height))
    bb2_surface.set_colorkey((0, 0, 0))

    gameDisplay.blit(pygame.image.load(file_path), (0, 0))
    for frame_idx in range(len(rel)):
        for data_idx in rel[frame_idx]:
            camera_id, id, tp, rel_x, rel_y, bb, name_list, kp = data_idx
            if frame_idx == 4: # True: # camera_id == chosen_camera_id and frame_idx == chosen_frame_idx and all(bb[:, 2] < 50.0):
                bb = np.array(bb, dtype=np.float64)
                # kp = np.array(kp, dtype=np.float64)
                # std_kp_names = ['thigh__L', 'leg__L', 'foot__L', 'arm__L', 'hand__L',
                # 'arm__R', 'hand__R', 'thigh__R', 'leg__R', 'foot__R', 'foreArm__R', 'foreArm__L']
                # name_list = [name_list[i].replace('root1', 'root').replace('man_', '').replace('crl_', '') for i in range(len(name_list))]
                # std_kp_list = []
                # for cur_kp_name in std_kp_names:
                #     if cur_kp_name not in name_list:
                #         print cur_kp_name, name_list, 2
                #         pdb.set_trace()
                #     std_kp_list.append(kp[name_list.index(cur_kp_name)])
                # kp = np.array(std_kp_list)
                # print kp
                # pdb.set_trace()
                # points = [(int(kp[i, 0]), int(VIEW_HEIGHT-kp[i, 1])) for i in range(len(kp))]
                # for cur_point in points:
                #     pygame.draw.circle(bb_surface, BB_COLOR, cur_point, 4)
                # gameDisplay.blit(bb_surface, (0, 0))

                if depth_bounding_boxes:
                    points = [(int(bb[i, 0]), int(bb[i, 1])) for i in range(8)]
                    # base
                    pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
                    pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
                    pygame.draw.line(bb_surface, BB_COLOR, points[1], points[2])
                    pygame.draw.line(bb_surface, BB_COLOR, points[2], points[3])
                    pygame.draw.line(bb_surface, BB_COLOR, points[3], points[0])
                    # top
                    pygame.draw.line(bb_surface, BB_COLOR, points[4], points[5])
                    pygame.draw.line(bb_surface, BB_COLOR, points[5], points[6])
                    pygame.draw.line(bb_surface, BB_COLOR, points[6], points[7])
                    pygame.draw.line(bb_surface, BB_COLOR, points[7], points[4])
                    # base-top
                    pygame.draw.line(bb_surface, BB_COLOR, points[0], points[4])
                    pygame.draw.line(bb_surface, BB_COLOR, points[1], points[5])
                    pygame.draw.line(bb_surface, BB_COLOR, points[2], points[6])
                    pygame.draw.line(bb_surface, BB_COLOR, points[3], points[7])
                    gameDisplay.blit(bb_surface, (0, 0))
                else:
                    # 2D Bounding Box
                    p1 = min(bb[:, 0]), max(bb[:, 1])
                    p2 = max(bb[:, 0]), max(bb[:, 1])
                    p3 = min(bb[:, 0]), min(bb[:, 1])
                    p4 = max(bb[:, 0]), min(bb[:, 1])
                    pygame.draw.line(bb2_surface, BB2_COLOR, p1, p2)
                    pygame.draw.line(bb2_surface, BB2_COLOR, p1, p3)
                    pygame.draw.line(bb2_surface, BB2_COLOR, p2, p4)
                    pygame.draw.line(bb2_surface, BB2_COLOR, p3, p4)
                    gameDisplay.blit(bb2_surface, (0, 0))


                pygame.display.update()
                pygame.display.flip()

                pygame.event.pump()
    pygame.image.save(gameDisplay, "outs/screenshot_" + str(chosen_frame_idx) + ".jpg")
    pdb.set_trace()
                # time.sleep(1)
                # pdb.set_trace()
    pygame.quit()
                # import sys
                # sys.exit(0)
