import os, pdb
import shutil
import cPickle
import numpy as np

main_data_dir = '/home/demo/carla_sam_newest/carla/Output/'
main_save_dir = '/home/demo/Desktop/KP/Locs_KP/'
train_splits = [1, 2, 5]
val_split = 4
test_split = 3

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def convert_data(map_num, iteration_num=2):
    data_dir = main_data_dir + str(map_num) + '_' + str(iteration_num) + '/'
    save_path = main_save_dir
    if map_num in train_splits:
        save_path = save_path + 'train/'
    elif map_num == val_split:
        save_path = save_path + 'val/'
    elif map_num == test_split:
        save_path = save_path + 'test/'

    carla_save_dir = make_dir(save_path + 'carla/')
    carla_cars_save_dir = make_dir(save_path + 'carla_cars/')
    carla_peds_save_dir = make_dir(save_path + 'carla_peds/')
    carla_bicycles_save_dir = make_dir(save_path + 'carla_bicycles/')

    # with open(data_dir + 'absolute_data.pickle', 'r') as f:
    #     abs = cPickle.load(f)
    # ids, tps, s, t = [], [], [], []
    # for frame_idx in range(len(abs)):
    #     for data_idx in abs[frame_idx]:
    #         id, tp, x, y, pitch, yaw, roll, steer, throttle = data_idx
    #         if id not in ids:
    #             ids.append(id)
    #             s.append(np.zeros(len(abs)))
    #             t.append(np.zeros(len(abs)))
    #             tps.append(np.zeros(len(abs)))
    #         tps[ids.index(id)][frame_idx] = tp
    #         s[ids.index(id)][frame_idx] = steer
    #         t[ids.index(id)][frame_idx] = throttle


    with open(data_dir + 'relative_data.pickle', 'r') as f:
        rel = cPickle.load(f)
    x_list, y_list = [], []
    rel_x_list, rel_y_list = [], []
    last_names = None
    l = []

    for frame_idx in range(len(rel)):
        for data_idx in rel[frame_idx]:
            camera_id, id, tp, rel_x, rel_y, bb, name_list, kp = data_idx
            image_locs = np.array(bb.mean(axis=0)).squeeze()
            if tp == 2 and name_list != ['Vehicle_Base', 'Wheel_Front_Left', 'Wheel_Front_Right', 'Wheel_Rear_Right', 'Wheel_Rear_Left']:
                # l.append(name_list)
                # 15 Keypoints
                # ['Bike_Rig', 'BikeBodyGeo', 'RearWheelGeo', 'PedalsGeo', 'RightPedalGeo', 'LeftPedalGeo', 'HandlerGeo', 'FrontWheelGeo', 'HandlerRightSocket', 'HandlerLeftSocket', 'Seat', 'Wheel_R_R', 'Wheel_R_L', 'Wheel_F_R', 'Wheel_F_L']
                with open(carla_bicycles_save_dir + 'map_' + str(map_num) + '_camera_' + str(camera_id) + '.txt', 'a') as f:
                    f.write(' '.join([str(a) for a in np.array([frame_idx, id, 2, image_locs[0], image_locs[1], image_locs[2]] + np.array(kp).reshape(-1).tolist()).tolist()])+'\n')
            elif tp == 3:   # 12 keypoints
                std_kp_names = ['thigh__L', 'leg__L', 'foot__L', 'arm__L', 'hand__L',
                'arm__R', 'hand__R', 'thigh__R', 'leg__R', 'foot__R', 'foreArm__R', 'foreArm__L']
                # std_kp_names = ['root', 'thigh__L', 'leg__L', 'foot__L', 'toe__L', 'toeEnd__L', 'spine__C', 'spine01__C', 'shoulder__L', 'arm__L', 'foreArm__L', 'hand__L', 'handThumb__L', 'handThumb01__L', 'handThumb02__L', 'handThumbEnd__L', 'handIndex__L', 'handIndex01__L',
                # 'handIndex02__L', 'handIndexEnd__L', 'handMiddle__L', 'handMiddle01__L', 'handMiddle02__L', 'handMiddleEnd__L', 'handRing__L', 'handRing01__L', 'handRing02__L', 'handRingEnd__L', 'handPinky__L', 'handPinky01__L', 'handPinky02__L', 'handPinkyEnd__L', 'neck__C', 'shoulder__R', 'arm__R', 'foreArm__R', 'hand__R'
                # , 'handThumb01__R', 'handThumb02__R', 'handThumbEnd__R', 'handIndex__R', 'handIndex01__R', 'handIndex02__R', 'handIndexEnd__R', 'handMiddle__R', 'handMiddle01__R', 'handMiddle02__R', 'handMiddleEnd__R', 'handRing__R', 'handRing01__R', 'handRing02__R', 'handRingEnd__R', 'handPinky__R', 'handPinky01__R', 'handPinky02__R', 'handPinkyEnd__R', 'thigh__R'
                # , 'leg__R', 'foot__R', 'toe__R', 'toeEnd__R']
                name_list = [name_list[i].replace('root1', 'root').replace('man_', '').replace('crl_', '') for i in range(len(name_list))]
                std_kp_list = []
                for cur_kp_name in std_kp_names:
                    if cur_kp_name not in name_list:
                        print cur_kp_name, name_list, 3
                        pdb.set_trace()
                    std_kp_list.append(kp[name_list.index(cur_kp_name)])
                kp = std_kp_list
                with open(carla_peds_save_dir + 'map_' + str(map_num) + '_camera_' + str(camera_id) + '.txt', 'a') as f:
                    f.write(' '.join([str(a) for a in np.array([frame_idx, id, 3, image_locs[0], image_locs[1], image_locs[2]] + np.array(kp).reshape(-1).tolist()).tolist()])+'\n')
            elif tp == 1 and name_list != ['Bike_Rig', 'BikeBodyGeo', 'RearWheelGeo', 'PedalsGeo', 'RightPedalGeo', 'LeftPedalGeo', 'HandlerGeo', 'FrontWheelGeo', 'HandlerRightSocket', 'HandlerLeftSocket', 'Seat', 'Wheel_R_R', 'Wheel_R_L', 'Wheel_F_R', 'Wheel_F_L']\
                and name_list != ['Bike_Rig', 'BikeBodyGeo', 'RearWheelGeo', 'PedalsGeo', 'RightPedalGeo', 'LeftPedalGeo', 'HandlerGeo', 'HandlerRightSocket', 'HandlerLeftSocket', 'FrontWheelGeo', 'Seat', 'Wheel_R_R', 'Wheel_R_L', 'Wheel_F_R', 'Wheel_F_L']:
                # 5 keypoints
                std_car_names = ['Vehicle_Base', 'Wheel_Front_Left', 'Wheel_Front_Right', 'Wheel_Rear_Right', 'Wheel_Rear_Left']
                std_kp_list = []
                for cur_kp_name in std_car_names:
                    if cur_kp_name not in name_list:
                        print cur_kp_name, name_list, 2
                        pdb.set_trace()
                    std_kp_list.append(kp[name_list.index(cur_kp_name)])
                kp = std_kp_list
                with open(carla_cars_save_dir + 'map_' + str(map_num) + '_camera_' + str(camera_id) + '.txt', 'a') as f:
                    f.write(' '.join([str(a) for a in np.array([frame_idx, id, 1, image_locs[0], image_locs[1], image_locs[2]] + np.array(kp).reshape(-1).tolist()).tolist()])+'\n')
                # l.append(name_list)

    # u_l = []
    # for cur_list in l:
    #     if len(u_l) == 0:
    #         u_l.append(cur_list)
    #     else:
    #         is_found = False
    #         for cur_u_l in u_l:
    #             if cur_u_l == cur_list:
    #                 is_found = True
    #                 break
    #         if not is_found:
    #             u_l.append(cur_list)
    # print u_l
    # pdb.set_trace()
for map_num in range(1, 6):
    print map_num,
    convert_data(map_num)
print 'Processed Locations'


# a = ['man_root1', 'man_root1_man_hips__C', 'man_thigh__L', 'man_leg__L', 'man_foot__L', 'man_toe__L', 'man_toeEnd__L', 'man_spine__C', 'man_spine01__C', 'man_shoulder__L', 'man_arm__L', 'man_foreArm__L', 'man_hand__L', 'man_handThumb__L', 'man_handThumb01__L', 'man_handThumb02__L', 'man_handThumbEnd__L', 'man_handIndex__L', 'man_handIndex01__L',\
#  'man_handIndex02__L', 'man_handIndexEnd__L', 'man_handMiddle__L', 'man_handMiddle01__L', 'man_handMiddle02__L', 'man_handMiddleEnd__L', 'man_handRing__L', 'man_handRing01__L', 'man_handRing02__L', 'man_handRingEnd__L', 'man_handPinky__L', 'man_handPinky01__L', 'man_handPinky02__L', 'man_handPinkyEnd__L', 'man_neck__C', 'Generious_Head', 'man_shoulder__R', 'man_arm__R', 'man_foreArm__R', 'man_hand__R',
#  'man_handThumb__R', 'man_handThumb01__R', 'man_handThumb02__R', 'man_handThumbEnd__R', 'man_handIndex__R', 'man_handIndex01__R', 'man_handIndex02__R', 'man_handIndexEnd__R', 'man_handMiddle__R', 'man_handMiddle01__R', 'man_handMiddle02__R', 'man_handMiddleEnd__R', 'man_handRing__R', 'man_handRing01__R', 'man_handRing02__R', 'man_handRingEnd__R', 'man_handPinky__R', 'man_handPinky01__R', 'man_handPinky02__R', 'man_handPinkyEnd__R',
#  'man_thigh__R', 'man_leg__R', 'man_foot__R', 'man_toe__R', 'man_toeEnd__R']
#
# b = ['crl_root', 'crl_hips__C', 'crl_spine__C', 'crl_spine01__C', 'crl_neck__C', 'crl_Head__C', 'crl_eye__L', 'crl_eye__R', 'crl_shoulder__L', 'crl_arm__L', 'crl_foreArm__L', 'crl_hand__L', 'crl_handThumb__L', 'crl_handThumb01__L', 'crl_handThumb02__L', 'crl_handThumbEnd__L', 'crl_handIndex__L', 'crl_handIndex01__L', 'crl_handIndex02__L', 'crl_handIndexEnd__L', \
# 'crl_handMiddle__L', 'crl_handMiddle01__L', 'crl_handMiddle02__L', 'crl_handMiddleEnd__L', 'crl_handRing__L', 'crl_handRing01__L', 'crl_handRing02__L', 'crl_handRingEnd__L', 'crl_handPinky__L', 'crl_handPinky01__L', 'crl_handPinky02__L', 'crl_handPinkyEnd__L', 'crl_shoulder__R', 'crl_arm__R', 'crl_foreArm__R', 'crl_hand__R', 'crl_handThumb__R', 'crl_handThumb01__R', 'crl_handThumb02__R', 'crl_handThumbEnd__R', 'crl_handIndex__R', 'crl_handIndex01__R', 'crl_handIndex02__R', 'crl_handIndexEnd__R', 'crl_handMiddle__R',\
#  'crl_handMiddle01__R', 'crl_handMiddle02__R', 'crl_handMiddleEnd__R', 'crl_handRing__R', 'crl_handRing01__R', 'crl_handRing02__R', 'crl_handRingEnd__R', 'crl_handPinky__R', 'crl_handPinky01__R', 'crl_handPinky02__R', 'crl_handPinkyEnd__R', 'crl_thigh__L', 'crl_leg__L', 'crl_foot__L', 'crl_toe__L', 'crl_toeEnd__L', 'crl_thigh__R', 'crl_leg__R', 'crl_foot__R', 'crl_toe__R', 'crl_toeEnd__R']
#
# a = [a[i].replace('root1', 'root') for i in range(len(a))]
# a = [a[i].replace('man_', '') for i in range(len(a))]
# b = [b[i].replace('crl_', '') for i in range(len(b))]
# for b_i in b:
#     if b_i not in a:
#         print b_i
# print a
# print '*' * 20
# for a_i in a:
#     if a_i not in b:
#         print a_i
# print b
# print '*' * 20
# req = ['root', 'thigh__L', 'leg__L', 'foot__L', 'toe__L', 'toeEnd__L', 'spine__C', 'spine01__C', 'shoulder__L', 'arm__L', 'foreArm__L', 'hand__L', 'handThumb__L', 'handThumb01__L', 'handThumb02__L', 'handThumbEnd__L', 'handIndex__L', 'handIndex01__L',
# 'handIndex02__L', 'handIndexEnd__L', 'handMiddle__L', 'handMiddle01__L', 'handMiddle02__L', 'handMiddleEnd__L', 'handRing__L', 'handRing01__L', 'handRing02__L', 'handRingEnd__L', 'handPinky__L', 'handPinky01__L', 'handPinky02__L', 'handPinkyEnd__L', 'neck__C', 'shoulder__R', 'arm__R', 'foreArm__R', 'hand__R'
# , 'handThumb01__R', 'handThumb02__R', 'handThumbEnd__R', 'handIndex__R', 'handIndex01__R', 'handIndex02__R', 'handIndexEnd__R', 'handMiddle__R', 'handMiddle01__R', 'handMiddle02__R', 'handMiddleEnd__R', 'handRing__R', 'handRing01__R', 'handRing02__R', 'handRingEnd__R', 'handPinky__R', 'handPinky01__R', 'handPinky02__R', 'handPinkyEnd__R', 'thigh__R'
# , 'leg__R', 'foot__R', 'toe__R', 'toeEnd__R']
#
# print len(req)
# for req_i in req:
#     if req_i not in a:
#         print 'a'
#         print req_i
#     if req_i not in b:
#         print 'b'
#         print req_i
