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

random.seed(100)
np.random.seed(100)

sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
    sys.version_info.major,
    sys.version_info.minor,
    'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
import carla

import os
import glob

files = glob.glob('/home/demo/carla_sam_newest/carla/PythonAPI/examples/outs/*.png')
for f in files:
    os.remove(f)



VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
VIEW_FOV = 90

BB_COLOR = (248, 64, 24)

calibration = np.identity(3)
calibration[0, 2] = VIEW_WIDTH / 2.0
calibration[1, 2] = VIEW_HEIGHT / 2.0
calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))

camera_transform = carla.Transform(carla.Location(x=-2.5, z=1.5))


# =========================================================================
# -- Utility Functions ----------------------------------------------------
# =========================================================================


def return_type(cur_type):
    if cur_type in ['vehicle.bh.crossbike', 'vehicle.kawasaki.ninja', 'vehicle.diamondback.century', 'vehicle.yamaha.yzf', 'vehicle.harley-davidson.low rider', 'vehicle.gazelle.omafiets']:
        return 2
    elif 'pedestrian' in cur_type:
        return 3
    else:
        return 1

def retrieve_agents_absolute_data(vehicle, ids, controls):
    # Return absolute data for vehicles, bicycles and walkers
   type = return_type(vehicle.type_id)
   absolute_locs = vehicle.get_transform().location
   abs_x, abs_y = absolute_locs.x, absolute_locs.y
   absolute_orients = vehicle.get_transform().rotation
   abs_pitch, abs_yaw, abs_roll = absolute_orients.pitch, absolute_orients.yaw, absolute_orients.roll
   if type == 1:
       control = controls[ids.index(vehicle.id)].get_control()
       throttle = control.throttle
       steer = control.steer
       return vehicle.id, type, abs_x, abs_y, abs_pitch, abs_yaw, abs_roll, steer, throttle
   else:
       return vehicle.id, type, abs_x, abs_y, abs_pitch, abs_yaw, abs_roll, None, None


# ==========================================================================
# -- BoundingBoxesClient ---------------------------------------------------
# ==========================================================================


class BoundingBoxesClient(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_relative_data(vehicles, camera, camera_id):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """
        relative_data = []
        bounding_boxes = [BoundingBoxesClient.get_bounding_box(vehicle, camera) for vehicle in vehicles]
        valid_agents = [all(bb[:, 2]>0) and  all(0<=bb[:, 0]) and all(bb[:, 0]< VIEW_WIDTH) and all(0<=bb[:, 1]) and all(bb[:, 1]<VIEW_HEIGHT) for bb in bounding_boxes]
        for agent_idx in range(len(vehicles)):
            if valid_agents[agent_idx]:
                vehicle = vehicles[agent_idx]
                id = vehicle.id
                type = return_type(vehicle.type_id)
                absolute_locs = vehicle.get_transform().location
                absolute_orients = vehicle.get_transform().rotation
                rel_x, rel_y = absolute_locs.x - camera.get_transform().location.x, absolute_locs.y - camera.get_transform().location.y

                bb = bounding_boxes[agent_idx]
                name_list, relative_kp = BoundingBoxesClient.get_keypoints(vehicle, camera)
                relative_data.append([camera_id, id, type, rel_x, rel_y, bb, name_list, relative_kp])
        return relative_data

    @staticmethod
    def get_keypoints(vehicle, camera):
        """
        Returns 3D Keypoints for a vehicle based on camera view.
        """

        absolute_locs = vehicle.get_transform().location
        kp = vehicle.get_keypoints().keypoints
        name_list, relative_keypoints = [], []
        for name, val in kp:
            val *= 0.01
            name_list.append(name)
            relative_keypoints.append([absolute_locs.x-val.x, absolute_locs.y-val.y, absolute_locs.z-val.z])

        kp_cords = BoundingBoxesClient._create_kp_points(vehicle, relative_keypoints)
        kp_cords_x_y_z = BoundingBoxesClient._vehicle_to_sensor_KP(kp_cords, vehicle, camera, name_list, relative_keypoints)[:3, :]
        kp_cords_y_minus_z_x = np.concatenate([kp_cords_x_y_z[1, :], -kp_cords_x_y_z[2, :], kp_cords_x_y_z[0, :]])
        kp_box = np.transpose(np.dot(calibration, kp_cords_y_minus_z_x))
        camera_kp_box = np.concatenate([kp_box[:, 0] / kp_box[:, 2], kp_box[:, 1] / kp_box[:, 2], kp_box[:, 2]], axis=1)
        return name_list, camera_kp_box

    @staticmethod
    def _create_kp_points(vehicle, relative_keypoints):
        """
        Returns 3D Keypoint for a vehicle.
        """
        cords = np.zeros((len(relative_keypoints), 4))
        for cur_point_idx, cur_point in enumerate(relative_keypoints):
            cords[cur_point_idx, :] = np.array([cur_point[0], cur_point[1], cur_point[2], 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor_KP(cords, vehicle, sensor, name_list, relative_keypoints):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = BoundingBoxesClient._vehicle_to_world_KP(cords, vehicle, name_list, relative_keypoints)
        sensor_cord = BoundingBoxesClient._world_to_sensor_KP(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world_KP(cords, vehicle, name_list, relative_keypoints):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """
        bb_transform = carla.Transform(vehicle.bounding_box.location) # carla.Location(0, 0, 0)) # loc[0], loc[1], loc[2]))
        bb_vehicle_matrix = BoundingBoxesClient.get_matrix(bb_transform)
        vehicle_world_matrix = BoundingBoxesClient.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor_KP(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = BoundingBoxesClient.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords


    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = BoundingBoxesClient._create_bb_points(vehicle)
        cords_x_y_z = BoundingBoxesClient._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = BoundingBoxesClient._vehicle_to_world(cords, vehicle)
        sensor_cord = BoundingBoxesClient._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = BoundingBoxesClient.get_matrix(bb_transform)
        vehicle_world_matrix = BoundingBoxesClient.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = BoundingBoxesClient.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(2.0)

world = client.get_world()

settings = world.get_settings()
settings.synchronous_mode = True
world.apply_settings(settings)
world.set_weather(carla.WeatherParameters.ClearNoon)

try:
    # blueprint = random.choice(world.get_blueprint_library().filter('walker.*'))
    blueprints_vehicles = world.get_blueprint_library().filter('vehicle.*')
    cur_bicycle_idx = 17 # random.choice([6, 8, 9, 13, 17, 21])
    blueprint = blueprints_vehicles[cur_bicycle_idx]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
    npc = world.try_spawn_actor(blueprint, spawn_point)
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
    camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
    camera_bp.set_attribute('fov', str(VIEW_FOV))
    camera_transform = carla.Transform(carla.Location(x=spawn_point.location.x-3, y=spawn_point.location.y-1.5, z=spawn_point.location.z))
    camera = world.spawn_actor(camera_bp, camera_transform)
    camera.calibration = calibration
    camera.listen(lambda image: image.save_to_disk('/home/demo/carla_sam_newest/carla/PythonAPI/examples/outs/%06d.png' % image.frame))

    pygame.init()
    pygame_clock = pygame.time.Clock()

    relative_data = []
    for i in range(5):
       world.tick()
       pygame_clock.tick(20)
       relative_data.append(BoundingBoxesClient.get_relative_data([npc], camera, camera.id))

    with open('/home/demo/carla_sam_newest/carla/PythonAPI/examples/outs/relative_data.pickle', 'wb') as rel_file:
        cPickle.dump(relative_data, rel_file)
finally:
    npc.destroy()
    camera.destroy()
    settings.synchronous_mode = False
    world.apply_settings(settings)


# import numpy as np, pdb
# from PIL import Image
# import matplotlib.pyplot as plt
# import scipy.misc
# import os
# from tqdm import tqdm
#
# def make_dir(save_path):
#     if not os.path.exists(os.path.join(save_path)):
#      os.makedirs(os.path.join(save_path))
#
# for map_num in tqdm(xrange(1, 6)):
#     ss_path = '/home/demo/carla_sam_newest/carla/Output/' + str(map_num) + '_2/SemanticSegmentation/'
#     save_path = '/home/demo/carla_sam_newest/carla/Output/' + str(map_num) + '_2/Background/'
#     make_dir(save_path)
#     for cur_file in os.listdir(ss_path):
#         if not os.path.exists(save_path + cur_file):
#             dp_img = np.array(Image.open(ss_path + cur_file), dtype=np.float32)
#             q = np.ones((len(dp_img), len(dp_img[0])))
#             for x in range(len(dp_img)):
#                 for y in range(len(dp_img[x])):
#                     if all(dp_img[x, y, :3] == (0, 0, 142)):
#                         q[x, y] = 0
#                     elif all(dp_img[x, y, :3] == (220, 20, 60)):
#                         q[x, y] = 0
#                     elif all(dp_img[x, y, :3] == (0, 0, 0)):
#                         q[x, y] = 0
#             scipy.misc.imsave(save_path + cur_file, q)
# # plt.imshow(q)
# # plt.show()
# print min(dp_img[:, :, 0]), max(dp_img[:, :, 0])
