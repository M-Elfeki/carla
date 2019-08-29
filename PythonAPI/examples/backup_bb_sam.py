import os
import sys
import pdb
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

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

random.seed(1)
np.random.seed(1)

sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
    sys.version_info.major,
    sys.version_info.minor,
    'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
import carla

VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
VIEW_FOV = 90

BB_COLOR = (248, 64, 24)

calibration = np.identity(3)
calibration[0, 2] = VIEW_WIDTH / 2.0
calibration[1, 2] = VIEW_HEIGHT / 2.0
calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))

camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))


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




class ColorModifierWrapper(object):
    """ We can not instanciate a Carla image from the python side, so this is
        an helper to convert what's out of the semantic segmentation now
    """

    def __init__(self, carla_image):
        self.carla_image = carla_image
        self.original_colors = [ColorModifierWrapper.clone_color(carla_image[i])
                                for i in range(len(carla_image))]
        self.color_map = {r:carla.Color(r = (r*13+16) % 256,
                                        g = (142+r*21) % 256,
                                        b = (89+r*47) % 256)
                          for r in range(256)}
        self.blank = carla.Color(r=255,b=255,g=255)
        self.black = carla.Color(r=0,b=0,g=0)

    @staticmethod
    def clone_color(color):
        return carla.Color(r=color.r, g=color.g, b=color.b, a=color.a)

    def view_as_semantic_seg(self):
        for i, color in enumerate(self.original_colors):
            new_color = ColorModifierWrapper.clone_color(color)
            if color.r >=16 and color.r < 128:
                new_color.r = 4 # pedestrian
            if color.r >=128 and color.r < 240:
                new_color.r = 10 # vehicle
            self.carla_image[long(i)] = new_color
        self.carla_image.convert(carla.ColorConverter.CityScapesPalette)
        return self.carla_image


    def view_as_instance_seg(self, rel_data=None):
        for i, color in enumerate(self.original_colors):
            if color.r < 16 and color.r != 4:
                self.carla_image[long(i)] = self.blank # Not a pedestrian nor a car
            elif color.r == 4:
                self.carla_image[long(i)] = self.black # rider
            else:
                self.carla_image[long(i)] = self.color_map[color.r]
        # ids_img = np.zeros((VIEW_HEIGHT, VIEW_WIDTH))
        # for x in range(VIEW_HEIGHT):
        #     for y in range(VIEW_WIDTH):
        #         i = long(x * y)
        #         if self.original_colors[i].r > 15:
        #             ids_img[x, y] = self.original_colors[i].r
        #             self.carla_image[i] = self.color_map[self.original_colors[i].r]
        #         else:
        #             self.carla_image[i] = self.blank
        # print len(list(set(ids_img.flatten().tolist())))

        # bbs, ids, types = [], [], []
        # ids_img = np.zeros((VIEW_HEIGHT, VIEW_WIDTH))
        # for cur_rel_data in rel_data:
        #     camera_id, id, type, rel_x, rel_y, bb, name_list, relative_kp = cur_rel_data
        #     bb = np.array(bb, dtype=np.float64)
        #     p1 = (min(bb[:, 0]), max(bb[:, 1]))
        #     p2 = (max(bb[:, 0]), max(bb[:, 1]))
        #     p3 = (min(bb[:, 0]), min(bb[:, 1]))
        #     p4 = (max(bb[:, 0]), min(bb[:, 1]))
        #     bbs.append(Polygon([p1, p2, p3, p4]))
        #     ids.append(id)
        #     types.append(type)
        # for x in range(VIEW_HEIGHT):
        #     for y in range(VIEW_WIDTH):
        #         i = long(x * y)
        #         if self.original_colors[i] >= 16:
        #             cur_point = Point(x, y)
        #             for idx, cur_bb in enumerate(bbs):
        #                 if cur_bb.contains(cur_point):
        #                     ids_img[x, y] = ids[idx]
        pdb.set_trace()

        # if rel_data is not None:
        #
        #     for x in range(VIEW_HEIGHT):
        #         for y in range(VIEW_WIDTH):
        #             i = long(x * y)
        #             cur_point = Point(x, y)
        #             for cur_bb in bbs:
        #                 if self.carla_image[long(i)] == self.black and cur_bb.contains(cur_point):
        #
        #             # If self.carla_image[long(i)] == self.black
        #             # Check if x&y are inside a bounding box--> belong to the same trajectory


        return self.carla_image



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
        kp_box = np.transpose(np.dot(camera.calibration, kp_cords_y_minus_z_x))
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
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
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

# =========================================================================
# -- SynchronousClient ----------------------------------------------------
# =========================================================================


class SynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self):
        self.client = None
        self.world = None
        self.main_path = '../../Output/'

        self.actor_list = []
        self.controls, self.ids = [], []
        self.all_relative, self.all_absolute, self.relative_instances = [], [], []
        self.rgb_queues, self.ss_queues, self.depth_queues, self.lidar_queues, self.rgb_imgs, self.ss_imgs, self.depth_imgs, self.lidar_data = [], [], [], [], [], [], [], []

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_agents(self):
        vehicles, walkers = [], []
        blueprints_vehicles = self.world.get_blueprint_library().filter('vehicle.*')
        blueprints_walkers = self.world.get_blueprint_library().filter('walker.*')

        # Find vehicle and walker spawn points
        vehicles_spawn_points = self.world.get_map().get_spawn_points()
        walkers_spawn_points = []
        for transform in vehicles_spawn_points:
            cur_sidewalk = self.world.get_map().get_waypoint(transform.location, lane_type=carla.LaneType.Sidewalk)
            if cur_sidewalk is not None:
                walkers_spawn_points.append(cur_sidewalk.transform)

        # Spawn Vehicles and Walkers
        num_vehicles, num_walkers = 0, 0
        for transform in vehicles_spawn_points:
            # blueprint 17 is prohipited (race bike [vehicle.diamondback.century]), because it does not fit its bounding boxes
            if random.random() < 0.3:
                cur_bicycle_idx = random.choice([6, 8, 9, 13, 21])
                blueprint = blueprints_vehicles[cur_bicycle_idx]
            else:
                blueprint = random.choice(blueprints_vehicles)
                while blueprint.id == 'vehicle.diamondback.century':
                    blueprint = random.choice(blueprints_vehicles)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')
            vehicle = self.world.try_spawn_actor(blueprint, transform)
            if vehicle is not None:
                self.actor_list.append(vehicle)
                vehicle.set_autopilot()
                num_vehicles += 1
                self.ids.append(vehicle.id)
                self.controls.append(vehicle)
                vehicles.append(vehicle)
        for transform in walkers_spawn_points:
            blueprint = random.choice(blueprints_walkers)
            location = transform.location
            location.z = 1.84
            speed = random.random() * 1.75 + 0.25
            rotation = transform.rotation
            if (abs(rotation.pitch) <= 10 or 350 <= abs(rotation.pitch)) and (80 <= abs(rotation.yaw) <= 100 or 260 <= abs(rotation.yaw) <= 280):
                if random.random() < 0.5:
                    rotation.yaw += 180
            elif (abs(rotation.yaw) <= 10 or 350 <= abs(rotation.yaw)) and (80 <= abs(rotation.pitch) <= 100 or 260 <= abs(rotation.pitch) <= 280):
                if random.random() < 0.5:
                    rotation.pitch += 180
            else:
                continue
            new_transformation = carla.Transform(location, rotation)
            walker_control = carla.WalkerControl()
            walker_control.speed = speed
            walker_control.direction = rotation.get_forward_vector()
            walker = self.world.try_spawn_actor(blueprint, new_transformation)
            if walker is None:
                continue
            walker.apply_control(walker_control)
            self.actor_list.append(walker)
            walkers.append(walker)
            num_walkers += 1
        print('spawned %d actors and %d vehicles' % (num_walkers, num_vehicles))
        return walkers, vehicles

    def camera_blueprints(self):
        """
        Returns camera blueprint.
        """

        camera_ss_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        camera_ss_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_ss_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_ss_bp.set_attribute('fov', str(VIEW_FOV))
        camera_rgb_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_rgb_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_rgb_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_rgb_bp.set_attribute('fov', str(VIEW_FOV))
        camera_dp_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        camera_dp_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_dp_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_dp_bp.set_attribute('fov', str(VIEW_FOV))
        lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('channels', '16')
        lidar_bp.set_attribute('range', '5000')
        lidar_bp.set_attribute('points_per_second', '222000')
        return camera_ss_bp, camera_rgb_bp, camera_dp_bp, lidar_bp


    def setup_cameras(self, vehicles, vehicle_limit):
        """
        Spawns actor-cameras to be used to render the views.
        Sets calibration for client-side boxes rendering.
        """

        random.shuffle(vehicles)
        camera_vehicles = []
        i = 0
        while len(camera_vehicles) < vehicle_limit:
            if return_type(vehicles[i].type_id) == 1:
                camera_vehicles.append(vehicles[i])
            i += 1

        camera_ss_bp, camera_rgb_bp, camera_dp_bp, lidar_bp = self.camera_blueprints()
        camera_ss_list, camera_rgb_list, camera_dp_list, lidar_list = [], [], [], []
        for vehicle_idx in xrange(vehicle_limit):
            camera_vehicle = camera_vehicles[vehicle_idx]

            camera_ss_list.append(self.world.spawn_actor(camera_ss_bp, camera_transform, attach_to=camera_vehicle))
            camera_ss_list[-1].listen(self.ss_queues[vehicle_idx].put)
            camera_ss_list[-1].calibration = calibration

            camera_rgb_list.append(self.world.spawn_actor(camera_rgb_bp, camera_transform, attach_to=camera_vehicles[vehicle_idx]))
            camera_rgb_list[-1].listen(self.rgb_queues[vehicle_idx].put)
            camera_rgb_list[-1].calibration = calibration

            camera_dp_list.append(self.world.spawn_actor(camera_dp_bp, camera_transform, attach_to=camera_vehicle))
            camera_dp_list[-1].listen(self.depth_queues[vehicle_idx].put)
            camera_dp_list[-1].calibration = calibration

            lidar_list.append(self.world.spawn_actor(lidar_bp, camera_transform, attach_to=camera_vehicle))
            lidar_list[-1].listen(self.lidar_queues[vehicle_idx].put)

            self.actor_list.append(camera_rgb_list[-1])
            self.actor_list.append(camera_ss_list[-1])
            self.actor_list.append(camera_dp_list[-1])
            self.actor_list.append(lidar_list[-1])
        return camera_rgb_list, camera_vehicles

    def game_loop(self, args):
        """
        Main program loop.
        """

        map_num, iteration_num, time_limit, vehicle_limit, start_idx = args.map_num, args.iteration_num, args.time_limit, args.vehicle_limit, args.start_idx

        def make_dir(save_path, dir_name):
            if not os.path.exists(os.path.join(save_path, dir_name)):
             os.makedirs(os.path.join(save_path, dir_name))

        save_path = self.main_path + str(map_num) + '_' + str(iteration_num) + '/'

        make_dir(save_path, 'Depth')
        make_dir(save_path, 'Original_Depth')
        make_dir(save_path, 'LogarithmicDepth')
        make_dir(save_path, 'LIDAR')
        make_dir(save_path, 'RGB')
        make_dir(save_path, 'SemanticSegmentation')
        make_dir(save_path, 'InstanceSegmentation')

        for _ in range(vehicle_limit):
            self.rgb_queues.append(queue.Queue())
            self.ss_queues.append(queue.Queue())
            self.depth_queues.append(queue.Queue())
            self.lidar_queues.append(queue.Queue())

            self.rgb_imgs.append([])
            self.ss_imgs.append([])
            self.depth_imgs.append([])
            self.lidar_data.append([])

        pygame.init()

        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(2000.0)
        self.world = self.client.load_world('Town0' + str(map_num))
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        self.set_synchronous_mode(True)

        try:
            walkers, vehicles = self.setup_agents()
            sensor_cameras, camera_vehicles = self.setup_cameras(vehicles, vehicle_limit)

            print 'Start Simulation'
            pygame_clock = pygame.time.Clock()
            for timeCounter in tqdm(xrange(time_limit + start_idx)):
                frame = self.world.tick()
                pygame_clock.tick_busy_loop(20)

                if timeCounter >= start_idx:
                    absolute_data = []
                    for cur_vehicle in vehicles:
                        absolute_data.append(retrieve_agents_absolute_data(cur_vehicle, self.ids, self.controls))
                    for cur_walker in walkers:
                        absolute_data.append(retrieve_agents_absolute_data(cur_walker, self.ids, self.controls))
                    self.all_absolute.append(absolute_data)

                    relative_data, relative_data_instances = [], []
                    for vehicle_idx in range(len(camera_vehicles)):
                        camera_vehicle = camera_vehicles[vehicle_idx]
                        sensor_camera = sensor_cameras[vehicle_idx]
                        other_agents = [vehicle for vehicle in vehicles if vehicle.id != camera_vehicle.id]
                        [other_agents.append(walker) for walker in walkers]
                        cur_rel_data = BoundingBoxesClient.get_relative_data(other_agents, sensor_camera, camera_vehicle.id)
                        relative_data.extend(cur_rel_data)
                        relative_data_instances.append(cur_rel_data)
                    self.all_relative.append(relative_data)
                    self.relative_instances.append(relative_data_instances)

                for vehicle_idx in range(len(camera_vehicles)):
                    rgb_queue = self.rgb_queues[vehicle_idx]
                    depth_queue = self.depth_queues[vehicle_idx]
                    lidar_queue = self.lidar_queues[vehicle_idx]
                    ss_queue = self.ss_queues[vehicle_idx]

                    while not rgb_queue.empty():
                        image = rgb_queue.get()
                        if timeCounter >= start_idx:
                            self.rgb_imgs[vehicle_idx].append(image)
                        if image.frame_number == frame:
                            break
                    while not depth_queue.empty():
                        image = depth_queue.get()
                        if timeCounter >= start_idx:
                            self.depth_imgs[vehicle_idx].append(image)
                        if image.frame_number == frame:
                            break
                    while not ss_queue.empty():
                        image = ss_queue.get()
                        if timeCounter >= start_idx:
                            self.ss_imgs[vehicle_idx].append(image)
                        if image.frame_number == frame:
                            break
                    while not lidar_queue.empty():
                        image = lidar_queue.get()
                        if timeCounter >= start_idx:
                            self.lidar_data[vehicle_idx].append(image)
                        if image.frame_number == frame:
                            break

            print 'Start Data Writes'
            with open(save_path + 'absolute_data.pickle', 'wb') as abs_file:
                cPickle.dump(self.all_absolute, abs_file)
            with open(save_path + 'relative_data.pickle', 'wb') as rel_file:
                cPickle.dump(self.all_relative, rel_file)
            for cur_id_idx in tqdm(xrange(vehicle_limit)):
                cur_id = camera_vehicles[cur_id_idx].id
                if len(self.rgb_imgs[cur_id_idx]) > 0 and not (len(self.rgb_imgs[cur_id_idx]) == len(self.depth_imgs[cur_id_idx]) == len(self.ss_imgs[cur_id_idx]) == len(self.lidar_data[cur_id_idx])):
                    with open(save_path + 'error_log.txt', 'a') as f:
                        f.write('{} {} {} {} {} {} {}\n'.format(map_num, iteration_num, cur_id, len(self.rgb_imgs[cur_id_idx]), len(self.depth_imgs[cur_id_idx]), len(self.ss_imgs[cur_id_idx]), len(self.lidar_data[cur_id_idx])))
                for idx, cur_img in enumerate(self.rgb_imgs[cur_id_idx]):
                    cur_img.save_to_disk(save_path + '/RGB/camera-%d-image-%04d_a.png' % (cur_id, idx))
                for idx, cur_img in enumerate(self.depth_imgs[cur_id_idx]):
                    cur_img.save_to_disk(save_path + '/Original_Depth/camera-%d-image-%04d_c.png' % (cur_id, idx))
                    cur_img.save_to_disk(save_path + '/Depth/camera-%d-image-%04d_c.png' % (cur_id, idx), carla.ColorConverter.Depth)
                    cur_img.convert(carla.ColorConverter.LogarithmicDepth)
                    cur_img.save_to_disk(save_path + '/LogarithmicDepth/camera-%d-image-%04d_b.png' % (cur_id, idx))

                for idx, cur_img in enumerate(self.ss_imgs[cur_id_idx]):
                    # Save the segmentation image
                    wrapper = ColorModifierWrapper(cur_img)
                    cur_img = wrapper.view_as_semantic_seg()
                    cur_img.save_to_disk(save_path + '/SemanticSegmentation/camera-%d-image-%04d_d.png' % (cur_id, idx))
                    cur_img = wrapper.view_as_instance_seg(self.relative_instances[idx][cur_id_idx])
                    cur_img.save_to_disk(save_path + '/InstanceSegmentation/camera-%d-image-%04d_e.png' % (cur_id, idx))
                for idx, cur_img in enumerate(self.lidar_data[cur_id_idx]):
                    cur_img.save_to_disk(save_path + '/LIDAR/camera-%d-image-%000d' % (cur_id, idx))

        finally:
            self.set_synchronous_mode(False)
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            pygame.quit()
            print 'Fininshed', map_num, iteration_num

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_num', type=int, default=3)
    parser.add_argument('--iteration_num', type=int, default=3,
                        help='Simulation Round')
    parser.add_argument('--time_limit', type=int, default=120,
                        help='Number of Frames to simulate')
    parser.add_argument('--vehicle_limit', type=int, default=50,
                        help='Number of cameras to record')
    parser.add_argument('--start_idx', type=int, default=20,
                        help='Number of frames to allow in the beginning before starting simulation Used to set all vehicles before starting recording')
    args = parser.parse_args()
    print(args)

    try:
        client = SynchronousClient()
        client.game_loop(args)
    except KeyboardInterrupt:
        pass
    finally:
        print '#' * 20


if __name__ == '__main__':
    main()
