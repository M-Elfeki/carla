# """ Create 2 cars, a npc at the same place and record
#     the scene in RGB and instance segmentation mode
# """
# #!/usr/bin/env python
#
# # Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# # Barcelona (UAB).
# #
# # This work is licensed under the terms of the MIT license.
# # For a copy, see <https://opensource.org/licenses/MIT>.
#
# import glob
# import os
# import sys
#
# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass
#
# import carla
#
# import random
#
# try:
#     import pygame
# except ImportError:
#     raise RuntimeError('cannot import pygame, make sure pygame package is installed')
#
# try:
#     import numpy as np
# except ImportError:
#     raise RuntimeError('cannot import numpy, make sure numpy package is installed')
#
# try:
#     import queue
# except ImportError:
#     import Queue as queue
#
# class ColorModifierWrapper(object):
#     """ We can not instanciate a Carla image from the python side, so this is
#         an helper to convert what's out of the semantic segmentation now
#     """
#
#     def __init__(self, carla_image):
#         self.carla_image = carla_image
#         self.original_colors = [ColorModifierWrapper.clone_color(carla_image[i])
#                                 for i in range(len(carla_image))]
#         self.color_map = {r:carla.Color(r = (r*13+16) % 256,
#                                         g = (142+r*21) % 256,
#                                         b = (89+r*47) % 256)
#                           for r in range(256)}
#         self.blank = carla.Color(r=255,b=255,g=255)
#         self.black = carla.Color(r=0,b=0,g=0)
#
#     @staticmethod
#     def clone_color(color):
#         return carla.Color(r=color.r, g=color.g, b=color.b, a=color.a)
#
#     def view_as_semantic_seg(self):
#         for i, color in enumerate(self.original_colors):
#             new_color = ColorModifierWrapper.clone_color(color)
#             if color.r >=16 and color.r < 128:
#                 new_color.r = 4 # pedestrian
#             if color.r >=128 and color.r < 240:
#                 new_color.r = 10 # vehicle
#             self.carla_image[long(i)] = new_color
#         self.carla_image.convert(carla.ColorConverter.CityScapesPalette)
#         return self.carla_image
#
#
#     def view_as_instance_seg(self):
#         for i, color in enumerate(self.original_colors):
#             if color.r < 16 and color.r != 4:
#                 self.carla_image[long(i)] = self.blank # Not a pedestrian nor a car
#             elif color.r == 4:
#                 self.carla_image[long(i)] = self.black # rider
#             else:
#                 self.carla_image[long(i)] = self.color_map[color.r]
#         return self.carla_image
#
#
#
# class CarlaSyncMode(object):
#     """
#     Context manager to synchronize output from different sensors. Synchronous
#     mode is enabled as long as we are inside this context
#
#         with CarlaSyncMode(world, sensors) as sync_mode:
#             while True:
#                 data = sync_mode.tick(timeout=1.0)
#
#     """
#
#     def __init__(self, world, *sensors):
#         self.world = world
#         self.sensors = sensors
#         self.frame = None
#         self._queues = []
#
#     def __enter__(self):
#         settings = self.world.get_settings()
#         settings.synchronous_mode = True
#         self.frame = self.world.apply_settings(settings)
#
#         def make_queue(register_event):
#             q = queue.Queue()
#             register_event(q.put)
#             self._queues.append(q)
#
#         make_queue(self.world.on_tick)
#         for sensor in self.sensors:
#             make_queue(sensor.listen)
#         return self
#
#     def tick(self, timeout):
#         self.frame = self.world.tick()
#         data = [self._retrieve_data(q, timeout) for q in self._queues]
#         assert all(x.frame == self.frame for x in data)
#         return data
#
#     def __exit__(self, *args, **kwargs):
#         settings = self.world.get_settings()
#         settings.synchronous_mode = False
#         self.world.apply_settings(settings)
#
#     def _retrieve_data(self, sensor_queue, timeout):
#         while True:
#             data = sensor_queue.get(timeout=timeout)
#             if data.frame == self.frame:
#                 return data
#
#
# def draw_image(surface, image, blend=False):
#     array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
#     array = np.reshape(array, (image.height, image.width, 4))
#     array = array[:, :, :3]
#     array = array[:, :, ::-1]
#     image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
#     if blend:
#         image_surface.set_alpha(100)
#     surface.blit(image_surface, (0, 0))
#
#
# def get_font():
#     fonts = [x for x in pygame.font.get_fonts()]
#     default_font = 'ubuntumono'
#     font = default_font if default_font in fonts else fonts[0]
#     font = pygame.font.match_font(font)
#     return pygame.font.Font(font, 14)
#
#
# def should_quit():
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             return True
#         elif event.type == pygame.KEYUP:
#             if event.key == pygame.K_ESCAPE:
#                 return True
#     return False
#
#
#
#
#
#
# def main():
#     actor_list = []
#     pygame.init()
#
#     display = pygame.display.set_mode(
#         (800, 600),
#         pygame.HWSURFACE | pygame.DOUBLEBUF)
#     font = get_font()
#     clock = pygame.time.Clock()
#
#     client = carla.Client('localhost', 2000)
#     client.set_timeout(2.0)
#
#     world = client.get_world()
#
#     try:
#         m = world.get_map()
#         start_pose = random.choice(m.get_spawn_points())
#         waypoint = m.get_waypoint(start_pose.location)
#
#         blueprint_library = world.get_blueprint_library()
#
#         my_vehicle_choice = random.choice(blueprint_library.filter('vehicle.bh.crossbike'))
#
#
#
#         vehicle = world.spawn_actor(
#             my_vehicle_choice,
#             start_pose)
#         actor_list.append(vehicle)
#         vehicle.set_simulate_physics(False)
#         bike = vehicle
#
#         second_start_pose = carla.Transform(
#             carla.Location(x=start_pose.location.x-10,
#                             y = start_pose.location.y, z = start_pose.location.z)
#             )
#
#         pedestrian_choice = random.choice(blueprint_library.filter('walker.pedestrian.014'))
#         vehicle = world.spawn_actor(
#             pedestrian_choice,
#             second_start_pose)
#         actor_list.append(vehicle)
#         vehicle.set_simulate_physics(False)
#
#         camera_rgb = world.spawn_actor(
#             blueprint_library.find('sensor.camera.rgb'),
#             carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
#             attach_to=vehicle)
#         actor_list.append(camera_rgb)
#
#         camera_semseg = world.spawn_actor(
#             blueprint_library.find('sensor.camera.semantic_segmentation'),
#             carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
#             attach_to=vehicle)
#         actor_list.append(camera_semseg)
#
#         # Create a synchronous mode context.
#         with CarlaSyncMode(world, camera_rgb, camera_semseg) as sync_mode:
#             while True:
#                 if should_quit():
#                     return
#                 clock.tick()
#
#                 # import pdb; pdb.set_trace()
#                 # Advance the simulation and wait for the data.
#                 snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=2.0)
#
#                 # # Choose the next waypoint and update the car location.
#                 # waypoint = random.choice(waypoint.next(1.5))
#                 # vehicle.set_transform(waypoint.transform)
#                 wrapper = ColorModifierWrapper(image_semseg)
#                 fps = 1.0 / snapshot.timestamp.delta_seconds
#
#                 # Draw the display.
#                 # draw_image(display, image_rgb)
#                 draw_image(display, wrapper.view_as_instance_seg(), blend=True)
#                 display.blit(
#                     font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
#                     (8, 10))
#                 display.blit(
#                     font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
#                     (8, 28))
#                 pygame.display.flip()
#
#     finally:
#
#         print('destroying actors.')
#         for actor in actor_list:
#             actor.destroy()
#
#         pygame.quit()
#         print('done.')
#
#
# if __name__ == '__main__':
#
#     try:
#
#         main()
#
#     except KeyboardInterrupt:
#         print('\nCancelled by user. Bye!')



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
from PIL import Image
import matplotlib.pyplot as plt

random.seed(1)
np.random.seed(1)


main_path = '/home/demo/carla_sam_newest/carla/Output/1_1/InstanceSegmentation/'
while True:
    result_ids = np.load(main_path+random.choice(os.listdir(main_path)))
    vis_results = result_ids
    # ids = list(set(result_ids.flatten().tolist()))
    # step = int(255.0/float(len(ids)))
    # vis_color = range(step, 256, step)
    # vis_results = np.zeros_like(result_ids)
    # for idx, cur_id in enumerate(ids):
    #     if cur_id == 0:
    #         continue
    #     vis_results[result_ids==cur_id] = vis_color[idx]
    plt.imshow(np.array(vis_results, dtype=np.int32)); plt.show()
