CARLA
=====



Building CARLA
--------------

Use `git clone` or download the project from this page. Note that the master
branch contains the latest fixes and features, for the latest stable code may be
best to switch to the `stable` branch.

Then follow the instruction at [How to build on Linux][buildlinuxlink] or
[How to build on Windows][buildwindowslink].

Unfortunately we don't have official instructions to build on Mac yet, please
check the progress at [issue #150][issue150].

[buildlinuxlink]: http://carla.readthedocs.io/en/latest/how_to_build_on_linux
[buildwindowslink]: http://carla.readthedocs.io/en/latest/how_to_build_on_windows
[issue150]: https://github.com/carla-simulator/carla/issues/150


Simulation Code
---------------

Since we aim at representing the state with various modalities, we provided different sensors to the simulation that are recorded simultaneously. The simulation code is runs from "PythonAPI/examples/bb_spawn_npc.py"

During the simulation, cameras are calibrated with the same internsic parameters as in Cityscapes dataset, and three main agent categories are spawned: 1. vehicles, 2. bikes/bicycles and 3. pedestrians. When we run the simulation, we randomly choose 50 spawned vehicles and assign cameras to them so they record their surrounding.

For each of the agents there are two types of data: relative and absolute. Absolute data are their category, id and position and orientation information relative to an absolute frame of reference that is generalized for all agents (corresponding to GPS positions). However, relative data are the position, orientation, bounding boxes and keypoints relative to the position of the capturing camera. In this case the same agent can be captured by different cameras having different relative locations but the same absolute one in all of them.

We introduced three major modifications in Carla base. The first is retrieving the keypoints for all agents, which we did by modifying the skeleton interface and introducing as a feature in class "vehicle". The second modification was introducing instance segmentation on top of the existed semantic segmentation. This was done by retrieving object ids and modifying the semantic segmentation wrapper to act as both semantic and instance segmentation sensor. The final modification involves making the motion of pedestrians more realistic. Since walkers are not running autonomously in Carla, we modified the spawn wrapper to introduce spawn points for walkers. Then we initialized a random fixed speed for each walker with a direction that is smartly aligns with the sidewalks within the simulation.

Thus far, when running the simulation, we record both relative and absolute information for each object. We also record the camera information for each relative item. Finally we store the sensors data (rgb, depth, semantic segmentation, lidar and absolute and relative position information) in queues, where they are stored in the desk after the simulation ends to avoid introducing unstability or time-lags to the simulation while running.
