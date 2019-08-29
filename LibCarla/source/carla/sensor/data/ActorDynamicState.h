// Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "carla/geom/Transform.h"
#include "carla/geom/Vector3D.h"
#include "carla/rpc/ActorId.h"
#include "carla/rpc/TrafficLightState.h"
#include "carla/rpc/VehicleControl.h"
#include "carla/rpc/WalkerControl.h"
#include "carla/rpc/Keypoints.h"

#include <cstdint>
#include <stdio.h>

namespace carla {
namespace sensor {
namespace data {

namespace detail {

#pragma pack(push, 1)
  class PackedVehicleControl {
  public:

    PackedVehicleControl() = default;

    PackedVehicleControl(const rpc::VehicleControl &control)
      : throttle(control.throttle),
        steer(control.steer),
        brake(control.brake),
        hand_brake(control.hand_brake),
        reverse(control.reverse),
        manual_gear_shift(control.manual_gear_shift),
        gear(control.gear) {}

    operator rpc::VehicleControl() const {
      return {throttle, steer, brake, hand_brake, reverse, manual_gear_shift, gear};
    }

  private:

    float throttle;
    float steer;
    float brake;
    bool hand_brake;
    bool reverse;
    bool manual_gear_shift;
    int32_t gear;
  };

#pragma pack(pop)

#pragma pack(push, 1)
  struct VehicleData {
    VehicleData() = default;

    PackedVehicleControl control;
    float speed_limit;
    rpc::TrafficLightState traffic_light_state;
    bool has_traffic_light;
    rpc::ActorId traffic_light_id;
  };
#pragma pack(pop)

#pragma pack(push, 1)
  class PackedWalkerControl {
  public:

    PackedWalkerControl() = default;

    PackedWalkerControl(const rpc::WalkerControl &control)
      : direction{control.direction.x, control.direction.y, control.direction.z},
        speed(control.speed),
        jump(control.jump) {}

    operator rpc::WalkerControl() const {
      return {geom::Vector3D{direction[0u], direction[1u], direction[2u]}, speed, jump};
    }

  private:

    float direction[3u];
    float speed;
    bool jump;
  };

#pragma pack(pop)

#pragma pack(push, 1)
  struct TrafficLightData {
    TrafficLightData() = default;

    rpc::TrafficLightState state;
    float green_time;
    float yellow_time;
    float red_time;
    float elapsed_time;
    bool time_is_frozen;
    uint32_t pole_index;
  };
#pragma pack(pop)

#pragma pack(push, 1)
  struct PackedKeypoint {
    // Helper structure that stores the name of a bone
    // (up to 24 chars) and its position (a float 3)
    PackedKeypoint() = default;

    char name[24];
    float location[3];
  };
#pragma pack(pop)

#pragma pack(push, 1)
struct PackedKeypoints {
  // Helper class that stores up to 72 keypoints (yeah it's quite heavy,
  // This is 2kB right there...)

  PackedKeypoints() = default;
  PackedKeypoint keypoints[72];
  int count=0;


  void addKeypoint(std::string name, float x, float y, float z){
      if (count >= 72){
          return;
      }
      strncpy(keypoints[count].name, name.c_str(), sizeof(keypoints[count]));
      keypoints[count].location[0] = x;
      keypoints[count].location[1] = y;
      keypoints[count].location[2] = z;
      count ++;
  }

  operator rpc::Keypoints() const {
      rpc::Keypoints kps;
      for (int i=0; i<count; i++)
      {
          geom::Vector3D v(keypoints[i].location[0], keypoints[i].location[1],
                     keypoints[i].location[2]);
          std::string name = std::string(keypoints[i].name);
          std::pair<std::string, geom::Vector3D> pair1(name, v);
          kps.keypoints.push_back(pair1);
      }
    return kps;
  };

};
#pragma pack(pop)


} // namespace detail

#pragma pack(push, 1)

  /// Dynamic state of an actor at a certain frame.
  struct ActorDynamicState {

    ActorId id;

    geom::Transform transform;

    geom::Vector3D velocity;

    geom::Vector3D angular_velocity;

    geom::Vector3D acceleration;

    union TypeDependentState {
      detail::TrafficLightData traffic_light_data;
      detail::VehicleData vehicle_data;
      detail::PackedWalkerControl walker_control;
    } state;

    detail::PackedKeypoints keypoints;

  };

#pragma pack(pop)

// static_assert(
//     sizeof(ActorDynamicState) == 204u,
//     "Invalid ActorDynamicState size! "
//     "If you modified this class please update the size here, else you may "
//     "comment this assert, but your platform may have compatibility issues "
//     "connecting to other platforms.");

} // namespace data
} // namespace sensor
} // namespace carla
