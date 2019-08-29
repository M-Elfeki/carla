// Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "carla/MsgPack.h"
#include "carla/geom/Vector3D.h"
#include <utility>
#include <vector>

namespace carla {
namespace rpc {

  class Keypoints {
  // The keypoints are just a vector of pairs <string, Vector3D>
  public:

    Keypoints() = default;

    std::vector<std::pair<std::string, geom::Vector3D>> keypoints;

    MSGPACK_DEFINE_ARRAY(keypoints)

  };

} // namespace rpc
} // namespace carla
