/**
 * Copyright 2018 Massachusetts Institute of Technology
 *
 * @file utils.h
 * @author W. Nicholas Greene
 * @date 2018-04-17 11:38:36 (Tue)
 */

#pragma once

#include <string>

#include <Eigen/Core>

#include <opencv2/core/core.hpp>

#include <ros/ros.h>

#include <image_transport/image_transport.h>

namespace dso_ros {

/**
 * @brief Find a parameter or fail.
 *
 * Copied from fsw/fla_utils/param_utils.h.
 */
template <typename T>
void getParamOrFail(const ros::NodeHandle& nh, const std::string& name, T* val) {
  if (!nh.getParam(name, *val)) {
    ROS_ERROR("Failed to find parameter: %s", nh.resolveName(name, true).c_str());
    exit(1);
  }
  return;
}

template <>
inline void getParamOrFail(const ros::NodeHandle& nh, const std::string& name, uint32_t* val) {
  int tmp;
  if (!nh.getParam(name, tmp)) {
    ROS_ERROR("Failed to find parameter: %s", nh.resolveName(name, true).c_str());
    exit(1);
  }
  *val = tmp;
  return;
}

void publishDepthMap(const image_transport::CameraPublisher& pub,
                     const std::string& frame_id,
                     double time, const Eigen::Matrix3f& K,
                     const cv::Mat1f& depth_est) {
  // Publish depthmap.
  std_msgs::Header header;
  header.stamp.fromSec(time);
  header.frame_id = frame_id;

  sensor_msgs::CameraInfo::Ptr cinfo(new sensor_msgs::CameraInfo);
  cinfo->header = header;
  cinfo->height = depth_est.rows;
  cinfo->width = depth_est.cols;
  cinfo->distortion_model = "plumb_bob";
  cinfo->D = {0.0, 0.0, 0.0, 0.0, 0.0};
  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      cinfo->K[ii*3 + jj] = K(ii, jj);
      cinfo->P[ii*4 + jj] = K(ii, jj);
      cinfo->R[ii*3 + jj] = 0.0;
    }
  }
  cinfo->P[3] = 0.0;
  cinfo->P[7] = 0.0;
  cinfo->P[11] = 0.0;
  cinfo->R[0] = 1.0;
  cinfo->R[4] = 1.0;
  cinfo->R[8] = 1.0;

  cv_bridge::CvImage depth_cvi(header, "32FC1", depth_est);

  pub.publish(depth_cvi.toImageMsg(), cinfo);

  return;
}

}  // namespace dso_ros
