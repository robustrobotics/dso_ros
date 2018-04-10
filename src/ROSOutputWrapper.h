/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once

#include <string>
#include <limits>
#include <deque>
#include <vector>

#include <std_msgs/Float32.h>
#include <nav_msgs/Path.h>

#include <image_transport/image_transport.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <cv_bridge/cv_bridge.h>

#include "boost/thread.hpp"
#include "util/MinimalImage.h"
#include "IOWrapper/Output3DWrapper.h"



#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"

namespace dso
{

class FrameHessian;
class CalibHessian;
class FrameShell;


namespace IOWrap
{

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

class ROSOutputWrapper : public Output3DWrapper
{
public:
  inline ROSOutputWrapper(const ros::NodeHandle& nh):
      nh_(nh)
        {
            printf("OUT: Created ROSOutputWrapper\n");

            tf_listener_ = std::make_shared<tf2_ros::TransformListener>(tf_buffer_);
            it_ = std::make_shared<image_transport::ImageTransport>(nh);
            depth_pub_ = it_->advertiseCamera("depth_registered/image_rect", 5);
            metric_depth_pub_ = it_->advertiseCamera("metric/depth_registered/image_rect", 5);
            coarse_metric_depth_pub_ = it_->advertiseCamera("metric/coarse/depth_registered/image_rect", 5);
            scale_pub_ = nh_.advertise<std_msgs::Float32>("metric/scale", 5);
            scaled_pose_history_pub_ = nh_.advertise<nav_msgs::Path>("metric/scaled_pose_history", 5);
            metric_pose_history_pub_ = nh_.advertise<nav_msgs::Path>("metric/pose_history", 5);
        }

        virtual ~ROSOutputWrapper()
        {
            printf("OUT: Destroyed ROSOutputWrapper\n");
        }

        virtual void publishGraph(const std::map<uint64_t, Eigen::Vector2i, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> &connectivity) override
        {
            // printf("OUT: got graph with %d edges\n", (int)connectivity.size());

            // int maxWrite = 5;

            // for(const std::pair<uint64_t,Eigen::Vector2i> &p : connectivity)
            // {
            //     int idHost = p.first>>32;
            //     int idTarget = p.first & ((uint64_t)0xFFFFFFFF);
            //     printf("OUT: Example Edge %d -> %d has %d active and %d marg residuals\n", idHost, idTarget, p.second[0], p.second[1]);
            //     maxWrite--;
            //     if(maxWrite==0) break;
            // }
        }



        virtual void publishKeyframes( std::vector<FrameHessian*> &frames, bool final, CalibHessian* HCalib) override
        {
            // for(FrameHessian* f : frames)
            // {
            //     printf("OUT: KF %d (%s) (id %d, tme %f): %d active, %d marginalized, %d immature points. CameraToWorld:\n",
            //            f->frameID,
            //            final ? "final" : "non-final",
            //            f->shell->incoming_id,
            //            f->shell->timestamp,
            //            (int)f->pointHessians.size(), (int)f->pointHessiansMarginalized.size(), (int)f->immaturePoints.size());
            //     std::cout << f->shell->camToWorld.matrix3x4() << "\n";


            //     int maxWrite = 5;
            //     for(PointHessian* p : f->pointHessians)
            //     {
            //         printf("OUT: Example Point x=%.1f, y=%.1f, idepth=%f, idepth std.dev. %f, %d inlier-residuals\n",
            //                p->u, p->v, p->idepth_scaled, sqrt(1.0f / p->idepth_hessian), p->numGoodResiduals );
            //         maxWrite--;
            //         if(maxWrite==0) break;
            //     }
            // }
        }

        void publishCamPose(const uint32_t id, const double time,
                            const SE3& pose, CalibHessian* HCalib) override {
          boost::lock_guard<boost::mutex> lock(pose_mtx_);

          geometry_msgs::TransformStamped tf;

          tf.header.stamp.fromSec(time);
          tf.header.frame_id = dso_world_frame_;
          tf.child_frame_id = dso_cam_frame_;
          tf.transform.rotation.w = pose.unit_quaternion().w();
          tf.transform.rotation.x = pose.unit_quaternion().x();
          tf.transform.rotation.y = pose.unit_quaternion().y();
          tf.transform.rotation.z = pose.unit_quaternion().z();
          tf.transform.translation.x = pose.translation()(0);
          tf.transform.translation.y = pose.translation()(1);
          tf.transform.translation.z = pose.translation()(2);
          tf_pub_.sendTransform(tf);

          // Save calibration.
          calib_hessian_ = HCalib;

          // Get corresponding metric camera pose.
          geometry_msgs::TransformStamped metric_tf;
          try {
            metric_tf = tf_buffer_.lookupTransform(metric_world_frame_,
                                                   metric_cam_frame_,
                                                   tf.header.stamp,
                                                   ros::Duration(1.0/15));
          } catch (tf2::TransformException &ex) {
            ROS_ERROR("%s", ex.what());
            return;
          }

          Sophus::SE3d metric_pose(
              Eigen::Quaterniond(metric_tf.transform.rotation.w,
                                 metric_tf.transform.rotation.x,
                                 metric_tf.transform.rotation.y,
                                 metric_tf.transform.rotation.z),
              Eigen::Vector3d(metric_tf.transform.translation.x,
                              metric_tf.transform.translation.y,
                              metric_tf.transform.translation.z));

          // Add to history (make sure after takeoff).
          float metric_trans = std::numeric_limits<float>::max();
          if (metric_pose_history_.size() > 0) {
            metric_trans = (metric_pose_history_.back().translation() -
                            metric_pose.translation()).norm();
          }

          if ((-metric_pose.translation()(1) > metric_takeoff_thresh_) &&
              (metric_trans > min_metric_inc_trans_)) {
            pose_history_.push_back(pose);
            metric_pose_history_.push_back(metric_pose);

            while (pose_history_.size() > max_pose_history_) {
              pose_history_.pop_front();
              metric_pose_history_.pop_front();
            }
          }

          return;
        }

        virtual void pushLiveFrame(FrameHessian* image) override
        {
            // can be used to get the raw image / intensity pyramid.
        }

        virtual void pushDepthImage(MinimalImageB3* image) override
        {
            // can be used to get the raw image with depth overlay.
        }
        virtual bool needPushDepthImage() override
        {
            return false;
        }

        virtual void pushDepthImageFloat(MinimalImageF* image, FrameHessian* KF ) override
        {
          boost::lock_guard<boost::mutex> lock(pose_mtx_);

            // printf("OUT: Predicted depth for KF %d (id %d, time %f, internal frame-ID %d). CameraToWorld:\n",
            //        KF->frameID,
            //        KF->shell->incoming_id,
            //        KF->shell->timestamp,
            //        KF->shell->id);
            // std::cout << KF->shell->camToWorld.matrix3x4() << "\n";

            // int maxWrite = 5;
            // for(int y=0;y<image->h;y++)
            // {
            //     for(int x=0;x<image->w;x++)
            //     {
            //         if(image->at(x,y) <= 0) continue;

            //         printf("OUT: Example Idepth at pixel (%d,%d): %f.\n", x,y,image->at(x,y));
            //         maxWrite--;
            //         if(maxWrite==0) break;
            //     }
            //     if(maxWrite==0) break;
            // }

          if (calib_hessian_ == nullptr) {
            return;
          }

          cv::Mat1f depthmap(image->h, image->w, std::numeric_limits<float>::quiet_NaN());
          for (int ii = 0; ii < image->h; ++ii) {
            for (int jj = 0; jj < image->w; ++jj) {
              if (image->at(jj, ii) <= 0) {
                continue;
              }
              depthmap(ii, jj) = 1.0f / image->at(jj, ii);
            }
          }

          Eigen::Matrix3f K(Eigen::Matrix3f::Zero());
          K(0, 0) = calib_hessian_->fxl();
          K(1, 1) = calib_hessian_->fyl();
          K(0, 2) = calib_hessian_->cxl();
          K(1, 2) = calib_hessian_->cyl();
          K(2, 2) = 1.0f;

          publishDepthMap(depth_pub_, dso_cam_frame_, KF->shell->timestamp, K,
                          depthmap);

          if (publish_metric_depthmap_) {
            cv::Mat1f metric_depthmap(depthmap.clone());
            float scale = getScale();

            if (std::isnan(scale)) {
              return;
            }

            for (int ii = 0; ii < metric_depthmap.rows; ++ii) {
              for (int jj = 0; jj < metric_depthmap.cols; ++jj) {
                metric_depthmap(ii, jj) *= scale;
              }
            }

            publishDepthMap(metric_depth_pub_, metric_cam_frame_,
                            KF->shell->timestamp, K, metric_depthmap);

            if (publish_coarse_metric_depthmap_) {
              // Create coarse depthmap.
              cv::Mat1f coarse_metric_depthmap =
                  getCoarseDepthmap(metric_depthmap, coarse_level_);
              Eigen::Matrix3f Klvl(K);
              Klvl /= (1 << coarse_level_);
              Klvl(2, 2) = 1.0f;
              publishDepthMap(coarse_metric_depth_pub_, metric_cam_frame_,
                              KF->shell->timestamp, Klvl,
                              coarse_metric_depthmap);
            }

            // Publish scale.
            std_msgs::Float32::Ptr float_msg(new std_msgs::Float32());
            float_msg->data = scale;
            scale_pub_.publish(float_msg);

            // Publish pose history.
            nav_msgs::Path::Ptr metric_pose_history_msg(new nav_msgs::Path());
            metric_pose_history_msg->header.stamp.fromSec(KF->shell->timestamp);
            metric_pose_history_msg->header.frame_id = metric_world_frame_;
            metric_pose_history_msg->poses.resize(metric_pose_history_.size());
            for (int ii = 0; ii < metric_pose_history_.size(); ++ii) {
              metric_pose_history_msg->poses[ii].pose.position.x =
                  metric_pose_history_[ii].translation()(0);
              metric_pose_history_msg->poses[ii].pose.position.y =
                  metric_pose_history_[ii].translation()(1);
              metric_pose_history_msg->poses[ii].pose.position.z =
                  metric_pose_history_[ii].translation()(2);

              metric_pose_history_msg->poses[ii].pose.orientation.w =
                  metric_pose_history_[ii].unit_quaternion().w();
              metric_pose_history_msg->poses[ii].pose.orientation.x =
                  metric_pose_history_[ii].unit_quaternion().x();
              metric_pose_history_msg->poses[ii].pose.orientation.y =
                  metric_pose_history_[ii].unit_quaternion().y();
              metric_pose_history_msg->poses[ii].pose.orientation.z =
                  metric_pose_history_[ii].unit_quaternion().z();
            }
            metric_pose_history_pub_.publish(metric_pose_history_msg);

            nav_msgs::Path::Ptr scaled_pose_history_msg(new nav_msgs::Path());
            scaled_pose_history_msg->header.stamp.fromSec(KF->shell->timestamp);
            scaled_pose_history_msg->header.frame_id = metric_world_frame_;
            scaled_pose_history_msg->poses.resize(pose_history_.size());

            // Weird alignment issues with Eigen::Quaternion going on, hence the
            // roundabout way of computing the poses relative to the first pose in the
            // history.
            Eigen::Quaternion<double, Eigen::DontAlign> quat0(pose_history_.front().unit_quaternion());
            Eigen::Vector3d trans0(pose_history_.front().translation());
            Eigen::Quaterniond quat0inv(quat0.inverse());
            Eigen::Vector3d trans0inv(-(quat0inv * trans0));
            for (int ii = 0; ii < pose_history_.size(); ++ii) {
              // Convert to metric frame.
              Eigen::Quaterniond quat(quat0inv * pose_history_[ii].unit_quaternion());
              Eigen::Vector3d trans(quat0inv * pose_history_[ii].translation() + trans0inv);
              trans *= scale;

              Eigen::Quaterniond scaled_quat(
                  metric_pose_history_.front().unit_quaternion() * quat);
              Eigen::Vector3d scaled_trans(
                  metric_pose_history_.front().unit_quaternion() * trans +
                  metric_pose_history_.front().translation());

              scaled_pose_history_msg->poses[ii].pose.position.x = scaled_trans(0);
              scaled_pose_history_msg->poses[ii].pose.position.y = scaled_trans(1);
              scaled_pose_history_msg->poses[ii].pose.position.z = scaled_trans(2);

              scaled_pose_history_msg->poses[ii].pose.orientation.w = scaled_quat.w();
              scaled_pose_history_msg->poses[ii].pose.orientation.x = scaled_quat.x();
              scaled_pose_history_msg->poses[ii].pose.orientation.y = scaled_quat.y();
              scaled_pose_history_msg->poses[ii].pose.orientation.z = scaled_quat.z();
            }
            scaled_pose_history_pub_.publish(scaled_pose_history_msg);
          }

          return;
        }

  float getScale() {
    float scale = std::numeric_limits<float>::quiet_NaN();

    ROS_ASSERT(pose_history_.size() == metric_pose_history_.size());

    int num_poses = pose_history_.size();

    float total_metric_trans = 0.0f;
    if (num_poses > 1) {
      total_metric_trans = (metric_pose_history_.back().translation() -
                            metric_pose_history_.front().translation()).norm();
    }

    if (total_metric_trans < min_metric_trans_) {
      ROS_ERROR("Not enough metric translation! (%f < %f)",
                total_metric_trans, min_metric_trans_);
      return std::numeric_limits<float>::quiet_NaN();
    }

    // Solve least squares to estimate scale.
    Eigen::VectorXf trans(num_poses * 3);
    Eigen::VectorXf metric_trans(num_poses * 3);

    // Weird alignment issues with Eigen::Quaternion going on, hence the
    // roundabout way of computing the poses relative to the first pose in the
    // history.
    Eigen::Quaternion<double, Eigen::DontAlign> quat0(pose_history_.front().unit_quaternion());
    Eigen::Vector3d trans0(pose_history_.front().translation());
    Eigen::Quaterniond quat0inv(quat0.inverse());
    Eigen::Vector3d trans0inv(-(quat0inv * trans0));

    Eigen::Quaternion<double, Eigen::DontAlign>
        metric_quat0(metric_pose_history_.front().unit_quaternion());
    Eigen::Vector3d metric_trans0(metric_pose_history_.front().translation());
    Eigen::Quaterniond metric_quat0inv(metric_quat0.inverse());
    Eigen::Vector3d metric_trans0inv(-(metric_quat0inv * metric_trans0));
    for (int ii = 0; ii < num_poses; ++ii) {
      Eigen::Vector3d rel_trans(quat0inv * pose_history_[ii].translation() + trans0inv);
      trans(3 * ii + 0) = rel_trans(0);
      trans(3 * ii + 1) = rel_trans(1);
      trans(3 * ii + 2) = rel_trans(2);

      Eigen::Vector3d metric_rel_trans(
          metric_quat0inv * metric_pose_history_[ii].translation() + metric_trans0inv);
      metric_trans(3 * ii + 0) = metric_rel_trans(0);
      metric_trans(3 * ii + 1) = metric_rel_trans(1);
      metric_trans(3 * ii + 2) = metric_rel_trans(2);
    }

    scale = metric_trans.dot(trans) / trans.dot(trans);

    float live_scale = metric_trans.tail<3>().dot(trans.tail<3>()) /
        trans.tail<3>().dot(trans.tail<3>());

    ROS_ERROR("LIVE_SCALE(%i): %f", num_poses, live_scale);
    ROS_ERROR("SCALE(%i): %f", num_poses, scale);

    if (scale <= 0.0f) {
      ROS_ERROR("Negative scale estimated (%f)! Resetting history!", scale);
      pose_history_.clear();
      metric_pose_history_.clear();
      scale = std::numeric_limits<float>::quiet_NaN();
    } else if (std::fabs(live_scale - scale) / scale > scale_divergence_factor_) {
      ROS_ERROR("Scale divergence! Resetting history!");
      pose_history_.clear();
      metric_pose_history_.clear();
      scale = std::numeric_limits<float>::quiet_NaN();
    }

    return scale;
  }

  cv::Mat1f getCoarseDepthmap(const cv::Mat1f& depthmap,
                              const int level) {
    std::vector<cv::Mat1f> depthmap_pyr(level + 1);
    depthmap_pyr[0] = depthmap.clone();

    for (int lvl = 1; lvl <= level; ++lvl) {
      int hlvl = depthmap_pyr[lvl - 1].rows >> 1;
      int wlvl = depthmap_pyr[lvl - 1].cols >> 1;
      depthmap_pyr[lvl] = cv::Mat1f(hlvl, wlvl, std::numeric_limits<float>::quiet_NaN());
      for (int ii = 0; ii < hlvl; ++ii) {
        for (int jj = 0; jj < wlvl; ++jj) {
          int ii_prev = ii << 1;
          int jj_prev = jj << 1;

          float depth_sum = 0.0f;
          int depth_count = 0;

          float depth = depthmap_pyr[lvl - 1](ii_prev, jj_prev);
          if (!std::isnan(depth)) {
            depth_sum += depth;
            depth_count++;
          }

          depth = depthmap_pyr[lvl - 1](ii_prev, jj_prev + 1);
          if (!std::isnan(depth)) {
            depth_sum += depth;
            depth_count++;
          }

          depth = depthmap_pyr[lvl - 1](ii_prev + 1, jj_prev);
          if (!std::isnan(depth)) {
            depth_sum += depth;
            depth_count++;
          }

          depth = depthmap_pyr[lvl - 1](ii_prev + 1, jj_prev + 1);
          if (!std::isnan(depth)) {
            depth_sum += depth;
            depth_count++;
          }

          if (depth_count > 0) {
            depthmap_pyr[lvl](ii, jj) = depth_sum / depth_count;
          }
        }
      }
    }

    cv::Mat1f coarse_depthmap = depthmap_pyr.back();
    if (fill_holes_) {
      cv::Mat1f filled_depthmap(coarse_depthmap.clone());
      for (int ii = fill_radius_; ii < coarse_depthmap.rows - fill_radius_; ++ii) {
        for (int jj = fill_radius_; jj < coarse_depthmap.cols - fill_radius_; ++jj) {
          float depth_center = coarse_depthmap(ii, jj);
          if (!std::isnan(depth_center)) {
            continue;
          }

          float depth_sum = 0.0f;
          int depth_count = 0;
          for (int wii = -fill_radius_; wii <= fill_radius_; ++wii) {
            for (int wjj = -fill_radius_; wjj <= fill_radius_; ++wjj) {
              float depth = coarse_depthmap(ii + wii, jj + wjj);
              if (!std::isnan(depth)) {
                depth_sum += depth;
                depth_count++;
              }
            }
          }

          if (depth_count > min_depths_to_fill) {
            filled_depthmap(ii, jj) = depth_sum / depth_count;
          }
        }
      }

      coarse_depthmap = filled_depthmap;
    }

    if (do_morph_close_) {
      // Apply closing operator to connect fragmented components.
      cv::Mat struct_el(morph_close_size_, morph_close_size_,
                        cv::DataType<uint8_t>::type, cv::Scalar(1));
      cv::morphologyEx(coarse_depthmap, coarse_depthmap, cv::MORPH_CLOSE, struct_el);
    }

    return coarse_depthmap;
  }

 private:
  boost::mutex pose_mtx_;

  ros::NodeHandle nh_;

  std::string dso_cam_frame_{"dso_cam"};
  std::string dso_world_frame_{"dso_world"};

  std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformBroadcaster tf_pub_;

  std::shared_ptr<image_transport::ImageTransport> it_;

  image_transport::CameraPublisher depth_pub_;

  // Scale estimation stuff.
  bool publish_metric_depthmap_ = true;
  image_transport::CameraPublisher metric_depth_pub_;

  bool publish_coarse_metric_depthmap_ = true;
  uint32_t coarse_level_ = 3;
  image_transport::CameraPublisher coarse_metric_depth_pub_;

  bool fill_holes_ = true;
  int fill_radius_ = 3;
  int min_depths_to_fill = 3; // Need this many depths in window to fill.

  bool do_morph_close_ = false;
  int morph_close_size_ = 5;

  ros::Publisher scale_pub_;
  ros::Publisher scaled_pose_history_pub_;
  ros::Publisher metric_pose_history_pub_;

  std::deque<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d> > pose_history_;
  std::deque<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d> > metric_pose_history_;

  float min_metric_inc_trans_ = 0.25f; // Camera must move this much in metric space to contribute to scale.
  float min_metric_trans_ = 2.0f;  // Camera must have move this much in metric space to contribute to scale.
  float scale_divergence_factor_ = 0.10f; // If diff between estimated scale and live scale exceeds this, reset window.
  uint32_t max_pose_history_ = 200;
  std::string metric_cam_frame_{"camera"};
  std::string metric_world_frame_{"camera_world"};
  float metric_takeoff_thresh_ = 1.5f;

  CalibHessian* calib_hessian_ = nullptr;
};

}

}
