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
#include <unordered_map>

#include <std_msgs/Float32.h>
#include <nav_msgs/Path.h>

#include <image_transport/image_transport.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include "boost/thread.hpp"
#include "util/MinimalImage.h"
#include "IOWrapper/Output3DWrapper.h"

#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"

#include "./utils.h"

namespace dso
{

class FrameHessian;
class CalibHessian;
class FrameShell;


namespace IOWrap
{

class ROSOutputWrapper : public Output3DWrapper
{
 public:
  struct Params {
    Params() {}

    // Frames.
    std::string dso_cam_frame{"dso_cam"};
    std::string dso_world_frame{"dso_world"};
    std::string metric_cam_frame{"camera"};
    std::string metric_world_frame{"camera_world"};

    // Scale stuff.
    bool publish_metric_depthmap = true;

    float min_metric_inc_trans = 0.25f; // Camera must move this much in metric space to contribute to scale.
    float min_metric_trans = 2.0f;  // Camera must have move this much in metric space to contribute to scale.
    float max_scaled_trans_diff = 2.5f; // If scaled dso position is farther from metric than this, reset.
    float scale_divergence_factor = 0.10f; // If diff between estimated scale and live scale exceeds this, reset window.
    uint32_t max_pose_history = 200;
    float metric_takeoff_thresh = 1.5f;
    float max_angle_diff = 0.0f;

    bool publish_scaled_cloud = true;

    // Regularization stuff.
    bool publish_coarse_metric_depthmap = true;
    uint32_t coarse_level = 3;

    bool match_source_resolution = true;

    bool fill_holes = true;
    int fill_radius = 3;
    int min_depths_to_fill = 3; // Need this many depths in window to fill.

    bool do_morph_close = false;
    int morph_close_size = 5;

    bool do_median_filter = true;
    int median_filter_size = 5;
  };

  struct KeyFrame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    uint32_t id;
    double time;
    Eigen::Quaternion<double, Eigen::DontAlign> quat;
    Eigen::Vector3d trans;
    pcl::PointCloud<pcl::PointXYZ> cloud;
  };

  inline ROSOutputWrapper(const ros::NodeHandle& nh,
                          const Params& params = Params()):
      nh_(nh), params_(params)
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
            scaled_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("metric/scaled_cloud", 5);
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

        void publishKeyframes(std::vector<FrameHessian*> &frames,
                              bool final,
                              CalibHessian* HCalib) override {
          boost::lock_guard<boost::mutex> lock(pose_mtx_);

            // for(FrameHessian* f : frames) {
            //     printf("OUT: KF %d (%s) (id %d, tme %f): %d active, %d marginalized, %d immature points. CameraToWorld:\n",
            //            f->frameID,
            //            final ? "final" : "non-final",
            //            f->shell->incoming_id,
            //            f->shell->timestamp,
            //            (int)f->pointHessians.size(), (int)f->pointHessiansMarginalized.size(), (int)f->immaturePoints.size());
            //     std::cout << f->shell->camToWorld.matrix3x4() << "\n";


            //     int maxWrite = 5;
            //     for(PointHessian* p : f->pointHessians) {
            //         printf("OUT: Example Point x=%.1f, y=%.1f, idepth=%f, idepth std.dev. %f, %d inlier-residuals\n",
            //                p->u, p->v, p->idepth_scaled, sqrt(1.0f / p->idepth_hessian), p->numGoodResiduals );
            //         maxWrite--;
            //         if(maxWrite==0) break;
            //     }
            // }
        }

        void publishKeyframes(std::vector<FrameHessian*> &frames,
                              const std::vector<SE3, Eigen::aligned_allocator<SE3> >& poses,
                              bool final,
                              CalibHessian* HCalib) override {
          boost::lock_guard<boost::mutex> lock(pose_mtx_);
            if (params_.publish_scaled_cloud) {
              float scale = getScale();

              if (std::isnan(scale)) {
                return;
              }

              Eigen::Matrix3f K(Eigen::Matrix3f::Zero());
              K(0, 0) = HCalib->fxl();
              K(1, 1) = HCalib->fyl();
              K(0, 2) = HCalib->cxl();
              K(1, 2) = HCalib->cyl();
              K(2, 2) = 1.0f;
              Eigen::Matrix3f Kinv(K.inverse());

              // Create/update data for each active keyframe.
              for (int ii = 0; ii < frames.size(); ++ii) {
                const auto& frame = frames[ii];
                auto& kf = keyframes_[frame->shell->id];
                if (kf == nullptr) {
                  // Create new kf.
                  kf = std::make_shared<KeyFrame>();
                }

                kf->id = frame->shell->id;
                kf->time = frame->shell->timestamp;
                kf->quat = poses[ii].unit_quaternion();
                kf->trans = poses[ii].translation();
                kf->cloud.clear();

                // Create point cloud.
                for (PointHessian* p : frame->pointHessians) {
                  float idepth_var_max = 1; // TODO(wng): Make this a param.
                  float idepth_var = 1.0 / p->idepth_hessian;
                  if ((p->idepth > 0.0) && (idepth_var < idepth_var_max)) {
                    Eigen::Vector3d u_hom(p->u, p->v, 1.0f);
                    Eigen::Vector3d p_cam(Kinv.cast<double>() * u_hom / p->idepth);
                    kf->cloud.points.emplace_back(p_cam(0), p_cam(1), p_cam(2));
                  }
                }

                for (PointHessian* p : frame->pointHessiansMarginalized) {
                  float idepth_var_max = 1; // TODO(wng): Make this a param.
                  float idepth_var = 1.0 / p->idepth_hessian;
                  if ((p->idepth > 0.0) && (idepth_var < idepth_var_max)) {
                    Eigen::Vector3d u_hom(p->u, p->v, 1.0f);
                    Eigen::Vector3d p_cam(Kinv.cast<double>() * u_hom / p->idepth);
                    kf->cloud.points.emplace_back(p_cam(0), p_cam(1), p_cam(2));
                  }
                }

                kf->cloud.width = kf->cloud.points.size();
                kf->cloud.height = 1;
                kf->cloud.is_dense = false;

                keyframes_[frame->shell->id] = kf; // Overwrite just in case.
              }

              // Create full point cloud from all keyframes relative to last DSO
              // frame in the history which we then publish relative to the last
              // metric pose in the history (i.e. treat the cloud like a
              // depthmap).
              pcl::PointCloud<pcl::PointXYZ> cloud;

              // Weird alignment issues with Eigen::Quaternion going on, hence
              // the roundabout way of computing the points relative to the last
              // pose in the history.
              Eigen::Quaternion<double, Eigen::DontAlign> quat0(pose_history_.back().unit_quaternion());
              Eigen::Vector3d trans0(pose_history_.back().translation());
              Eigen::Quaterniond quat0inv(quat0.inverse());
              Eigen::Vector3d trans0inv(-(quat0inv * trans0));
              double time0 = pose_history_time_.back();

	      uint32_t num_close_pts = 0;	      
              for (const auto& kv : keyframes_) {
                const auto& kf = kv.second;
                for (int pidx = 0; pidx < kf->cloud.points.size(); ++pidx) {
                  Eigen::Vector3d p_cam(kf->cloud.points[pidx].x,
                                        kf->cloud.points[pidx].y,
                                        kf->cloud.points[pidx].z);

                  Eigen::Vector3d p_world(kf->quat * p_cam + kf->trans);
                  Eigen::Vector3d p0(quat0inv * p_world + trans0inv);
                  p0 *= scale;

		  const float close_ratio_dist = 5.0f;
		  if (p0.norm() < close_ratio_dist) {
		    num_close_pts++;
		  }
		  
                  cloud.points.emplace_back(p0(0), p0(1), p0(2));
                }
              }

              if (cloud.points.size() == 0) {
                return;
              }
	      
              cloud.width = cloud.points.size();
              cloud.height = 1;
              cloud.is_dense = false;

              sensor_msgs::PointCloud2::Ptr scaled_cloud_msg(new sensor_msgs::PointCloud2());
              pcl::toROSMsg(cloud, *scaled_cloud_msg);

              scaled_cloud_msg->header = std_msgs::Header();
              scaled_cloud_msg->header.stamp.fromSec(time0);
              scaled_cloud_msg->header.frame_id = params_.metric_cam_frame;

	      // If ratio of close points to far points exceeds
	      // threshold, don't publish because it might be a bad
	      // initialization.
	      const float max_close_ratio = 0.1f;
	      const float close_ratio = static_cast<float>(num_close_pts) / cloud.points.size();
	      if (close_ratio < max_close_ratio) {		
		scaled_cloud_pub_.publish(scaled_cloud_msg);			      				
	      } else {
		ROS_ERROR("[dso_ros] close_ratio failed (%f > %f), censoring output.",
			  close_ratio, max_close_ratio);
	      }
          }

          return;
        }

        void publishCamPose(const uint32_t id, const double time,
                            const SE3& pose, CalibHessian* HCalib) override {
          boost::lock_guard<boost::mutex> lock(pose_mtx_);

          geometry_msgs::TransformStamped tf;

          tf.header.stamp.fromSec(time);
          tf.header.frame_id = params_.dso_world_frame;
          tf.child_frame_id = params_.dso_cam_frame;
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
            metric_tf = tf_buffer_.lookupTransform(params_.metric_world_frame,
                                                   params_.metric_cam_frame,
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

          if ((-metric_pose.translation()(1) > params_.metric_takeoff_thresh) &&
              (metric_trans > params_.min_metric_inc_trans)) {
            pose_history_time_.push_back(time);
            pose_history_.push_back(pose);
            metric_pose_history_.push_back(metric_pose);

            while (pose_history_.size() > params_.max_pose_history) {
              pose_history_time_.pop_front();
              pose_history_.pop_front();
              metric_pose_history_.pop_front();
            }
          }

          // Check angle between current pose and last pose in history.
          float angle_diff = std::numeric_limits<float>::max();
          if (metric_pose_history_.size() > 0) {
            Eigen::Matrix4d curr_in_last(metric_pose_history_.back().matrix().inverse() *
                                         metric_pose.matrix());
            Eigen::AngleAxisd aa(curr_in_last.block<3, 3>(0, 0));
            angle_diff = std::fabs(aa.angle() * 180.0 / M_PI);
          }

          if (angle_diff > params_.max_angle_diff) {
            ROS_ERROR("ANGLE_DIFF: Exceeds threshold! Resetting history! (%f > %f)",
                      angle_diff, params_.max_angle_diff);
            metric_pose_history_.clear();
            pose_history_.clear();
            keyframes_.clear();
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

          dso_ros::publishDepthMap(depth_pub_, params_.dso_cam_frame, KF->shell->timestamp, K,
                                   depthmap);

          if (params_.publish_metric_depthmap) {
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

            dso_ros::publishDepthMap(metric_depth_pub_, params_.metric_cam_frame,
                                     KF->shell->timestamp, K, metric_depthmap);

            if (params_.publish_coarse_metric_depthmap) {
              // Create coarse depthmap.
              cv::Mat1f coarse_metric_depthmap =
                  getCoarseDepthmap(metric_depthmap, params_.coarse_level);
              Eigen::Matrix3f Klvl(K);
              Klvl /= (1 << params_.coarse_level);
              Klvl(2, 2) = 1.0f;

              if (params_.match_source_resolution) {
                std::vector<cv::Mat1f> upsampled_depthmaps(params_.coarse_level + 1);
                upsampled_depthmaps.back() = coarse_metric_depthmap.clone();
                for (int lvl = params_.coarse_level - 1; lvl >= 0; lvl--) {
                  cv::pyrUp(upsampled_depthmaps[lvl + 1], upsampled_depthmaps[lvl]);
                  // int wlvl = depthmap.cols >> lvl;
                  // int hlvl = depthmap.rows >> lvl;
                  // upsampled_depthmaps[lvl] =
                  //     cv::Mat1f(hlvl, wlvl, std::numeric_limits<float>::quiet_NaN());
                  // for (int ii = 0; ii < hlvl; ++ii) {
                  //   for (int jj = 0; jj < wlvl; ++jj) {
                  //     float depth = upsampled_depthmaps[lvl + 1](ii >> 1, jj >> 1);
                  //     if (!std::isnan(depth) && (depth > 0.0f)) {
                  //       upsampled_depthmaps[lvl](ii, jj) = depth;
                  //     }
                  //   }
                  // }
                }
                coarse_metric_depthmap = upsampled_depthmaps.front();
                Klvl = K;
              }

              dso_ros::publishDepthMap(coarse_metric_depth_pub_, params_.metric_cam_frame,
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
            metric_pose_history_msg->header.frame_id = params_.metric_world_frame;
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
            scaled_pose_history_msg->header.frame_id = params_.metric_world_frame;
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

    if (total_metric_trans < params_.min_metric_trans) {
      ROS_ERROR("Not enough metric translation! (%f < %f)",
                total_metric_trans, params_.min_metric_trans);
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

    // Convert last pose to metric frame.
    Eigen::Quaterniond quatn(quat0inv * pose_history_.back().unit_quaternion());
    Eigen::Vector3d transn(quat0inv * pose_history_.back().translation() + trans0inv);
    transn *= scale;

    Eigen::Quaterniond scaled_quat(
        metric_pose_history_.front().unit_quaternion() * quatn);
    Eigen::Vector3d scaled_trans(
        metric_pose_history_.front().unit_quaternion() * transn +
        metric_pose_history_.front().translation());
    float trans_diff = (scaled_trans - metric_pose_history_.back().translation()).norm();

    if (scale <= 0.0f) {
      ROS_ERROR("Negative scale estimated (%f)! Resetting DSO!", scale);
      pose_history_.clear();
      metric_pose_history_.clear();
      keyframes_.clear();
      scale = std::numeric_limits<float>::quiet_NaN();
      setting_fullResetRequested = true;
    } else if (std::fabs(live_scale - scale) / scale > params_.scale_divergence_factor) {
      ROS_ERROR("Scale divergence! Resetting history!");
      pose_history_.clear();
      keyframes_.clear();
      metric_pose_history_.clear();
      scale = std::numeric_limits<float>::quiet_NaN();
    } else if (trans_diff > params_.max_scaled_trans_diff) {
      ROS_ERROR("Trans diff exceeds threshold ! (%f > %f)! Resetting DSO!",
                trans_diff, params_.max_scaled_trans_diff);
      pose_history_.clear();
      metric_pose_history_.clear();
      keyframes_.clear();
      scale = std::numeric_limits<float>::quiet_NaN();
      setting_fullResetRequested = true;
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
    if (params_.fill_holes) {
      cv::Mat1f filled_depthmap(coarse_depthmap.clone());
      for (int ii = params_.fill_radius; ii < coarse_depthmap.rows - params_.fill_radius; ++ii) {
        for (int jj = params_.fill_radius; jj < coarse_depthmap.cols - params_.fill_radius; ++jj) {
          float depth_center = coarse_depthmap(ii, jj);
          if (!std::isnan(depth_center)) {
            continue;
          }

          float depth_sum = 0.0f;
          int depth_count = 0;
          for (int wii = -params_.fill_radius; wii <= params_.fill_radius; ++wii) {
            for (int wjj = -params_.fill_radius; wjj <= params_.fill_radius; ++wjj) {
              float depth = coarse_depthmap(ii + wii, jj + wjj);
              if (!std::isnan(depth)) {
                depth_sum += depth;
                depth_count++;
              }
            }
          }

          if (depth_count > params_.min_depths_to_fill) {
            filled_depthmap(ii, jj) = depth_sum / depth_count;
          }
        }
      }

      coarse_depthmap = filled_depthmap;
    }

    if (params_.do_morph_close) {
      // Apply closing operator to connect fragmented components.
      cv::Mat struct_el(params_.morph_close_size, params_.morph_close_size,
                        cv::DataType<uint8_t>::type, cv::Scalar(1));
      cv::morphologyEx(coarse_depthmap, coarse_depthmap, cv::MORPH_CLOSE, struct_el);
    }

    if (params_.do_median_filter) {
      cv::medianBlur(coarse_depthmap, coarse_depthmap, params_.median_filter_size);
    }

    return coarse_depthmap;
  }

 private:
  boost::mutex pose_mtx_;

  ros::NodeHandle nh_;
  Params params_;

  std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformBroadcaster tf_pub_;

  std::shared_ptr<image_transport::ImageTransport> it_;

  image_transport::CameraPublisher depth_pub_;

  // Scale estimation stuff.
  image_transport::CameraPublisher metric_depth_pub_;
  image_transport::CameraPublisher coarse_metric_depth_pub_;

  ros::Publisher scale_pub_;
  ros::Publisher scaled_pose_history_pub_;
  ros::Publisher metric_pose_history_pub_;

  ros::Publisher scaled_cloud_pub_;

  std::deque<double> pose_history_time_; // Timestamps in pose history.
  std::deque<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d> > pose_history_;
  std::deque<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d> > metric_pose_history_;

  using KeyFrameMap = std::unordered_map<uint32_t, std::shared_ptr<KeyFrame> >;
  KeyFrameMap keyframes_;

  CalibHessian* calib_hessian_ = nullptr;
};

}

}
