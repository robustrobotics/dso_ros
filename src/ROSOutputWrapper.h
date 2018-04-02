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

#include <image_transport/image_transport.h>
#include <tf2_ros/transform_broadcaster.h>
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

            it_ = std::make_shared<image_transport::ImageTransport>(nh);
            depth_pub_ = it_->advertiseCamera("depth_registered/image_rect", 5);
        }

        virtual ~ROSOutputWrapper()
        {
            printf("OUT: Destroyed ROSOutputWrapper\n");
        }

        virtual void publishGraph(const std::map<uint64_t, Eigen::Vector2i, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> &connectivity) override
        {
            printf("OUT: got graph with %d edges\n", (int)connectivity.size());

            int maxWrite = 5;

            for(const std::pair<uint64_t,Eigen::Vector2i> &p : connectivity)
            {
                int idHost = p.first>>32;
                int idTarget = p.first & ((uint64_t)0xFFFFFFFF);
                printf("OUT: Example Edge %d -> %d has %d active and %d marg residuals\n", idHost, idTarget, p.second[0], p.second[1]);
                maxWrite--;
                if(maxWrite==0) break;
            }
        }



        virtual void publishKeyframes( std::vector<FrameHessian*> &frames, bool final, CalibHessian* HCalib) override
        {
            for(FrameHessian* f : frames)
            {
                printf("OUT: KF %d (%s) (id %d, tme %f): %d active, %d marginalized, %d immature points. CameraToWorld:\n",
                       f->frameID,
                       final ? "final" : "non-final",
                       f->shell->incoming_id,
                       f->shell->timestamp,
                       (int)f->pointHessians.size(), (int)f->pointHessiansMarginalized.size(), (int)f->immaturePoints.size());
                std::cout << f->shell->camToWorld.matrix3x4() << "\n";


                int maxWrite = 5;
                for(PointHessian* p : f->pointHessians)
                {
                    printf("OUT: Example Point x=%.1f, y=%.1f, idepth=%f, idepth std.dev. %f, %d inlier-residuals\n",
                           p->u, p->v, p->idepth_scaled, sqrt(1.0f / p->idepth_hessian), p->numGoodResiduals );
                    maxWrite--;
                    if(maxWrite==0) break;
                }
            }
        }

        void publishCamPose(const uint32_t id, const double time,
                            const SE3& pose, CalibHessian* HCalib) override {
          boost::lock_guard<boost::mutex> lock(pose_mtx_);

          geometry_msgs::TransformStamped tf;

          tf.header.stamp.fromSec(time);
          tf.header.frame_id = "dso_world";
          tf.child_frame_id = "dso_cam";
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

            publishDepthMap(depth_pub_, "dso_cam", KF->shell->timestamp, K,
                            depthmap);

            return;
        }

 private:
  boost::mutex pose_mtx_;

  ros::NodeHandle nh_;

  tf2_ros::TransformBroadcaster tf_pub_;

  std::shared_ptr<image_transport::ImageTransport> it_;

  image_transport::CameraPublisher depth_pub_;

  CalibHessian* calib_hessian_ = nullptr;
};

}

}
