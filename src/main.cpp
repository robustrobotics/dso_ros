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





#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "util/settings.h"
#include "FullSystem/FullSystem.h"
#include "util/Undistort.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"

#include <boost/filesystem.hpp>

#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseStamped.h>
#include "cv_bridge/cv_bridge.h"

#include <fla_msgs/ProcessStatus.h>

#include "./ROSOutputWrapper.h"
#include "./utils.h"

std::string calib = "";
std::string vignetteFile = "";
std::string gammaFile = "";
int mode = 1;

using namespace dso;
namespace bfs = boost::filesystem;

void parseArgument(char* arg)
{
	int option;
	char buf[1000];

	if(1==sscanf(arg,"quiet=%d",&option))
	{
		if(option==1)
		{
			setting_debugout_runquiet = true;
			printf("QUIET MODE, I'll shut up!\n");
		}
		return;
	}


	if(1==sscanf(arg,"nolog=%d",&option))
	{
		if(option==1)
		{
			setting_logStuff = false;
			printf("DISABLE LOGGING!\n");
		}
		return;
	}

	if(1==sscanf(arg,"nogui=%d",&option))
	{
		if(option==1)
		{
			disableAllDisplay = true;
			printf("NO GUI!\n");
		}
		return;
	}
	if(1==sscanf(arg,"nomt=%d",&option))
	{
		if(option==1)
		{
			multiThreading = false;
			printf("NO MultiThreading!\n");
		}
		return;
	}
	if(1==sscanf(arg,"calib=%s",buf))
	{
		calib = buf;
		printf("loading calibration from %s!\n", calib.c_str());
		return;
	}
	if(1==sscanf(arg,"vignette=%s",buf))
	{
		vignetteFile = buf;
		printf("loading vignette from %s!\n", vignetteFile.c_str());
		return;
	}

	if(1==sscanf(arg,"gamma=%s",buf))
	{
		gammaFile = buf;
		printf("loading gammaCalib from %s!\n", gammaFile.c_str());
		return;
	}

	if(1==sscanf(arg,"mode=%d",&option))
	{

		mode = option;
		if(option==0)
		{
			printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
		}
		if(option==1)
		{
			printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
			setting_photometricCalibration = 0;
			setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
		}
		if(option==2)
		{
			printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
			setting_photometricCalibration = 0;
			setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_minGradHistAdd=3;
		}
		return;
	}

	printf("could not parse argument \"%s\"!!\n", arg);
}




FullSystem* fullSystem = 0;
Undistort* undistorter = 0;
int frameID = 0;
double last_update_sec = 0.0;

void vidCb(const sensor_msgs::ImageConstPtr img)
{
	cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
	assert(cv_ptr->image.type() == CV_8U);
	assert(cv_ptr->image.channels() == 1);


	if(setting_fullResetRequested || fullSystem->isLost)
	{
		std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
		delete fullSystem;
		for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();
		fullSystem = new FullSystem();
		fullSystem->linearizeOperation=false;
		fullSystem->outputWrapper = wraps;
	    if(undistorter->photometricUndist != 0)
	    	fullSystem->setGammaFunction(undistorter->photometricUndist->getG());
		setting_fullResetRequested=false;
	}

	MinimalImageB minImg((int)cv_ptr->image.cols, (int)cv_ptr->image.rows,(unsigned char*)cv_ptr->image.data);
	ImageAndExposure* undistImg = undistorter->undistort<unsigned char>(&minImg, 1,0, 1.0f);
  undistImg->timestamp = img->header.stamp.toSec();
	fullSystem->addActiveFrame(undistImg, frameID);
	frameID++;
	delete undistImg;

  last_update_sec = ros::Time::now().toSec();

  return;
}


int node_id;
double heart_beat_dt;
double alarm_timeout;
double fail_timeout;
bool fla_calib = false;
ros::Publisher heart_beat_pub;

void heartBeat(const ros::TimerEvent&) {
  double now = ros::Time::now().toSec();

  fla_msgs::ProcessStatus::Ptr ps(new fla_msgs::ProcessStatus);

  ps->id = node_id;
  ps->pid = getpid();

  if (now - last_update_sec > alarm_timeout)  {
    ps->status = fla_msgs::ProcessStatus::ALARM;
    // ps->arg = Status::ALARM_TIMEOUT; // Time since last update longer than expected.
  } else if (now - last_update_sec > fail_timeout) {
    ps->status = fla_msgs::ProcessStatus::FAIL;
    // ps->arg = Status::FAIL_TIMEOUT; // Time since last update probably error.
  } else {
    ps->status = fla_msgs::ProcessStatus::READY;
    // ps->arg = Status::GOOD; // All good.
  }

  heart_beat_pub.publish(ps);

  return;
}

int main( int argc, char** argv )
{
  // Setup ROS.
	ros::init(argc, argv, "dso_live");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  ros::Subscriber imgSub = nh.subscribe("image", 1, &vidCb);


  // Parse command line args.
	for(int i=1; i<argc;i++) parseArgument(argv[i]);

	setting_desiredImmatureDensity = 1000;
	setting_desiredPointDensity = 1200;
	setting_minFrames = 5;
	setting_maxFrames = 7;
	setting_maxOptIterations=4;
	setting_minOptIterations=1;
	setting_logStuff = false;
	setting_kfGlobalWeight = 1.3;

  // Parse FLA params.
  dso_ros::getParamOrFail(pnh, "fla/node_id", &node_id);
  dso_ros::getParamOrFail(pnh, "fla/heart_beat_dt", &heart_beat_dt);
  dso_ros::getParamOrFail(pnh, "fla/alarm_timeout", &alarm_timeout);
  dso_ros::getParamOrFail(pnh, "fla/fail_timeout", &fail_timeout);
  dso_ros::getParamOrFail(pnh, "fla/use_fla_calib", &fla_calib);

  if (fla_calib) {
    // Read FLA calibration params.
    int width, height;
    dso_ros::getParamOrFail(nh, "/samros/camera/image_width", &width);
    dso_ros::getParamOrFail(nh, "/samros/camera/image_height", &height);

    double fx, fy, cx, cy;
    dso_ros::getParamOrFail(nh, "/samros/camera/intrinsics/fu", &fx);
    dso_ros::getParamOrFail(nh, "/samros/camera/intrinsics/fv", &fy);
    dso_ros::getParamOrFail(nh, "/samros/camera/intrinsics/pu", &cx);
    dso_ros::getParamOrFail(nh, "/samros/camera/intrinsics/pv", &cy);

    double k1, k2, p1, p2, k3;
    dso_ros::getParamOrFail(nh, "/samros/camera/distortion/k1", &k1);
    dso_ros::getParamOrFail(nh, "/samros/camera/distortion/k2", &k2);
    dso_ros::getParamOrFail(nh, "/samros/camera/distortion/p1", &p1);
    dso_ros::getParamOrFail(nh, "/samros/camera/distortion/p2", &p2);
    dso_ros::getParamOrFail(nh, "/samros/camera/distortion/k3", &k3);

    // Create temporary DSO calibration file.
    std::string tmp_file = (bfs::temp_directory_path() / bfs::unique_path()).native() + ".yaml";
    ROS_INFO("Creating FLA camera calibration file %s", tmp_file.c_str());

    std::ofstream tmp_file_stream;
    tmp_file_stream.open(tmp_file.c_str());
    tmp_file_stream << std::setprecision(15);
    tmp_file_stream << fx << " " << fy << " " << " " << cx << " " << cy << " " <<
        k1 << " " << k2 << " " << p1 << " " << p2 << std::endl;
    tmp_file_stream << width << " " << height << std::endl;
    tmp_file_stream << "crop" << std::endl;
    tmp_file_stream << width << " " << height << std::endl;

    calib = tmp_file;
  }

  ros::Timer heart_beat(nh.createTimer(ros::Duration(heart_beat_dt),
                                       &heartBeat));
  heart_beat_pub = nh.advertise<fla_msgs::ProcessStatus>("/globalstatus", 1);

  // Parse ROS params.
  IOWrap::ROSOutputWrapper::Params params;
  dso_ros::getParamOrFail(pnh, "frames/dso_cam_frame", &params.dso_cam_frame);
  dso_ros::getParamOrFail(pnh, "frames/dso_world_frame", &params.dso_world_frame);
  dso_ros::getParamOrFail(pnh, "frames/metric_cam_frame", &params.metric_cam_frame);
  dso_ros::getParamOrFail(pnh, "frames/metric_world_frame", &params.metric_world_frame);

  dso_ros::getParamOrFail(pnh, "scale/publish_metric_depthmap", &params.publish_metric_depthmap);
  dso_ros::getParamOrFail(pnh, "scale/min_metric_inc_trans", &params.min_metric_inc_trans);
  dso_ros::getParamOrFail(pnh, "scale/min_metric_trans", &params.min_metric_trans);
  dso_ros::getParamOrFail(pnh, "scale/scale_divergence_factor", &params.scale_divergence_factor);
  dso_ros::getParamOrFail(pnh, "scale/max_pose_history", &params.max_pose_history);
  dso_ros::getParamOrFail(pnh, "scale/metric_takeoff_thresh", &params.metric_takeoff_thresh);

  dso_ros::getParamOrFail(pnh, "regularization/publish_coarse_metric_depthmap", &params.publish_coarse_metric_depthmap);
  dso_ros::getParamOrFail(pnh, "regularization/coarse_level", &params.coarse_level);
  dso_ros::getParamOrFail(pnh, "regularization/match_source_resolution", &params.match_source_resolution);
  dso_ros::getParamOrFail(pnh, "regularization/fill_holes", &params.fill_holes);
  dso_ros::getParamOrFail(pnh, "regularization/fill_radius", &params.fill_radius);
  dso_ros::getParamOrFail(pnh, "regularization/min_depths_to_fill", &params.min_depths_to_fill);
  dso_ros::getParamOrFail(pnh, "regularization/do_morph_close", &params.do_morph_close);
  dso_ros::getParamOrFail(pnh, "regularization/morph_close_size", &params.morph_close_size);
  dso_ros::getParamOrFail(pnh, "regularization/do_median_filter", &params.do_median_filter);
  dso_ros::getParamOrFail(pnh, "regularization/median_filter_size", &params.median_filter_size);

  undistorter = Undistort::getUndistorterForFile(calib, gammaFile, vignetteFile);

  setGlobalCalib(
      (int)undistorter->getSize()[0],
      (int)undistorter->getSize()[1],
      undistorter->getK().cast<float>());


  fullSystem = new FullSystem();
  fullSystem->linearizeOperation=false;


  // if(!disableAllDisplay)
  //   fullSystem->outputWrapper.push_back(new IOWrap::PangolinDSOViewer(
  //       (int)undistorter->getSize()[0],
  //       (int)undistorter->getSize()[1]));


  fullSystem->outputWrapper.push_back(new IOWrap::ROSOutputWrapper(nh, params));


  if(undistorter->photometricUndist != 0)
    fullSystem->setGammaFunction(undistorter->photometricUndist->getG());

  ros::spin();

  for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
  {
    ow->join();
    delete ow;
  }

  delete undistorter;
  delete fullSystem;

	return 0;
}

