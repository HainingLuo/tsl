#ifndef TSL_TSL_NODE_H
#define TSL_TSL_NODE_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <tsl/SimAdjust.h>
#include <tsl/SimReset.h>
#include <tsl/segmenter.h>
#include <tsl/camera.h>


#include "tsl/tsl.h"

using PointCloudMsg = sensor_msgs::PointCloud2;

class TslNode
{
private:
    // topics
    std::string result_frame_;
    std::string rgb_topic_;
    std::string depth_topic_;
    std::string camera_info_topic_;
    std::string unity_reset_service_;
    std::string unity_adjust_service_;

    ros::NodeHandle nh_;
    ros::Publisher result_states_pub_;
    ros::ServiceClient adjust_client
    Tsl tsl;

    // camera class
    Camera camera;

    // segmentation parameters
    int hue_min_;
    int hue_max_;
    int sat_min_;
    int sat_max_;
    int val_min_;
    int val_max_;
    

public:
    TslNode();
    // ~TslNode();
    void PointCloudCallback(const PointCloudMsg::ConstPtr& msg);
    void RGBDCallback(const sensor_msgs::ImageConstPtr& rgb_msg, const sensor_msgs::ImageConstPtr& depth_msg, const sensor_msgs::CameraInfoConstPtr& info_msg);
};

#endif // TSL_TSL_NODE_H