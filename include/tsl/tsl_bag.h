#ifndef TSL_BAG_H
#define TSL_BAG_H

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <rosbag/query.h>
#include <rosbag/message_instance.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <iostream>
#include <fstream>
#include <yaml-cpp/yaml.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <tsl/SimAdjust.h>
#include <tsl/SimReset.h>
#include <tsl/segmenter.h>
#include <tsl/camera.h>

#include <tsl/gmm.h>
#include "tsl/tsl.h"
#include "tsl/util.h"

// #include "Python.h"
#include <pybind11/embed.h>
#include <pybind11/numpy.h>


using PointCloudMsg = sensor_msgs::PointCloud2;
using PointCloud = pcl::PointCloud<pcl::PointXYZ>;

namespace py = pybind11;

class TslBag
{
private:
    // topics
    std::string result_frame_;
    std::string rgb_topic_;
    std::string depth_topic_;
    std::string camera_info_topic_;
    std::string unity_reset_service_;
    std::string unity_adjust_service_;
    std::string unity_predict_service_;
    std::string robot_frame;
    std::string camera_frame;
    std::string bag_path;
    std::string bag_config_path;
    std::string eyelet_topic_;
    std::string eyelet_init_topic_;
    std::string aglet_topic_;
    std::string result_pc_topic_;
    std::string pkg_path_;

    // 
    bool viusalisation;
    int num_state_points;
    int num_messages;
    std::vector<float> cam_pose;
    cv::Point resolution;
    float rope_length;
    float rope_radius;
    std::vector<int> skip_frames;
    std::vector<float> aglet_1_position, aglet_2_position;

    ros::NodeHandle nh_;
    ros::Publisher result_img_pub_;
    ros::Publisher segmented_pc_pub_;
    ros::Publisher result_states_pub_;
    ros::Publisher aglet_pub_;
    ros::ServiceClient adjust_client;
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

    // plot parameters
    bool use_plot_;
    std::string plot_topic_;
    float plot_x_min_;
    float plot_x_max_;
    float plot_y_min_;
    float plot_y_max_;
    
    // segmenter
    ImageSegmenter hsv_segmenter_;

    // functions
    Eigen::MatrixXf InitialiseStates(const cv::Mat& image, const cv::Mat& depth, const std::string method);
    // void ProcessImage(const sensor_msgs::ImageConstPtr& rgb_msg);
    void ProcessImage(cv::Mat& image);
    cv::Mat ImageToCvMat(const sensor_msgs::ImageConstPtr& msg);
    cv::Mat DepthToCvMat(const sensor_msgs::ImageConstPtr& msg);
    Eigen::MatrixXf Retrieve3dPointsDownSampled(const std::vector<cv::Point>& pixelCoordinates, const cv::Mat& depth);
    Eigen::MatrixXf Retrieve3dPoints(const std::vector<cv::Point>& pixelCoordinates, const cv::Mat& depth);

public:
    TslBag();
    // ~TslBag();
    void PointCloudCallback(const PointCloudMsg::ConstPtr& msg);
    void RGBDCallback(const sensor_msgs::ImageConstPtr& rgb_msg, const sensor_msgs::ImageConstPtr& depth_msg);
};

#endif // TSL_BAG_H