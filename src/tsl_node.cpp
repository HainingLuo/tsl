#include "tsl/tsl_node.h"

TslNode::TslNode():
    nh_{}
{
    // get parameters
    // get image and info topics
    nh_.getParam("/tsl/rgb_topic", rgb_topic_);
    nh_.getParam("/tsl/depth_topic", depth_topic_);
    nh_.getParam("/tsl/camera_info_topic", camera_info_topic_);
    nh_.getParam("/tsl/result_frame", result_frame_);
    nh_.getParam("/tsl/unity_reset", unity_reset_service_);
    nh_.getParam("/tsl/unity_adjust", unity_adjust_service_);

    // get segmentation parameters
    nh_.getParam("/tsl/segmentation/hue_min", hue_min_);
    nh_.getParam("/tsl/segmentation/hue_max", hue_max_);
    nh_.getParam("/tsl/segmentation/sat_min", sat_min_);
    nh_.getParam("/tsl/segmentation/sat_max", sat_max_);
    nh_.getParam("/tsl/segmentation/val_min", val_min_);
    nh_.getParam("/tsl/segmentation/val_max", val_max_);


    // initialise publishers
    result_states_pub_ = nh_.advertise<PointCloudMsg>("/tsl/result_states", 10);

    // initialise subscribers
    ros::Subscriber sub = nh_.subscribe<PointCloudMsg>("/tsl/segmented_pc", 10, 
                            &TslNode::PointCloudCallback, this);

    // initialise services

    // initialise the camera class
    sensor_msgs::CameraInfoConstPtr info_msg = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(camera_info_topic_, nh_);
    Eigen::Matrix3d intrinsicMatrix;
    intrinsicMatrix << info_msg->K[0], info_msg->K[1], info_msg->K[2],
                        info_msg->K[3], info_msg->K[4], info_msg->K[5],
                        info_msg->K[6], info_msg->K[7], info_msg->K[8];
    camera = Camera(intrinsicMatrix);

    // synchronised subscribers for rgb and depth images
    messsage_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh_, rgb_topic_, 1);
    messsage_filters::Subscriber<sensor_msgs::Image> depth_sub(nh_, depth_topic_, 1);
    messsage_filters::Subscriber<sensor_msgs::CameraInfo> info_sub(nh_, camera_info_topic_, 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, 
                                        sensor_msgs::Image, sensor_msgs::CameraInfo> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), rgb_sub, depth_sub, info_sub);
    sync.registerCallback(boost::bind(&TslNode::RGBDCallback, this, _1, _2, _3));
    
    // initialise the image segmentation
    auto hsv_segmenter_ = ImageSegmenter(hue_min_, hue_max_, sat_min_, sat_max_, val_min_, val_max_);

    // initialise the states
    InitialiseStates();
    
    // initialise the tsl class
    tsl = Tsl();

    // reset the simulation
    ros::ServiceClient reset_client = nh.serviceClient<tsl::SimReset>("/unity_reset");
    tsl::SimReset reset_srv;
    // convert the eigen matrix to a posearray message

    ROS_INFO_STREAM("Tsl node initialised");
    ros::spin();
}

void TslNode::InitialiseStates()
{
    // call the initial states server on python side to initialise config
    ros::ServiceClient initial_states_client = nh.serviceClient<tsl::SimAdjust>("/tsl/get_initial_states");
    tsl::SimAdjust init_srv;
    if (initial_states_client.call(init_srv)) {
        // covert the srv response to an eigen matrix
        
        // save the states to the tsl class

        ROS_INFO("got initial states");
    } else {
        ROS_ERROR("Failed to call service get_initial_states");
    }
}

void TslNode::RGBDCallback(const sensor_msgs::ImageConstPtr& rgb_msg, 
                            const sensor_msgs::ImageConstPtr& depth_msg, 
                            const sensor_msgs::CameraInfoConstPtr& info_msg)
{
    // color segmentation of the rgb image



    // convert the rgb and depth images to pcl point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    // convert the pcl point cloud to an eigen matrix
    MatrixXf X(cloud->size(), 3);

    // get observation and action
    // send action to simulation
    // get simmulation predicted states
    // call cpd with observation, action and simmulation predicted states
    // get cpd predicted states back
    // send cpd predicted states to simulation controller
    // get simmulation states
    // update states
}

void TslNode::PointCloudCallback(const PointCloudMsg::ConstPtr& msg)
{
    //// psuedo code
    // get observation and action
    // send action to simulation
    // get simmulation predicted states
    // call cpd with observation, action and simmulation predicted states
    // get cpd predicted states back
    // send cpd predicted states to simulation controller
    // get simmulation states
    // update states

    // convert the point cloud message to a pcl point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);

    // convert the pcl point cloud to an eigen matrix
    MatrixXf X(cloud->size(), 3);
    for (int i=0; i<cloud->size(); i++) {
        X(i, 0) = cloud->points[i].x;
        X(i, 1) = cloud->points[i].y;
        X(i, 2) = cloud->points[i].z;
    }

    // get prediction from the physics engine
    MatrixXf Y_pred;

    // run the cpd algorithm
    MatrixXf Y = tsl.step(X, Y_pred);

    // convert the eigen matrix to a pcl point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr result_states (new pcl::PointCloud<pcl::PointXYZ>);
    for (int i=0; i<Y.rows(); i++) {
        pcl::PointXYZ point;
        point.x = Y(i, 0);
        point.y = Y(i, 1);
        point.z = Y(i, 2);
        result_states->push_back(point);
    }

    // convert the pcl point cloud to a point cloud message
    PointCloudMsg::Ptr result_states_msg (new PointCloudMsg);
    pcl::toROSMsg(*result_states, *result_states_msg);
    result_states_msg->header.frame_id = result_frame_;

    // publish the point cloud message
    result_states_pub_.publish(result_states_msg);
}


int main(int argc, char **argv) {
    ros::init(argc, argv, "tsl_node");
    ROS_INFO_STREAM("Initialising tsl node");
    TslNode tsl_node;
    return 0;
}