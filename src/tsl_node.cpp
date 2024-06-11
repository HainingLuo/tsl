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
    adjust_client = nh_.serviceClient<tsl::SimAdjust>("/unity_adjust");

    // initialise the camera class
    sensor_msgs::CameraInfoConstPtr info_msg = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(camera_info_topic_, nh_);
    Eigen::Matrix3d intrinsicMatrix;
    intrinsicMatrix << info_msg->K[0], info_msg->K[1], info_msg->K[2],
                        info_msg->K[3], info_msg->K[4], info_msg->K[5],
                        info_msg->K[6], info_msg->K[7], info_msg->K[8];
    camera = Camera(intrinsicMatrix);

    // synchronised subscribers for rgb and depth images
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh_, rgb_topic_, 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh_, depth_topic_, 1);
    // message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub(nh_, camera_info_topic_, 1);
    // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, 
    //                                     sensor_msgs::Image, sensor_msgs::CameraInfo> MySyncPolicy;
    // message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), rgb_sub, depth_sub, info_sub);
    // sync.registerCallback(boost::bind(&TslNode::RGBDCallback, this, _1, _2, _3));
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, 
                                        sensor_msgs::Image> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), rgb_sub, depth_sub);
    sync.registerCallback(boost::bind(&TslNode::RGBDCallback, this, _1, _2));
    
    // initialise the image segmentation
    hsv_segmenter_ = ImageSegmenter(hue_min_, hue_max_, sat_min_, sat_max_, val_min_, val_max_);
    
    // initialise the tsl class
    tsl = Tsl();

    // initialise the states
    tsl.Y = InitialiseStates();

    // reset the simulation
    ros::ServiceClient reset_client = nh_.serviceClient<tsl::SimReset>("/unity_reset");
    tsl::SimReset reset_srv;
    // convert the eigen matrix to a posearray message
    for (int i=0; i<tsl.Y.rows(); i++) {
        geometry_msgs::Pose pose;
        pose.position.x = tsl.Y(i, 0);
        pose.position.y = tsl.Y(i, 1);
        pose.position.z = tsl.Y(i, 2);
        reset_srv.request.states_est.poses.push_back(pose);
    }
    // call the reset service
    if (reset_client.call(reset_srv)) {
        // save the states
        // for (int i=0; i<reset_srv.response.states_est.size(); i++) {
        //     tsl.Y(i, 0) = reset_srv.response.states_est[i].x;
        //     tsl.Y(i, 1) = reset_srv.response.states_est[i].y;
        //     tsl.Y(i, 2) = reset_srv.response.states_est[i].z;
        // }   
        // no need to save, same as the initial states     
    } else {
        ROS_ERROR("Failed to call service reset");
    }

    ROS_INFO_STREAM("Tsl node initialised");
    ros::spin();
}

Eigen::MatrixXf TslNode::InitialiseStates()
{
    // call the initial states server on python side to initialise config
    ros::ServiceClient initial_states_client = nh_.serviceClient<tsl::SimAdjust>("/tsl/get_initial_states");
    tsl::SimAdjust init_srv;
    Eigen::MatrixXf states;
    if (initial_states_client.call(init_srv)) {
        // covert the srv response to an eigen matrix
        for (int i=0; i<init_srv.response.states_sim.poses.size(); i++) {
            states(i, 0) = init_srv.response.states_sim.poses[i].position.x;
            states(i, 1) = init_srv.response.states_sim.poses[i].position.y;
            states(i, 2) = init_srv.response.states_sim.poses[i].position.z;
        }

        ROS_INFO("got initial states");
    } else {
        ROS_ERROR("Failed to call service get_initial_states");
    }
    return states;
}

void TslNode::RGBDCallback(const sensor_msgs::ImageConstPtr& rgb_msg, 
                            const sensor_msgs::ImageConstPtr& depth_msg)
{
    // color segmentation of the rgb image
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    // cv::Mat mask = hsv_segmenter_.segmentImage(cv_ptr->image);
    std::vector<cv::Point> pixelCoordinates = hsv_segmenter_.retrievePoints(cv_ptr->image);

    // extract the segmented points from the depth image
    cv_bridge::CvImagePtr cv_depth_ptr;
    try {
        cv_depth_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    
    // // extract the non zero points from the mask
    // std::vector<Eigen::Vector2i> pixelCoordinates;
    // cv::findNonZero(mask, pixelCoordinates);

    // convert the pixel coordinates to 3D points
    PointCloud::Ptr points3D = camera.pixels2PointCloud(pixelCoordinates, cv_depth_ptr->image);
    // PoinCloud::Ptr points3D = camera.convertMaskToPointCloud(mask, cv_depth_ptr->image);
        
    // downsample the points
    PointCloud::Ptr cloud_filtered (new PointCloud);
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud (points3D);
    sor.setLeafSize (0.02f, 0.02f, 0.02f);
    sor.filter (*cloud_filtered);
    if (cloud_filtered->size() == 0) {
        ROS_WARN("No points in the downsampled point cloud!");
        return;
    }

    // convert the pcl point cloud to eigen matrix
    Matrix3Xf X = cloud_filtered->getMatrixXfMap().topRows(3);    
    
    // call cpd function
    tsl.step(X);

    // call unity adjust service
    tsl::SimAdjust adjust_srv;
    // convert the eigen matrix to a posearray message
    for (int i=0; i<tsl.Y.rows(); i++) {
        geometry_msgs::Pose pose;
        pose.position.x = tsl.Y(i, 0);
        pose.position.y = tsl.Y(i, 1);
        pose.position.z = tsl.Y(i, 2);
        adjust_srv.request.states_est.poses.push_back(pose);
    }
    // call the adjust service
    // TODO change the pose array to point cloud
    if (adjust_client.call(adjust_srv)) {
        // save the states
        for (int i=0; i<adjust_srv.response.states_sim.poses.size(); i++) {
            tsl.Y(i, 0) = adjust_srv.response.states_sim.poses[i].position.x;
            tsl.Y(i, 1) = adjust_srv.response.states_sim.poses[i].position.y;
            tsl.Y(i, 2) = adjust_srv.response.states_sim.poses[i].position.z;
        }        
    } else {
        ROS_ERROR("Failed to call service adjust");
    }

    // publish result states
    PointCloudMsg::Ptr result_states_msg (new PointCloudMsg);


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

    // run the cpd algorithm
    MatrixXf Y = tsl.step(X);

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