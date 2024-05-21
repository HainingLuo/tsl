#include "tsl/tsl_bag.h"

TslBag::TslBag():
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
    nh_.getParam("/tsl/bag_path", bag_path);
    nh_.getParam("/tsl/bag_config_path", bag_config_path);
    nh_.getParam("/tsl/cam_pose", cam_pose);
    nh_.getParam("/tsl/robot_frame", robot_frame);
    nh_.getParam("/tsl/camera_frame", camera_frame);
    nh_.getParam("/tsl/num_state_points", num_state_points);

    // get segmentation parameters
    nh_.getParam("/tsl/segmentation/hue_min", hue_min_);
    nh_.getParam("/tsl/segmentation/hue_max", hue_max_);
    nh_.getParam("/tsl/segmentation/sat_min", sat_min_);
    nh_.getParam("/tsl/segmentation/sat_max", sat_max_);
    nh_.getParam("/tsl/segmentation/val_min", val_min_);
    nh_.getParam("/tsl/segmentation/val_max", val_max_);


    // initialise publishers
    result_states_pub_ = nh_.advertise<PointCloudMsg>("/tsl/result_states", 10);

    // // initialise subscribers
    // ros::Subscriber sub = nh_.subscribe<PointCloudMsg>("/tsl/segmented_pc", 10, 
    //                         &TslBag::PointCloudCallback, this);

    // initialise services
    adjust_client = nh_.serviceClient<tsl::SimAdjust>("/unity_adjust");

    // load the ROS bag
    rosbag::Bag bag;
    bag.open(bag_path, rosbag::bagmode::Read);
    ROS_INFo_STREAM("Opened bag: " << bag_path);

    // read the number of rgb_topic_ and depth_topic_ messages in the bag
    // rosbag::View view(bag, rosbag::TopicQuery({rgb_topic_, depth_topic_}));
    rosbag::View view_image(bag, rosbag::TopicQuery({rgb_topic_}));
    rosbag::View view_depth(bag, rosbag::TopicQuery({depth_topic_}));
    num_messages = view.size();
    ROS_INFO_STREAM("Number of messages in the bag: " << num_messages);

    // static tf broadcaster for camera pose
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(cam_pose[0], cam_pose[1], cam_pose[2]));
    transform.setRotation(tf::Quaternion(cam_pose[3], cam_pose[4], cam_pose[5], cam_pose[6]));
    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "camera"));

    // load the camera info message
    rosbag::View view_info(bag, rosbag::TopicQuery({camera_info_topic_}));
    rosbag::View::iterator info_it = view_info.begin();
    rosbag::MessageInstance const info_m = *info_it;
    sensor_msgs::CameraInfo::ConstPtr info_msg = info_m.instantiate<sensor_msgs::CameraInfo>();
    Eigen::Matrix3d intrinsicMatrix;
    intrinsicMatrix << info_msg->K[0], info_msg->K[1], info_msg->K[2],
                        info_msg->K[3], info_msg->K[4], info_msg->K[5],
                        info_msg->K[6], info_msg->K[7], info_msg->K[8];
    camera = Camera(intrinsicMatrix);

    // load the first image message
    rosbag::View::iterator it = view_image.begin();
    rosbag::MessageInstance const m = *it;
    sensor_msgs::Image::ConstPtr image_msg = m.instantiate<sensor_msgs::Image>();
    cv::Mat init_img = ImageToCvMat(image_msg);
    resolution = cv::Point(init_img.cols, init_img.rows); // width, height

    // load the first depth message
    it = view_depth.begin();
    rosbag::MessageInstance const n = *it;
    sensor_msgs::Image::ConstPtr depth_msg = n.instantiate<sensor_msgs::Image>();
    cv::Mat init_depth = DepthToCvMat(depth_msg);

    // read /tsl_data/shoelacing_4_2.yaml
    std::string data_path = "/tsl_data";
    std::string bagfile_name = "shoelacing_4_2";
    std::string yaml_path = path.join(data_path, bagfile_name+".yaml");
    YAML::Node yaml = YAML::LoadFile(yaml_path);
    std::vector<cv::Vec4d> eyelets;
    for (int i = 0; i < yaml["eyelets"].size(); i++) {
        cv::Vec4d eyelet;
        eyelet[0] = yaml["eyelets"][i][0].as<double>();
        eyelet[1] = yaml["eyelets"][i][1].as<double>();
        eyelet[2] = yaml["eyelets"][i][2].as<double>();
        eyelet[3] = yaml["eyelets"][i][3].as<double>();
        eyelets.push_back(eyelet);
    }

    // // synchronised subscribers for rgb and depth images
    // message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh_, rgb_topic_, 1);
    // message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh_, depth_topic_, 1);
    // // message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub(nh_, camera_info_topic_, 1);
    // // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, 
    // //                                     sensor_msgs::Image, sensor_msgs::CameraInfo> MySyncPolicy;
    // // message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), rgb_sub, depth_sub, info_sub);
    // // sync.registerCallback(boost::bind(&TslBag::RGBDCallback, this, _1, _2, _3));
    // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, 
    //                                     sensor_msgs::Image> MySyncPolicy;
    // message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), rgb_sub, depth_sub);
    // sync.registerCallback(boost::bind(&TslBag::RGBDCallback, this, _1, _2));
    
    // initialise the image segmentation
    hsv_segmenter_ = ImageSegmenter(hue_min_, hue_max_, sat_min_, sat_max_, val_min_, val_max_);
    
    // initialise the tsl class
    tsl = Tsl();

    // initialise the states
    tsl.Y = InitialiseStates();

    // load and publish the eyelet poses
    // 
    eyelets_traj = np.load(path.join(data_path, bagfile_name+".npy"));
    eyelets.clear();
    geometry_msgs::PoseArray eyelet_msg;
    eyelet_msg.header.frame_id = camera_frame;
    for (int i = 0; i < eyelets_traj.size(); i++) {
        int eyelet_x = eyelets_traj[i][0][1] + eyelets_traj[i][0][3] / 2;
        int eyelet_y = eyelets_traj[i][0][0] + eyelets_traj[i][0][2] / 2;
        geometry_msgs::Pose p;
        cv::Point3d eyelet_3d = read_point_from_region(cv::Point(eyelet_x, eyelet_y), depth, 20, "middle", camera_intrinsics);
        cv::Point3d eyelet_offset(0, 0, 0.002);
        p.position.x = eyelet_3d.x + eyelet_offset.x;
        p.position.y = eyelet_3d.y + eyelet_offset.y;
        p.position.z = eyelet_3d.z + eyelet_offset.z;
        tf::Quaternion eyelet_orientation = tf::createQuaternionFromRPY(0, M_PI / 6 * pow(-1, i + 1), 0);
        p.orientation.x = eyelet_orientation.x();
        p.orientation.y = eyelet_orientation.y();
        p.orientation.z = eyelet_orientation.z();
        p.orientation.w = eyelet_orientation.w();
        eyelets.push_back(cv::Vec4d(eyelet_3d.x, eyelet_3d.y, eyelet_3d.z, eyelet_orientation));
        eyelet_msg.poses.push_back(p);
    }
    ros::Duration(1).sleep();
    eyelet_init_pub.publish(eyelet_msg);
    ROS_INFO("eyelet poses published");



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

    ROS_INFO_STREAM("Tsl bag node initialised");
    ros::spin();
}

Eigen::MatrixXf TslBag::InitialiseStates(const cv::Mat& init_img, const cv::Mat& init_depth)
{
    // get the segmented points
    Eigen::MatrixXf X = GetSegmentedPoints(init_img, init_depth);
    // apply gaussian mixture model to the points
    GaussianMixtureModel gmm = GaussianMixtureModel(num_state_points, maxIterations=10000);
    gmm.fit(X);
    // get the states from the gmm
    return gmm.getMeans();
}

void TslBag::ProcessImage(const sensor_msgs::ImageConstPtr& rgb_msg)
{
    // set region [270:270+154, 267:267+92, :] to zero
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    cv::Mat image = cv_ptr->image;
    cv::Mat roi = image(cv::Rect(267, 270, 92, 154));
    roi = cv::Mat::zeros(roi.size(), roi.type());    
}

void TslBag::RGBDCallback(const sensor_msgs::ImageConstPtr& rgb_msg, 
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
    PointCloud::Ptr points3D = camera.convertPixelsToPointCloud(pixelCoordinates, cv_depth_ptr->image);
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

void TslBag::PointCloudCallback(const PointCloudMsg::ConstPtr& msg)
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
    PoinCloud::Ptr cloud (new PoinCloud);
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
    PoinCloud::Ptr result_states (new PoinCloud);
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

Eigen::MatrixXf TslBag::GetSegmentedPoints(const cv::Mat& image, const cv::Mat& depth)
{
    // color segmentation of the rgb image
    cv::Mat mask = hsv_segmenter_.segmentImage(image);
    std::vector<cv::Point> pixelCoordinates = hsv_segmenter_.retrievePoints(image);

    // extract the segmented points from the depth image
    // convert the pixel coordinates to 3D points
    PointCloud::Ptr points3D = camera.convertPixelsToPointCloud(pixelCoordinates, depth);
    // PoinCloud::Ptr points3D = camera.convertMaskToPointCloud(mask, cv_depth_ptr->image);
    
    // downsample the points
    PointCloud::Ptr cloud_filtered (new PointCloud);
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud (points3D);
    sor.setLeafSize (0.02f, 0.02f, 0.02f);
    sor.filter (*cloud_filtered);

    // convert the pcl point cloud to eigen matrix
    Eigen::MatrixXf X = cloud_filtered->getMatrixXfMap().topRows(3);
    return X;
}

cv::Mat ImageToCvMat(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return cv::Mat();
    }
    return cv_ptr->image;
}

cv::Mat DepthToCvMat(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return cv::Mat();
    }
    return cv_ptr->image;
}


int main(int argc, char **argv) {
    ros::init(argc, argv, "tsl_bag");
    ROS_INFO_STREAM("Initialising tsl bag node");
    TslBag tsl_bag;
    return 0;
}