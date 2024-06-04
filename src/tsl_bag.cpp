#include "tsl/tsl_bag.h"

TslBag::TslBag():
    nh_{}
{
    // get parameters
    // get image and info topics
    nh_.getParam("/tsl_bag/rgb_topic", rgb_topic_);
    nh_.getParam("/tsl_bag/depth_topic", depth_topic_);
    nh_.getParam("/tsl_bag/camera_info_topic", camera_info_topic_);
    nh_.getParam("/tsl_bag/eyelet_init_topic", eyelet_init_topic_);
    nh_.getParam("/tsl_bag/eyelet_pose_topic", eyelet_topic_);
    nh_.getParam("/tsl_bag/aglet_pose_topic", aglet_topic_);
    nh_.getParam("/tsl_bag/result_pc_topic", result_pc_topic_);
    nh_.getParam("/tsl_bag/result_frame", result_frame_);
    nh_.getParam("/tsl_bag/unity_reset_service_", unity_reset_service_);
    nh_.getParam("/tsl_bag/unity_adjust_service_", unity_adjust_service_);
    nh_.getParam("/tsl_bag/unity_predict_service_", unity_predict_service_);
    nh_.getParam("/tsl_bag/bag_path", bag_path);
    nh_.getParam("/tsl_bag/bag_config_path", bag_config_path);
    nh_.getParam("/tsl_bag/cam_pose", cam_pose);
    nh_.getParam("/tsl_bag/robot_frame", robot_frame);
    nh_.getParam("/tsl_bag/camera_frame", camera_frame);
    nh_.getParam("/tsl_bag/num_state_points", num_state_points);
    nh_.getParam("/tsl_bag/rope_length", rope_length);
    nh_.getParam("/tsl_bag/rope_radius", rope_radius);
    nh_.getParam("/tsl_bag/skip_frames", skip_frames);
    nh_.getParam("/tsl_bag/visualisation", viusalisation);
    // get segmentation parameters
    nh_.getParam("/tsl_bag/segmentation/hue_min", hue_min_);
    nh_.getParam("/tsl_bag/segmentation/hue_max", hue_max_);
    nh_.getParam("/tsl_bag/segmentation/sat_min", sat_min_);
    nh_.getParam("/tsl_bag/segmentation/sat_max", sat_max_);
    nh_.getParam("/tsl_bag/segmentation/val_min", val_min_);
    nh_.getParam("/tsl_bag/segmentation/val_max", val_max_);


    // initialise publishers
    result_states_pub_ = nh_.advertise<PointCloudMsg>(result_pc_topic_, 10);
    ros::Publisher aglet_pub_ = nh_.advertise<geometry_msgs::PoseArray>(aglet_topic_, 10);
    ros::Publisher eyelet_init_pub = nh_.advertise<geometry_msgs::PoseArray>(eyelet_init_topic_, 10);

    // initialise subscribers
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_(tf_buffer_);
    // ros::Subscriber sub = nh_.subscribe<PointCloudMsg>("/tsl/segmented_pc", 10, 
    //                         &TslBag::PointCloudCallback, this);

    // initialise services
    adjust_client = nh_.serviceClient<tsl::SimAdjust>(unity_adjust_service_);

    // wait for 1 second
    ros::Duration(1).sleep();

    // load the ROS bag
    rosbag::Bag bag;
    bag.open(bag_path, rosbag::bagmode::Read);
    ROS_INFO_STREAM("Opened bag: " + bag_path);

    // read the number of rgb_topic_ and depth_topic_ messages in the bag
    // rosbag::View view(bag, rosbag::TopicQuery({rgb_topic_, depth_topic_}));
    rosbag::View view_image(bag, rosbag::TopicQuery({rgb_topic_}));
    rosbag::View view_depth(bag, rosbag::TopicQuery({depth_topic_}));
    num_messages = view_image.size()+view_depth.size();
    ROS_INFO_STREAM("Number of messages in the bag: " << num_messages);

    // static tf broadcaster for camera pose
    static tf2_ros::StaticTransformBroadcaster br;
    geometry_msgs::TransformStamped tf_msg;
    tf_msg.header.stamp = ros::Time::now();
    tf_msg.header.frame_id = robot_frame;
    tf_msg.child_frame_id = camera_frame;
    tf_msg.transform.translation.x = cam_pose[0];
    tf_msg.transform.translation.y = cam_pose[1];
    tf_msg.transform.translation.z = cam_pose[2];
    tf_msg.transform.rotation.x = cam_pose[3];
    tf_msg.transform.rotation.y = cam_pose[4];
    tf_msg.transform.rotation.z = cam_pose[5];
    tf_msg.transform.rotation.w = cam_pose[6];
    br.sendTransform(tf_msg);
    ROS_INFO_STREAM("Broadcasted camera pose");

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
    ROS_INFO_STREAM("Loaded camera info");

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
    // tsl.Y = InitialiseStates(init_img, init_depth);

    // read bag_config_path yaml file
    YAML::Node config = YAML::LoadFile(bag_config_path);

    // read the initial eyelet poses
    std::vector<geometry_msgs::Pose> eyelet_poses;
    for (int i=0; i<config["eyelets_init"].size(); i++) {
        geometry_msgs::Pose pose;
        pose = vec2PoseMsg(config["eyelets_init"][i].as<std::vector<float>>());
        // pose.position.x = config["eyelets_init"][i][0].as<float>();
        // pose.position.y = config["eyelets_init"][i][1].as<float>();
        // pose.position.z = config["eyelets_init"][i][2].as<float>();
        // pose.orientation.x = config["eyelets_init"][i][3].as<float>();
        // pose.orientation.y = config["eyelets_init"][i][4].as<float>();
        // pose.orientation.z = config["eyelets_init"][i][5].as<float>();
        // pose.orientation.w = config["eyelets_init"][i][6].as<float>();
        eyelet_poses.push_back(pose);
    }
    geometry_msgs::PoseArray eyelet_poses_msg;
    eyelet_poses_msg.header.frame_id = result_frame_;
    eyelet_poses_msg.poses = eyelet_poses;
    // read the initial aglet poses
    geometry_msgs::PoseArray aglet_poses_msg;
    geometry_msgs::Pose aglet_1_pose, aglet_2_pose;
    aglet_1_position = config["aglet_1_init"].as<std::vector<float>>();
    aglet_2_position = config["aglet_2_init"].as<std::vector<float>>();
    aglet_1_pose.position = vec2PointMsg(aglet_1_position);
    aglet_2_pose.position = vec2PointMsg(aglet_2_position);
    // aglet_1_pose.position.x = config["aglet_1_init"][0].as<float>();
    // aglet_1_pose.position.y = config["aglet_1_init"][1].as<float>();
    // aglet_1_pose.position.z = config["aglet_1_init"][2].as<float>();    
    // aglet_2_pose.position.x = config["aglet_2_init"][0].as<float>();
    // aglet_2_pose.position.y = config["aglet_2_init"][1].as<float>();
    // aglet_2_pose.position.z = config["aglet_2_init"][2].as<float>();
    aglet_poses_msg.header.frame_id = robot_frame;
    aglet_poses_msg.poses.push_back(aglet_1_pose);
    aglet_poses_msg.poses.push_back(aglet_2_pose);
    // read the initial states
    Eigen::MatrixXf initial_states(config["states_init"].size(), 3);
    for (int i=0; i<config["states_init"].size(); i++) {
        initial_states.row(i) << config["states_init"][i][0].as<float>(),
                                 config["states_init"][i][1].as<float>(),
                                 config["states_init"][i][2].as<float>();
    }
    tsl.Y = initial_states;

    // reset the simulation
    ros::ServiceClient reset_client = nh_.serviceClient<tsl::SimReset>(unity_reset_service_);
    tsl::SimReset reset_srv;
    // convert the eigen matrix to a posearray message
    for (int i=0; i<tsl.Y.rows(); i++) {
        geometry_msgs::Pose pose;
        pose.position.x = tsl.Y(i, 0);
        pose.position.y = tsl.Y(i, 1);
        pose.position.z = tsl.Y(i, 2);
        // geometry_msgs::Pose pose = eigenVec2PoseMsg(tsl.Y.row(i));
        reset_srv.request.states_est.poses.push_back(pose);
    }
    reset_srv.request.rope_length.data = rope_length;
    reset_srv.request.rope_radius.data = rope_radius;
    reset_srv.request.gripper_poses = aglet_poses_msg;
    reset_srv.request.gripper_states.data.push_back(0.0); // 0.0: open, 1.0: close
    reset_srv.request.gripper_states.data.push_back(0.0);
    // get the cam2rob transform
    geometry_msgs::TransformStamped cam2rob_msg;
    try {
        cam2rob_msg = tf_buffer_.lookupTransform(robot_frame, camera_frame, ros::Time(0));
    } catch (tf2::TransformException &ex) {
        ROS_WARN("%s", ex.what());
    }
    reset_srv.request.cam2rob = cam2rob_msg.transform;
    // call the reset service
    if (reset_client.call(reset_srv)) {
        // save the states
        // for (int i=0; i<reset_srv.response.states_est.size(); i++) {
        //     tsl.Y(i, 0) = reset_srv.response.states_est[i].x;
        //     tsl.Y(i, 1) = reset_srv.response.states_est[i].y;
        //     tsl.Y(i, 2) = reset_srv.response.states_est[i].z;
        // }   
        // no need to save, same as the initial states  
        ROS_INFO("Unity environment reset");
    } else {
        ROS_ERROR("Failed to call service reset");
    }

    // publish the initial eyelet and aglet poses
    eyelet_init_pub.publish(eyelet_poses_msg);
    aglet_pub_.publish(aglet_poses_msg);

    ROS_INFO_STREAM("Tsl bag node initialised");

    // start the bag loop
    // create a bag view for the rgb, depth and aglet topics
    rosbag::View view(bag, rosbag::TopicQuery({rgb_topic_, depth_topic_, aglet_topic_}));
    int count = 0;
    bool new_action = false;
    // create a dictionary to store the rgb, depth and aglet messages
    std::map<std::string, sensor_msgs::Image::ConstPtr> messages;
    messages["rgb"] = nullptr;
    messages["depth"] = nullptr;
    // create a pose array message to store the aglet poses
    std::vector<float> aglet_1_position_last_update = aglet_1_position;
    std::vector<float> aglet_2_position_last_update = aglet_2_position;
    // iterate through the messages
    for (rosbag::MessageInstance const m : view) {
        // check if count is in the skip frames
        if (std::find(skip_frames.begin(), skip_frames.end(), count) != skip_frames.end()) {
            count++;
            continue;
        }
        else {
            count++;
        }
        // process the message
        if (m.getTopic() == rgb_topic_) {
            messages["rgb"] = m.instantiate<sensor_msgs::Image>();
        } else if (m.getTopic() == depth_topic_) {
            messages["depth"] = m.instantiate<sensor_msgs::Image>();
        } else if (m.getTopic() == aglet_topic_) {
            geometry_msgs::PoseArray::ConstPtr aglet_msg = m.instantiate<geometry_msgs::PoseArray>();
            if (aglet_msg->poses[0].position.x != 0.0) {
                aglet_1_position = {aglet_msg->poses[0].position.x, 
                                    aglet_msg->poses[0].position.y, 
                                    aglet_msg->poses[0].position.z};
                aglet_poses_msg.poses[0] = aglet_msg->poses[0];
            }
            if (aglet_msg->poses[1].position.x != 0.0) {
                aglet_2_position = {aglet_msg->poses[1].position.x, 
                                    aglet_msg->poses[1].position.y, 
                                    aglet_msg->poses[1].position.z};
                aglet_poses_msg.poses[1] = aglet_msg->poses[1];
            }
            // check if this is a new action
            if (vecDist(aglet_1_position, aglet_1_position_last_update)>0.002 || 
                vecDist(aglet_2_position, aglet_2_position_last_update)>0.002) 
                new_action = true;
            aglet_pub_.publish(aglet_poses_msg);
        }
        // check if we have both rgb and depth messages
        if (messages["rgb"] != nullptr && messages["depth"] != nullptr) {
            // check if this is a new action
            if (new_action) {
                // update the last action
                aglet_1_position_last_update = aglet_1_position;
                aglet_2_position_last_update = aglet_2_position;
                // convert the rgb and depth messages to cv::Mat
                cv::Mat rgb_img = ImageToCvMat(messages["rgb"]);
                ProcessImage(rgb_img);
                cv::Mat depth_img = DepthToCvMat(messages["depth"]);
                // color segmentation of the rgb image
                std::vector<cv::Point> pixelCoordinates = hsv_segmenter_.retrievePoints(rgb_img);
                // get the segmented points
                Eigen::MatrixXf X = Retrieve3dPoints(pixelCoordinates, depth_img);
                // call the cpd algorithm
                Eigen::MatrixXf Y = tsl.step(X);
                // call unity adjust service
                tsl::SimAdjust adjust_srv;
                // convert the eigen matrix to a posearray message
                for (int i=0; i<Y.rows(); i++) {
                    geometry_msgs::Pose pose;
                    pose.position.x = tsl.Y(i, 0);
                    pose.position.y = tsl.Y(i, 1);
                    pose.position.z = tsl.Y(i, 2);
                    // geometry_msgs::Pose pose = eigenVec2PoseMsg(Y.row(i));
                    adjust_srv.request.states_est.poses.push_back(pose);
                }
                // call the adjust service
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
                // // publish result states
                // PointCloudMsg::Ptr result_states_msg (new PointCloudMsg);
                // pcl::PointCloud<pcl::PointXYZ> result_states;
                // for (int i=0; i<Y.rows(); i++) {
                //     pcl::PointXYZ point;
                //     point.x = Y(i, 0);
                //     point.y = Y(i, 1);
                //     point.z = Y(i, 2);
                //     result_states.push_back(point);
                // }
                // pcl::toROSMsg(result_states, *result_states_msg);
                // result_states_msg->header.frame_id = result_frame_;
                // result_states_pub_.publish(result_states_msg);
                // reset for the next action
                new_action = false;
                messages["rgb"] = nullptr;
                messages["depth"] = nullptr;

                // // visualise the states with opencv
                // if (viusalisation) {
                //     // cv::imshow("rgb", rgb_img);
                //     // create a blank image
                //     cv::Mat image = cv::Mat::zeros(resolution.y, resolution.x, CV_8UC3);
                //     // draw the pixelCoordinates on the image
                //     for (int i=0; i<pixelCoordinates.size(); i++) {
                //         cv::circle(image, pixelCoordinates[i], 1, cv::Scalar(0, 255, 0), -1);
                //     }
                //     // concatenate the rgb and image
                //     cv::Mat concat_img;
                //     cv::hconcat(rgb_img, image, concat_img);
                //     cv::imshow("result", concat_img);
                //     cv::waitKey(1);
                // }
            }
        }
        

    }
    // ros::spin();
}



Eigen::MatrixXf TslBag::InitialiseStates(const cv::Mat& init_img, const cv::Mat& init_depth)
{
    // // get the segmented points
    // Eigen::MatrixXf X = Retrieve3dPoints(init_img, init_depth);
    // // apply gaussian mixture model to the points
    // GaussianMixtureModel gmm = GaussianMixtureModel(num_state_points, maxIterations=10000);
    // gmm.fit(X);
    // // get the states from the gmm
    // return gmm.getMeans();
    return Eigen::MatrixXf();
}

void TslBag::ProcessImage(cv::Mat& image)
{
    // set region [270:270+154, 267:267+92, :] to zero
    cv::Mat roi = image(cv::Rect(267, 270, 92, 154));
    roi = cv::Mat::zeros(roi.size(), roi.type());
}

// void TslBag::ProcessImage(const sensor_msgs::ImageConstPtr& rgb_msg)
// {
//     // set region [270:270+154, 267:267+92, :] to zero
//     cv_bridge::CvImagePtr cv_ptr;
//     try {
//         cv_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
//     } catch (cv_bridge::Exception& e) {
//         ROS_ERROR("cv_bridge exception: %s", e.what());
//         return;
//     }
//     cv::Mat image = cv_ptr->image;
//     cv::Mat roi = image(cv::Rect(267, 270, 92, 154));
//     roi = cv::Mat::zeros(roi.size(), roi.type());    
// }

Eigen::MatrixXf TslBag::Retrieve3dPoints(const std::vector<cv::Point>& pixelCoordinates, const cv::Mat& depth)
{
    // extract the segmented points from the depth image
    PointCloud::Ptr points3D = camera.convertPixelsToPointCloud(pixelCoordinates, depth);
    
    // print the size of the 3D points
    // std::cout << "Size of 3D points: " << points3D->size() << std::endl;

    // downsample the points
    PointCloud::Ptr cloud_filtered(new PointCloud);
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(points3D);
    sor.setLeafSize(0.02f, 0.02f, 0.02f);
    sor.filter(*cloud_filtered);

    // print the size of the downsampled points
    // std::cout << "Size of downsampled points: " << cloud_filtered->size() << std::endl; 

    // convert the pcl point cloud to eigen matrix with 3 columns
    Eigen::MatrixXf X = cloud_filtered->getMatrixXfMap().topRows(3);
    return X.transpose();
}

cv::Mat TslBag::ImageToCvMat(const sensor_msgs::ImageConstPtr& msg)
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

cv::Mat TslBag::DepthToCvMat(const sensor_msgs::ImageConstPtr& msg)
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