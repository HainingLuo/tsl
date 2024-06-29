#include "tsl/tsl_bag.h"

TslBag::TslBag():
    nh_{}
{
    // get parameters
    // get topics
    nh_.getParam("/tsl_bag/rgb_topic", rgb_topic_);
    nh_.getParam("/tsl_bag/depth_topic", depth_topic_);
    nh_.getParam("/tsl_bag/camera_info_topic", camera_info_topic_);
    nh_.getParam("/tsl_bag/eyelet_init_topic", eyelet_init_topic_);
    nh_.getParam("/tsl_bag/eyelet_pose_topic", eyelet_topic_);
    nh_.getParam("/tsl_bag/aglet_pose_topic", aglet_topic_);
    nh_.getParam("/tsl_bag/segmented_pc_topic", segmented_pc_topic_);
    nh_.getParam("/tsl_bag/result_pc_topic", result_pc_topic_);
    // get frames
    nh_.getParam("/tsl_bag/result_frame", result_frame_);
    nh_.getParam("/tsl_bag/robot_frame", robot_frame);
    nh_.getParam("/tsl_bag/camera_frame", camera_frame);
    // get services
    nh_.getParam("/tsl_bag/unity_reset_service_", unity_reset_service_);
    nh_.getParam("/tsl_bag/unity_adjust_service_", unity_adjust_service_);
    nh_.getParam("/tsl_bag/unity_predict_service_", unity_predict_service_);
    // get paths
    nh_.getParam("/tsl_bag/bag_path", bag_path);
    nh_.getParam("/tsl_bag/bag_config_path", bag_config_path);
    // get tsl parameters
    nh_.getParam("/tsl_bag/cam_pose", cam_pose);
    nh_.getParam("/tsl_bag/num_state_points", num_state_points);
    nh_.getParam("/tsl_bag/rope_length", rope_length);
    nh_.getParam("/tsl_bag/rope_radius", rope_radius);
    nh_.getParam("/tsl_bag/skip_frames", skip_frames);
    nh_.getParam("/tsl_bag/visualisation", viusalisation);
    // cpd parameters
    nh_.getParam("/tsl_bag/alpha", tsl.alpha);
    nh_.getParam("/tsl_bag/beta", tsl.beta);
    nh_.getParam("/tsl_bag/gamma", tsl.gamma);
    nh_.getParam("/tsl_bag/tolerance", tsl.tolerance);
    nh_.getParam("/tsl_bag/max_iter", tsl.max_iter);
    nh_.getParam("/tsl_bag/mu", tsl.mu);
    nh_.getParam("/tsl_bag/k", tsl.k);
    // get segmentation parameters
    nh_.getParam("/tsl_bag/segmentation/hue_min", hue_min_);
    nh_.getParam("/tsl_bag/segmentation/hue_max", hue_max_);
    nh_.getParam("/tsl_bag/segmentation/sat_min", sat_min_);
    nh_.getParam("/tsl_bag/segmentation/sat_max", sat_max_);
    nh_.getParam("/tsl_bag/segmentation/val_min", val_min_);
    nh_.getParam("/tsl_bag/segmentation/val_max", val_max_);
    // get plot parameters
    nh_.getParam("/tsl_bag/plot/use_plot", use_plot_);
    nh_.getParam("/tsl_bag/plot/plot_topic", plot_topic_);
    nh_.getParam("/tsl_bag/plot/x_min", plot_x_min_);
    nh_.getParam("/tsl_bag/plot/x_max", plot_x_max_);
    nh_.getParam("/tsl_bag/plot/y_min", plot_y_min_);
    nh_.getParam("/tsl_bag/plot/y_max", plot_y_max_);
    // get the package path
    pkg_path_ = ros::package::getPath("tsl");
    tsl.pkg_path_ = pkg_path_;

    // initialise publishers
    result_img_pub_ = nh_.advertise<sensor_msgs::Image>(plot_topic_, 10);
    segmented_pc_pub_ = nh_.advertise<PointCloudMsg>(segmented_pc_topic_, 10);
    result_states_pub_ = nh_.advertise<PointCloudMsg>(result_pc_topic_, 10);
    ros::Publisher aglet_pub_ = nh_.advertise<geometry_msgs::PoseArray>(aglet_topic_, 10);
    ros::Publisher eyelet_init_pub = nh_.advertise<geometry_msgs::PoseArray>(eyelet_init_topic_, 10);

    // initialise subscribers
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_(tf_buffer_);

    // initialise services
    ros::ServiceClient reset_client = nh_.serviceClient<tsl::SimReset>(unity_reset_service_);
    adjust_client = nh_.serviceClient<tsl::SimAdjust>(unity_adjust_service_);

    // wait for 1 second
    ros::Duration(1).sleep();

    // load the ROS bag
    rosbag::Bag bag;
    bag.open(bag_path, rosbag::bagmode::Read);
    ROS_INFO_STREAM("Opened bag: " + bag_path);

    // read the number of rgb_topic_ and depth_topic_ messages in the bag
    rosbag::View view_image(bag, rosbag::TopicQuery({rgb_topic_}));
    rosbag::View view_depth(bag, rosbag::TopicQuery({depth_topic_}));
    num_messages = view_image.size()+view_depth.size();
    ROS_INFO_STREAM("Number of messages in the bag: " << num_messages);

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
    
    // initialise the image segmentation
    std::vector<int> roi_temp, roni_temp;
    nh_.getParam("/tsl_bag/segmentation/roi", roi_temp);
    nh_.getParam("/tsl_bag/segmentation/roni", roni_temp);
    cv::Rect roi(0, 0, resolution.x, resolution.y);
    if (roi_temp.size() != 4) {
        ROS_INFO("ROI not set, using the whole image");
    } else {
        roi = cv::Rect(roi_temp[0], roi_temp[1], roi_temp[2], roi_temp[3]);
    }
    int n_roni = roni_temp.size()/4;
    std::vector<cv::Rect> roni;
    for (int i=0; i<n_roni; i++) {
        cv::Rect rect(roni_temp[i*4], roni_temp[i*4+1], roni_temp[i*4+2], roni_temp[i*4+3]);
        roni.push_back(rect);
    }
    hsv_segmenter_ = ImageSegmenter(hue_min_, hue_max_, sat_min_, sat_max_, val_min_, val_max_, 
                                    roi, roni);

    // initialise the states
    std::string init_method;
    nh_.getParam("/tsl_bag/init_method", init_method);
    // get the segmented points
    cv::Mat init_mask = hsv_segmenter_.segmentImage(init_img);
    // call the initialisation method
    if (init_method == "GeneticAlgorithm") {
        std::vector<cv::Point> pixelCoordinates = hsv_segmenter_.findNonZero(init_mask);
        Eigen::MatrixXf X = camera.pixels2EigenMatDownSampled(pixelCoordinates, init_depth);
        tsl.InitialiseStatesGA(X, num_state_points);
    } else if (init_method == "SkeletonInterpolation") {
        // get the 3D coordinates of all the pixels
        std::vector<cv::Point> all_pixelCoordinates;
        for (int i=0; i<init_mask.rows; i++) {
            for (int j=0; j<init_mask.cols; j++) {
                all_pixelCoordinates.push_back(cv::Point(j, i));
            }
        }
        Eigen::MatrixXf coordinates3D = camera.pixels2EigenMat(all_pixelCoordinates, init_depth);
        tsl.InitialiseStatesSI(init_mask, coordinates3D, num_state_points);
    } else {
        ROS_ERROR("Invalid initialisation method");
    }

    // read the initial eyelet poses from the bag
    rosbag::View view_eyelet(bag, rosbag::TopicQuery({eyelet_topic_}));
    // rosbag::View::iterator eyelet_it = view_eyelet.begin();
    // loop through the eyelet messages till one that has a non empty pose array
    bool eyelet_found = false;
    geometry_msgs::PoseArray::ConstPtr eyelet_poses_msg_ptr;
    for (rosbag::MessageInstance const eyelet_m : view_eyelet) {
        // eyelet_it++;
        // rosbag::MessageInstance const eyelet_m = *eyelet_it;
        eyelet_poses_msg_ptr = eyelet_m.instantiate<geometry_msgs::PoseArray>();
        if (eyelet_poses_msg_ptr->poses.size() != 0) {
            eyelet_found = true;
            break;
        }
    }
    if (!eyelet_found) {
        ROS_ERROR("No eyelet poses found in the bag");
        // return;
    }
    geometry_msgs::PoseArray eyelet_poses_msg = *eyelet_poses_msg_ptr;
    std::vector<geometry_msgs::Pose> eyelet_poses;
    for (int i=0; i<eyelet_poses_msg.poses.size(); i++) {
        eyelet_poses.push_back(eyelet_poses_msg.poses[i]);
    }
    
    // read the initial aglet poses from the bag
    rosbag::View view_aglet(bag, rosbag::TopicQuery({aglet_topic_}));
    // rosbag::View::iterator aglet_it = view_aglet.begin();
    // loop through the aglet messages till one that has a pose array with 2 poses, both has non zero x
    bool aglet_found = false;
    geometry_msgs::PoseArray::ConstPtr aglet_poses_msg_ptr;
    // while (!aglet_found) {
    for (rosbag::MessageInstance const aglet_m : view_aglet) {
        // aglet_it++;
        // rosbag::MessageInstance const aglet_m = *aglet_it;
        aglet_poses_msg_ptr = aglet_m.instantiate<geometry_msgs::PoseArray>();
        if (aglet_poses_msg_ptr->poses.size() == 2 && 
            aglet_poses_msg_ptr->poses[0].position.x != 0.0 && 
            aglet_poses_msg_ptr->poses[1].position.x != 0.0) {
            aglet_found = true;
            break;
        }
    }
    if (!aglet_found) {
        ROS_ERROR("No aglet poses found in the bag");
        return;
    }
    geometry_msgs::PoseArray aglet_poses_msg = *aglet_poses_msg_ptr;
    std::vector<float> aglet_1_position = {aglet_poses_msg.poses[0].position.x, 
                                            aglet_poses_msg.poses[0].position.y, 
                                            aglet_poses_msg.poses[0].position.z};
    std::vector<float> aglet_2_position = {aglet_poses_msg.poses[1].position.x,
                                            aglet_poses_msg.poses[1].position.y,
                                            aglet_poses_msg.poses[1].position.z};
    // print the initial aglet poses
    ROS_INFO_STREAM("Initial aglet 1 position: " << aglet_1_position[0] << ", " << aglet_1_position[1] << ", " << aglet_1_position[2]);

    // // read bag_config_path yaml file
    // YAML::Node config = YAML::LoadFile(bag_config_path);

    // // read the initial eyelet poses
    // std::vector<geometry_msgs::Pose> eyelet_poses;
    // for (int i=0; i<config["eyelets_init"].size(); i++) {
    //     geometry_msgs::Pose pose;
    //     pose = vec2PoseMsg(config["eyelets_init"][i].as<std::vector<float>>());
    //     eyelet_poses.push_back(pose);
    // }
    // geometry_msgs::PoseArray eyelet_poses_msg;
    // eyelet_poses_msg.header.frame_id = result_frame_;
    // eyelet_poses_msg.poses = eyelet_poses;
    // // read the initial aglet poses
    // geometry_msgs::PoseArray aglet_poses_msg;
    // geometry_msgs::Pose aglet_1_pose, aglet_2_pose;
    // aglet_1_position = config["aglet_1_init"].as<std::vector<float>>();
    // aglet_2_position = config["aglet_2_init"].as<std::vector<float>>();
    // aglet_1_pose.position = vec2PointMsg(aglet_1_position);
    // aglet_2_pose.position = vec2PointMsg(aglet_2_position);
    // aglet_poses_msg.header.frame_id = robot_frame;
    // aglet_poses_msg.poses.push_back(aglet_1_pose);
    // aglet_poses_msg.poses.push_back(aglet_2_pose);
    aglet_1_position_last_update = aglet_1_position;
    aglet_2_position_last_update = aglet_2_position;

    // reset the simulation
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
    // get the cam2rob transform
    geometry_msgs::TransformStamped cam2rob_msg;
    cam2rob_msg.header.stamp = ros::Time::now();
    cam2rob_msg.header.frame_id = robot_frame;
    cam2rob_msg.child_frame_id = camera_frame;
    cam2rob_msg.transform.translation.x = cam_pose[0];
    cam2rob_msg.transform.translation.y = cam_pose[1];
    cam2rob_msg.transform.translation.z = cam_pose[2];
    cam2rob_msg.transform.rotation.x = cam_pose[3];
    cam2rob_msg.transform.rotation.y = cam_pose[4];
    cam2rob_msg.transform.rotation.z = cam_pose[5];
    cam2rob_msg.transform.rotation.w = cam_pose[6];
    reset_srv.request.cam2rob = cam2rob_msg.transform;
    reset_srv.request.rope_length.data = rope_length;
    reset_srv.request.rope_radius.data = rope_radius;
    reset_srv.request.gripper_poses = aglet_poses_msg;
    reset_srv.request.eyelet_poses = eyelet_poses_msg;
    reset_srv.request.gripper_states.data.push_back(0.0); // 0.0: open, 1.0: close
    reset_srv.request.gripper_states.data.push_back(0.0);
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
    // eyelet_init_pub.publish(eyelet_poses_msg);
    // aglet_pub_.publish(aglet_poses_msg);

    std::cout << "Tsl bag node initialised" << std::endl;

    // start the bag loop
    // create a bag view for the rgb, depth and aglet topics
    rosbag::View view(bag, rosbag::TopicQuery({rgb_topic_, depth_topic_, aglet_topic_}));
    // create a dictionary to store the rgb, depth and aglet messages
    std::map<std::string, sensor_msgs::Image::ConstPtr> messages;
    messages["rgb"] = nullptr;
    messages["depth"] = nullptr;
    // iterate through the messages
    for (rosbag::MessageInstance const m : view) {
        // return if the node is shutdown
        if (!ros::ok()) {
            return;
        }
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
                try {
                    // time the procedure
                    auto start = std::chrono::high_resolution_clock::now();

                    // update the last action
                    aglet_1_position_last_update = aglet_1_position;
                    aglet_2_position_last_update = aglet_2_position;
                    // convert the rgb and depth messages to cv::Mat
                    cv::Mat rgb_img = ImageToCvMat(messages["rgb"]);
                    cv::Mat depth_img = DepthToCvMat(messages["depth"]);
                    // color segmentation of the rgb image
                    std::vector<cv::Point> pixelCoordinates = hsv_segmenter_.retrievePoints(rgb_img);
                    // get the segmented points
                    Eigen::MatrixXf X = camera.pixels2EigenMatDownSampled(pixelCoordinates, depth_img);
                    // call the cpd algorithm
                    Eigen::MatrixXf Y = tsl.step(X);

                    // time cpd
                    auto cpd_pause = std::chrono::high_resolution_clock::now();

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
                    // reset for the next action
                    new_action = false;
                    messages["rgb"] = nullptr;
                    messages["depth"] = nullptr;
                    key_frame_count++;

                    // time adjust
                    auto stop = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> cpd_duration = cpd_pause-start;
                    std::chrono::duration<double> adjust_duration = stop-cpd_pause;
                    std::chrono::duration<double> frame_duration = stop-start;
                    frame_time_total += frame_duration.count();
                    std::cout << "\r" << 
                            "Frame " << std::setw(5) << std::setfill('0') << count << 
                            ": " << std::fixed << std::setprecision(3) << frame_duration.count() << "s" << 
                            " (cpd: " << std::fixed << std::setprecision(3) << cpd_duration.count() << "s)" <<
                            " (adjust: " << std::fixed << std::setprecision(3) << adjust_duration.count() << "s)" <<
                            std::flush;

                    // visualise the states with opencv
                    if (use_plot_) {
                        // cv::imshow("rgb", rgb_img);
                        // create a blank image
                        cv::Mat image = cv::Mat::zeros(resolution.y, resolution.x, CV_8UC3);
                        // draw the states on the image
                        for (int i=0; i<Y.rows(); i++) {
                            // rescale the tsl.Y from plot_x_min_ to plot_x_max_ to 0 to resolution.x
                            float x = (Y(i, 0)-plot_x_min_)/(plot_x_max_-plot_x_min_)*resolution.x;
                            float y = (Y(i, 1)-plot_y_min_)/(plot_y_max_-plot_y_min_)*resolution.y;
                            cv::circle(image, cv::Point((int) x, (int) y), 1, cv::Scalar(0, 255, 0), -1);
                        }
                        // concatenate the rgb and image
                        cv::Mat concat_img;
                        cv::hconcat(rgb_img, image, concat_img);
                        // publish the result image to result_img_pub_
                        sensor_msgs::ImagePtr result_img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", concat_img).toImageMsg();
                        result_img_pub_.publish(result_img_msg);
                        
                        // cv::imshow("result", concat_img);
                        // cv::waitKey(1);
                    }

                    // publish X for debug
                    PointCloudMsg::Ptr X_msg (new PointCloudMsg);
                    pcl::PointCloud<pcl::PointXYZ> X_cloud;
                    for (int i=0; i<X.rows(); i++) {
                        pcl::PointXYZ point;
                        point.x = X(i, 0);
                        point.y = X(i, 1);
                        point.z = X(i, 2);
                        X_cloud.push_back(point);
                    }
                    pcl::toROSMsg(X_cloud, *X_msg);
                    X_msg->header.frame_id = result_frame_;
                    segmented_pc_pub_.publish(X_msg);

                    // publish the result states
                    PointCloudMsg::Ptr result_states_msg (new PointCloudMsg);
                    pcl::PointCloud<pcl::PointXYZ> result_states;
                    for (int i=0; i<Y.rows(); i++) {
                        pcl::PointXYZ point;
                        point.x = Y(i, 0);
                        point.y = Y(i, 1);
                        point.z = Y(i, 2);
                        result_states.push_back(point);
                    }
                    pcl::toROSMsg(result_states, *result_states_msg);
                    result_states_msg->header.frame_id = result_frame_;
                    result_states_pub_.publish(result_states_msg);
                } catch (const std::exception& e) {
                    ROS_ERROR_STREAM("Exception caught: " << e.what());
                    // close the bag
                    bag.close();
                    return;
                }
            }
        }
    }
    // print the average frame time
    std::cout << std::endl;
    std::cout << "Average frame time: " << frame_time_total/count << std::endl;
    std::cout << "Average key frame time: " << frame_time_total/key_frame_count << std::endl;
}


// Eigen::MatrixXf TslBag::InitialiseStates(const cv::Mat& init_img, const cv::Mat& init_depth, const std::string method)
// {
//     // Initialize the Python interpreter
//     py::scoped_interpreter guard{};
//     py::module sys = py::module::import("sys");
//     sys.attr("path").attr("insert")(0, pkg_path_ +"/scripts");
//     // Import the Python module
//     py::module module = py::module::import("estimate_initial_states");

//     // get the segmented points
//     cv::Mat mask = hsv_segmenter_.segmentImage(init_img);
//     std::vector<cv::Point> pixelCoordinates = hsv_segmenter_.findNonZero(mask);
//     // Convert num_state_points to a Python variable
//     py::int_ num_state_points_var(num_state_points);

//     py::object result;
//     // Call the Python function and pass the NumPy array and num_state_points as arguments
//     if (method == "GeneticAlgorithm") {
//         // get the 3D points
//         Eigen::MatrixXf X = camera.pixels2EigenMatDownSampled(pixelCoordinates, init_depth);
//         // Convert Eigen::MatrixXf to NumPy array
//         Eigen::MatrixXf X_transposed = X.transpose();
//         py::array np_array({X.rows(), X.cols()}, X_transposed.data());
//         result = module.attr("estimate_initial_states_ga")(np_array, num_state_points_var);
//     }
//     else if (method == "SkeletonInterpolation") {
//         // convert the mask to a NumPy array
//         py::array_t<uint8_t> np_array({mask.rows, mask.cols}, mask.data);
//         // get the 3D coordinates of all the pixels
//         std::vector<cv::Point> all_pixelCoordinates;
//         for (int i=0; i<mask.rows; i++) {
//             for (int j=0; j<mask.cols; j++) {
//                 all_pixelCoordinates.push_back(cv::Point(j, i));
//             }
//         }
//         // get the 3D points
//         Eigen::MatrixXf coordinates3D = camera.pixels2EigenMat(all_pixelCoordinates, init_depth);
//         // convert the 3D coordinates to a NumPy array
//         Eigen::MatrixXf coordinates3D_transposed = coordinates3D.transpose();
//         // extract the data from the Eigen matrix
//         int D = coordinates3D.cols();
//         py::array_t<float> np_array_3d({mask.rows, mask.cols, D}, coordinates3D_transposed.data());
//         //
//         result = module.attr("estimate_initial_states_si")(np_array, np_array_3d, num_state_points_var);
//     }
//     else {
//         // print error
//         std::cerr << "Invalid method: " << method << std::endl;
//         return Eigen::MatrixXf();
//     }

//     // Convert the result to a C++ variable
//     Eigen::MatrixXf output = Eigen::Map<Eigen::MatrixXf>(
//         static_cast<float*>(result.cast<py::array_t<float>>().request().ptr),
//         result.cast<py::array_t<float>>().shape(1),
//         result.cast<py::array_t<float>>().shape(0)
//     );

//     return output.transpose();
// }


int main(int argc, char **argv) {
    ros::init(argc, argv, "tsl_bag");
    ROS_INFO_STREAM("Initialising tsl bag node");
    TslBag tsl_bag;
    return 0;
}