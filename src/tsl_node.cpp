#include "tsl/tsl_node.h"

TslNode::TslNode():
    nh_{}
{
    // get parameters
    // get topics
    nh_.getParam("/tsl_node/rgb_topic", rgb_topic_);
    nh_.getParam("/tsl_node/depth_topic", depth_topic_);
    nh_.getParam("/tsl_node/camera_info_topic", camera_info_topic_);
    nh_.getParam("/tsl_node/eyelet_pose_topic", eyelet_topic_);
    nh_.getParam("/tsl_node/aglet_pose_topic", aglet_topic_);
    nh_.getParam("/tsl_node/segmented_pc_topic", segmented_pc_topic_);
    nh_.getParam("/tsl_node/result_pc_topic", result_pc_topic_);
    // get frames
    nh_.getParam("/tsl_node/result_frame", result_frame_);
    nh_.getParam("/tsl_node/robot_frame", robot_frame);
    nh_.getParam("/tsl_node/camera_frame", camera_frame);
    // get services
    nh_.getParam("/tsl_node/unity_reset_service_", unity_reset_service_);
    nh_.getParam("/tsl_node/unity_adjust_service_", unity_adjust_service_);
    // get tsl parameters
    nh_.getParam("/tsl_node/cam_pose", cam_pose);
    nh_.getParam("/tsl_node/num_state_points", num_state_points);
    nh_.getParam("/tsl_node/rope_length", rope_length);
    nh_.getParam("/tsl_node/rope_radius", rope_radius);
    nh_.getParam("/tsl_node/visualisation", viusalisation);
    // cpd parameters
    nh_.getParam("/tsl_node/alpha", tsl.alpha);
    nh_.getParam("/tsl_node/beta", tsl.beta);
    nh_.getParam("/tsl_node/gamma", tsl.gamma);
    nh_.getParam("/tsl_node/tolerance", tsl.tolerance);
    nh_.getParam("/tsl_node/max_iter", tsl.max_iter);
    nh_.getParam("/tsl_node/mu", tsl.mu);
    nh_.getParam("/tsl_node/k", tsl.k);
    // get segmentation parameters
    nh_.getParam("/tsl_node/segmentation/hue_min", hue_min_);
    nh_.getParam("/tsl_node/segmentation/hue_max", hue_max_);
    nh_.getParam("/tsl_node/segmentation/sat_min", sat_min_);
    nh_.getParam("/tsl_node/segmentation/sat_max", sat_max_);
    nh_.getParam("/tsl_node/segmentation/val_min", val_min_);
    nh_.getParam("/tsl_node/segmentation/val_max", val_max_);
    // get plot parameters
    nh_.getParam("/tsl_node/plot/use_plot", use_plot_);
    nh_.getParam("/tsl_node/plot/plot_topic", plot_topic_);
    nh_.getParam("/tsl_node/plot/x_min", plot_x_min_);
    nh_.getParam("/tsl_node/plot/x_max", plot_x_max_);
    nh_.getParam("/tsl_node/plot/y_min", plot_y_min_);
    nh_.getParam("/tsl_node/plot/y_max", plot_y_max_);
    // get the package path
    pkg_path_ = ros::package::getPath("tsl");
    tsl.pkg_path_ = pkg_path_;

    // initialise publishers
    result_img_pub_ = nh_.advertise<sensor_msgs::Image>(plot_topic_, 10);
    segmented_pc_pub_ = nh_.advertise<PointCloudMsg>(segmented_pc_topic_, 10);
    result_states_pub_ = nh_.advertise<PointCloudMsg>(result_pc_topic_, 10);
    // ros::Publisher aglet_pub_ = nh_.advertise<geometry_msgs::PoseArray>(aglet_topic_, 10);

    // initialise subscribers
    ros::Subscriber aglet_sub = nh_.subscribe<geometry_msgs::PoseArray>(aglet_topic_, 10,
                            &TslNode::AgletCallback, this);
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_(tf_buffer_);

    // initialise services
    ros::ServiceClient reset_client = nh_.serviceClient<tsl::SimReset>(unity_reset_service_);
    adjust_client = nh_.serviceClient<tsl::SimAdjust>(unity_adjust_service_);

    // wait for 1 second
    ros::Duration(1).sleep();

    // initialise the camera class
    sensor_msgs::CameraInfoConstPtr info_msg = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(camera_info_topic_, nh_);
    Eigen::Matrix3d intrinsicMatrix;
    intrinsicMatrix << info_msg->K[0], info_msg->K[1], info_msg->K[2],
                        info_msg->K[3], info_msg->K[4], info_msg->K[5],
                        info_msg->K[6], info_msg->K[7], info_msg->K[8];
    camera = Camera(intrinsicMatrix);
    ROS_INFO_STREAM("Loaded camera info");

    // synchronised subscribers for rgb and depth images
    // message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh_, rgb_topic_, 1);
    message_filters::Subscriber<sensor_msgs::CompressedImage> rgb_sub(nh_, rgb_topic_, 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh_, depth_topic_, 1);
    // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, 
    //                                     sensor_msgs::Image> MySyncPolicy;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::CompressedImage, 
                                        sensor_msgs::Image> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), rgb_sub, depth_sub);
    sync.registerCallback(boost::bind(&TslNode::RGBDCallback, this, _1, _2));
    
    // wait for the first RGB and depth messages
    // Wait for the first RGB message
    ROS_INFO_STREAM("Waiting for the first RGB message");
    // sensor_msgs::Image::ConstPtr rgb_msg = 
            // ros::topic::waitForMessage<sensor_msgs::Image>(rgb_topic_, nh_);
    sensor_msgs::CompressedImage::ConstPtr rgb_msg = 
            ros::topic::waitForMessage<sensor_msgs::CompressedImage>(rgb_topic_, nh_);
    // cv::Mat init_img = ImageToCvMat(rgb_msg);
    cv::Mat init_img = CompressedImageToCvMat(rgb_msg);
    resolution = cv::Point(init_img.cols, init_img.rows); // width, height
    // Wait for the first depth message
    ROS_INFO_STREAM("Waiting for the first depth message");
    sensor_msgs::Image::ConstPtr depth_msg = 
            ros::topic::waitForMessage<sensor_msgs::Image>(depth_topic_, nh_);
    cv::Mat init_depth = DepthToCvMat(depth_msg);

    // initialise the image segmentation
    std::vector<int> roi_temp, roni_temp;
    nh_.getParam("/tsl_node/segmentation/roi", roi_temp);
    nh_.getParam("/tsl_node/segmentation/roni", roni_temp);
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
    nh_.getParam("/tsl_node/init_method", init_method);
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
    // tsl.Y = InitialiseStates(init_img, init_depth, init_method);

    // wait for the first eyelet message
    ROS_INFO_STREAM("Waiting for the first eyelet message");
    geometry_msgs::PoseArray::ConstPtr eyelet_poses_msg;
    while (ros::ok()) {
        eyelet_poses_msg = ros::topic::waitForMessage<geometry_msgs::PoseArray>(eyelet_topic_, nh_);
        if (eyelet_poses_msg->poses.size() > 0) {
            break;
        }
    }
    // read the initial eyelet poses
    std::vector<geometry_msgs::Pose> eyelet_poses;
    for (int i=0; i<eyelet_poses_msg->poses.size(); i++) {
        eyelet_poses.push_back(eyelet_poses_msg->poses[i]);
    }

    // wait for the first aglet message
    ROS_INFO_STREAM("Waiting for the first aglet message");
    geometry_msgs::PoseArray::ConstPtr aglet_poses_msg;
    while (ros::ok()) {
        aglet_poses_msg = ros::topic::waitForMessage<geometry_msgs::PoseArray>(aglet_topic_, nh_);
        if (aglet_poses_msg->poses.size() <2 || aglet_poses_msg->poses[0].position.x == 0 ||
            aglet_poses_msg->poses[1].position.x == 0) {
            continue;
        }
        break;
    }
    // read the initial aglet poses
    aglet_1_position = std::vector<float>{aglet_poses_msg->poses[0].position.x, 
                                            aglet_poses_msg->poses[0].position.y, 
                                            aglet_poses_msg->poses[0].position.z};
    aglet_2_position = std::vector<float>{aglet_poses_msg->poses[1].position.x,
                                            aglet_poses_msg->poses[1].position.y,
                                            aglet_poses_msg->poses[1].position.z};
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
        reset_srv.request.states_est.poses.push_back(pose);
    }
    // get the cam2rob transform
    geometry_msgs::TransformStamped cam2rob_msg;
    try {
        cam2rob_msg = tf_buffer_.lookupTransform(robot_frame, camera_frame, ros::Time(0));
    } catch (tf2::TransformException &ex) {
        ROS_WARN("%s", ex.what());
    }
    reset_srv.request.cam2rob = cam2rob_msg.transform;
    reset_srv.request.rope_length.data = rope_length;
    reset_srv.request.rope_radius.data = rope_radius;
    // aglet poses here should be in camera frame for complex reasons
    geometry_msgs::PoseArray aglet_poses_msg_cam_frame = transformPoseArray(*aglet_poses_msg, getInverseTransform(cam2rob_msg));
    reset_srv.request.gripper_poses = aglet_poses_msg_cam_frame;
    reset_srv.request.eyelet_poses = *eyelet_poses_msg;
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
    } else {
        ROS_ERROR("Failed to call service reset");
    }

    ROS_INFO_STREAM("Tsl node initialised");
    ros::spin();
}
    
void TslNode::AgletCallback(const geometry_msgs::PoseArray::ConstPtr& msg)
{
    // get the aglet positions
    aglet_1_position = std::vector<float>{msg->poses[0].position.x, 
                                            msg->poses[0].position.y, 
                                            msg->poses[0].position.z};
    aglet_2_position = std::vector<float>{msg->poses[1].position.x,
                                            msg->poses[1].position.y,
                                            msg->poses[1].position.z};
    // check if there is a new action
    if (vecDist(aglet_1_position, aglet_1_position_last_update)>0.002 || 
        vecDist(aglet_2_position, aglet_2_position_last_update)>0.002) {
        new_action = true;
    }
}

// void TslNode::RGBDCallback(const sensor_msgs::ImageConstPtr& rgb_msg, 
//                             const sensor_msgs::ImageConstPtr& depth_msg)
void TslNode::RGBDCallback(const sensor_msgs::CompressedImageConstPtr& rgb_msg, 
                            const sensor_msgs::ImageConstPtr& depth_msg)
{
    // return if the node is shutdown
    if (!ros::ok() || updating) {
        return;
    }
    updating = true;
    // increment the frame count
    frame_count++;
    // check if there is a new action
    if (true) {
        aglet_1_position_last_update = aglet_1_position;
        aglet_2_position_last_update = aglet_2_position;
        key_frame_count++;
        new_action = false;

        // time the procedure
        auto start = std::chrono::high_resolution_clock::now();

        // convert the rgb and depth messages to cv::Mat
        // cv::Mat rgb_img = ImageToCvMat(rgb_msg);
        cv::Mat rgb_img = CompressedImageToCvMat(rgb_msg);
        cv::Mat depth_img = DepthToCvMat(depth_msg);
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

        // time adjust
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpd_duration = cpd_pause-start;
        std::chrono::duration<double> adjust_duration = stop-cpd_pause;
        std::chrono::duration<double> frame_duration = stop-start;
        frame_time_total += frame_duration.count();
        std::cout << "\r" << 
                "Frame " << std::setw(5) << std::setfill('0') << frame_count << 
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
    }

    updating = false;
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