#include <vector>
#include <Eigen/Dense>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/Pose.h>

geometry_msgs::Point vec2PointMsg(const std::vector<float>& vec) {
    if (vec.size() != 3) {
        // Handle error: vector size is less than 3
        // You can throw an exception or return a default point
        // For simplicity, let's return a point with all coordinates set to 0
        return geometry_msgs::Point();
    }

    geometry_msgs::Point point;
    point.x = vec[0];
    point.y = vec[1];
    point.z = vec[2];
    return point;
}

geometry_msgs::Pose vec2PoseMsg(const std::vector<float>& vec) {
    if (vec.size() == 7) {
        geometry_msgs::Pose pose;
        pose.position.x = vec[0];
        pose.position.y = vec[1];
        pose.position.z = vec[2];
        pose.orientation.x = vec[3];
        pose.orientation.y = vec[4];
        pose.orientation.z = vec[5];
        pose.orientation.w = vec[6];
        return pose;
    } else if (vec.size() == 6) {
        geometry_msgs::Pose pose;
        pose.position.x = vec[0];
        pose.position.y = vec[1];
        pose.position.z = vec[2];
        // convert Euler angles to quaternion
        // Assuming the input is roll, pitch, yaw
        tf2::Quaternion q;
        q.setRPY(vec[3], vec[4], vec[5]);
        pose.orientation.x = q.x();
        pose.orientation.y = q.y();
        pose.orientation.z = q.z();
        pose.orientation.w = q.w();
        return pose;
    } else {
        // Handle error: vector size is neither 6 nor 7
        // You can throw an exception or return a default pose
        // For simplicity, let's return a pose with all coordinates set to 0
        return geometry_msgs::Pose();
    }
}

geometry_msgs::Point eigenVec2PointMsg(const Eigen::VectorXf &vec) {
    if (vec.size() != 3) {
        // Handle error: vector size is less than 3
        // You can throw an exception or return a default point
        // For simplicity, let's return a point with all coordinates set to 0
        return geometry_msgs::Point();
    }

    geometry_msgs::Point point;
    point.x = vec[0];
    point.y = vec[1];
    point.z = vec[2];
    return point;
}

geometry_msgs::Pose eigenVec2PoseMsg(const Eigen::VectorXf &vec) {
    if (vec.size() == 7) {
        geometry_msgs::Pose pose;
        pose.position.x = vec[0];
        pose.position.y = vec[1];
        pose.position.z = vec[2];
        pose.orientation.x = vec[3];
        pose.orientation.y = vec[4];
        pose.orientation.z = vec[5];
        pose.orientation.w = vec[6];
        return pose;
    } else if (vec.size() == 6) {
        geometry_msgs::Pose pose;
        pose.position.x = vec[0];
        pose.position.y = vec[1];
        pose.position.z = vec[2];
        // convert Euler angles to quaternion
        // Assuming the input is roll, pitch, yaw
        tf2::Quaternion q;
        q.setRPY(vec[3], vec[4], vec[5]);
        pose.orientation.x = q.x();
        pose.orientation.y = q.y();
        pose.orientation.z = q.z();
        pose.orientation.w = q.w();
        return pose;
    } else {
        // Handle error: vector size is neither 6 nor 7
        // You can throw an exception or return a default pose
        // For simplicity, let's return a pose with all coordinates set to 0
        return geometry_msgs::Pose();
    }
}

float vecDist(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    if (vec1.size() != vec2.size()) {
        // Handle error: vectors have different sizes
        // You can throw an exception or return a default distance
        // For simplicity, let's return -1
        return -1;
    }

    float dist = 0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        dist += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    }
    return sqrt(dist);
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

cv::Mat CompressedImageToCvMat(const sensor_msgs::CompressedImageConstPtr& msg)
{
    cv::Mat matrix = cv::imdecode(cv::Mat(msg->data),1);
    return matrix;
    // cv_bridge::CvImagePtr cv_ptr;
    // try {
    //     cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    // } catch (cv_bridge::Exception& e) {
    //     ROS_ERROR("cv_bridge exception: %s", e.what());
    //     return cv::Mat();
    // }
    // return cv_ptr->image;
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

geometry_msgs::Pose transformToPose(const geometry_msgs::Transform& transform) {
    geometry_msgs::Pose pose;
    pose.orientation = transform.rotation;
    pose.position.x = transform.translation.x;
    pose.position.y = transform.translation.y;
    pose.position.z = transform.translation.z;
    return pose;
}

geometry_msgs::Transform poseToTransform(const geometry_msgs::Pose& pose) {
    geometry_msgs::Transform transform;
    transform.rotation = pose.orientation;
    transform.translation.x = pose.position.x;
    transform.translation.y = pose.position.y;
    transform.translation.z = pose.position.z;
    return transform;
}

geometry_msgs::TransformStamped getInverseTransform(const geometry_msgs::TransformStamped& transform) {
    // Convert the transform to Eigen and get the inverse
    Eigen::Isometry3d eigen_transform;
    tf2::fromMsg(transformToPose(transform.transform), eigen_transform);
    Eigen::Isometry3d inverse_transform = eigen_transform.inverse();
    // Convert the inverse transform back to geometry_msgs::TransformStamped
    geometry_msgs::TransformStamped inverse;
    inverse.header = transform.header;
    inverse.child_frame_id = transform.header.frame_id;
    inverse.header.frame_id = transform.child_frame_id;
    geometry_msgs::Pose inverse_pose = tf2::toMsg(inverse_transform);
    // convert the pose to transform
    inverse.transform = poseToTransform(inverse_pose);
    return inverse;
}

geometry_msgs::Pose transformPose(const geometry_msgs::Pose& pose, const geometry_msgs::TransformStamped& transform) 
{
    // Convert the pose to Eigen
    Eigen::Isometry3d eigen_pose;
    tf2::fromMsg(pose, eigen_pose);
    // Convert the transform to Eigen
    Eigen::Isometry3d eigen_transform;
    tf2::fromMsg(transformToPose(transform.transform), eigen_transform);
    // Transform the pose
    Eigen::Isometry3d transformed_pose = eigen_transform * eigen_pose;
    // Convert the transformed pose back to geometry_msgs::Pose
    geometry_msgs::Pose transformed;
    transformed = tf2::toMsg(transformed_pose);
    return transformed;
}

geometry_msgs::PoseArray transformPoseArray(const geometry_msgs::PoseArray& pose_array, const geometry_msgs::TransformStamped& transform) 
{
    geometry_msgs::PoseArray transformed_array;
    transformed_array.header = pose_array.header;
    for (const auto& pose : pose_array.poses) {
        geometry_msgs::Pose transformed_pose = transformPose(pose, transform);
        transformed_array.poses.push_back(transformed_pose);
    }
    return transformed_array;
}