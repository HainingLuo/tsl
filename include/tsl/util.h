#include <vector>
#include <tf2/LinearMath/Quaternion.h>
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