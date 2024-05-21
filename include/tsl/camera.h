#include <Eigen/Dense>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class Camera {
public:
    // default constructor
    Camera() {};

    Camera(const Eigen::Matrix3d& intrinsicMatrix) : intrinsicMatrix_(intrinsicMatrix) {}

    Eigen::MatrixXf convertPixelsTo3D(const std::vector<Eigen::Vector2i>& pixelCoordinates,
                                      const cv::Mat& depthImage) {
        Eigen::MatrixXf points3D(pixelCoordinates.size(), 3);
        for (int i = 0; i < pixelCoordinates.size(); i++) {
            int x = pixelCoordinates[i].x();
            int y = pixelCoordinates[i].y();

            double depth = depthImage.at<double>(y, x);
            Eigen::Vector3d point3D = depthTo3D(x, y, depth);
            points3D.row(i) = point3D.cast<float>().transpose();
        }
        return points3D;
    }
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr convertPixelsToPointCloud(const std::vector<cv::Point>& pixelCoordinates,
                                                                  const cv::Mat& depthImage) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZ>);

        for (const auto& pixel : pixelCoordinates) {
            int x = pixel.x;
            int y = pixel.y;

            double depth = depthImage.at<double>(y, x);
            Eigen::Vector3d point3D = depthTo3D(x, y, depth);

            pcl::PointXYZ pclPoint;
            pclPoint.x = point3D.x();
            pclPoint.y = point3D.y();
            pclPoint.z = point3D.z();

            pointCloud->push_back(pclPoint);
        }

        return pointCloud;
    }

private:
    Eigen::Matrix3d intrinsicMatrix_;
    int width_;  // width of the images
    int height_; // height of the images

    Eigen::Vector3d depthTo3D(int x, int y, double depth) {
        Eigen::Vector3d point3D;
        point3D.x() = (x - intrinsicMatrix_(0, 2)) * depth / intrinsicMatrix_(0, 0);
        point3D.y() = (y - intrinsicMatrix_(1, 2)) * depth / intrinsicMatrix_(1, 1);
        point3D.z() = depth;
        return point3D;
    }
};
