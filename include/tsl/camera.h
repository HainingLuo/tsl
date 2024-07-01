#include <Eigen/Dense>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using PointCloudMsg = sensor_msgs::PointCloud2;
using PointCloud = pcl::PointCloud<pcl::PointXYZ>;

class Camera {
public:
    Camera() {};

    Camera(const Eigen::Matrix3d& intrinsicMatrix) : intrinsicMatrix_(intrinsicMatrix) {}

    Eigen::Matrix3d intrinsicMatrix_;
    
    /**
     * Convert a list of pixel coordinates to a point cloud.
     * 
     * @param pixelCoordinates: list of pixel coordinates in the format (u, v)
     * @param depthImage: depth image
     * @return point cloud
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr pixels2PointCloud(
        const std::vector<cv::Point>& pixelCoordinates,
        const cv::Mat& depthImage) 
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZ>);

        for (const auto& pixel : pixelCoordinates) {
            int u = pixel.x;
            int v = pixel.y;

            double depth = readDepth(u, v, depthImage);
            Eigen::Vector3d point3D = pixel2Point(u, v, depth);

            pcl::PointXYZ pclPoint;
            pclPoint.x = point3D.x();
            pclPoint.y = point3D.y();
            pclPoint.z = point3D.z();

            pointCloud->push_back(pclPoint);
        }
        return pointCloud;
    }

    /**
     * Convert a list of pixel coordinates to an eigen matrix.
     * 
     * @note: Pixels are assumed to be in the format (u, v) 
     * where u is the column index and v is the row index.
     * @param pixelCoordinates: list of pixel coordinates
     * @param depthImage: depth image
     * @return eigen matrix with 3 columns
     */
    Eigen::MatrixXf pixels2EigenMat(const std::vector<cv::Point>& pixelCoordinates, 
                                    const cv::Mat& depthImage)
    {
        /*
        Convert a list of pixel coordinates to 3D points.
        Pixels are assumed to be in the format (u, v) where u is the column index and v is the row index.
        */
        // // extract the segmented points from the depth image
        // PointCloud::Ptr points3D = pixels2PointCloud(pixelCoordinates, depthImage);
        // // convert the pcl point cloud to eigen matrix with 3 columns
        // Eigen::MatrixXf eigen_mat = points3D->getMatrixXfMap().topRows(3);
        // return eigen_mat.transpose();
        Eigen::MatrixXf points3D(pixelCoordinates.size(), 3);
        for (int i = 0; i < pixelCoordinates.size(); i++) {
            int u = pixelCoordinates[i].x;
            int v = pixelCoordinates[i].y;

            double depth = readDepth(u, v, depthImage);
            Eigen::Vector3d point3D = pixel2Point(u, v, depth);
            points3D.row(i) = point3D.cast<float>().transpose();
        }
        return points3D;
    }

    /**
     * Convert a list of pixel coordinates to a downsampled point cloud.
     * 
     * @param pixelCoordinates: list of pixel coordinates in the format (u, v)
     * @param depth: depth image
     * @param leafSize: leaf size for downsampling
     * @return downsampled point cloud
     */
    Eigen::MatrixXf pixels2EigenMatDownSampled(const std::vector<cv::Point>& pixelCoordinates, 
                                                const cv::Mat& depth,
                                                const float leafSize = 0.02f)
    {
        // extract the segmented points from the depth image
        PointCloud::Ptr points3D = pixels2PointCloud(pixelCoordinates, depth);

        // downsample the points
        PointCloud::Ptr cloud_filtered(new PointCloud);
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(points3D);
        sor.setLeafSize(leafSize, leafSize, leafSize);
        sor.filter(*cloud_filtered);

        // convert the pcl point cloud to eigen matrix with 3 columns
        Eigen::MatrixXf eigen_mat = cloud_filtered->getMatrixXfMap().topRows(3);
        return eigen_mat.transpose();
    }

private:
    int width_;  // width of the images
    int height_; // height of the images

    /**
     * Get the depth value at a pixel
     * 
     * @param u: column index of the pixel
     * @param v: row index of the pixel
     * @param depthImage: depth image
     * @param region: check depth value in a region around the pixel
     * @param mode: 0 for mean depth, 1 for median depth
     * @return depth value at the pixel
    */
    double readDepth(int u, int v, const cv::Mat& depthImage, int region=3, int mode=1) 
    {
        std::vector<double> depth;
        for (int i = -region; i <= region; i++) {
            for (int j = -region; j <= region; j++) {
                int d_t = 0.001*depthImage.at<u_int16_t>(v + i, u + j);
                if (d_t != 0 && !std::isnan(d_t)) {
                    depth.push_back(d_t);
                }
            }
        }
        if (depth.size() == 0) {
            return -1;
        }
        else {
            if (mode == 0) {
                return std::accumulate(depth.begin(), depth.end(), 0.0) / depth.size();
            }
            else {
                std::sort(depth.begin(), depth.end());
                return depth[depth.size() / 2];
            }
        }
        
        // double depth = 0.001*depthImage.at<u_int16_t>(v, u);
        // // remove nan and zero depth values
        // if (depth == 0 || std::isnan(depth)) {
        //     return -1;
        // }
        // return depth;        
    }

    /**
     * Convert a pixel coordinate and depth value to a 3D point.
     * 
     * @param u: column index of the pixel
     * @param v: row index of the pixel
     * @param depth: depth value at the pixel
     * @return 3D point in the camera frame
    */
    Eigen::Vector3d pixel2Point(int u, int v, double depth) 
    {
        Eigen::Vector3d point3D;
        point3D.x() = (u - intrinsicMatrix_(0, 2)) * depth / intrinsicMatrix_(0, 0);
        point3D.y() = (v - intrinsicMatrix_(1, 2)) * depth / intrinsicMatrix_(1, 1);
        point3D.z() = depth;
        return point3D;
    }
};
