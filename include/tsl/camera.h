#include <Eigen/Dense>
#include <vector>

class Camera {
public:
    Camera(const Eigen::Matrix3d& intrinsicMatrix) : intrinsicMatrix_(intrinsicMatrix) {}

    std::vector<Eigen::Vector3d> convertPixelsTo3D(const std::vector<Eigen::Vector2i>& pixelCoordinates,
                                                   const std::vector<Eigen::Vector3d>& rgbImage,
                                                   const std::vector<double>& depthImage) {
        std::vector<Eigen::Vector3d> points3D;
        for (const auto& pixel : pixelCoordinates) {
            int x = pixel.x();
            int y = pixel.y();

            double depth = depthImage[y * width_ + x];
            Eigen::Vector3d point3D = depthTo3D(x, y, depth);
            points3D.push_back(point3D);
        }
        return points3D;
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
