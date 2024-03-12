#include <opencv2/opencv.hpp>

class ImageSegmenter {
public:
    // default constructor
    ImageSegmenter() {};

    ImageSegmenter(int hMin, int hMax, int sMin, int sMax, int vMin, int vMax)
        : hMin_(hMin), hMax_(hMax), sMin_(sMin), sMax_(sMax), vMin_(vMin), vMax_(vMax) {

        hMin1_ = hMin_ > hMax_ ? 0 : hMin_;
        hMax1_ = hMin_ > hMax_ ? 180 : hMax_;
        sMin1_ = sMin_ > sMax_ ? 0 : sMin_;
        sMax1_ = sMin_ > sMax_ ? 255 : sMax_;
        vMin1_ = vMin_ > vMax_ ? 0 : vMin_;
        vMax1_ = vMin_ > vMax_ ? 255 : vMax_;
    }

    cv::Mat segmentImage(const cv::Mat& inputImage) {
        cv::Mat hsvImage;
        cv::cvtColor(inputImage, hsvImage, cv::COLOR_BGR2HSV);

        cv::Mat mask1, mask2;
        cv::inRange(hsvImage, cv::Scalar(hMin_, sMin_, vMin_), cv::Scalar(hMax1_, sMax1_, vMax1_), mask1);
        cv::inRange(hsvImage, cv::Scalar(hMin1_, sMin1_, vMin1_), cv::Scalar(hMax_, sMax_, vMax_), mask2);
        cv::Mat mask = mask1 | mask2;

        return mask;
    }

    std::vector<cv::Point> retrievePoints(const cv::Mat& inputImage) {
        cv::Mat hsvImage;
        cv::cvtColor(inputImage, hsvImage, cv::COLOR_BGR2HSV);

        cv::Mat mask1, mask2;
        cv::inRange(hsvImage, cv::Scalar(hMin_, sMin_, vMin_), cv::Scalar(hMax1_, sMax1_, vMax1_), mask1);
        cv::inRange(hsvImage, cv::Scalar(hMin1_, sMin1_, vMin1_), cv::Scalar(hMax_, sMax_, vMax_), mask2);
        cv::Mat mask = mask1 | mask2;

        std::vector<cv::Point> pixelCoordinates;
        cv::findNonZero(mask, pixelCoordinates);

        return pixelCoordinates;
    }

private:
    int hMin_;
    int hMin1_;
    int hMax_;
    int hMax1_;
    int sMin_;
    int sMin1_;
    int sMax_;
    int sMax1_;
    int vMin_;
    int vMin1_;
    int vMax_;
    int vMax1_;
};
