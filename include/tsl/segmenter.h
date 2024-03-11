#include <opencv2/opencv.hpp>

class ImageSegmenter {
public:
    ImageSegmenter(int hMin, int hMax, int sMin, int sMax, int vMin, int vMax)
        : hMin_(hMin), hMax_(hMax), sMin_(sMin), sMax_(sMax), vMin_(vMin), vMax_(vMax) {}

    cv::Mat segmentImage(const cv::Mat& inputImage) {
        cv::Mat hsvImage;
        cv::cvtColor(inputImage, hsvImage, cv::COLOR_BGR2HSV);

        cv::Mat mask;
        cv::inRange(hsvImage, cv::Scalar(hMin_, sMin_, vMin_), cv::Scalar(hMax_, sMax_, vMax_), mask);

        cv::Mat segmented;
        inputImage.copyTo(segmented, mask);

        return segmented;
    }

private:
    int hMin_;
    int hMax_;
    int sMin_;
    int sMax_;
    int vMin_;
    int vMax_;
};
