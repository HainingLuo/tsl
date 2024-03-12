#include <opencv2/opencv.hpp>

class ImageSegmenter {
public:
    ImageSegmenter(int hMin, int hMax, int sMin, int sMax, int vMin, int vMax)
        : hMin_(hMin), hMax_(hMax), sMin_(sMin), sMax_(sMax), vMin_(vMin), vMax_(vMax) {

        hMin1_ = 0 if hMin_ > hMax_ else hMin_;
        hMax1_ = 180 if hMin_ > hMax_ else hMax_;
        sMin1_ = 0 if sMin_ > sMax_ else sMin_;
        sMax1_ = 255 if sMin_ > sMax_ else sMax_;
        vMin1_ = 0 if vMin_ > vMax_ else vMin_;
        vMax1_ = 255 if vMin_ > vMax_ else vMax_;
    }

    cv::Mat segmentImage(const cv::Mat& inputImage) {
        cv::Mat hsvImage;
        cv::cvtColor(inputImage, hsvImage, cv::COLOR_BGR2HSV);

        cv::Mat mask1;
        cv::inRange(hsvImage, cv::Scalar(hMin_, sMin_, vMin_), cv::Scalar(hMax1_, sMax1_, vMax1_), mask1);
        cv::inRange(hsvImage, cv::Scalar(hMin1_, sMin1_, vMin1_), cv::Scalar(hMax_, sMax_, vMax_), mask2);
        cv::Mat mask = mask1 | mask2;

        return mask;
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
