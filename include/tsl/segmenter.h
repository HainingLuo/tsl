#include <opencv2/opencv.hpp>

class ImageSegmenter {
public:
    ImageSegmenter() {};

    ImageSegmenter(int hMin, int hMax, int sMin, int sMax, int vMin, int vMax
        , cv::Rect roi = cv::Rect(0, 0, 0, 0),
        std::vector<cv::Rect> roni = std::vector<cv::Rect>())
        : hMin_(hMin), hMax_(hMax), sMin_(sMin), sMax_(sMax), vMin_(vMin), vMax_(vMax) {

        hMin1_ = hMin_ > hMax_ ? 0 : hMin_;
        hMax1_ = hMin_ > hMax_ ? 180 : hMax_;
        sMin1_ = sMin_ > sMax_ ? 0 : sMin_;
        sMax1_ = sMin_ > sMax_ ? 255 : sMax_;
        vMin1_ = vMin_ > vMax_ ? 0 : vMin_;
        vMax1_ = vMin_ > vMax_ ? 255 : vMax_;

        roi_ = roi;
        roni_ = roni;
    }

    /**
     * Segment the input image based on the HSV color space.
     * 
     * @param inputImage: input image in BGR format
     * @return binary mask of the segmented image
    */
    cv::Mat segmentImage(const cv::Mat& inputImage) {
        cv::Mat processedImage = inputImage;
        // Crop the image if a region of interest is specified
        if (roi_.area() > 0) {
            processedImage = inputImage(roi_);
        }
        // set to blank if regions of no interest are specified
        for (const auto& roi : roni_) {
            processedImage(roi) = cv::Scalar(0, 0, 0);
        }
        cv::Mat hsvImage;
        cv::cvtColor(inputImage, hsvImage, cv::COLOR_BGR2HSV);
        cv::Mat mask;
        cv::inRange(hsvImage, cv::Scalar(hMin_, sMin_, vMin_), cv::Scalar(hMax_, sMax_, vMax_), mask);
        return mask;
    }

    /**
     * Retrieve the pixel coordinates of the non-zero pixels in the masked image.
     * 
     * @param inputImage: input image in BGR format
     * @return vector of pixel coordinates
    */
    std::vector<cv::Point> retrievePoints(const cv::Mat& inputImage) {
        /* Find the non-zero pixels in the masked image.
            Output is a vector of cv::Point objects. */
        cv::Mat mask = segmentImage(inputImage);
        return findNonZero(mask);
    }

    /**
     * Find the non-zero pixels in a binary image.
     * 
     * @param inputImage: binary image
     * @return vector of pixel coordinates
    */
    std::vector<cv::Point> findNonZero(const cv::Mat& inputImage) {
        std::vector<cv::Point> pixelCoordinates;
        cv::findNonZero(inputImage, pixelCoordinates);
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
    cv::Rect roi_; // Region of interest
    std::vector<cv::Rect> roni_; // List of regions of no interest
};
