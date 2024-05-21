#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

class GaussianMixtureModel {
public:
    GaussianMixtureModel(int numComponents, int maxIterations = 100) : numComponents(numComponents) {
        means.resize(numComponents);
    }

    void fit(const MatrixXf& points) {
        int numPoints = points.rows();
        int numDimensions = points.cols();

        // Initialize the GMM parameters
        covariances.resize(numComponents, MatrixXd::Identity(numDimensions, numDimensions));
        weights.resize(numComponents, 1.0 / numComponents);

        // Run the Expectation Maximization algorithm
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            // Expectation step
            MatrixXf responsibilities(numPoints, numComponents);
            for (int i = 0; i < numPoints; i++) {
                VectorXd point = points.row(i).transpose();
                for (int j = 0; j < numComponents; j++) {
                    responsibilities(i, j) = weights[j] * multivariateNormalPDF(point, means[j], covariances[j]);
                }
                responsibilities.row(i) /= responsibilities.row(i).sum();
            }

            // Maximization step
            for (int j = 0; j < numComponents; j++) {
                double sumResponsibilities = responsibilities.col(j).sum();
                means[j] = (points.transpose() * responsibilities.col(j)) / sumResponsibilities;
                covariances[j] = MatrixXd::Zero(numDimensions, numDimensions);
                for (int i = 0; i < numPoints; i++) {
                    VectorXd point = points.row(i).transpose();
                    VectorXd diff = point - means[j];
                    covariances[j] += responsibilities(i, j) * (diff * diff.transpose());
                }
                covariances[j] /= sumResponsibilities;
                weights[j] = sumResponsibilities / numPoints;
            }
        }
    }

    MatrixXf getMeans() const {
        MatrixXf result(means.size(), means[0].size());
        for (int i = 0; i < means.size(); i++) {
            result.row(i) = means[i].transpose();
        }
        return result;
    }

private:
    int numComponents;
    std::vector<VectorXd> means;
    std::vector<MatrixXd> covariances;
    std::vector<double> weights;
    const int maxIterations = 100;

    double multivariateNormalPDF(const VectorXd& x, const VectorXd& mean, const MatrixXd& covariance) {
        double exponent = -0.5 * (x - mean).transpose() * covariance.inverse() * (x - mean);
        double coefficient = std::pow(2 * M_PI, -x.size() / 2.0) * std::pow(covariance.determinant(), -0.5);
        return coefficient * std::exp(exponent);
    }
};