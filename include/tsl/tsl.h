//  definition for the cpd class
//
//  Created by Haining Luo.

#ifndef TSL_H
#define TSL_H

#include <iostream>
#include <vector>
#include <numeric>
#include <Eigen/Dense>
#include <chrono>

using Eigen::Matrix3Xf;
using Eigen::MatrixXf;
using Eigen::VectorXf;

class Tsl
{
// private:

public:
    // Tsl(float alpha, float beta, float gamma, float tolerance, int max_iter, double mu, int k);
    
    // ~Tsl();
    float beta = 0.5; // gaussian kernel width
    float gamma = 1.2; //1.0; // weight of the LLE error
    float alpha = 1.0; // 0.5; // weight of the CPD error
    float tolerance = 1e-4; // tolerance for convergence
    // float tolerance = 0.0002; // tolerance for convergence
    int max_iter = 100; // 50; // maximum number of iterations
    double mu = 0.1; // outlier weight
    int k = 8; // number of nearest neighbours // 6 

    MatrixXf Y; // control points
    void CPD(const MatrixXf &X);
    MatrixXf step(const MatrixXf &X);
};


#endif // TSL_H