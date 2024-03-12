//  definition for the cpd class
//
//  Created by Haining Luo.

#ifndef TSL_H
#define TSL_H

#include <iostream>
#include <vector>
#include <numeric>
#include <Eigen/Dense>

using Eigen::Matrix3Xf;
using Eigen::MatrixXf;
using Eigen::VectorXf;

class Tsl
{
private:
    int max_iter; // maximum number of iterations
    double mu; // outlier weight
    MatrixXf Y; // control points

public:
    Tsl(/* args */);
    
    // ~Tsl();

    void CPD(const MatrixXf &X);
    MatrixXf step(const MatrixXf &X);
};


#endif // TSL_H