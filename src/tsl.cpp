// created by Haining Luo

#include "tsl/tsl.h"

Tsl::Tsl(/* args */)
{
    std::cout << "Tsl class initialised" << std::endl;
}

void Tsl::CPD(const MatrixXf &X)
{
    // X: (N x 3) list of downsampled point cloud points
    // Y: (M x 3) list of control points
    // returns: (M x 3) list of points after cpd

    // PARAMS: should be ros params
    float beta = 1.0; // gaussian kernel width
    float gamma = 1.0; // weight of the LLE error
    float alpha = 0.5; // weight of the CPD error
    float tolerance = 1e-4; // tolerance for convergence
    // float tolerance = 0.0002; // tolerance for convergence
    max_iter = 50; // maximum number of iterations
    mu = 0.1; // outlier weight
    int k = 6; // number of nearest neighbours

    int N = X.rows();
    int M = Y.rows();
    int D = X.cols();

    // make a copy of the previous Y
    MatrixXf Y_prev = Y;

    // compute the distancem matrix dis_xy (M x N)
    MatrixXf dis_xy(M, N);
    for (int i=0; i<M; i++) {
        for (int j=0; j<N; j++) {
            dis_xy(i, j) = (Y.row(i) - X.row(j)).squaredNorm();
        }
    }

    // compute the distance matrix dis_yy (M x M)
    MatrixXf dis_yy = MatrixXf::Zero(M, M);
    for (int i=0; i<M; i++) {
        for (int j=i; j<M; j++) {
            dis_yy(i, j) = (Y.row(i) - Y.row(j)).squaredNorm(), dis_yy(j, i) = dis_yy(i, j);
        }
    }
    // std::cout << "dis_yy: " << dis_yy << std::endl;

    // initialise gaussian variance sigma2
    float sigma2 = dis_xy.sum() / D*M*N;
    std::cout << "sigma2 init: " << sigma2 << std::endl;

    // initialise gaussian kernel G
    // MatrixXf G(M, M);
    MatrixXf G = (-dis_yy / (2 * beta * beta)).array().exp();

    // initialise LLE matrix L
    MatrixXf L = MatrixXf::Zero(M, M);
    // MatrixXf W = MatrixXf::Zero(M, k);
    for (int i=0; i<M; i++) {
        // find the k nearest neighbours of Y(i, :)
        VectorXf dis_yy_i = dis_yy.col(i);
        std::vector<int> idx(dis_yy_i.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin()+k+1, idx.end(), 
                        [&dis_yy_i](int i1, int i2) {return dis_yy_i(i1) < dis_yy_i(i2);});

        // compute the weight matrix L
        MatrixXf Z = MatrixXf::Zero(k, D);
        for (int j=0; j<k; j++) {
            Z.row(j) = Y.row(idx[j+1]) - Y.row(i);
        }

        MatrixXf C = Z * Z.transpose();
        // TODO: cdcpd has a regulation term here
        VectorXf w = C.llt().solve(VectorXf::Ones(k));
        // L.row(i) = w/w.sum();
        for (int j=0; j<k; j++) {
            L(i, idx[j]) = w(j) / w.sum();
        }
    }
    // MatrixXf M_mat = (W.transpose() * W) - W.transpose() - W;
    // M_mat.diagonal().array() += 1;

    // initialise LLE matrix H (M x M)
    MatrixXf H = (MatrixXf::Identity(M, M) - L).transpose() * (MatrixXf::Identity(M, M) - L); // from trackdlo
    // std::cout << "H: " << H << std::endl;

    // start iterations
    bool converged = false;
    for (int i=0; i<max_iter; i++) {
        float sigma2_prev = sigma2;
        // update distance matrix dis_xy
        for (int i=0; i<M; i++) {
            for (int j=0; j<N; j++) {
                dis_xy(i, j) = (Y.row(i) - X.row(j)).squaredNorm();
            }
        }

        // E-step: estimate correspondences
        // Estimate the membership probability matrix P (M x N)
        MatrixXf P = (-0.5 * dis_xy / sigma2).array().exp();
        float c = pow((2 * M_PI * sigma2), static_cast<float>(D)/2) * mu / (1 - mu) * 
                    static_cast<float>(M)/N;
        P = P.array().rowwise() / (P.colwise().sum().array() + c);
        // std::cout << "P: " << P << std::endl;

        // M-step: estimate transformation
        VectorXf Pt1 = P.colwise().sum();
        VectorXf P1 = P.rowwise().sum();
        float Np = P1.sum();
        MatrixXf PX = P * X;

        MatrixXf p1d = P1.asDiagonal();

        MatrixXf A = (P1.asDiagonal() * G) + alpha * sigma2 * MatrixXf::Identity(M, M) + sigma2 * gamma * (H * G);

        MatrixXf B =
            PX - (p1d + sigma2 * gamma * H) * Y_prev;

        MatrixXf W = (A).householderQr().solve(B);
        // std::cout << "W: " << W << std::endl;

        MatrixXf Y_new = Y_prev + G * W;
        // std::cout << "Y_new: " << Y_new << std::endl;

        // update sigma2
        float trXtdPt1X = (X.transpose() * Pt1.asDiagonal() * X).trace();
        float trPXtT = (PX.transpose() * Y_new).trace();
        float trTtdP1T = (Y_new.transpose() * P1.asDiagonal() * Y_new).trace();
        std::cout << "trXtdPt1X: " << trXtdPt1X << std::endl;
        std::cout << "trPXtT: " << trPXtT << std::endl;
        std::cout << "trTtdP1T: " << trTtdP1T << std::endl;
        sigma2 = (trXtdPt1X - 2*trPXtT + trTtdP1T) / (Np * D);
        std::cout << "sigma2: " << sigma2 << std::endl;

        // if (sigma2 <= 0) {
        // sigma2 = tolerance / 10;
        // }

        // check for convergence
        float dis = (Y_new - Y).rowwise().norm().sum()/M;
        std::cout << "dis: " << dis << std::endl;
        Y = Y_new;
        if (std::abs(sigma2 - sigma2_prev)<tolerance)
        // if (dis < tolerance)
        {
            converged = true;
            std::cout << "Converged after " << i << " iterations" << std::endl;
            return;
        }
    }
    std::cout << "Finished without convergence" << std::endl;
}

MatrixXf Tsl::step(const MatrixXf &X)
{
    // X: (N x 3) list of downsampled point cloud points
    // Y: (M x 3) list of control points
    // returns: (M x 3) list of points after cpd
    // check if Y is empty
    if (Y.rows() == 0) {
        std::cout << "Y is empty" << std::endl;
        // initialise Y
        Y = X;
        return Y;
    }
    else {
        CPD(X);
        return Y;
    }

}