// created by Haining Luo

#include "tsl/tsl.h"

// Tsl::Tsl(float alpha, float beta, float gamma, float tolerance, int max_iter, double mu, int k)
// {
//     this->alpha = alpha;
//     this->beta = beta;
//     this->gamma = gamma;
//     this->tolerance = tolerance;
//     this->max_iter = max_iter;
//     this->mu = mu;
//     this->k = k;
//     std::cout << "Tsl class initialised" << std::endl;
// }

void Tsl::CPD(const MatrixXf &X)
{
    // X: (N x 3) list of downsampled point cloud points
    // Y: (M x 3) list of control points
    // returns: (M x 3) list of points after cpd

    // PARAMS: should be ros params
    float beta = 0.5; // gaussian kernel width
    float gamma = 1.2; //1.0; // weight of the LLE error
    float alpha = 1.0; // 0.5; // weight of the CPD error
    float tolerance = 1e-4; // tolerance for convergence
    // float tolerance = 0.0002; // tolerance for convergence
    max_iter = 100; // 50; // maximum number of iterations
    mu = 0.1; // outlier weight
    int k = 8; // number of nearest neighbours // 6 

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
    float sigma2 = dis_xy.sum() / (D*M*N);
    // std::cout << "sigma2 init: " << sigma2 << std::endl;

    // initialise gaussian kernel G
    // MatrixXf G(M, M);
    MatrixXf G = (-dis_yy / (2 * beta * beta)).array().exp();

    // initialise LLE matrix L
    // From https://cs.nyu.edu/~roweis/lle/algorithm.html
    float regulation_term = 1e-3;
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
        // add regulation term
        if (G.trace() > 0) {
            C.diagonal().array() += regulation_term * G.trace();
        } 
        VectorXf w = C.llt().solve(VectorXf::Ones(k));
        // L.row(i) = w/w.sum();
        for (int j=0; j<k; j++) {
            L(i, idx[j]) = w(j) / w.sum();
        }
    }
    // MatrixXf M_mat = (W.transpose() * W) - W.transpose() - W;
    // M_mat.diagonal().array() += 1;

    // initialise LLE matrix H (M x M)
    MatrixXf H = (MatrixXf::Identity(M, M) - L).transpose() * (MatrixXf::Identity(M, M) - L);
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
        VectorXf Pt1 = P.colwise().sum(); // Pt1 (N)
        VectorXf P1 = P.rowwise().sum(); // P1 (M)
        float Np = P1.sum();
        MatrixXf PX = P * X; // PX (M x D)

        MatrixXf p1d = P1.asDiagonal();

        // A (M x M)
        MatrixXf A = (P1.asDiagonal() * G) + alpha * sigma2 * MatrixXf::Identity(M, M) + sigma2 * gamma * (H * G) + 10*G;
        // MatrixXf A = (P1.asDiagonal() * G) + alpha * sigma2 * MatrixXf::Identity(M, M) + sigma2 * gamma * (H * G);

        // B (M x D)
        MatrixXf B = PX - (p1d + sigma2 * gamma * H) * Y_prev;

        MatrixXf W = (A).householderQr().solve(B); // W (M x D)
        // std::cout << "W: " << W << std::endl;

        MatrixXf Y_new = Y_prev + G * W; // Y_new (M x D)
        // std::cout << "Y_new: " << Y_new << std::endl;

        // update sigma2
        // float trXtdPt1X = (X.transpose() * Pt1.asDiagonal() * X).trace();
        // float trPXtT = (PX.transpose() * Y_new).trace();
        // float trTtdP1T = (Y_new.transpose() * P1.asDiagonal() * Y_new).trace();
        Eigen::VectorXf xPxtemp = (X.array()*X.array()).rowwise().sum();
        double trXtdPt1X = Pt1.dot(xPxtemp);
        // std::cout << "trXtdPt1X: " << trXtdPt1X << std::endl;
        Eigen::VectorXf yPytemp = (Y_new.array()*Y_new.array()).rowwise().sum();
        double trTtdP1T = P1.dot(yPytemp);
        // std::cout << "trTtdP1T: " << trTtdP1T << std::endl;
        double trPXtT = (Y_new.transpose() * PX).trace();
        // std::cout << "trPXtT: " << trPXtT << std::endl;
        sigma2 = (trXtdPt1X - 2*trPXtT + trTtdP1T) / (Np * D);
        // std::cout << "sigma2: " << sigma2 << std::endl;

        // if (sigma2 <= 0) {
        // sigma2 = tolerance / 10;
        // }

        // check for convergence
        float dis = (Y_new - Y).rowwise().norm().sum()/M;
        // std::cout << "dis: " << dis << std::endl;
        Y = Y_new;
        if (std::abs(sigma2 - sigma2_prev)<tolerance)
        // if (dis < tolerance)
        {
            converged = true;
            // std::cout << "Converged after " << i << " iterations" << std::endl;
            return;
        }
    }
    // std::cout << "Finished without convergence" << std::endl;
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


/** 
 * Initialise the states of the rope with Genetic Algorithm 
 * 
 * @param X: downsampled point cloud in the form of a matrix (N x 3)
 * @param num_state_points: number of control points
 * @return Y: control points in the form of a matrix (M x 3)
*/
void Tsl::InitialiseStatesGA(const Eigen::MatrixXf& X, const int num_state_points)
{
    // Initialize the Python interpreter
    py::scoped_interpreter guard{};
    py::module sys = py::module::import("sys");
    sys.attr("path").attr("insert")(0, pkg_path_ +"/scripts");
    // Import the Python module
    py::module module = py::module::import("estimate_initial_states");

    // Convert num_state_points to a Python variable
    py::int_ num_state_points_var(num_state_points);
    // Convert Eigen::MatrixXf to NumPy array
    Eigen::MatrixXf X_transposed = X.transpose();
    py::array np_array({X.rows(), X.cols()}, X_transposed.data());
    py::object result = module.attr("estimate_initial_states_ga")(np_array, num_state_points_var);

    // Convert the result to a C++ variable
    Eigen::MatrixXf output = Eigen::Map<Eigen::MatrixXf>(
        static_cast<float*>(result.cast<py::array_t<float>>().request().ptr),
        result.cast<py::array_t<float>>().shape(1),
        result.cast<py::array_t<float>>().shape(0)
    );
    Y = output.transpose();
}

/** 
 * Initialise the states of the rope with skeletonisation
 * 
 * @param mask: binary mask of the shoelace
 * @param coordinates3D: 3D coordinates of the segmented pixels (? x 3) (not downsampled)
 * @param num_state_points: number of control points
 * @return Y: control points in the form of a matrix (M x 3)
*/
void Tsl::InitialiseStatesSI(const cv::Mat& mask, 
                                        const Eigen::MatrixXf& coordinates3D, 
                                        const int num_state_points)
{
    // Initialize the Python interpreter
    py::scoped_interpreter guard{};
    py::module sys = py::module::import("sys");
    sys.attr("path").attr("insert")(0, pkg_path_ +"/scripts");
    // Import the Python module
    py::module module = py::module::import("estimate_initial_states");

    // Convert num_state_points to a Python variable
    py::int_ num_state_points_var(num_state_points);
    // convert the mask to a NumPy array
    py::array_t<uint8_t> np_array({mask.rows, mask.cols}, mask.data);
    // convert the 3D coordinates to a NumPy array
    Eigen::MatrixXf coordinates3D_transposed = coordinates3D.transpose();
    int D = coordinates3D.cols();
    py::array_t<float> np_array_3d({mask.rows, mask.cols, D}, coordinates3D_transposed.data());
    py::object result = module.attr("estimate_initial_states_si")(np_array, np_array_3d, 
                                                        num_state_points_var);

    // Convert the result to a C++ variable
    Eigen::MatrixXf output = Eigen::Map<Eigen::MatrixXf>(
        static_cast<float*>(result.cast<py::array_t<float>>().request().ptr),
        result.cast<py::array_t<float>>().shape(1),
        result.cast<py::array_t<float>>().shape(0)
    );
    Y = output.transpose();
}
