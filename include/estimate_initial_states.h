#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

typedef cv::Point Point_2D;

// struct Point_2D {
//     double x;
//     double y;

//     Point_2D(double x, double y) : x(x), y(y) {}
// };

struct Rect {
    Point_2D start;
    Point_2D end;

    Rect(Point_2D start, Point_2D end) : start(start), end(end) {}
};

bool onSegment(Point_2D p, Point_2D q, Point_2D r) {
    if ((q.x <= std::max(p.x, r.x)) && (q.x >= std::min(p.x, r.x)) &&
        (q.y <= std::max(p.y, r.y)) && (q.y >= std::min(p.y, r.y))) {
        return true;
    }
    return false;
}

int orientation(Point_2D p, Point_2D q, Point_2D r) {
    // To find the orientation of an ordered triplet (p, q, r)
    // The function returns the following values:
    // 0: Collinear points
    // 1: Clockwise points
    // 2: Counterclockwise points

    // See https://www.geeksforgeeks.org/orientation-3-ordered-points/ for details of the below formula.

    double val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if (val > 0) {
        // Clockwise orientation
        return 1;
    } else if (val < 0) {
        // Counterclockwise orientation
        return 2;
    } else {
        // Collinear orientation
        return 0;
    }
}

bool doIntersect(Point_2D p1, Point_2D q1, Point_2D p2, Point_2D q2) {
    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);

    if ((o1 != o2) && (o3 != o4)) {
        return true;
    }

    if ((o1 == 0) && onSegment(p1, p2, q1)) {
        return true;
    }

    if ((o2 == 0) && onSegment(p1, q2, q1)) {
        return true;
    }

    if ((o3 == 0) && onSegment(p2, p1, q2)) {
        return true;
    }

    if ((o4 == 0) && onSegment(p2, q1, q2)) {
        return true;
    }

    return false;
}

std::vector<Point_2D> build_rect(Point_2D pt1, Point_2D pt2, double width) {
    double line_angle = std::atan2(pt2.y - pt1.y, pt2.x - pt1.x);
    double angle1 = line_angle + M_PI / 2;
    double angle2 = line_angle - M_PI / 2;
    Point_2D rect_pt1(pt1.x + width / 2.0 * std::cos(angle1), pt1.y + width / 2.0 * std::sin(angle1));
    Point_2D rect_pt2(pt1.x + width / 2.0 * std::cos(angle2), pt1.y + width / 2.0 * std::sin(angle2));
    Point_2D rect_pt3(pt2.x + width / 2.0 * std::cos(angle1), pt2.y + width / 2.0 * std::sin(angle1));
    Point_2D rect_pt4(pt2.x + width / 2.0 * std::cos(angle2), pt2.y + width / 2.0 * std::sin(angle2));

    return {rect_pt1, rect_pt2, rect_pt4, rect_pt3};
}

bool check_rect_overlap(Rect rect1, Rect rect2) {
    if (rect1.start.x > rect2.end.x || rect1.end.x < rect2.start.x) {
        return false;
    }
    if (rect1.start.y > rect2.end.y || rect1.end.y < rect2.start.y) {
        return false;
    }
    return true;
}

double compute_cost(const std::vector<Point_2D>& chain1, const std::vector<Point_2D>& chain2, double w_e, double w_c, int mode) {
    // start + start
    if (mode == 0) {
        double cost_euclidean = std::sqrt(std::pow(chain1[0].x - chain2[0].x, 2) + std::pow(chain1[0].y - chain2[0].y, 2));
        double cost_curvature_1 = std::acos((chain1[0].x - chain2[0].x) * (chain1[1].x - chain1[0].x) + (chain1[0].y - chain2[0].y) * (chain1[1].y - chain1[0].y)) / (std::sqrt(std::pow(chain1[0].x - chain1[1].x, 2) + std::pow(chain1[0].y - chain1[1].y, 2)) * cost_euclidean);
        double cost_curvature_2 = std::acos((chain1[0].x - chain2[0].x) * (chain2[0].x - chain2[1].x) + (chain1[0].y - chain2[0].y) * (chain2[0].y - chain2[1].y)) / (std::sqrt(std::pow(chain2[0].x - chain2[1].x, 2) + std::pow(chain2[0].y - chain2[1].y, 2)) * cost_euclidean);
        double total_cost = w_e * cost_euclidean + w_c * (std::abs(cost_curvature_1) + std::abs(cost_curvature_2)) / 2.0;
        return total_cost;
    }
    // start + end
    else if (mode == 1) {
        double cost_euclidean = std::sqrt(std::pow(chain1[0].x - chain2.back().x, 2) + std::pow(chain1[0].y - chain2.back().y, 2));
        double cost_curvature_1 = std::acos((chain1[0].x - chain2.back().x) * (chain1[1].x - chain1[0].x) + (chain1[0].y - chain2.back().y) * (chain1[1].y - chain1[0].y)) / (std::sqrt(std::pow(chain1[0].x - chain1[1].x, 2) + std::pow(chain1[0].y - chain1[1].y, 2)) * cost_euclidean);
        double cost_curvature_2 = std::acos((chain1[0].x - chain2.back().x) * (chain2.back().x - chain2[chain2.size() - 2].x) + (chain1[0].y - chain2.back().y) * (chain2.back().y - chain2[chain2.size() - 2].y)) / (std::sqrt(std::pow(chain2.back().x - chain2[chain2.size() - 2].x, 2) + std::pow(chain2.back().y - chain2[chain2.size() - 2].y, 2)) * cost_euclidean);
        double total_cost = w_e * cost_euclidean + w_c * (std::abs(cost_curvature_1) + std::abs(cost_curvature_2)) / 2.0;
        return total_cost;
    }
    // end + start
    else if (mode == 2) {
        double cost_euclidean = std::sqrt(std::pow(chain1.back().x - chain2[0].x, 2) + std::pow(chain1.back().y - chain2[0].y, 2));
        double cost_curvature_1 = std::acos((chain2[0].x - chain1.back().x) * (chain1.back().x - chain1[chain1.size() - 2].x) + (chain2[0].y - chain1.back().y) * (chain1.back().y - chain1[chain1.size() - 2].y)) / (std::sqrt(std::pow(chain1.back().x - chain1[chain1.size() - 2].x, 2) + std::pow(chain1.back().y - chain1[chain1.size() - 2].y, 2)) * cost_euclidean);
        double cost_curvature_2 = std::acos((chain2[0].x - chain1.back().x) * (chain2[1].x - chain2[0].x) + (chain2[0].y - chain1.back().y) * (chain2[1].y - chain2[0].y)) / (std::sqrt(std::pow(chain2[0].x - chain2[1].x, 2) + std::pow(chain2[0].y - chain2[1].y, 2)) * cost_euclidean);
        double total_cost = w_e * cost_euclidean + w_c * (std::abs(cost_curvature_1) + std::abs(cost_curvature_2)) / 2.0;
        return total_cost;
    }
    // end + end
    else {
        double cost_euclidean = std::sqrt(std::pow(chain1.back().x - chain2.back().x, 2) + std::pow(chain1.back().y - chain2.back().y, 2));
        double cost_curvature_1 = std::acos((chain2.back().x - chain1.back().x) * (chain1.back().x - chain1[chain1.size() - 2].x) + (chain2.back().y - chain1.back().y) * (chain1.back().y - chain1[chain1.size() - 2].y)) / (std::sqrt(std::pow(chain1.back().x - chain1[chain1.size() - 2].x, 2) + std::pow(chain1.back().y - chain1[chain1.size() - 2].y, 2)) * cost_euclidean);
        double cost_curvature_2 = std::acos((chain2.back().x - chain1.back().x) * (chain2[chain2.size() - 2].x - chain2.back().x) + (chain2.back().y - chain1.back().y) * (chain2[chain2.size() - 2].y - chain2.back().y)) / (std::sqrt(std::pow(chain2.back().x - chain2[chain2.size() - 2].x, 2) + std::pow(chain2.back().y - chain2[chain2.size() - 2].y, 2)) * cost_euclidean);
        double total_cost = w_e * cost_euclidean + w_c * (std::abs(cost_curvature_1) + std::abs(cost_curvature_2)) / 2.0;
        return total_cost;
    }
}

// implement linear sum assignment algorithm
// reference: https://en.wikipedia.org/wiki/Hungarian_algorithm
void linear_sum_assignment(const std::vector<std::vector<double>>& cost_matrix, std::vector<int>& row_idx, std::vector<int>& col_idx) {
    int n = cost_matrix.size();
    std::vector<std::vector<double>> cost_matrix_copy = cost_matrix;
    std::vector<int> u(n, 0);
    std::vector<int> v(n, 0);
    std::vector<int> p(n, 0);
    std::vector<int> way(n, 0);
    for (int i = 1; i < n; i++) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(n, std::numeric_limits<double>::max());
        std::vector<char> used(n, false);
        do {
            used[j0] = true;
            int i0 = p[j0];
            double delta = std::numeric_limits<double>::max();
            int j1 = 0;
            for (int j = 1; j < n; j++) {
                if (!used[j]) {
                    double cur = cost_matrix_copy[i0][j] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for (int j = 0; j < n; j++) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }
    row_idx.resize(n);
    col_idx.resize(n);
    for (int j = 0; j < n; j++) {
        row_idx[p[j]] = j;
    }
    for (int i = 0; i < n; i++) {
        col_idx[i] = p[i];
    }
}

// partial implementation of paper "Deformable One-Dimensional Object Detection for Routing and Manipulation"
// paper link: https://ieeexplore.ieee.org/abstract/document/9697357
std::vector<std::vector<Point_2D>> estimate_initial_states(cv::Mat mask, int img_scale, float seg_length, int max_curvature, bool visualize_process) {

    cv::Mat smoothed_im;

    // Smooth image
    cv::Mat im(mask.size(), CV_8UC1, mask.data);
    cv::blur(im, smoothed_im, cv::Size(2, 2));

    // Perform skeletonization
    cv::Mat gray;
    cv::threshold(smoothed_im, gray, 100, 255, cv::THRESH_BINARY);
    cv::Mat skeleton;
    cv::ximgproc::thinning(gray, skeleton, cv::ximgproc::THINNING_ZHANGSUEN);

    // Extract contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(skeleton, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // Traverse contours and extract chains
    std::vector<std::vector<cv::Point>> chains;
    for (const auto& contour : contours) {
        // double c_area = cv::contourArea(contour);
        cv::Mat chain_mask = cv::Mat::zeros(gray.size(), CV_8UC1);

        cv::Point last_segment_dir;
        std::vector<cv::Point> chain;
        cv::Point cur_seg_start_point;

        for (int i = 0; i < contour.size(); i++) {
            // Special case: if reached the end of current contour, append chain if not empty
            if (i == contour.size() - 1) {
                if (!chain.empty()) {
                    chains.push_back(chain);
                }
                break;
            }

            // Graph for visualization
            cv::line(chain_mask, contour[i], contour[i + 1], cv::Scalar(255), 1);

            // If haven't initialized, perform initialization
            if (cur_seg_start_point == cv::Point()) {
                cur_seg_start_point = contour[i];
            }

            // Keep traversing until reach segment length
            if (cv::norm(contour[i] - cur_seg_start_point) <= seg_length) {
                continue;
            }

            cv::Point cur_seg_end_point = contour[i];

            // Record segment direction
            cv::Point cur_segment_dir = cur_seg_end_point - cur_seg_start_point;
            if (last_segment_dir == cv::Point()) {
                last_segment_dir = cur_segment_dir;
            }

            // If direction hasn't changed much, add current segment to chain
            else if (cv::norm(cur_segment_dir) * cv::norm(last_segment_dir) != 0 &&
                     cv::norm(cur_segment_dir.dot(last_segment_dir)) /
                         (cv::norm(cur_segment_dir) * cv::norm(last_segment_dir)) >=
                         std::cos(max_curvature / 180 * CV_PI)) {
                // If chain is empty, append both start and end point
                if (chain.empty()) {
                    chain.push_back(cur_seg_start_point);
                    chain.push_back(cur_seg_end_point);
                }
                // If chain is not empty, only append end point
                else {
                    chain.push_back(cur_seg_end_point);
                }

                // Next start point <- end point
                cur_seg_start_point = cur_seg_end_point;

                // Update last segment direction
                last_segment_dir = cur_segment_dir;
            }

            // If direction changed, start a new chain
            else {
                // Append current chain to chains if chain is not empty
                if (!chain.empty()) {
                    chains.push_back(chain);
                }

                // Reinitialize all variables
                last_segment_dir = cv::Point();
                chain.clear();
                cur_seg_start_point = cv::Point();
            }
        }
    }

    std::cout << "Finished contour traversal. Pruning extracted chains..." << std::endl;
    std::cout << "Number of chains: " << chains.size() << std::endl;


    std::vector<double> all_chain_length;
    std::unordered_map<std::pair<Point_2D, Point_2D>, Rect> line_seg_to_rect_dict;
    double rect_width = 3;

    for (auto& chain : chains) {
        double chain_length = 0;
        for (int i = 0; i < chain.size() - 1; i++) {
            chain_length += std::sqrt(std::pow(chain[i + 1].x - chain[i].x, 2) + std::pow(chain[i + 1].y - chain[i].y, 2));
        }
        all_chain_length.push_back(chain_length);

        for (int i = 0; i < chain.size() - 1; i++) {
            Point_2D start = chain[i];
            Point_2D end = chain[i + 1];
            line_seg_to_rect_dict[std::make_pair(start, end)] = Rect(start, end);
        }
    }

    std::sort(all_chain_length.begin(), all_chain_length.end());
    std::vector<std::vector<Point_2D>> sorted_chains;
    for (int i = 0; i < all_chain_length.size(); i++) {
        sorted_chains.push_back(chains[i]);
    }

    std::vector<std::vector<Point_2D>> pruned_chains;
    for (int i = 0; i < chains.size(); i++) {
        std::vector<std::vector<Point_2D>> leftover_chains;
        std::vector<Point_2D> cur_chain = sorted_chains.back();
        sorted_chains.pop_back();

        for (int j = 0; j < sorted_chains.size(); j++) {
            std::vector<Point_2D> test_chain = sorted_chains[j];
            std::vector<Point_2D> new_test_chain;
            for (int l = 0; l < test_chain.size() - 1; l++) {
                Rect rect_test_seg = line_seg_to_rect_dict[std::make_pair(test_chain[l], test_chain[l + 1])];
                bool no_overlap = true;
                for (int k = 0; k < cur_chain.size() - 1; k++) {
                    Rect rect_cur_seg = line_seg_to_rect_dict[std::make_pair(cur_chain[k], cur_chain[k + 1])];
                    if (check_rect_overlap(rect_cur_seg, rect_test_seg)) {
                        no_overlap = false;
                        break;
                    }
                }
                if (no_overlap) {
                    if (new_test_chain.empty()) {
                        new_test_chain.push_back(test_chain[l]);
                        new_test_chain.push_back(test_chain[l + 1]);
                    } else {
                        new_test_chain.push_back(test_chain[l + 1]);
                    }
                }
            }
            leftover_chains.push_back(new_test_chain);
        }

        if (!cur_chain.empty()) {
            pruned_chains.push_back(cur_chain);
        }

        all_chain_length.clear();
        for (auto& chain : leftover_chains) {
            if (!chain.empty()) {
                double chain_length = 0;
                for (int i = 0; i < chain.size() - 1; i++) {
                    chain_length += std::sqrt(std::pow(chain[i + 1].x - chain[i].x, 2) + std::pow(chain[i + 1].y - chain[i].y, 2));
                }
                all_chain_length.push_back(chain_length);
            } else {
                all_chain_length.push_back(0);
            }
        }

        std::sort(all_chain_length.begin(), all_chain_length.end());
        sorted_chains.clear();
        for (int i = 0; i < all_chain_length.size(); i++) {
            sorted_chains.push_back(leftover_chains[i]);
        }
    }

    std::cout << "Finished pruning. Merging remaining chains..." << std::endl;

    // if (visualize_process) {
    //     cv::Mat mask(gray.rows, gray.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    //     for (const auto& chain : pruned_chains) {
    //         cv::Scalar color(int(std::rand() % 200) + 55, int(std::rand() % 200) + 55, int(std::rand() % 200) + 55);
    //         for (int i = 0; i < chain.size() - 1; i++) {
    //             cv::line(mask, chain[i], chain[i + 1], color, 1);
    //         }
    //     }
    //     cv::imshow("after pruning", mask);
    //     while (true) {
    //         int key = cv::waitKey(10);
    //         if (key == 27) {  // escape
    //             cv::destroyAllWindows();
    //             break;
    //         }
    //     }
    // }
    if (pruned_chains.size() == 1) {
        return pruned_chains;
    }

    int matrix_size = 2 * pruned_chains.size() + 2;
    std::vector<std::vector<double>> cost_matrix(matrix_size, std::vector<double>(matrix_size, 0));
    double w_e = 0.001;
    double w_c = 1;

    for (int i = 0; i < pruned_chains.size(); i++) {
        for (int j = 0; j < pruned_chains.size(); j++) {
            if (i == j) {
                cost_matrix[2 * i][2 * j] = 100000;
                cost_matrix[2 * i][2 * j + 1] = 100000;
                cost_matrix[2 * i + 1][2 * j] = 100000;
                cost_matrix[2 * i + 1][2 * j + 1] = 100000;
            } else {
                cost_matrix[2 * i][2 * j] = compute_cost(pruned_chains[i], pruned_chains[j], w_e, w_c, 0);
                cost_matrix[2 * i][2 * j + 1] = compute_cost(pruned_chains[i], pruned_chains[j], w_e, w_c, 1);
                cost_matrix[2 * i + 1][2 * j] = compute_cost(pruned_chains[i], pruned_chains[j], w_e, w_c, 2);
                cost_matrix[2 * i + 1][2 * j + 1] = compute_cost(pruned_chains[i], pruned_chains[j], w_e, w_c, 3);
            }
        }
    }

    for (int i = 0; i < matrix_size; i++) {
        cost_matrix[i][matrix_size - 1] = 1000;
        cost_matrix[i][matrix_size - 2] = 1000;
        cost_matrix[matrix_size - 1][i] = 1000;
        cost_matrix[matrix_size - 2][i] = 1000;
        cost_matrix[matrix_size - 2][matrix_size - 2] = 100000;
    }

    std::vector<int> row_idx;
    std::vector<int> col_idx;
    linear_sum_assignment(cost_matrix, row_idx, col_idx);
    int cur_idx = col_idx[row_idx.back()];
    std::vector<std::vector<Point_2D>> ordered_chains;

    cv::Mat mask(gray.rows, gray.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    while (true) {
        int cur_chain_idx = cur_idx / 2;
        std::vector<Point_2D> cur_chain = pruned_chains[cur_chain_idx];

        if (cur_idx % 2 == 1) {
            std::reverse(cur_chain.begin(), cur_chain.end());
        }
        ordered_chains.push_back(cur_chain);

        int next_idx;
        if (cur_idx % 2 == 0) {
            next_idx = col_idx[cur_idx + 1];
        } else {
            next_idx = col_idx[cur_idx - 1];
        }

        if (next_idx == matrix_size - 1 || next_idx == matrix_size - 2) {
            break;
        }
        cur_idx = next_idx;
    }

    std::cout << "Finished merging." << std::endl;

    // if (visualize_process) {
    //     cv2::Mat mask(gray.rows, gray.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    //     for (int i = 0; i < ordered_chains.size(); i++) {
    //         std::vector<Point_2D> chain = ordered_chains[i];
    //         cv::Scalar color(int(std::rand() % 200) + 55, int(std::rand() % 200) + 55, int(std::rand() % 200) + 55);
    //         for (int j = 0; j < chain.size() - 1; j++) {
    //             cv::line(mask, chain[j], chain[j + 1], color, 1);
    //         }

    //         cv2::imshow("after merging", mask);
    //         while (true) {
    //             int key = cv2::waitKey(10);
    //             if (key == 27) {  // escape
    //                 cv2.destroyAllWindows();
    //                 break;
    //             }
    //         }

    //         if (i == ordered_chains.size() - 1) {
    //             break;
    //         }

    //         Point_2D pt1 = ordered_chains[i].back();
    //         Point_2D pt2 = ordered_chains[i + 1].front();
    //         cv::line(mask, pt1, pt2, cv::Scalar(255, 255, 255), 2);
    //         cv::circle(mask, pt1, 3, cv::Scalar(255, 255, 255));
    //         cv::circle(mask, pt2, 3, cv::Scalar(255, 255, 255));
    //     }
    // }

    return ordered_chains;
}

Eigen::Matrix3d rotation_matrix_from_vectors(Eigen::Vector3d vec1, Eigen::Vector3d vec2) {
    Eigen::Vector3d a = vec1.normalized();
    Eigen::Vector3d b = vec2.normalized();
    Eigen::Vector3d v = a.cross(b);
    double c = a.dot(b);
    double s = v.norm();
    Eigen::Matrix3d kmat;
    kmat << 0, -v(2), v(1),
            v(2), 0, -v(0),
            -v(1), v(0), 0;
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity() + kmat + kmat * kmat * ((1 - c) / (s * s));
    return rotation_matrix;
}

std::vector<Marker> ndarray2MarkerArray(Eigen::MatrixXd Y, std::string marker_frame, std::vector<double> node_color, std::vector<double> line_color) {
    std::vector<Marker> results;
    for (int i = 0; i < Y.rows(); i++) {
        Marker cur_node_result;
        cur_node_result.header.frame_id = marker_frame;
        cur_node_result.type = Marker::SPHERE;
        cur_node_result.action = Marker::ADD;
        cur_node_result.ns = "node_results" + std::to_string(i);
        cur_node_result.id = i;

        cur_node_result.pose.position.x = Y(i, 0);
        cur_node_result.pose.position.y = Y(i, 1);
        cur_node_result.pose.position.z = Y(i, 2);
        cur_node_result.pose.orientation.w = 1.0;
        cur_node_result.pose.orientation.x = 0.0;
        cur_node_result.pose.orientation.y = 0.0;
        cur_node_result.pose.orientation.z = 0.0;

        cur_node_result.scale.x = 0.01;
        cur_node_result.scale.y = 0.01;
        cur_node_result.scale.z = 0.01;
        cur_node_result.color.r = node_color[0];
        cur_node_result.color.g = node_color[1];
        cur_node_result.color.b = node_color[2];
        cur_node_result.color.a = node_color[3];

        results.push_back(cur_node_result);

        if (i == Y.rows() - 1) {
            break;
        }

        Marker cur_line_result;
        cur_line_result.header.frame_id = marker_frame;
        cur_line_result.type = Marker::CYLINDER;
        cur_line_result.action = Marker::ADD;
        cur_line_result.ns = "line_results" + std::to_string(i);
        cur_line_result.id = i;

        cur_line_result.pose.position.x = (Y.row(i) + Y.row(i + 1)) / 2.0;
        cur_line_result.pose.position.y = (Y.row(i) + Y.row(i + 1)) / 2.0;
        cur_line_result.pose.position.z = (Y.row(i) + Y.row(i + 1)) / 2.0;

        Eigen::Matrix3d rot_matrix = rotation_matrix_from_vectors(Eigen::Vector3d(0, 0, 1), (Y.row(i + 1) - Y.row(i)).normalized());
        Eigen::Quaterniond q(rot_matrix);
        cur_line_result.pose.orientation.w = q.w();
        cur_line_result.pose.orientation.x = q.x();
        cur_line_result.pose.orientation.y = q.y();
        cur_line_result.pose.orientation.z = q.z();
        cur_line_result.scale.x = 0.005;
        cur_line_result.scale.y = 0.005;
        cur_line_result.scale.z = (Y.row(i + 1) - Y.row(i)).norm();
        cur_line_result.color.r = line_color[0];
        cur_line_result.color.g = line_color[1];
        cur_line_result.color.b = line_color[2];
        cur_line_result.color.a = line_color[3];

        results.push_back(cur_line_result);
    }

    return results;
}
