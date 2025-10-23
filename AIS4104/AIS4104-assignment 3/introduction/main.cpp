#include <Eigen/Dense>
#include <iostream>

static constexpr double deg_to_rad = EIGEN_PI / 180.0;

// ====================================== T.1 a) ======================================
Eigen::VectorXd std_vector_to_eigen(const std::vector<double> &v)
{
    Eigen::VectorXd eigen_vector(v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        eigen_vector(i) = v[i];
    }
    return eigen_vector;
}

// ====================================== T.1 b) ======================================
bool is_average_below_eps(const std::vector<double> &values, double eps = 10e-7, uint8_t n_values = 5u)
{
    if (values.size() < n_values)
        return false;

    double sum = 0.0;
    for (size_t i = values.size() - n_values; i < values.size(); ++i)
    {
        sum += values[i];
    }

    double average = sum / n_values;
    return average <= eps;
}

// ====================================== T.1 c) ======================================
// Equation (3.62) page 87, MR pre-print 2019
// From Assignment 1
Eigen::Matrix4d transformation_matrix(const Eigen::Matrix3d &r, const Eigen::Vector3d &p)
{
    Eigen::Matrix4d matrix;
    matrix << r(0, 0), r(0, 1), r(0, 2), p(0),
        r(1, 0), r(1, 1), r(1, 2), p(1),
        r(2, 0), r(2, 1), r(2, 2), p(2),
        0, 0, 0, 1;
    return matrix;
}

// Equation on page 72, MR pre-print 2019
// From Assignment 1
Eigen::Matrix3d rotate_x(double radians)
{
    const double cos = std::cos(radians), sin = std::sin(radians);
    Eigen::Matrix3d R;
    R << 1, 0, 0,
        0, cos, -sin,
        0, sin, cos;
    return R;
}

// Equation on page 72, MR pre-print 2019
// From Assignment 1
Eigen::Matrix3d rotate_z(double radians)
{
    const double cos = std::cos(radians);
    const double sin = std::sin(radians);
    Eigen::Matrix3d R;
    R << cos, -sin, 0,
        sin, cos, 0,
        0, 0, 1;
    return R;
}

// From assignment 2
static inline Eigen::Matrix4d A(double a, double alpha, double d, double theta)
{
    Eigen::Matrix4d T = transformation_matrix(rotate_z(theta), {0, 0, d});
    T = T * transformation_matrix(Eigen::Matrix3d::Identity(), {a, 0, 0});
    T = T * transformation_matrix(rotate_x(alpha), {0, 0, 0});
    return T;
}

std::pair<Eigen::Matrix4d, std::vector<Eigen::VectorXd>> ur3e_space_chain()
{
    static const double a_link[6] = {0.0, -0.24355, -0.21320, 0.0, 0.0, 0.0};
    static const double d_link[6] = {0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.09210};
    static const double alpha_link[6] = {EIGEN_PI / 2, 0.0, 0.0, EIGEN_PI / 2, -EIGEN_PI / 2, 0.0};

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    Eigen::Vector3d o = Eigen::Vector3d::Zero();
    Eigen::Vector3d z = Eigen::Vector3d::UnitZ();
    Eigen::Matrix<double, 6, 1> S;

    std::vector<Eigen::VectorXd> S_list;
    S_list.reserve(6);

    // Definition 3.24 on page 102v, MR pre-print 2019
    for (int i = 0; i < 6; ++i)
    {
        S.head<3>() = z;
        S.tail<3>() = -z.cross(o);

        Eigen::VectorXd Si_vec(6);
        Si_vec = S;
        S_list.push_back(Si_vec);

        T = T * A(a_link[i], alpha_link[i], d_link[i], 0.0);
        o = T.block<3, 1>(0, 3);
        z = T.block<3, 3>(0, 0) * Eigen::Vector3d::UnitZ();
    }

    // Equation (4.6) on page 136, MR pre-print 2019
    // Home configuration M equals the T because M is the end-effector pose when all joint variables are 0.
    return {T, S_list};
}

// ====================================== T.1 d) ======================================
// Equation (3.30) page 75, MR pre-print 2019
// From Assignment 1
Eigen::Matrix3d skew_symmetric(Eigen::Vector3d vec)
{
    Eigen::Matrix3d skewMatrix;
    skewMatrix << 0.0, -vec(2), vec(1),
        vec(2), 0.0, -vec(0),
        -vec(1), vec(0), 0.0;
    return skewMatrix;
}

// Equation (3.88) on page 103, MR pre-print 2019
// From assignment 2
Eigen::Matrix4d matrix_exponential(const Eigen::Vector3d &w, const Eigen::Vector3d &v, double theta)
{
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d w_skew = skew_symmetric(w);

    Eigen::Matrix3d R = I + std::sin(theta) * w_skew +
                        (1 - std::cos(theta)) * (w_skew * w_skew);

    Eigen::Matrix3d V = I * theta + (1 - std::cos(theta)) * w_skew +
                        (theta - std::sin(theta)) * (w_skew * w_skew);
    Eigen::Vector3d p = V * v;

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = p;

    return T;
}

Eigen::Matrix4d ur3e_space_fk(const Eigen::VectorXd &joint_positions)
{
    auto [M, S_list] = ur3e_space_chain();
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

    // Equation (4.14) on page 140, MR pre-print 2019
    for (int i = 0; i < 6; ++i)
    {
        const Eigen::Vector3d w = S_list[i].head<3>();
        const Eigen::Vector3d v = S_list[i].tail<3>();
        const double theta = joint_positions[i];
        T *= matrix_exponential(w, v, theta);
    }

    // Equation (4.6) on page 136, MR pre-print 2019
    return T * M;
}

// ====================================== T.1 e) ======================================
// Definition (3.20) on page 98, MR pre-print 2019
// From assignment 2
Eigen::MatrixXd adjoint_matrix(const Eigen::Matrix4d &T)
{
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Vector3d p = T.block<3, 1>(0, 3);

    Eigen::MatrixXd Ad(6, 6);
    Ad.setZero();
    Ad.block<3, 3>(0, 0) = R;
    Ad.block<3, 3>(3, 0) = skew_symmetric(p) * R;
    Ad.block<3, 3>(3, 3) = R;
    return Ad;
}

// Equation (4.16) on page 147, MR pre-print 2019
std::pair<Eigen::Matrix4d, std::vector<Eigen::VectorXd>> ur3e_body_chain()
{
    auto [M, S_list] = ur3e_space_chain();

    Eigen::Matrix3d R = M.block<3, 3>(0, 0);
    Eigen::Vector3d p = M.block<3, 1>(0, 3);
    Eigen::Matrix4d M_inv = Eigen::Matrix4d::Identity();
    M_inv.block<3, 3>(0, 0) = R.transpose();
    M_inv.block<3, 1>(0, 3) = -R.transpose() * p;

    Eigen::MatrixXd Ad_Minv = adjoint_matrix(M_inv);

    // Equation (4.16) on page 147, MR pre-print 2019<
    std::vector<Eigen::VectorXd> B_list(6);
    for (int i = 0; i < 6; ++i)
    {
        B_list[i] = Ad_Minv * S_list[i];
    }

    return {M, B_list};
}

// ====================================== T.1 f) ======================================
Eigen::Matrix4d ur3e_body_fk(const Eigen::VectorXd &joint_positions)
{
    auto [M, B_list] = ur3e_body_chain();

    Eigen::Matrix4d T = M;
    for (int i = 0; i < 6; ++i)
    {
        const Eigen::Vector3d w = B_list[i].head<3>();
        const Eigen::Vector3d v = B_list[i].tail<3>();
        const double theta = joint_positions[i];
        T *= matrix_exponential(w, v, theta);
    }
    return T;
}

// ====================================== T.1 g) ======================================
// Equations (B.3)–(B.5) on page 579, MR pre-print 2019
// From assignment 2
Eigen::Vector3d euler_zyx_from_rotation_matrix(const Eigen::Matrix3d &R)
{
    const double r11 = R(0, 0), r21 = R(1, 0), r31 = R(2, 0);
    const double r12 = R(0, 1), r22 = R(1, 1);
    const double r32 = R(2, 1), r33 = R(2, 2);

    const double eps = 1e-12;
    Eigen::Vector3d ang;

    if (std::abs(r31 + 1.0) < eps)
    {
        ang[1] = EIGEN_PI / 2.0;
        ang[0] = 0.0;
        ang[2] = std::atan2(r12, r22);
    }
    else if (std::abs(r31 - 1.0) < eps)
    {
        ang[1] = -EIGEN_PI / 2.0;
        ang[0] = 0.0;
        ang[2] = -std::atan2(r12, r22);
    }
    else
    {
        const double c = std::sqrt(r11 * r11 + r21 * r21);
        ang[1] = std::atan2(-r31, c);
        ang[0] = std::atan2(r21, r11);
        ang[2] = std::atan2(r32, r33);
    }
    return ang;
}

// From assignment 2
void print_pose(const std::string &label, const Eigen::Matrix4d &T)
{
    const auto R = T.block<3, 3>(0, 0);
    const auto p = T.block<3, 1>(0, 3);
    const Eigen::Vector3d e_deg = euler_zyx_from_rotation_matrix(R) * (180.0 / EIGEN_PI);
    std::cout << label << "\n"
              << "Euler ZYX (deg): " << e_deg.transpose() << "\n"
              << "Position:        " << p.transpose() << "\n\n";
}

void ur3e_test_fk()
{
    std::cout << "Forward kinematics tests" << std::endl;

    print_pose("Space FK (0,0,0,0,0,0)",
               ur3e_space_fk(std_vector_to_eigen({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}) * deg_to_rad));
    print_pose("Body FK  (0,0,0,0,0,0)",
               ur3e_body_fk(std_vector_to_eigen({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}) * deg_to_rad));
    std::cout << std::endl;

    print_pose("Space FK (0,0,0,-90,0,0)",
               ur3e_space_fk(std_vector_to_eigen({0.0, 0.0, 0.0, -90.0, 0.0, 0.0}) * deg_to_rad));
    print_pose("Body FK  (0,0,0,-90,0,0)",
               ur3e_body_fk(std_vector_to_eigen({0.0, 0.0, 0.0, -90.0, 0.0, 0.0}) * deg_to_rad));
    std::cout << std::endl;

    print_pose("Space FK (0,0,-180,0,0,0)",
               ur3e_space_fk(std_vector_to_eigen({0.0, 0.0, -180.0, 0.0, 0.0, 0.0}) * deg_to_rad));
    print_pose("Body FK  (0,0,-180,0,0,0)",
               ur3e_body_fk(std_vector_to_eigen({0.0, 0.0, -180.0, 0.0, 0.0, 0.0}) * deg_to_rad));
    std::cout << std::endl;

    print_pose("Space FK (0,0,-90,0,0,0)",
               ur3e_space_fk(std_vector_to_eigen({0.0, 0.0, -90.0, 0.0, 0.0, 0.0}) * deg_to_rad));
    print_pose("Body FK  (0,0,-90,0,0,0)",
               ur3e_body_fk(std_vector_to_eigen({0.0, 0.0, -90.0, 0.0, 0.0, 0.0}) * deg_to_rad));
}

int main()
{
    ur3e_test_fk();
}
