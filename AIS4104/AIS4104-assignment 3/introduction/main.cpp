#include <Eigen/Dense>

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
Eigen::Matrix4d ur3e_space_fk(const Eigen::VectorXd &joint_positions)
{
}