#include <Eigen/Dense>
#include <iostream>

static constexpr double deg_to_rad = EIGEN_PI / 180.0;
static constexpr double rad_to_deg = 180.0 / EIGEN_PI;

// ============================ From previous assignments =============================
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

// From assignment 2
// Equations (3.53-3.54) on page 84, MR pre-print 2019
std::pair<Eigen::Vector3d, double> matrix_logarithm(const Eigen::Matrix3d &R)
{
    double theta = std::acos(std::clamp((R.trace() - 1.0) * 0.5, -1.0, 1.0));
    double s = std::sin(theta);
    Eigen::Vector3d w((R(2, 1) - R(1, 2)) / (2 * s),
                      (R(0, 2) - R(2, 0)) / (2 * s),
                      (R(1, 0) - R(0, 1)) / (2 * s));
    w.normalize();
    return {w, theta};
}

// Equation on page 72, MR pre-print 2019
// From Assignment 1
Eigen::Matrix3d rotate_y(double radians)
{
    const double cos = std::cos(radians);
    const double sin = std::sin(radians);
    Eigen::Matrix3d R;
    R << cos, 0, sin,
        0, 1, 0,
        -sin, 0, cos;
    return R;
}

// Equation on page 577, MR pre-print 2019
// From Assignment 1
Eigen::Matrix3d rotation_matrix_from_euler_zyx(const Eigen::Vector3d &e)
{
    return rotate_z(e[0]) * rotate_y(e[1]) * rotate_x(e[2]);
}

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
// Equation (4.16) on page 147, MR pre-print 2019
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
void ur3e_test_fk()
{
    std::cout << "=================================== Forward kinematics tests ===================================" << std::endl;

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

// ====================================== T.2 a) ======================================
// Section (6.2.1) on page 225, MR pre-print 2019
std::pair<uint32_t, double> newton_raphson_root_find(
    const std::function<double(double)> &f,
    double x_0,
    double dx_0 = 0.5,
    double eps = 10e-7)
{
    const uint32_t max_iterations = 1000;
    double x = x_0;

    for (uint32_t i = 0; i < max_iterations; ++i)
    {
        double fx = f(x);

        if (std::abs(fx) < eps)
            return {i, x};

        // Approximate derivative using central difference
        // Equation taken from https://en.wikipedia.org/wiki/Finite_difference
        double dfdx = (f(x + dx_0) - f(x - dx_0)) / (2.0 * dx_0);

        // Newton–Raphson update
        double x_new = x - fx / dfdx;

        if (std::abs(x_new - x) < eps)
            return {i + 1, x_new};

        x = x_new;
    }

    return {max_iterations, x};
}

// ====================================== T.2 b) ======================================
// Gradient Descent equation taken from https://en.wikipedia.org/wiki/Gradient_descent
std::pair<uint32_t, double> gradient_descent_minimize(
    const std::function<double(double)> &f,
    double x_0,
    double gamma = 0.1, // η
    double dx_0 = 0.5,
    double eps = 10e-7)
{
    const uint32_t max_iterations = 1000;
    double x = x_0;

    for (uint32_t i = 0; i < max_iterations; ++i)
    {
        // Approximate derivative using central difference
        // Equation taken from https://en.wikipedia.org/wiki/Finite_difference
        const double grad = (f(x + dx_0) - f(x - dx_0)) / (2.0 * dx_0);

        if (std::abs(grad) < eps)
            return {i, x};

        const double x_new = x - gamma * grad; // Gradient equation (a_n+1 = a_n - η∇f(a_n))

        if (std::abs(x_new - x) < eps)
            return {i + 1, x_new};

        x = x_new;
    }

    return {max_iterations, x};
}

// ====================================== T.2 c) ======================================
void test_newton_raphson_root_find(const std::function<double(double)> &f, double x0)
{
    auto [iterations, x_hat] = newton_raphson_root_find(f, x0);
    std::cout << "NR root f, x0=" << x0 << " -> it=" << iterations << " x=" << x_hat << " f(x)=" << f(x_hat) << std::endl;
}

void test_gradient_descent_minimize(const std::function<double(double)> &f, double x0)
{
    auto [iterations, x_hat] = gradient_descent_minimize(f, x0);
    std::cout << "GD root f, x0=" << x0 << " -> it=" << iterations << " x=" << x_hat << " f(x)=" << f(x_hat) << std::endl;
}

void test_optimizations()
{
    std::cout << "====================================== Root finding tests ======================================" << std::endl;
    auto f1 = [](double x)
    {
        return (x - 3.0) * (x - 3.0) - 1.0;
    };
    test_newton_raphson_root_find(f1, -20.0);
    test_gradient_descent_minimize(f1, -20.0);
}

// ====================================== T.3 a) ======================================
// Equation (5.11) on page 178, MR pre-print 2019
Eigen::MatrixXd ur3e_space_jacobian(const Eigen::VectorXd &current_joint_positions)
{
    auto [M, S_list] = ur3e_space_chain();

    Eigen::MatrixXd space_jacobian(6, 6);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

    // Equation 5.11 on page 178, MR pre-print 2019
    for (int i = 0; i < 6; ++i)
    {
        space_jacobian.col(i) = adjoint_matrix(T) * S_list[i];

        const Eigen::Vector3d w = S_list[i].head<3>();
        const Eigen::Vector3d v = S_list[i].tail<3>();
        T = T * matrix_exponential(w, v, current_joint_positions[i]);
    }

    return space_jacobian;
}

// ====================================== T.3 b) ======================================
// Equation (5.18) on page 183, MR pre-print 2019
Eigen::MatrixXd ur3e_body_jacobian(const Eigen::VectorXd &current_joint_positions)
{
    auto [M, B_list] = ur3e_body_chain();

    Eigen::MatrixXd body_jacobian(6, 6);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

    body_jacobian.col(5) = B_list[5];

    // Equation (5.18) on page 183, MR pre-print 2019
    for (int i = 4; i >= 0; --i) // Second to last joint
    {
        const Eigen::Vector3d w = B_list[i + 1].head<3>();
        const Eigen::Vector3d v = B_list[i + 1].tail<3>();
        T = T * matrix_exponential(w, v, -current_joint_positions[i + 1]);

        body_jacobian.col(i) = adjoint_matrix(T) * B_list[i];
    }

    return body_jacobian;
}

// ====================================== T.3 c) ======================================
void ur3e_test_jacobian(const Eigen::VectorXd &joint_positions)
{
    Eigen::Matrix4d tsb = ur3e_body_fk(joint_positions);
    auto [m, space_screws] = ur3e_space_chain();
    Eigen::MatrixXd jb = ur3e_body_jacobian(joint_positions);
    Eigen::MatrixXd js = ur3e_space_jacobian(joint_positions);
    Eigen::MatrixXd ad_tsb = adjoint_matrix(tsb);
    Eigen::MatrixXd ad_tbs = adjoint_matrix(tsb.inverse());
    std::cout << "Jb: " << std::endl
              << jb << std::endl
              << "Ad_tbs*Js:" << std::endl
              << ad_tbs * js << std::endl
              << std::endl;
    std::cout << "Js: " << std::endl
              << js << std::endl
              << "Ad_tsb*Jb:" << std::endl
              << ad_tsb * jb << std::endl
              << std::endl;
    std::cout << "d Jb: " << std::endl
              << jb - ad_tbs * js << std::endl
              << std::endl;
    std::cout << "d Js: " << std::endl
              << js - ad_tsb * jb << std::endl
              << std::endl;
}
void ur3e_test_jacobian()
{
    std::cout << std::endl
              << "==================================== Jacobian matrix tests =====================================" << std::endl;
    ur3e_test_jacobian(std_vector_to_eigen(std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}) *
                       deg_to_rad);
    ur3e_test_jacobian(std_vector_to_eigen(std::vector<double>{45.0, -20.0, 10.0, 2.5, 30.0, -50.0}) *
                       deg_to_rad);
}

// ====================================== T.4 a) ======================================
// Equations found on page 228, Mr pre-print 2019
std::pair<size_t, Eigen::VectorXd> ur3e_ik_body(const Eigen::Matrix4d &t_sd, const Eigen::VectorXd &current_joint_positions, double gamma = 1e-2, double v_e = 4e-3, double w_e = 4e-3)
{
    Eigen::VectorXd joint_positions = current_joint_positions;
    size_t iteration = 0;

    while (true)
    {
        const Eigen::Matrix4d tsb = ur3e_body_fk(joint_positions);

        // T_bd = T_sb^{-1} * T_sd
        const Eigen::Matrix4d T_bd = tsb.inverse() * t_sd;

        const auto [screw_axis, theta] = matrix_logarithm(T_bd); // This is wrong, since it only works for 3x3 rotation matrices
        const Eigen::VectorXd Vb = screw_axis * theta;           // [ω_b; v_b]
        const Eigen::Vector3d w_b = Vb.head<3>();
        const Eigen::Vector3d v_b = Vb.tail<3>();

        // For small ε_ω, ε_v
        if (w_b.norm() < w_e && v_b.norm() < v_e)
            return {iteration, joint_positions};

        const Eigen::MatrixXd Jb = ur3e_body_jacobian(joint_positions);

        const Eigen::MatrixXd Jb_plus =
            Jb.completeOrthogonalDecomposition().pseudoInverse();

        // θ^{i+1} = θ^{i} + J_b^+ * V_b
        const Eigen::VectorXd dq = Jb_plus * Vb;

        joint_positions += dq;
        ++iteration;
    }
}

// ====================================== T.4 b) ======================================
void ur3e_ik_test_pose(const Eigen::Vector3d &pos, const Eigen::Vector3d &zyx, const Eigen::VectorXd &j0)
{
    std::cout << "Test from pose" << std::endl;
    Eigen::Matrix4d t_sd = transformation_matrix(rotation_matrix_from_euler_zyx(zyx), pos);
    auto [iterations, j_ik] = ur3e_ik_body(t_sd, j0);
    Eigen::Matrix4d t_ik = ur3e_body_fk(j_ik);
    print_pose("IK pose", t_ik);
    print_pose("Desired pose", t_sd);
    std::cout << "Converged after " << iterations << " iterations" << std::endl;
    std::cout << "J_0: " << j0.transpose() * rad_to_deg << std::endl;
    std::cout << "J_ik: " << j_ik.transpose() * rad_to_deg << std::endl
              << std::endl;
}

void ur3e_ik_test_configuration(const Eigen::VectorXd &joint_positions, const Eigen::VectorXd &j0)
{
    std::cout << "Test from configuration" << std::endl;
    Eigen::Matrix4d t_sd = ur3e_space_fk(joint_positions);
    auto [iterations, j_ik] = ur3e_ik_body(t_sd, j0);
    Eigen::Matrix4d t_ik = ur3e_body_fk(j_ik);
    print_pose(" IK pose", t_ik);
    print_pose("Desired pose", t_sd);
    std::cout << "Converged after " << iterations << " iterations" << std::endl;
    std::cout << "J_0: " << j0.transpose() * rad_to_deg << std::endl;
    std::cout << "J_d: " << joint_positions.transpose() * rad_to_deg << std::endl;
    std::cout << "J_ik: " << j_ik.transpose() * rad_to_deg << std::endl
              << std::endl;
}

void ur3e_ik_test()
{
    Eigen::VectorXd j_t0 = std_vector_to_eigen(std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}) *
                           deg_to_rad;
    Eigen::VectorXd j_t1 = std_vector_to_eigen(std::vector<double>{0.0, 0.0, -89.0, 0.0, 0.0, 0.0}) *
                           deg_to_rad;
    ur3e_ik_test_pose(Eigen::Vector3d{0.3289, 0.22315, 0.36505}, Eigen::Vector3d{0.0, 90.0, -90.0} * deg_to_rad, j_t0);
    ur3e_ik_test_pose(Eigen::Vector3d{0.3289, 0.22315, 0.36505}, Eigen::Vector3d{0.0, 90.0, -90.0} * deg_to_rad, j_t1);
    Eigen::VectorXd j_t2 = std_vector_to_eigen(std::vector<double>{50.0, -30.0, 20, 0.0, -30.0, 50.0}) * deg_to_rad;
    Eigen::VectorXd j_d1 = std_vector_to_eigen(std::vector<double>{45.0, -20.0, 10.0, 2.5, 30.0,
                                                                   -50.0}) *
                           deg_to_rad;
    ur3e_ik_test_configuration(j_d1, j_t0);
    ur3e_ik_test_configuration(j_d1, j_t2);
}

int main()
{
    ur3e_test_fk();
    test_optimizations();
    ur3e_test_jacobian();
    ur3e_ik_test();
}
