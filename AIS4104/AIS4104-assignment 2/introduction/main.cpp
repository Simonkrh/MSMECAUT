#include <iostream>
#include <Eigen/Dense>
#include <numbers>
#include <cmath>

// ====================================== T.1 a) ======================================
// Equations (B.3)–(B.5) on page 579, MR pre-print 2019
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

// Equation on page 577, MR pre-print 2019
// From Assignment 1
Eigen::Matrix3d rotation_matrix_from_euler_zyx(const Eigen::Vector3d &e)
{
    return rotate_z(e[0]) * rotate_y(e[1]) * rotate_x(e[2]);
}

void test_euler_zyx(double alpha, double beta, double gamma)
{
    std::cout << "========================== T.1 a) ==========================" << "\n";

    Eigen::Vector3d original(alpha, beta, gamma);

    Eigen::Matrix3d rotationMatrixFromEuler = rotation_matrix_from_euler_zyx(original);

    Eigen::Vector3d eulerFromMatrix = euler_zyx_from_rotation_matrix(rotationMatrixFromEuler);

    std::cout << "Original angles: " << original.transpose() << "\n";
    std::cout << "Recovered angles: " << eulerFromMatrix.transpose() << "\n";
    std::cout << "Difference: " << (original - eulerFromMatrix).transpose() << "\n\n";
}

// ====================================== T.1 b) ======================================
// Equation (3.70) on page 97, MR pre-print 2019
Eigen::VectorXd twist(const Eigen::Vector3d &w, const Eigen::Vector3d &v)
{
    Eigen::VectorXd V_s(6);
    V_s << w, v;
    return V_s;
}

void test_twist()
{
    std::cout << "========================== T.1 b) ==========================" << "\n";

    Eigen::Vector3d w(1.0, 2.0, 3.0);
    Eigen::Vector3d v(4.0, 5.0, 6.0);

    Eigen::VectorXd V_s = twist(w, v);

    std::cout << "Twist vector:\n"
              << V_s.transpose() << std::endl;
    std::cout << "Expected: 1 2 3 4 5 6 " << "\n\n";
}

// ====================================== T.1 c) ======================================
// Equation on page 101, MR pre-print 2019
Eigen::VectorXd screw_axis(const Eigen::Vector3d &q, const Eigen::Vector3d &s, double h)
{
    const Eigen::Vector3d &shat = s;
    Eigen::VectorXd S(6);
    S.head<3>() = shat;
    S.tail<3>() = -shat.cross(q) + h * shat;
    return S;
}

void test_screw_axis()
{
    std::cout << "========================== T.1 c) ==========================" << "\n";

    Eigen::Vector3d shat(0, 0, 1);
    Eigen::Vector3d q(1, 0, 0);
    double h = 0.0;

    Eigen::VectorXd S = screw_axis(q, shat, h);
    std::cout << "Srew = [" << S.transpose() << " ]\n";
    std::cout << "Expected: [ 0 0 1  0 -1 0 ] " << "\n\n";
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

// Definition (3.20) on page 98, MR pre-print 2019
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

void test_adjoint_matrix()
{
    std::cout << "========================== T.1 d) ==========================" << "\n";

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T(0, 3) = 1.0;
    T(1, 3) = 2.0;
    T(2, 3) = 3.0;

    Eigen::MatrixXd Ad = adjoint_matrix(T);

    std::cout << "Adjoint matrix =\n"
              << Ad << "\n";

    Eigen::MatrixXd expected(6, 6);
    expected << 1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0,
        0, -3, 2, 1, 0, 0,
        3, 0, -1, 0, 1, 0,
        -2, 1, 0, 0, 0, 1;

    std::cout << "Expected =\n"
              << expected << "\n\n";
}

// ====================================== T.1 e) ======================================
double cot(double x)
{
    return std::cos(x) / std::sin(x);
}

void test_cot()
{
    std::cout << "========================== T.1 e) ==========================" << "\n";

    std::cout << "cot(pi/4) = " << cot(EIGEN_PI / 4) << "\n";
    std::cout << "Expected = 1\n"
              << "\n\n";
}

// ====================================== T.2 a) ======================================
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

// Equation (3.98) on page 108, MR pre-print 2019
void changeWrenchFrame()
{
    Eigen::Vector3d f_w(-30, 0, 0);       // force in world
    Eigen::Vector3d m_s(0, 0, 2);         // torque in sensor
    Eigen::Vector3d e_ws_deg(60, -60, 0); // ZYX Euler angles from sensor to world

    Eigen::Vector3d e_ws = e_ws_deg * (EIGEN_PI / 180.0);

    Eigen::Matrix3d R_ws =
        rotate_y(e_ws[0]) * rotate_z(e_ws[1]) * rotate_x(e_ws[2]);

    Eigen::Matrix4d T_ws = transformation_matrix(R_ws, Eigen::Vector3d::Zero());

    Eigen::MatrixXd Ad_ws = adjoint_matrix(T_ws);

    Eigen::Vector3d m_w = R_ws * m_s;

    Eigen::VectorXd F_w(6);
    F_w << m_w, f_w;

    Eigen::VectorXd F_s = Ad_ws.transpose() * F_w;
    Eigen::Vector3d f_s = F_s.tail<3>();

    std::cout << "========================== T.2 a) ==========================\n";
    std::cout << "f_w: " << f_w.transpose() << "\n";
    std::cout << "m_w: " << m_w.transpose() << "\n";
    std::cout << "f_s: " << f_s.transpose() << "\n";
    std::cout << "m_s: " << m_s.transpose() << "\n\n";
}

// ====================================== T.2 b) ======================================
// Example (3.28) on page 108-109, MR pre-print 2019
void sensor_wrench_example_3_28()
{
    // Fh = (0,0,0, 0,-5,0), Fa = (0,0,0, 0,0,1)
    Eigen::VectorXd Fh = twist(Eigen::Vector3d::Zero(), Eigen::Vector3d(0, -5, 0));
    Eigen::VectorXd Fa = twist(Eigen::Vector3d::Zero(), Eigen::Vector3d(0, 0, 1));

    Eigen::Matrix4d Thf;
    Thf << 1, 0, 0, -0.1,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    Eigen::Matrix4d Taf;
    Taf << 1, 0, 0, -0.25,
        0, 0, 1, 0,
        0, -1, 0, 0,
        0, 0, 0, 1;

    Eigen::MatrixXd Ad_thf = adjoint_matrix(Thf);
    Eigen::MatrixXd Ad_taf = adjoint_matrix(Taf);

    Eigen::VectorXd Ff = Ad_thf.transpose() * Fh + Ad_taf.transpose() * Fa;

    std::cout << "========================== T.2 b) ==========================\n";
    std::cout << "F_f: " << Ff.transpose() << "\n";
    std::cout << "Expected: 0 0 -0.75  0 -6 0 \n\n";
}

// ====================================== T.3 a) ======================================
// Equation on page 72, MR pre-print 2019
// From Assignment 1
Eigen::Matrix3d rotation_matrix_from_axis_angle(const Eigen::Vector3d &axis,
                                                double degrees)
{
    Eigen::Vector3d u = axis.normalized();
    const double radians = degrees * EIGEN_PI / 180.0;
    const double cos = std::cos(radians);
    const double sin = std::sin(radians);

    Eigen::Matrix3d matrix;
    matrix << cos + u.x() * u.x() * (1 - cos), u.x() * u.y() * (1 - cos) - u.z() * sin, u.x() * u.z() * (1 - cos) + u.y() * sin,
        u.y() * u.x() * (1 - cos) + u.z() * sin, cos + u.y() * u.y() * (1 - cos), u.y() * u.z() * (1 - cos) - u.x() * sin,
        u.z() * u.x() * (1 - cos) - u.y() * sin, u.z() * u.y() * (1 - cos) + u.x() * sin, cos + u.z() * u.z() * (1 - cos);

    return matrix;
}

// Equation (3.51) on page 82, MR pre-print 2019
Eigen::Matrix3d matrix_exponential(const Eigen::Vector3d &w, double theta)
{
    return rotation_matrix_from_axis_angle(w, theta * 180.0 / EIGEN_PI);
}

void test_matrix_exponential_SO3()
{
    std::cout << "========================== T.3 a) ==========================\n";

    // Rotation of 90 degree about z-axis
    Eigen::Vector3d w(0, 0, 1);
    double theta = EIGEN_PI / 2.0;

    Eigen::Matrix3d R = matrix_exponential(w, theta);

    std::cout << "R =\n"
              << R << "\n";
    std::cout << "Expected:\n"
              << "0 -1  0\n"
              << "1  0  0\n"
              << "0  0  1 \n\n";
}

// ====================================== T.3 b) ======================================
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

void test_matrix_logarithm_SO3()
{
    std::cout << "========================== T.3 b) ==========================\n";

    // Rotation of 90 degrees about z-axis
    Eigen::Vector3d w_true(0, 0, 1);
    double theta_true = EIGEN_PI / 2.0;

    Eigen::Matrix3d R = matrix_exponential(w_true, theta_true);

    auto [w_hat, theta] = matrix_logarithm(R);

    std::cout << "Expected axis:   " << w_true.transpose() << "\n";
    std::cout << "Expected angle:  " << theta_true << "\n";
    std::cout << "Recovered axis:  " << w_hat.transpose() << "\n";
    std::cout << "Recovered angle: " << theta << "\n\n";
}

// ====================================== T.3 c) ======================================
// Equation (3.88) on page 103, MR pre-print 2019
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

void test_matrix_exponential_SE3()
{
    std::cout << "========================== T.3 c) ==========================\n";

    Eigen::Vector3d w(0, 0, 1);    // rotate about z
    Eigen::Vector3d v(1, 0, 0);    // translation direction
    double theta = EIGEN_PI / 2.0; // 90 degrees

    Eigen::Matrix4d T = matrix_exponential(w, v, theta);

    std::cout << "T =\n"
              << T << "\n\n";
    std::cout << "Expected R =\n"
              << "0 -1  0\n"
              << "1  0  0\n"
              << "0  0  1\n\n";
}

/// ====================================== T.3 d) ======================================
// Equations (3.91–3.92) on page 104, MR pre-print 2019
std::pair<Eigen::VectorXd, double> matrix_logarithm(const Eigen::Matrix4d &T)
{
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Vector3d p = T.block<3, 1>(0, 3);

    auto [w, theta] = matrix_logarithm(R);
    Eigen::VectorXd S(6);

    if (theta < 1e-12)
    {
        S << Eigen::Vector3d::Zero(), p;
        return {S, p.norm()};
    }

    Eigen::Matrix3d W = skew_symmetric(w);

    Eigen::Matrix3d Ginv = (1.0 / theta) * Eigen::Matrix3d::Identity() - 0.5 * W + (1.0 / theta - 0.5 / std::tan(theta / 2.0)) * (W * W);

    Eigen::Vector3d v = Ginv * p;

    S << w, v;
    return {S, theta};
}

void test_matrix_logarithm_SE3()
{
    std::cout << "========================== T.3 d) ==========================\n";

    Eigen::Vector3d w(0, 0, 1), v(1, 0, 0);
    double theta = EIGEN_PI / 2.0;
    Eigen::Matrix4d T = matrix_exponential(w, v, theta);

    auto [S, th] = matrix_logarithm(T);

    std::cout << "S^T = " << S.transpose() << "\n";
    std::cout << "theta = " << th << "\n";
    std::cout << "Expected S^T = 0 0 1 1 0 0\n";
    std::cout << "Expected theta = " << EIGEN_PI / 2.0 << "\n\n";
}

// ====================================== T.4 a) ======================================
void print_pose(const std::string &label, const Eigen::Matrix4d &T)
{
    const auto R = T.block<3, 3>(0, 0);
    const auto p = T.block<3, 1>(0, 3);
    const Eigen::Vector3d e_deg = euler_zyx_from_rotation_matrix(R) * (180.0 / EIGEN_PI);
    std::cout << label << "\n"
              << "Euler ZYX (deg): " << e_deg.transpose() << "\n"
              << "Position:        " << p.transpose() << "\n\n";
}

void test_print_pose()
{
    std::cout << "========================== T.4 a) ==========================\n";
    Eigen::Vector3d e_deg(90.0, 30.0, -45.0);
    Eigen::Matrix3d R = rotate_z(e_deg[0] * EIGEN_PI / 180.0) * rotate_y(e_deg[1] * EIGEN_PI / 180.0) * rotate_x(e_deg[2] * EIGEN_PI / 180.0);
    Eigen::Matrix4d T = transformation_matrix(R, Eigen::Vector3d(1, 2, 3));

    print_pose("Pose T", T);

    std::cout << "Expected Euler ZYX (deg):  90  30  -45\n";
    std::cout << "Expected Position:         1  2   3\n\n";
}

// ====================================== T.4 b) ======================================
inline Eigen::Matrix4d Tx(double a)
{
    return transformation_matrix(Eigen::Matrix3d::Identity(), Eigen::Vector3d(a, 0, 0));
}

// Equation (4.4-4.5) on page 135, MR pre-print 2019
Eigen::Matrix4d planar_3r_fk_transform(const std::vector<double> &joint_positions)
{
    const double L1 = 10.0, L2 = 10.0, L3 = 10.0;

    const double q1d = joint_positions.size() > 0 ? joint_positions[0] : 0.0;
    const double q2d = joint_positions.size() > 1 ? joint_positions[1] : 0.0;
    const double q3d = joint_positions.size() > 2 ? joint_positions[2] : 0.0;

    const double q1 = q1d * EIGEN_PI / 180.0;
    const double q2 = q2d * EIGEN_PI / 180.0;
    const double q3 = q3d * EIGEN_PI / 180.0;

    Eigen::Matrix4d T =
        transformation_matrix(rotate_z(q1), Eigen::Vector3d::Zero()) * Tx(L1) *
        transformation_matrix(rotate_z(q2), Eigen::Vector3d::Zero()) * Tx(L2) *
        transformation_matrix(rotate_z(q3), Eigen::Vector3d::Zero()) * Tx(L3);

    return T;
}

void test_planar_3r_fk()
{
    std::cout << "========================== T.4 b) ==========================\n";

    const std::vector<std::vector<double>> J = {
        {0.0, 0.0, 0.0},
        {90.0, 0.0, 0.0},
        {0.0, 90.0, 0.0},
        {0.0, 0.0, 90.0},
        {10.0, -15.0, 2.75}};
    const char *names[] = {"j1", "j2", "j3", "j4", "j5"};

    for (int i = 0; i < (int)J.size(); ++i)
    {
        Eigen::Matrix4d T = planar_3r_fk_transform(J[i]);
        print_pose(std::string("Pose ") + names[i], T);
    }
}

// ====================================== T.4 c) ======================================
Eigen::Matrix4d planar_3r_fk_screw(const std::vector<double> &joint_positions)
{
    const double L1 = 10.0, L2 = 10.0, L3 = 10.0;

    const double q1 = (joint_positions.size() > 0 ? joint_positions[0] : 0.0) * EIGEN_PI / 180.0;
    const double q2 = (joint_positions.size() > 1 ? joint_positions[1] : 0.0) * EIGEN_PI / 180.0;
    const double q3 = (joint_positions.size() > 2 ? joint_positions[2] : 0.0) * EIGEN_PI / 180.0;

    const Eigen::Vector3d omega(0, 0, 1);
    const Eigen::VectorXd S1 = screw_axis(Eigen::Vector3d(0, 0, 0), omega, 0.0);
    const Eigen::VectorXd S2 = screw_axis(Eigen::Vector3d(L1, 0, 0), omega, 0.0);
    const Eigen::VectorXd S3 = screw_axis(Eigen::Vector3d(L1 + L2, 0, 0), omega, 0.0);

    const Eigen::Matrix4d M = transformation_matrix(Eigen::Matrix3d::Identity(),
                                                    Eigen::Vector3d(L1 + L2 + L3, 0, 0));

    const Eigen::Matrix4d T1 = matrix_exponential(S1.head<3>(), S1.tail<3>(), q1);
    const Eigen::Matrix4d T2 = matrix_exponential(S2.head<3>(), S2.tail<3>(), q2);
    const Eigen::Matrix4d T3 = matrix_exponential(S3.head<3>(), S3.tail<3>(), q3);

    return T1 * T2 * T3 * M;
}

void test_planar_3r_fk_screw()
{
    std::cout << "========================== T.4 c) ==========================\n";

    const std::vector<std::vector<double>> tests = {
        {0.0, 0.0, 0.0},
        {90.0, 0.0, 0.0},
        {0.0, 90.0, 0.0},
        {0.0, 0.0, 90.0},
        {10.0, -15.0, 2.75}};
    const char *name[] = {"j1", "j2", "j3", "j4", "j5"};

    for (int i = 0; i < (int)tests.size(); ++i)
    {
        Eigen::Matrix4d T_chain = planar_3r_fk_transform(tests[i]);
        Eigen::Matrix4d T_poe = planar_3r_fk_screw(tests[i]);

        std::cout << "Pose " << name[i] << " (chain):\n";
        print_pose("", T_chain);

        std::cout << "Pose " << name[i] << " (PoE):\n";
        print_pose("", T_poe);

        std::cout << "||T_chain - T_PoE||_max = "
                  << (T_chain - T_poe).cwiseAbs().maxCoeff() << "\n\n";
    }
}

// ====================================== T.5 a) ======================================
static inline Eigen::Matrix4d A(double a, double alpha, double d, double theta)
{
    Eigen::Matrix4d T = transformation_matrix(rotate_z(theta), {0, 0, d});
    T = T * transformation_matrix(Eigen::Matrix3d::Identity(), {a, 0, 0});
    T = T * transformation_matrix(rotate_x(alpha), {0, 0, 0});
    return T;
}

Eigen::Matrix4d ur3e_fk_screw(const std::vector<double> &q_deg)
{
    static const double a_link[6] = {0.0, -0.24355, -0.21320, 0.0, 0.0, 0.0};
    static const double d_link[6] = {0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.09210};
    static const double alpha_link[6] = {EIGEN_PI / 2, 0.0, 0.0, EIGEN_PI / 2, -EIGEN_PI / 2, 0.0};

    auto rad = [](double deg)
    { return deg * EIGEN_PI / 180.0; };

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    Eigen::Vector3d o = Eigen::Vector3d::Zero();
    Eigen::Vector3d z = Eigen::Vector3d::UnitZ();
    Eigen::Matrix<double, 6, 1> S[6];

    // Equation (4.7) on page 137, MR pre-print 2019
    for (int i = 0; i < 6; ++i)
    {
        S[i].head<3>() = z;
        S[i].tail<3>() = -z.cross(o);

        T = T * A(a_link[i], alpha_link[i], d_link[i], 0.0);
        o = T.block<3, 1>(0, 3);
        z = T.block<3, 3>(0, 0) * Eigen::Vector3d::UnitZ();
    }
    const Eigen::Matrix4d M = T; // Equation (4.6) on page 136, MR pre-print 2019

    // Equation (4.10) on page 138, MR pre-print 2019
    Eigen::Matrix4d G = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 6; ++i)
    {
        const double th = (i < (int)q_deg.size()) ? rad(q_deg[i]) : 0.0;
        G *= matrix_exponential(S[i].head<3>(), S[i].tail<3>(), th);
    }
    return G * M;
}

void test_ur3e_fk_screw()
{
    std::cout << "========================== T.5 a) ==========================\n";
    std::vector<std::vector<double>> J = {
        {0, 0, 0, -90, 0, 0},
        {0, -180, 0, 0, 0, 0},
        {0, -90, 0, 0, 0, 0}};
    const char *name[] = {"j1", "j2", "j3"};
    for (int i = 0; i < 3; ++i)
    {
        Eigen::Matrix4d T = ur3e_fk_screw(J[i]);
        print_pose(std::string("UR3e ") + name[i], T);
    }
}

// ====================================== T.5 b) ======================================
// Equation (C.1) on page 587, MR pre-print 2019
// Equations (C.2-C.5) on page 592, MR pre-print 2019
Eigen::Matrix4d ur3e_fk_transform(const std::vector<double> &q_deg)
{
    static const double a[6] = {0.0, -0.24355, -0.21320, 0.0, 0.0, 0.0};
    static const double d[6] = {0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.09210};
    static const double alpha[6] = {EIGEN_PI / 2, 0.0, 0.0, EIGEN_PI / 2, -EIGEN_PI / 2, 0.0};

    auto rad = [](double deg)
    { return deg * EIGEN_PI / 180.0; };

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 6; ++i)
    {
        const double theta = (i < (int)q_deg.size()) ? rad(q_deg[i]) : 0.0;
        T *= A(a[i], alpha[i], d[i], theta);
    }
    return T;
}

void test_ur3e_fk_transform_vs_screw()
{
    std::cout << "========================== T.5 b) ==========================\n";

    const std::vector<std::vector<double>> tests = {
        {0, 0, 0, -90, 0, 0},
        {0, -180, 0, 0, 0, 0},
        {0, -90, 0, 0, 0, 0}};
    const char *name[] = {"j1", "j2", "j3"};

    for (int i = 0; i < (int)tests.size(); ++i)
    {
        Eigen::Matrix4d T_screw = ur3e_fk_screw(tests[i]);
        Eigen::Matrix4d T_dh = ur3e_fk_transform(tests[i]);

        std::cout << "UR3e " << name[i] << " (PoE):\n";
        print_pose("", T_screw);

        std::cout << "UR3e " << name[i] << " (DH):\n";
        print_pose("", T_dh);

        double err = (T_screw - T_dh).cwiseAbs().maxCoeff();
        std::cout << "||T_PoE - T_DH||_max = " << err << "\n\n";
    }
}

// ====================================== main ======================================
int main()
{
    // T.1 a)
    test_euler_zyx(EIGEN_PI / 4.0, EIGEN_PI / 6.0, -EIGEN_PI / 3.0);

    // T.1 b)
    test_twist();

    // T.1 c)
    test_screw_axis();

    // T.1 d)
    test_adjoint_matrix();

    // T.1 e)
    test_cot();

    // T.2 a)
    changeWrenchFrame();

    // T.2 b)
    sensor_wrench_example_3_28();

    // T.3 a)
    test_matrix_exponential_SO3();

    // T.3 b)
    test_matrix_logarithm_SO3();

    // T.3 c)
    test_matrix_exponential_SE3();

    // T.3 d)
    test_matrix_logarithm_SE3();

    // T.4 a)
    test_print_pose();

    // T.4 b)
    test_planar_3r_fk();

    // T.4 c)
    test_planar_3r_fk_screw();

    // T.5 a)
    test_ur3e_fk_screw();

    // T.5 b)
    test_ur3e_fk_transform_vs_screw();
}