#include <iostream>
#include <Eigen/Dense>
#include <numbers>
#include <cmath>

constexpr double M_PI = std::numbers::pi;

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
        ang[1] = M_PI / 2.0;
        ang[0] = 0.0;
        ang[2] = std::atan2(r12, r22);
    }
    else if (std::abs(r31 - 1.0) < eps)
    {
        ang[1] = -M_PI / 2.0;
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

// Equation on page 65, MR pre-print 2019
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

// Equation on page 65, MR pre-print 2019
// From Assignment 1
Eigen::Matrix3d rotate_y(double radians)
{
    const double cos = std::cos(radians), sin = std::sin(radians);
    Eigen::Matrix3d R;
    R << cos, 0, sin,
        0, 1, 0,
        -sin, 0, cos;
    return R;
}

// Equation on page 65, MR pre-print 2019
// From Assignment 1
Eigen::Matrix3d rotate_z(double radians)
{
    const double cos = std::cos(radians), sin = std::sin(radians);
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
// Equation (3.74) on page 97, MR pre-print 2019
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

    std::cout << "cot(pi/4) = " << cot(M_PI / 4) << "\n";
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

    Eigen::Vector3d e_ws = e_ws_deg * (M_PI / 180.0);

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
// Equation on page 65, MR pre-print 2019
// From Assignment 1
Eigen::Matrix3d rotation_matrix_from_axis_angle(const Eigen::Vector3d &axis,
                                                double degrees)
{
    Eigen::Vector3d u = axis.normalized();
    const double radians = degrees * M_PI / 180.0;
    const double cos = std::cos(radians);
    const double sin = std::sin(radians);

    Eigen::Matrix3d matrix;
    matrix << cos + u.x() * u.x() * (1 - cos), u.x() * u.y() * (1 - cos) - u.z() * sin, u.x() * u.z() * (1 - cos) + u.y() * sin,
        u.y() * u.x() * (1 - cos) + u.z() * sin, cos + u.y() * u.y() * (1 - cos), u.y() * u.z() * (1 - cos) - u.x() * sin,
        u.z() * u.x() * (1 - cos) - u.y() * sin, u.z() * u.y() * (1 - cos) + u.x() * sin, cos + u.z() * u.z() * (1 - cos);

    return matrix;
}

// Equation (3.51), page 81, MR pre-print 2019
Eigen::Matrix3d matrix_exponential(const Eigen::Vector3d &w, double theta)
{
    return rotation_matrix_from_axis_angle(w, theta * 180.0 / M_PI);
}

void test_matrix_exponential()
{
    std::cout << "========================== T.3 a) ==========================\n";

    // Rotation of 90° about z-axis
    Eigen::Vector3d w(0, 0, 1);
    double theta = M_PI / 2.0;

    Eigen::Matrix3d R = matrix_exponential(w, theta);

    std::cout << "R =\n"
              << R << "\n";
    std::cout << "Expected:\n"
              << "0 -1  0\n"
              << "1  0  0\n"
              << "0  0  1 \n\n";
}

// ====================================== main ======================================
int main()
{
    // T.1 a)
    test_euler_zyx(M_PI / 4, M_PI / 6, -M_PI / 3);

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
    test_matrix_exponential();
}