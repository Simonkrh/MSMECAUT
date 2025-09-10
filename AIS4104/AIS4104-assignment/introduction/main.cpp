#include <iostream>

#include <Eigen/Dense>

// T.1 a)
// Equation (3.30) page 75, MR pre-print 2019
Eigen::Matrix3d skew_symmetric(Eigen::Vector3d vec)
{
    Eigen::Matrix3d skewMatrix;
    skewMatrix << 0.0, -vec(2), vec(1),
        vec(2), 0.0, -vec(0),
        -vec(1), vec(0), 0.0;
    return skewMatrix;
}

// T.1 b)
void skew_symmetric_test()
{
    Eigen::Matrix3d skew_matrix = skew_symmetric(Eigen::Vector3d{0.5, 0.5, 0.707107});
    std::cout << "Skew-symmetric matrix: " << std::endl;
    std::cout << skew_matrix << std::endl;
    std::cout << "Skew-symmetric matrix transposition: " << std::endl;
    std::cout << -skew_matrix.transpose() << std::endl;
}

// T.2 a)
// Equation (3.16) page 65, MR pre-print 2019
Eigen::Matrix3d rotation_matrix_from_frame_axes(const Eigen::Vector3d &x, const Eigen::Vector3d &y,
                                                const Eigen::Vector3d &z)
{
    Eigen::Matrix3d matrix;
    matrix << x.normalized(), y.normalized(), z.normalized();
    return matrix;
}

// T.2 b)
// Equation on page 65, MR pre-print 2019
Eigen::Matrix3d rotate_x(double degrees)
{
    const double radians = degrees * M_PI / 180.0;
    Eigen::Matrix3d matrix;
    matrix << 1, 0, 0,
        0, cos(radians), -sin(radians),
        0, sin(radians), cos(radians);
    return matrix;
}

// T.2 c)
// Equation on page 65, MR pre-print 2019
Eigen::Matrix3d rotate_y(double degrees)
{
    const double radians = degrees * M_PI / 180.0;
    Eigen::Matrix3d matrix;
    matrix << cos(radians), 0, sin(radians),
        0, 1, 0,
        -sin(radians), 0, cos(radians);
    return matrix;
}

// T.2 d)
// Equation on page 65, MR pre-print 2019
Eigen::Matrix3d rotate_z(double degrees)
{
    const double radians = degrees * M_PI / 180.0;
    Eigen::Matrix3d matrix;
    matrix << cos(radians), -sin(radians), 0,
        sin(radians), cos(radians), 0,
        0, 0, 1;
    return matrix;
}

// T.2 e)
// Equation on page 65, MR pre-print 2019
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

// T.2 f)
// Equation on page 577, MR pre-print 2019
Eigen::Matrix3d rotation_matrix_from_euler_zyx(const Eigen::Vector3d &e)
{
    return rotate_z(e[0]) * rotate_y(e[1]) * rotate_x(e[2]);
}

// T.2 g)
void rotation_matrix_test()
{
    Eigen::Matrix3d rot = rotation_matrix_from_euler_zyx(Eigen::Vector3d{45.0, -45.0, 90.0});
    Eigen::Matrix3d rot_aa = rotation_matrix_from_axis_angle(Eigen::Vector3d{0.8164966, 0.0, 0.5773503}, 120.0);
    Eigen::Matrix3d rot_fa = rotation_matrix_from_frame_axes(Eigen::Vector3d{0.5, 0.5, 0.707107},
                                                             Eigen::Vector3d{-0.5, -0.5, 0.707107}, Eigen::Vector3d{0.707107, -0.707107, 0.0});
    std::cout << "Rotation matrix from Euler: " << std::endl;
    std::cout << rot << std::endl
              << std::endl;
    std::cout << "Rotation matrix from axis-angle pair: " << std::endl;
    std::cout << rot_aa << std::endl
              << std::endl;
    std::cout << "Rotation matrix from frame axes: " << std::endl;
    std::cout << rot_fa << std::endl
              << std::endl;
}

// T.3 a)
Eigen::Matrix4d transformation_matrix(const Eigen::Matrix3d &r, const Eigen::Vector3d &p)
{
    Eigen::Matrix4d matrix;
    // implement the necessary equations and functionality.
    return matrix;
}

// T.3 b)
void transformation_matrix_test()
{
    Eigen::Matrix3d r = rotation_matrix_from_euler_zyx(Eigen::Vector3d{45, -45.0, 90.0});
    Eigen::Vector3d v{1.0, -2.0, 3.0};
    std::cout << "transformation_matrix: " << std::endl;
    std::cout << transformation_matrix(r, v) << std::endl;
}

int main()
{
    skew_symmetric_test();
    rotation_matrix_test();
    return 0;
}
