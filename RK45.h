#pragma once
#include <Eigen/Core>

Eigen::VectorXd RK45(Eigen::VectorXd init, double t0, double t,
	std::function<Eigen::VectorXd(const Eigen::VectorXd&, const double&)> df, double tol = 1e-3, double step_min = 1e-4, double step_max = 1e-2);