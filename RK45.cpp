#include "RK45.h"

using namespace Eigen;

Eigen::VectorXd RK45(Eigen::VectorXd y, double t, double T, std::function<Eigen::VectorXd(const Eigen::VectorXd&, const double&)> f, double tol, double smin, double smax)
{
	VectorXd k1, k2, k3, k4, k5, k6, R, y_ = y;
	double h = 0.05;
	while (true) {
		if (t + h > T)
			h = T - t;
		k1 = h * f(y, t);
		k2 = h * f((y * 4 + k1) / 4, (t * 4 + h) / 4);
		k3 = h * f((y * 32 + k1 * 3 + k2 * 9) / 32, (t * 8 + h * 3) / 8);
		k4 = h * f((y * 2197 + k1 * 1932 - k2 * 7200 + k3 * 7296) / 2197, (t * 13 + h * 12) / 13);
		k5 = h * f((y * 4104 + k1 * 8341 - k2 * 32832 + k3 * 29440 - k4 * 845) / 4104, t + h);
		k6 = h * f((y * 20520 - k1 * 6080 + k2 * 41040 - k3 * 28352 + k4 * 9295 - k5 * 5643) / 20520, (t * 2 + h) / 2);

		R = (4285545 * k1 - 46193664 * k3 - 45653660 * k4 + 30855924 * k5 + 56101680 * k6) / 771398100;
		double err = R.maxCoeff();
		if (h <= smin || err / h < 2 * tol) {
			t += h;
			y_ = y;
			y += (33440 * k1 + 146432 * k3 + 142805 * k4 - 50787 * k5 + 10260 * k6) / 282150;
			if (t >= T)
				break;
		}
		h *= pow(h / err * tol, 1/4.);
		h = std::min(smax, std::max(h, smin));
	}
	return y + (t - T)*(y_ - y) / h;
}
