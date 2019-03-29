#include <omp.h>
#include <ctime>
#include <complex>
#include <fstream>
#include <iostream>
#include "RK45.h"
#include "pi.h"

inline float sign(float x) { return x == 0.f ? 0.f : (x > 0.f ? 1.f : -1.f); }

using namespace std;
using namespace Eigen;

constexpr double x0 = 0;
constexpr double v0 = 0;
constexpr double omega = 2 * pi;
constexpr double omega0 = omega * 3 / 2;
constexpr double beta = omega0 / 4;

constexpr double T = 2 * pi / omega;

constexpr double gamma_min = 1.08;
constexpr double gamma_max = 1.09;
constexpr double gamma_step = (gamma_max - gamma_min) / 1024;//65536

constexpr double t_min = 512;
constexpr double t_max = t_min + 128;
unsigned long sample_rate = static_cast<unsigned long>(ceil(32. / T));
double t_step = 1. / sample_rate;

constexpr double tol = 1e-3;
constexpr double h_min = 1e-3;
constexpr double h_max = 1e-2;

unsigned long t_num = static_cast<unsigned long>(t_max - t_min) * sample_rate;
unsigned long BATCH = static_cast<unsigned long>(omp_get_num_procs() * 128);
unsigned long batch_num = static_cast<unsigned long>(ceil((gamma_max - gamma_min) / gamma_step / BATCH));
unsigned long Gamma_num = BATCH * batch_num;

#define SIZE_INFO(T) cout<<"Size of "<<#T<<" is "<<sizeof(T)<<" bytes"<<endl

int main() {
	SIZE_INFO(int);
	SIZE_INFO(unsigned);
	SIZE_INFO(long);
	SIZE_INFO(unsigned long);
	SIZE_INFO(float);
	SIZE_INFO(double);
	SIZE_INFO(std::complex<float>);
	
	cout << "\n\n" << endl;

	cout << "Signal length: " << t_num << " (t: " << t_min << "s - " << t_max << "s ( " << static_cast<unsigned long>(t_max - t_min) << " s) x " << sample_rate << " samples per second)" << endl;
	cout << "Gamma: [ " << gamma_min << " , " << (gamma_min + Gamma_num * gamma_step) << " ) with step: " << gamma_step << endl;
	cout << "Num of Simulations: " << batch_num * BATCH << endl;
	cout << "\nNum procs using: " << omp_get_num_procs() << endl;
	cout << "size of Batch: " << BATCH << endl;
	cout << "Num of Batch: " << batch_num << endl;
	cout << "\n\n" << endl;

	system("pause");

	std::ofstream info("info.dat", std::ios::binary);
	std::ofstream file_raw_x("raw_x.dat", std::ios::binary);
	std::ofstream file_raw_v("raw_v.dat", std::ios::binary);

	info.write((char*)&sample_rate, sizeof(unsigned long));
	info.write((char*)&Gamma_num, sizeof(unsigned long));
	info.write((char*)&t_num, sizeof(unsigned long));

	info.close();

	auto raw_x = new float[t_num * BATCH]; //cache
	auto raw_v = new float[t_num * BATCH]; //cache
	VectorXd init(2); //cache
	init << x0, v0;

	std::time_t Start_time = std::time(0);
	for (unsigned long batch_id = 0; batch_id < batch_num; ++batch_id) {
		std::time_t t_result = std::time(nullptr);
		cout << "Start Batch " << batch_id + 1 << " / " << batch_num
			<< " (gamma: " << gamma_min + gamma_step * batch_id * BATCH << " - " << gamma_min + gamma_step * ((batch_id + 1) * BATCH - 1) << ")"
			<< " time: " << std::asctime(std::localtime(&t_result)) << std::flush;
#pragma omp parallel for
		for (long id = 0; id < static_cast<long>(BATCH); ++id) {
			unsigned long gamma_id = static_cast<unsigned long>(batch_id * BATCH + id);
			double gamma = gamma_min + gamma_id * gamma_step;

			auto f = [=](auto Y, auto t) {
				double x = Y[0], v = Y[1];
				/*dx/dt*/ Y[0] = v;
				/*dv/dt*/ Y[1] = -omega0 * omega0 * sin(x) - 2 * beta * v + gamma * omega0*omega0 * cos(omega*t);
				return Y;
			};

			auto y = RK45(init, 0, t_min, f, tol, h_min, h_max);
			auto t = t_min;
			for (unsigned t_id = 0; t_id < t_num; ++t_id) {
				raw_x[id*t_num + t_id] = static_cast<float>(y[0]);
				raw_v[id*t_num + t_id] = static_cast<float>(y[1]);
				y = RK45(y, t, t + t_step, f, tol, h_min, h_max);
				t += t_step;
			}
		}

		cout << "Finished RK45, start saving raw data" << endl;

		file_raw_x.write((char*)raw_x, t_num*BATCH * sizeof(float));
		file_raw_v.write((char*)raw_v, t_num*BATCH * sizeof(float));

		file_raw_x.flush();
		file_raw_v.flush();

		unsigned long ETA = static_cast<unsigned long>(static_cast<double>(std::time(0) - Start_time) / (batch_id + 1)*(batch_num - batch_id - 1));
		cout << "Finished Calculation of Batch " << batch_id + 1 << " / " << batch_num
			<< "  (gamma: " << gamma_min + gamma_step * batch_id * BATCH << " - " << gamma_min + gamma_step * ((batch_id + 1) * BATCH - 1) << ")"
			<< "  ETA: " << ETA / 3600 << ":" << (ETA % 3600) / 60 << ":" << ETA % 3600 % 60 << "\n\n" << endl;
	}

	file_raw_x.close();
	file_raw_v.close();

	delete[] raw_x;
	delete[] raw_v;

	return 0;

}