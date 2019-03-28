#include <omp.h>
#include <ctime>
#include <complex>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "RK45.h"
#include "FFT.h"
#include "pi.h"

#pragma comment(lib,"FFT.lib")

inline float sign(float x) { return x == 0.f ? 0.f : (x > 0.f ? 1.f : -1.f); }

using namespace std;
using namespace cv;
using namespace Eigen;

constexpr double x0 = 0;
constexpr double v0 = 0;
constexpr double omega = 2 * pi;
constexpr double omega0 = omega * 3 / 2;
constexpr double beta = omega0 / 4;

constexpr double T = 2 * pi / omega;

constexpr double gamma_min = 1.10;
constexpr double gamma_max = 1.20;
constexpr double gamma_step = (gamma_max - gamma_min) / 65536;//65536

constexpr double t_min = 512;
constexpr double t_max = t_min + 2048;
constexpr double sample_rate = 64. / T;
constexpr double t_step = 1 / sample_rate;

int main() {
	unsigned long N = static_cast<unsigned long>((t_max - t_min))*static_cast<unsigned long>(sample_rate);
	unsigned long BATCH = static_cast<unsigned long>(omp_get_num_procs() * 128);
	unsigned long batch_num = static_cast<unsigned long>(ceil((gamma_max - gamma_min) / gamma_step / BATCH));

	cout << "Num procs using: " << omp_get_num_procs() << endl;
	cout << "Signal length: " << N << " (t: " << t_min << "s - " << t_max << "s x " << sample_rate << "sample per second)" << endl;
	cout << "Num of Simulations: " << batch_num * BATCH << endl;
	cout << "size of Batch: " << BATCH << endl;
	cout << "Num of Batch: " << batch_num << endl;
	cout << "Gamma: [ " << gamma_min << " , " << (gamma_min + batch_num * gamma_step*BATCH) << " ) with step: " << gamma_step << endl;
	cout << "\n\n" << endl;

	Mat fft_result_norm(BATCH*batch_num, N / 2 + 1, CV_32FC1);
	Mat out_position(BATCH*batch_num, 1 + 3142 * 2, CV_32FC1, Scalar(0));
	Mat out_speed(BATCH*batch_num, 1 + 4000 * 2, CV_32FC1, Scalar(0));


	auto raw_x = new float[N*BATCH]; //cache
	auto raw_v = new float[N*BATCH]; //cache
	std::complex<float>* fft_result = new std::complex<float>[(N / 2 + 1)*BATCH]; //cache
	VectorXd init(2); //cache
	init << x0, v0;
	
	for (unsigned long batch_id = 0; batch_id < batch_num; ++batch_id) {
		std::time_t t_result = std::time(nullptr);
		cout << "Start Batch " << batch_id + 1 << " / " << batch_num
			<< " (gamma: " << gamma_min + gamma_step * batch_id * BATCH << " - " << gamma_min + gamma_step * ((batch_id + 1) * BATCH - 1) << ")"
			<< " time: " << std::asctime(std::localtime(&t_result)) << endl;
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

			auto y = RK45(init, 0, t_min, f, 1e-3, 5e-4, 1e-3);
			auto t = t_min;
			for (unsigned t_id = 0; t_id < N; ++t_id) {
				raw_x[id*N + t_id] = static_cast<float>(y[0]);
				raw_v[id*N + t_id] = static_cast<float>(y[1]);
				y = RK45(y, t, t + t_step, f, 1e-3, 5e-4, 1e-3);
				t += t_step;
			}
		}

		cout << "Finished RK45, start FFT" << endl;

		FFT(raw_x, fft_result, N, BATCH);

		cout << "Finished FFT, start data refine" << endl;

#pragma omp parallel for
		for (long id = 0; id < static_cast<long>(BATCH); ++id) {
			unsigned long gamma_id = static_cast<unsigned long>(batch_id * BATCH + id);

			auto p_x = out_position.ptr<float>(gamma_id);
			auto p_v = out_speed.ptr<float>(gamma_id);
			auto p_fft = fft_result_norm.ptr<float>(gamma_id);

			double max_x = 0;
			double max_v = 0;
			double max_fft = 0;

			for (auto q = fft_result + (N / 2 + 1)*id, end = q + N / 2 + 1; q < end; ++q, ++p_fft) {
				*p_fft = norm(*q);
				if (*p_fft > max_fft)
					max_fft = *p_fft;
			}
			for (auto x = raw_x + N * id, v = raw_v + N * id, end = x + N; x < end; x += static_cast<unsigned long>(sample_rate), v += static_cast<unsigned long>(sample_rate)) {
				auto t = static_cast<unsigned long>(3142 + round(abs(1000 * fmod(*x, pi)))*sign(*x));
				if (0 <= t && t < 3142 * 2 + 1) {
					p_x[t] = p_x[t] + 1;
					if (p_x[t] > max_x) max_x = p_x[t];
				}
				t = static_cast<unsigned long>(4000 + round(abs(100 * (*v)))*sign(*v));
				if (0 <= t && t < 4000 * 2 + 1) {
					p_v[t] = p_v[t] + 1;
					if (p_v[t] > max_v) max_v = p_v[t];
				}
			}

			p_x = out_position.ptr<float>(gamma_id);
			p_v = out_speed.ptr<float>(gamma_id);
			p_fft = fft_result_norm.ptr<float>(gamma_id);

			for (auto end = p_fft + (N / 2 + 1); p_fft < end; ++p_fft)
				*p_fft = (10.f + log10f(static_cast<float>(*p_fft / max_fft))) / 10.f*255.f;
			for (auto end = p_x + 1 + 3142 * 2; p_x < end; ++p_x)
				*p_x = static_cast<float>(*p_x / max_x)*255.f;
			for (auto end = p_v + 1 + 3142 * 2; p_v < end; ++p_v)
				*p_v = static_cast<float>(*p_v / max_v)*255.f;
		}

		cout << "Finished Calculation of Batch " << batch_id + 1 << " / " << batch_num
			<< " (gamma: " << gamma_min + gamma_step * batch_id * BATCH << " - " << gamma_min + gamma_step * ((batch_id + 1) * BATCH - 1) << ")\n\n" << endl;
	}

	delete[] raw_x;
	delete[] raw_v;
	delete[] fft_result;

	Mat out;
	fft_result_norm.convertTo(out, CV_8UC1);
	imwrite("FFT.png", out);
	out_position.convertTo(out, CV_8UC1);
	imwrite("position.png", out);
	out_speed.convertTo(out, CV_8UC1);
	imwrite("speed.png", out);

	return 0;

}