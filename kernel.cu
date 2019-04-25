#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <string> 
#include <array>
#include <iomanip>
#include <chrono>
#include <fstream>

typedef std::chrono::high_resolution_clock Clock;

#define pi 3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481117450284102701938521f
#define omega (2.f*pi)
#define omega0 (omega*3.f/2.f)
#define beta (omega0/4.f)

#define roll_num 16
#define wrarp_num 32
#define grid_num 4
#define time 10
#define h_exp_3 5
#define t_step_exp 8

#define batch_size ( wrarp_num * grid_num * 32)
#define t_step (1.f/(1<<t_step_exp))
#define h_level_3 (1<<h_exp_3)
#define h (1.f/(1<<(h_exp_3*3+t_step_exp)))

#define t_size (sizeof(unsigned long) * batch_size)
#define p_size (sizeof(float)*(4 + 2 * roll_num) * batch_size)


__device__ void __RK4(unsigned long t_index, float x, float v, float *outx, float *outv, float y, float dy);
__device__ void RK4(unsigned long t_index, float x, float v, float *outx, float *outv, float y, float dy);
void RK4_CPU(unsigned t_index, float y, float dy, float& x, float& v);

__global__ void kernel(unsigned long *index, float * p) {
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	for (unsigned i = 0; i < roll_num; ++i) {
		__RK4(index[id] + i,
			p[batch_size*(2 * i + 2) + id], p[batch_size*(2 * i + 3) + id],
			p + batch_size * (2 * i + 4) + id, p + batch_size * (2 * i + 5) + id,
			p[id], p[batch_size + id]);
	}
}

struct workinggroup
{
	float *p_h, *p_d;
	unsigned long *t_index_h, *t_index_d;
	cudaStream_t stream;
	int gpu;
	workinggroup(int gpu_id) :gpu(gpu_id) {
		t_index_h = (unsigned long*)malloc(t_size);
		p_h = (float*)malloc(p_size);
		cudaSetDevice(gpu);
		cudaStreamCreate(&stream);
		cudaMalloc((void**)&t_index_d, t_size);
		cudaMalloc((void**)&p_d, p_size);
	}
	~workinggroup() {
		free(t_index_h);
		free(p_h);
		cudaSetDevice(gpu);
		cudaStreamDestroy(stream);
		cudaFree(t_index_d);
		cudaFree(p_d);
	}
	inline unsigned long &t_index(unsigned long index) { return t_index_h[index]; }

	inline float &gamma(unsigned long index) { return p_h[0 * batch_size + index]; }
	inline float &dgamma(unsigned long index) { return p_h[1 * batch_size + index]; }

	inline float &x(unsigned long t_index, unsigned long index) { return p_h[(2 * t_index + 2) * batch_size + index]; }
	inline float &v(unsigned long t_index, unsigned long index) { return p_h[(2 * t_index + 3) * batch_size + index]; }

	inline float &x_0(unsigned long index) { return x(0, index); }
	inline float &v_0(unsigned long index) { return v(0, index); }
	inline float &x_n(unsigned long index) { return x(roll_num, index); }
	inline float &v_n(unsigned long index) { return v(roll_num, index); }

	inline void lunch_plan() {
		cudaSetDevice(gpu);
		cudaMemcpyAsync((void*)t_index_d, (void*)t_index_h, t_size, cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync((void*)p_d, (void*)p_h, p_size, cudaMemcpyHostToDevice, stream);

		kernel <<<grid_num, wrarp_num * 32, 0, stream >>> (t_index_d, p_d);

		cudaMemcpyAsync((void*)p_h, (void*)p_d, p_size, cudaMemcpyDeviceToHost, stream);
	}

	inline void sync() { cudaSetDevice(gpu); cudaStreamSynchronize(stream); }
	inline bool check() { cudaSetDevice(gpu); return cudaStreamQuery(stream) != cudaErrorNotReady; }
};

struct info
{
	unsigned long gpu;			//gpu using
	unsigned long t_file;		//time step per file
	unsigned long t_num;		//time step per second
	unsigned long t_max;		//time range [0, t_max]
	unsigned long t_n;			//total sample number
	unsigned long t_current;	//time step finished
	unsigned long gamma_num;	//testing gamma's number
	float init_gamma;			//gamma start
	float d_gamma;				//gamma step
	float x0;					//init x
	float v0;					//init v
};

int main(int argc, char *argv[])
{
	info para;
	bool continue_working = false;
	unsigned long init_t = 0;
	unsigned long start_t = 0;
	unsigned long file_id = 0;
	float gamma_range = 0.2;

	para.t_num = (1 << t_step_exp);
	para.t_file = 128 * para.t_num;
	para.gpu = 0;
	para.x0 = 0;
	para.v0 = 0;
	para.t_max = 10;
	para.init_gamma = 1.f;

	para.d_gamma = gamma_range / (batch_size * 3);
	para.t_n = para.t_max * para.t_num;
	para.gamma_num = batch_size * 3;

	if (argc > 1) {
		for (int i = 0; i < argc; ++i) {
			std::string arg = argv[i];
			if (arg == "-h" || arg == "--help" || arg == "-?") {
				std::cout << "-g:\tset the gpu for calculate" << std::endl;
				std::cout << "-x:\tset init x0" << std::endl;
				std::cout << "-v:\tset init v0" << std::endl;
				std::cout << "-t:\tset time range" << std::endl;
				std::cout << "-y:\tset minimal gamma" << std::endl;
				std::cout << "-yr:\tset the gamma searching range (d_gamma = gamma_range / " << para.gamma_num << ")" << std::endl;
				std::cout << "-h, --help, -?: " << std::endl;
				return 0;
			}
			else if (arg == "-g")
				para.gpu = std::stoi(argv[++i]);
			else if (arg == "-x")
				para.x0 = std::stof(argv[++i]);
			else if (arg == "-v")
				para.v0 = std::stof(argv[++i]);
			else if (arg == "-t") {
				para.t_max = std::stoul(argv[++i]);
				para.t_n = para.t_max * para.t_num;
			}
			else if (arg == "-y")
				para.init_gamma = std::stof(argv[++i]);
			else if (arg == "-yr") {
				gamma_range = std::stof(argv[++i]);
				para.d_gamma = gamma_range / (batch_size * 3);
			}
		}
	}
	else {
		std::ifstream info_file("info.dat", std::ios::binary);
		if (info_file.is_open()) {
			info tmp;
			info_file.read((char*)&tmp, sizeof(info));
			if (tmp.t_current != tmp.t_n) {
				continue_working = true;
				start_t = (tmp.t_current / roll_num);
				init_t = start_t * roll_num - 1;
				file_id = init_t / tmp.t_file;
				para = tmp;
				std::cout << "Find unfinished task! Auto start continue working from t id " << start_t * roll_num << " ...\n(disable by adding any arg)\n\n\n\n" << std::endl;
			}
		}
	}
	


	std::cout << "Starting calculation:\n" << "\tx0: " << para.x0 << "   v0: " << para.v0
		<< "\n\tgamma: [ " << para.init_gamma << " , " << (para.init_gamma + para.gamma_num*para.d_gamma) << " ) -- " << para.gamma_num << " steps with step length: " << para.d_gamma
		<< "\n\tt_max: " << para.t_max << "\n\ton GPU " << para.gpu << "\n\n" << std::endl;

	float *tmp_data = new float[para.gamma_num * roll_num * 2];

	if (continue_working) {
		std::ifstream data(std::string("data") + std::to_string(file_id) + std::string(".dat"), std::ios::binary);
		data.seekg(para.gamma_num * 2 * (init_t % para.t_file) * sizeof(float));
		data.read((char*)tmp_data, para.gamma_num * 2 * sizeof(float));
	}

	workinggroup a(para.gpu), b(para.gpu), c(para.gpu);
	workinggroup *g[] = { &a, &b, &c };

	for (int i = 0; i < 3; ++i) {
		for (unsigned long j = 0; j < batch_size; ++j) {
			g[i]->t_index(j) = start_t * roll_num;
			g[i]->gamma(j) = para.init_gamma;
			g[i]->dgamma(j) = para.d_gamma * (batch_size*i + j);
			g[i]->x_0(j) = continue_working ? tmp_data[2 * (batch_size * i + j)] : para.x0;
			g[i]->v_0(j) = continue_working ? tmp_data[2 * (batch_size * i + j) + 1] : para.v0;
		}
		g[i]->lunch_plan();
	}

	auto start = Clock::now();
	for (unsigned long t = start_t; t < para.t_n / roll_num; ++t) {
		auto s = Clock::now();
		para.t_current = (t + 1) * roll_num;
		for (int i = 0; i < 3; ++i) {
			g[i]->sync();
			for (unsigned long j = 0; j < batch_size; ++j) for (unsigned long k = 0; k < roll_num; ++k) {
				tmp_data[2 * (para.gamma_num * k + i * batch_size + j)] = g[i]->x(k + 1, j);
				tmp_data[2 * (para.gamma_num * k + i * batch_size + j) + 1] = g[i]->v(k + 1, j);
			}
			if (t != (para.t_n / roll_num - 1)) {
				for (unsigned long j = 0; j < batch_size; ++j) {
					g[i]->t_index(j) = para.t_current;
					g[i]->x_0(j) = g[i]->x_n(j);
					g[i]->v_0(j) = g[i]->v_n(j);
				}
				g[i]->lunch_plan();
			}

		}
		
		std::ofstream data_file;
		data_file.open(std::string("data") + std::to_string(t * roll_num / para.t_file) + std::string(".dat"), std::ios::binary | std::ios::out | std::ios::in);
		if (data_file.is_open())
			data_file.seekp(t * roll_num % para.t_file * para.gamma_num * 2 * sizeof(float));
		else
			data_file.open(std::string("data") + std::to_string(t * roll_num / para.t_file) + std::string(".dat"), std::ios::binary | std::ios::out);
		data_file.write((char*)tmp_data, para.gamma_num * roll_num * 2 * sizeof(float));
		data_file.close();

		std::ofstream info_file("info.dat", std::ios::binary);
		info_file.write((char*)&para, sizeof(para));
		info_file.close();

		auto e = Clock::now();

		std::cout.setf(std::ios::fixed);
		std::cout << "Finish calculation: 0 - " << ((t + 1) * roll_num - 1) << "\t(current t ID: " << (t + 1) * roll_num << ") / " << para.t_n
			<< " |\tfrom " << std::setprecision(3) << t * roll_num * t_step << "s to " << std::setprecision(3) << (t + 1) * roll_num * t_step << "s |\tused "
			<< std::setprecision(2) << (static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count()) / 1000000) << "ms\t("
			<< std::setprecision(4) << ((t + 1) * roll_num * 100.f / para.t_n) << "%  used: "
			<< std::setprecision(3)
			<< (static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(e - start).count()) / 3600000000000) << "hours  ETA: "
			<< std::setprecision(3)
			<< (para.t_n - (t + 1) * roll_num) *(static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(e - start).count()) / ((t + 1 - start_t) * roll_num) / 3600000000000) << " hours)"
			<< std::endl;
	}
	delete[] tmp_data;
	return 0;
}

__device__ void __RK4(unsigned long t_index, float x, float v, float *outx, float *outv, float y, float dy) {
	/**********************************************************************
		(v' , x') = f((v+dv+ddv , x+dx+ddx) , t+dt): gamma = gamma + dgamma
		ot = omega * t;
		odt = omega * dt;

		bv = -2*beta* v
		y_cos_t = gamma*omega0^2* cos(ot)
		y_sin_t = gamma*omega0^2* sin(ot)
		dy_cos_t = dgamma*omega0^2* cos(ot)
		dy_sin_t = dgamma*omega0^2* sin(ot)
		cos_x = omega0^2* cos(x)
		sin_x = omega0^2* sin(x)

		tmp_1 = bv + y_cos_t * cos(odt)
		tmp_2 = - y_sin_t * sin(odt) + dy_cos_t * cos(odt)
		tmp_3 = - dy_sin_t * sin(odt)

		v'	=	(h^1)			tmp_1 - sin_x * cos(dx) * cos(ddx)
				(h^2)			tmp_2 - cos_x * sin(dx) * cos(ddx) - 2*beta * dv
				(h^3)			tmp_3 - cos_x * cos(dx) * sin(ddx) + sin_x * sin(dx) * sin(ddx) - 2*beta * ddv


		x'	=	(h^1)			+ v
				(h^2)			+ dv
				(h^2)			+ ddv
	**********************************************************************/

	const float T = fmodf(t_step * t_index, 2 * pi / omega);
	const float betav = -2.f * beta * v;
	const float beta2 = -2.f * beta;
	const float cos_x = omega0 * omega0 * cosf(x);
	const float sin_x = omega0 * omega0 * sinf(x);

	const float hv = h * (v);
	const float hhv = .5f * hv;

	float dx1 = 0, dv1 = 0;
	for (unsigned long i = 0; i < h_level_3; ++i) {
		float dx2 = 0, dv2 = 0;
		for (unsigned long j = 0; j < h_level_3; ++j) {
			float dx3 = 0, dv3 = 0;

			const float ot = omega * (T + (h * h_level_3 * h_level_3) * i + (h * h_level_3) * j);
			const float y_cos_t = y * (omega0 * omega0) * cosf(ot);
			const float y_sin_t = y * (omega0 * omega0) * sinf(ot);
			const float dy_cos_t = dy * (omega0 * omega0) * cosf(ot);
			const float dy_sin_t = dy * (omega0 * omega0) * sinf(ot);

			for (unsigned long k = 0; k < h_level_3; ++k) {
				float			dx2_ = 0, dx3_ = 0;
				float dv1_ = 0, dv2_ = 0, dv3_ = 0;
				float dx, ddx, dv, ddv, odt;
				float		   tmp_dx2, tmp_dx3;
				float tmp_dv1, tmp_dv2, tmp_dv3;
				float tmp_1, tmp_2, tmp_3;

				//K1
				odt = (omega * h) * k;
				tmp_1 = betav + y_cos_t * __cosf(odt);
				tmp_2 = dy_cos_t * __cosf(odt) - y_sin_t * __sinf(odt);
				tmp_3 = -dy_sin_t * __sinf(odt);

				dv = dv1;	ddv = dv2 + dv3;
				dx = dx1;	ddx = dx2 + dx3;
				dv1_ += h * (tmp_1 - sin_x * __cosf(dx) * __cosf(ddx));
				dv2_ += h * (tmp_2 - cos_x * __sinf(dx) * __cosf(ddx) - beta2 * dv);
				dv3_ += h * (tmp_3 - cos_x * __cosf(dx) * __sinf(ddx) + sin_x * __sinf(dx) * __sinf(ddx) - beta2 * ddv);
				dx2_ += h * dv;
				dx3_ += h * ddv;

				//t of K2 & K3
				odt = (omega * h) * (k + .5f);
				tmp_1 = betav + y_cos_t * __cosf(odt);
				tmp_2 = dy_cos_t * __cosf(odt) - y_sin_t * __sinf(odt);
				tmp_3 = -dy_sin_t * __sinf(odt);

				//K2
				dv = dv1 + .5f * dv1_;	ddv = (dv2 + .5f * dv2_) + (dv3 + .5f * dv3_);
				dx = dx1 + hhv;	ddx = (dx2 + .5f * dx2_) + (dx3 + .5f * dx3_);
				dv1_ += 2.f * (tmp_dv1 = h * (tmp_1 - sin_x * __cosf(dx) * __cosf(ddx)));
				dv2_ += 2.f * (tmp_dv2 = h * (tmp_2 - cos_x * __sinf(dx) * __cosf(ddx) - beta2 * dv));
				dv3_ += 2.f * (tmp_dv3 = h * (tmp_3 - cos_x * __cosf(dx) * __sinf(ddx) + sin_x * __sinf(dx) * __sinf(ddx) - beta2 * ddv));
				dx2_ += 2.f * (tmp_dx2 = h * dv);
				dx3_ += 2.f * (tmp_dx3 = h * ddv);

				//K3
				dv = dv1 + .5f * tmp_dv1;	ddv = (dv2 + .5f * tmp_dv2) + (dv3 + .5f * tmp_dv3);
				dx = dx1 + hhv;	ddx = (dx2 + .5f * tmp_dx2) + (dx3 + .5f * tmp_dx3);
				dv1_ += 2.f * (tmp_dv1 = h * (tmp_1 - sin_x * __cosf(dx) * __cosf(ddx)));
				dv2_ += 2.f * (tmp_dv2 = h * (tmp_2 - cos_x * __sinf(dx) * __cosf(ddx) - beta2 * dv));
				dv3_ += 2.f * (tmp_dv3 = h * (tmp_3 - cos_x * __cosf(dx) * __sinf(ddx) + sin_x * __sinf(dx) * __sinf(ddx) - beta2 * ddv));
				dx2_ += 2.f * (tmp_dx2 = h * dv);
				dx3_ += 2.f * (tmp_dx3 = h * ddv);

				//K4
				odt = (omega * h) * (k + 1);
				tmp_1 = betav + y_cos_t * __cosf(odt);
				tmp_2 = dy_cos_t * __cosf(odt) - y_sin_t * __sinf(odt);
				tmp_3 = -dy_sin_t * __sinf(odt);

				dv = dv1 + tmp_dv1;	ddv = (dv2 + tmp_dv2) + (dv3 + tmp_dv3);
				dx = dx1 + hv;	ddx = (dx2 + tmp_dx2) + (dx3 + tmp_dx3);
				dv1_ += h * (tmp_1 - sin_x * __cosf(dx) * __cosf(ddx));
				dv2_ += h * (tmp_2 - cos_x * __sinf(dx) * __cosf(ddx) - beta2 * dv);
				dv3_ += h * (tmp_3 - cos_x * __cosf(dx) * __sinf(ddx) + sin_x * __sinf(dx) * __sinf(ddx) - beta2 * ddv);
				dx2_ += h * dv;
				dx3_ += h * ddv;

				//sum
				dv1 += dv1_ / 6.f;
				dv2 += dv2_ / 6.f;
				dv3 += dv3_ / 6.f;
				dx1 += hv;
				dx2 += dx2_ / 6.f;
				dx3 += dx3_ / 6.f;
			}
			dx2 += dx3;
			dv2 += dv3;
		}
		dx1 += dx2;
		dv1 += dv2;
	}
	(*outx) = x + dx1;
	(*outv) = v + dv1;

}
__device__ void RK4(unsigned long t_index, float x, float v, float *outx, float *outv, float y, float dy) {
	/**********************************************************************
		(v' , x') = f((v+dv+ddv , x+dx+ddx) , t+dt): gamma = gamma + dgamma
		ot = omega * t;
		odt = omega * dt;

		bv = -2*beta* v
		y_cos_t = gamma*omega0^2* cos(ot)
		y_sin_t = gamma*omega0^2* sin(ot)
		dy_cos_t = dgamma*omega0^2* cos(ot)
		dy_sin_t = dgamma*omega0^2* sin(ot)
		cos_x = omega0^2* cos(x)
		sin_x = omega0^2* sin(x)

		tmp_1 = bv + y_cos_t * cos(odt)
		tmp_2 = - y_sin_t * sin(odt) + dy_cos_t * cos(odt)
		tmp_3 = - dy_sin_t * sin(odt)

		v'	=	(h^1)			tmp_1 - sin_x * cos(dx) * cos(ddx)
				(h^2)			tmp_2 - cos_x * sin(dx) * cos(ddx) - 2*beta * dv
				(h^3)			tmp_3 - cos_x * cos(dx) * sin(ddx) + sin_x * sin(dx) * sin(ddx) - 2*beta * ddv


		x'	=	(h^1)			+ v
				(h^2)			+ dv
				(h^2)			+ ddv
	**********************************************************************/

	const float T = fmodf(t_step * t_index, 2 * pi / omega);
	const float betav = -2.f * beta * v;
	const float beta2 = -2.f * beta;
	const float cos_x = omega0 * omega0 * cosf(x);
	const float sin_x = omega0 * omega0 * sinf(x);

	const float hv = h * (v);
	const float hhv = .5f * hv;

	float dx1 = 0, dv1 = 0;
	for (unsigned long i = 0; i < h_level_3; ++i) {
		float dx2 = 0, dv2 = 0;
		for (unsigned long j = 0; j < h_level_3; ++j) {
			float dx3 = 0, dv3 = 0;

			const float ot = omega * (T + (h * h_level_3 * h_level_3) * i + (h * h_level_3) * j);
			const float y_cos_t = y * (omega0 * omega0) * cosf(ot);
			const float y_sin_t = y * (omega0 * omega0) * sinf(ot);
			const float dy_cos_t = dy * (omega0 * omega0) * cosf(ot);
			const float dy_sin_t = dy * (omega0 * omega0) * sinf(ot);

			for (unsigned long k = 0; k < h_level_3; ++k) {
				float			dx2_ = 0, dx3_ = 0;
				float dv1_ = 0, dv2_ = 0, dv3_ = 0;
				float dx, ddx, dv, ddv, odt;
				float		   tmp_dx2, tmp_dx3;
				float tmp_dv1, tmp_dv2, tmp_dv3;
				float tmp_1, tmp_2, tmp_3;

				//K1
				odt = (omega * h) * k;
				tmp_1 = betav + y_cos_t * cosf(odt);
				tmp_2 = dy_cos_t * cosf(odt) - y_sin_t * sinf(odt);
				tmp_3 = -dy_sin_t * sinf(odt);

				dv = dv1;	ddv = dv2 + dv3;
				dx = dx1;	ddx = dx2 + dx3;
				dv1_ += h * (tmp_1 - sin_x * cosf(dx) * cosf(ddx));
				dv2_ += h * (tmp_2 - cos_x * sinf(dx) * cosf(ddx) - beta2 * dv);
				dv3_ += h * (tmp_3 - cos_x * cosf(dx) * sinf(ddx) + sin_x * sinf(dx) * sinf(ddx) - beta2 * ddv);
				dx2_ += h * dv;
				dx3_ += h * ddv;

				//t of K2 & K3
				odt = (omega * h) * (k + .5f);
				tmp_1 = betav + y_cos_t * cosf(odt);
				tmp_2 = dy_cos_t * cosf(odt) - y_sin_t * sinf(odt);
				tmp_3 = -dy_sin_t * sinf(odt);

				//K2
				dv = dv1 + .5f * dv1_;	ddv = (dv2 + .5f * dv2_) + (dv3 + .5f * dv3_);
				dx = dx1 + hhv;	ddx = (dx2 + .5f * dx2_) + (dx3 + .5f * dx3_);
				dv1_ += 2.f * (tmp_dv1 = h * (tmp_1 - sin_x * cosf(dx) * cosf(ddx)));
				dv2_ += 2.f * (tmp_dv2 = h * (tmp_2 - cos_x * sinf(dx) * cosf(ddx) - beta2 * dv));
				dv3_ += 2.f * (tmp_dv3 = h * (tmp_3 - cos_x * cosf(dx) * sinf(ddx) + sin_x * sinf(dx) * sinf(ddx) - beta2 * ddv));
				dx2_ += 2.f * (tmp_dx2 = h * dv);
				dx3_ += 2.f * (tmp_dx3 = h * ddv);

				//K3
				dv = dv1 + .5f * tmp_dv1;	ddv = (dv2 + .5f * tmp_dv2) + (dv3 + .5f * tmp_dv3);
				dx = dx1 + hhv;	ddx = (dx2 + .5f * tmp_dx2) + (dx3 + .5f * tmp_dx3);
				dv1_ += 2.f * (tmp_dv1 = h * (tmp_1 - sin_x * cosf(dx) * cosf(ddx)));
				dv2_ += 2.f * (tmp_dv2 = h * (tmp_2 - cos_x * sinf(dx) * cosf(ddx) - beta2 * dv));
				dv3_ += 2.f * (tmp_dv3 = h * (tmp_3 - cos_x * cosf(dx) * sinf(ddx) + sin_x * sinf(dx) * sinf(ddx) - beta2 * ddv));
				dx2_ += 2.f * (tmp_dx2 = h * dv);
				dx3_ += 2.f * (tmp_dx3 = h * ddv);

				//K4
				odt = (omega * h) * (k + 1);
				tmp_1 = betav + y_cos_t * cosf(odt);
				tmp_2 = dy_cos_t * cosf(odt) - y_sin_t * sinf(odt);
				tmp_3 = -dy_sin_t * sinf(odt);

				dv = dv1 + tmp_dv1;	ddv = (dv2 + tmp_dv2) + (dv3 + tmp_dv3);
				dx = dx1 + hv;	ddx = (dx2 + tmp_dx2) + (dx3 + tmp_dx3);
				dv1_ += h * (tmp_1 - sin_x * cosf(dx) * cosf(ddx));
				dv2_ += h * (tmp_2 - cos_x * sinf(dx) * cosf(ddx) - beta2 * dv);
				dv3_ += h * (tmp_3 - cos_x * cosf(dx) * sinf(ddx) + sin_x * sinf(dx) * sinf(ddx) - beta2 * ddv);
				dx2_ += h * dv;
				dx3_ += h * ddv;

				//sum
				dv1 += dv1_ / 6.f;
				dv2 += dv2_ / 6.f;
				dv3 += dv3_ / 6.f;
				dx1 += hv;
				dx2 += dx2_ / 6.f;
				dx3 += dx3_ / 6.f;
			}
			dx2 += dx3;
			dv2 += dv3;
		}
		dx1 += dx2;
		dv1 += dv2;
	}
	(*outx) = x + dx1;
	(*outv) = v + dv1;

}

void RK4_CPU(unsigned t_index, float y, float dy, float& x, float& v) {
	const float T = fmod(t_step * t_index, 2 * pi / omega);
	const float betav = -2.f * beta * v;
	const float beta2 = -2.f * beta;
	const float cos_x = omega0 * omega0 * cosf(x);
	const float sin_x = omega0 * omega0 * sinf(x);

	const float hv = h * v;
	const float hhv = .5f * hv;

	float dx1 = 0, dv1 = 0;
	for (unsigned long i = 0; i < h_level_3; ++i) {
		float dx2 = 0, dv2 = 0;
		for (unsigned long j = 0; j < h_level_3; ++j) {
			float dx3 = 0, dv3 = 0;

			const float ot = omega * (T + (h * h_level_3 * h_level_3) * i + (h * h_level_3) * j);
			const float y_cos_t = y * (omega0 * omega0) * cosf(ot);
			const float y_sin_t = y * (omega0 * omega0) * sinf(ot);
			const float dy_cos_t = dy * (omega0 * omega0) * cosf(ot);
			const float dy_sin_t = dy * (omega0 * omega0) * sinf(ot);

			for (unsigned long k = 0; k < h_level_3; ++k) {
				float			dx2_ = 0, dx3_ = 0;
				float dv1_ = 0, dv2_ = 0, dv3_ = 0;
				float dx, ddx, dv, ddv, odt;
				float		   tmp_dx2, tmp_dx3;
				float tmp_dv1, tmp_dv2, tmp_dv3;
				float tmp_1, tmp_2, tmp_3;

				//K1
				odt = (omega * h) * k;
				tmp_1 = betav + y_cos_t * cosf(odt);
				tmp_2 = dy_cos_t * cosf(odt) - y_sin_t * sinf(odt);
				tmp_3 = -dy_sin_t * sinf(odt);

				dv = dv1;	ddv = dv2 + dv3;
				dx = dx1;	ddx = dx2 + dx3;
				dv1_ += h * (tmp_1 - sin_x * cosf(dx) * cosf(ddx));
				dv2_ += h * (tmp_2 - cos_x * sinf(dx) * cosf(ddx) - beta2 * dv);
				dv3_ += h * (tmp_3 - cos_x * cosf(dx) * sinf(ddx) + sin_x * sinf(dx) * sinf(ddx) - beta2 * ddv);
				dx2_ += h * dv;
				dx3_ += h * ddv;

				//t of K2 & K3
				odt = (omega * h) * (k + .5f);
				tmp_1 = betav + y_cos_t * cosf(odt);
				tmp_2 = dy_cos_t * cosf(odt) - y_sin_t * sinf(odt);
				tmp_3 = -dy_sin_t * sinf(odt);

				//K2
				dv = dv1 + .5f * dv1_;	ddv = (dv2 + .5f * dv2_) + (dv3 + .5f * dv3_);
				dx = dx1 + hhv;	ddx = (dx2 + .5f * dx2_) + (dx3 + .5f * dx3_);
				dv1_ += 2.f * (tmp_dv1 = h * (tmp_1 - sin_x * cosf(dx) * cosf(ddx)));
				dv2_ += 2.f * (tmp_dv2 = h * (tmp_2 - cos_x * sinf(dx) * cosf(ddx) - beta2 * dv));
				dv3_ += 2.f * (tmp_dv3 = h * (tmp_3 - cos_x * cosf(dx) * sinf(ddx) + sin_x * sinf(dx) * sinf(ddx) - beta2 * ddv));
				dx2_ += 2.f * (tmp_dx2 = h * dv);
				dx3_ += 2.f * (tmp_dx3 = h * ddv);

				//K3
				dv = dv1 + .5f * tmp_dv1;	ddv = (dv2 + .5f * tmp_dv2) + (dv3 + .5f * tmp_dv3);
				dx = dx1 + hhv;	ddx = (dx2 + .5f * tmp_dx2) + (dx3 + .5f * tmp_dx3);
				dv1_ += 2.f * (tmp_dv1 = h * (tmp_1 - sin_x * cosf(dx) * cosf(ddx)));
				dv2_ += 2.f * (tmp_dv2 = h * (tmp_2 - cos_x * sinf(dx) * cosf(ddx) - beta2 * dv));
				dv3_ += 2.f * (tmp_dv3 = h * (tmp_3 - cos_x * cosf(dx) * sinf(ddx) + sin_x * sinf(dx) * sinf(ddx) - beta2 * ddv));
				dx2_ += 2.f * (tmp_dx2 = h * dv);
				dx3_ += 2.f * (tmp_dx3 = h * ddv);

				//K4
				odt = (omega * h) * (k + 1);
				tmp_1 = betav + y_cos_t * cosf(odt);
				tmp_2 = dy_cos_t * cosf(odt) - y_sin_t * sinf(odt);
				tmp_3 = -dy_sin_t * sinf(odt);

				dv = dv1 + tmp_dv1;	ddv = (dv2 + tmp_dv2) + (dv3 + tmp_dv3);
				dx = dx1 + hv;	ddx = (dx2 + tmp_dx2) + (dx3 + tmp_dx3);
				dv1_ += h * (tmp_1 - sin_x * cosf(dx) * cosf(ddx));
				dv2_ += h * (tmp_2 - cos_x * sinf(dx) * cosf(ddx) - beta2 * dv);
				dv3_ += h * (tmp_3 - cos_x * cosf(dx) * sinf(ddx) + sin_x * sinf(dx) * sinf(ddx) - beta2 * ddv);
				dx2_ += h * dv;
				dx3_ += h * ddv;

				//sum
				dv1 += dv1_ / 6.f;
				dv2 += dv2_ / 6.f;
				dv3 += dv3_ / 6.f;
				dx1 += hv;
				dx2 += dx2_ / 6.f;
				dx3 += dx3_ / 6.f;
			}
			dx2 += dx3;
			dv2 += dv3;
		}
		dx1 += dx2;
		dv1 += dv2;
	}
	x += dx1;
	v += dv1;
}