#include "FFT.h"
#include "cufft.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define C(X) if((X)!=cudaSuccess) return false

bool FFT(float* input, std::complex<float>* output, const unsigned long size, const unsigned long batch) {
	cufftReal *in, *in_h;
	cufftComplex *out, *out_h;

	if (cudaMalloc((void**)&in, size * batch * sizeof(cufftReal)) != cudaSuccess)
		return false;

	if (cudaMallocHost((void**)&in_h, size * batch * sizeof(cufftReal)) != cudaSuccess) {
		cudaFree(in);
		return false;
	}

	if (cudaMalloc((void**)&out, (size / 2 + 1) * batch * sizeof(cufftComplex)) != cudaSuccess) {
		cudaFree(in);
		cudaFreeHost(in_h);
		return false;
	}
	if (cudaMallocHost((void**)&out_h, (size / 2 + 1) * batch * sizeof(cufftComplex)) != cudaSuccess) {
		cudaFree(in);
		cudaFreeHost(in_h);
		cudaFree(out);
		return false;
	}

	auto p = in_h;

	for (unsigned long i = 0; i < size* batch; ++i)
		*p++ = (cufftReal)*input++;



	if (cudaMemcpy(in, in_h, size * batch * sizeof(cufftReal), cudaMemcpyHostToDevice) != cudaSuccess) {
		cudaFree(in);
		cudaFreeHost(in_h);
		cudaFree(out);
		cudaFreeHost(out_h);
		return false;
	}

	cufftHandle plan;
	//cufftPlan1d(&plan, size, CUFFT_R2C, batch);
	int n = size;
	cufftPlanMany(&plan, 1, &n,
		NULL, 1, size,
		NULL, 1, (size / 2 + 1), CUFFT_R2C, batch);
	cufftExecR2C(plan, in, out);
	cudaDeviceSynchronize();
	
	if (cudaMemcpy(out_h, out, (size / 2 + 1) * batch * sizeof(cufftComplex), cudaMemcpyDeviceToHost) != cudaSuccess) {
		cudaFree(in);
		cudaFreeHost(in_h);
		cudaFree(out);
		cudaFreeHost(out_h);
		return false;
	}

	cufftDestroy(plan);
	
	for (unsigned long i = 0; i < (size / 2 + 1) * batch; ++i)
		output[i] = std::complex<float>(out_h[i].x, out_h[i].y);

	cudaFree(in);
	cudaFreeHost(in_h);
	cudaFree(out);
	cudaFreeHost(out_h);
	return true;
}
