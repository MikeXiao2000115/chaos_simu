#pragma once
#include <complex>
bool __declspec (dllexport) FFT(float* input, std::complex<float>* output, const unsigned long size, const unsigned long batch);