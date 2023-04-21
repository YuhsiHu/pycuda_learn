#include <cuda_runtime.h>
#include <stdio.h>

/**
编写一个程序来了解分支处理机制
nvcc divergence_test.cu -o divergence_test
*/
__global__ void divergence_test_ker()
{
	if( threadIdx.x % 2 == 0)
		printf("threadIdx.x %d : This is an even thread.\n", threadIdx.x);
	else
		printf("threadIdx.x %d : This is an odd thread.\n", threadIdx.x);
}

__host__ int main()
{
	cudaSetDevice(0);
	divergence_test_ker<<<1, 32>>>();
	cudaDeviceSynchronize();
	cudaDeviceReset();
}