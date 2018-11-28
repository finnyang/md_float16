#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_fp16.h"
#include <stdio.h>
#include <iostream>
using namespace std;
 
#define CHECK(call) \
{ \
	const cudaError_t error = call; \
	if (error != cudaSuccess) \
	{ \
		printf("Error: %s: %d, ", __FILE__, __LINE__); \
		printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
		system("pause"); \
	} \
}
 
__global__ void myHalf2Add(half2 *a, half2 *b, half2 *c, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	c[i] = __hadd2(a[i], b[i]);
}
__global__ void float22Half2Vec(float2 * src, half2 *des, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	des[i] = __float22half2_rn(src[i]);
 
}
 
__global__ void half22Float2Vec(half2 *src, float2 *des, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	des[i] = __half22float2(src[i]);
	
}
 
int main()
{
	const int blocks = 128;
	const int threads = 128;
	size_t size = blocks*threads * 2;
	float *vec1 = new float[size];
	float *vec2 = new float[size];
	float *res = new float[size];
	for (size_t i = 0; i < size; i++)
	{
		vec2[i] = vec1[i] = i;
	}
	float * vecDev1, *vecDev2, *resDev;
	CHECK(cudaMalloc((void **)&vecDev1, size * sizeof(float)));
	CHECK(cudaMalloc((void **)&vecDev2, size * sizeof(float)));
	CHECK(cudaMalloc((void **)&resDev, size * sizeof(float)));
	CHECK(cudaMemcpy(vecDev1, vec1, size * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(vecDev2, vec2, size * sizeof(float), cudaMemcpyHostToDevice));

	half2 *vecHalf2Dev1, *vecHalf2Dev2, *resHalf2Dev;
	CHECK(cudaMalloc((void **)&vecHalf2Dev1, size * sizeof(float) / 2));
	CHECK(cudaMalloc((void **)&vecHalf2Dev2, size * sizeof(float) / 2));
	CHECK(cudaMalloc((void **)&resHalf2Dev, size * sizeof(float) / 2));


	float22Half2Vec <<<128, 128 >>> ((float2*)vecDev1, vecHalf2Dev1, size);
	float22Half2Vec <<<128, 128 >>> ((float2*)vecDev2, vecHalf2Dev2, size);
	myHalf2Add <<<128, 128 >>> (vecHalf2Dev1, vecHalf2Dev2, resHalf2Dev, size);
	half22Float2Vec <<<128, 128 >>>(resHalf2Dev, (float2*)resDev, size);

	//half22Float2Vec << <128, 128 >> >(vecHalf2Dev1, (float2*)resDev, size);
	//CHECK(cudaMemcpy(res, resDev, size * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(res, resDev, size * sizeof(float), cudaMemcpyDeviceToHost));

	for (int i = 0; i < 128; i++)//打印出前64个结果，并与CPU结果对比
	{
		cout << vec1[i] << " + " << vec2[i] << " = " << vec1[i] + vec2[i] << "  ?  " << res[i] << endl;
	}
	for (int i = 128 * 128; i < 128 * 128 + 128; i++)//打印出前64个结果，并与CPU结果对比
	{
		cout << vec1[i] << " + " << vec2[i] << " = " << vec1[i] + vec2[i] << "  ?  " << res[i] << endl;
	}
	delete[] vec1;
	delete[] vec2;
	delete[] res;
	CHECK(cudaFree(vecDev1));
	CHECK(cudaFree(vecDev2));
	CHECK(cudaFree(resDev));
	CHECK(cudaFree(vecHalf2Dev1));
	CHECK(cudaFree(vecHalf2Dev2));
	CHECK(cudaFree(resHalf2Dev));
	system("pause");
	return 0;
}
