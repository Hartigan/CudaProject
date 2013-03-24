
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Matrix.h"

#include <stdio.h>

cudaError_t MulMatrix(Matrix* a, Matrix* b, Matrix* &c);

__global__ void mulKernel(float *c, const float *a, const float *b, int length)
{
	int i = blockIdx.x;
	int j = blockIdx.y;

	float tmp = 0;

	for(int l=0;l<length;l++)
	{
		tmp += a[i*length+l]*b[l*length+j];
	}

	c[512*i+j] = tmp;
}

int main()
{
    const int arraySize = 512;
    Matrix* matrixA = new Matrix(arraySize,arraySize);
	Matrix* matrixB = new Matrix(arraySize,arraySize);
	Matrix* matrixC;

	for (int i=0;i<arraySize;i++)
	{
		for (int j=0;j<arraySize;j++)
		{
			matrixA->setValue(i,j,1.1);
			matrixB->setValue(i,j,1.1);
		}
	}

    // Add vectors in parallel.
    cudaError_t cudaStatus = MulMatrix(matrixA,matrixB,matrixC);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }


	delete matrixA;
	delete matrixB;
	delete matrixC;
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t MulMatrix(Matrix* a, Matrix* b, Matrix* &c)
{
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, a->getDimY() * b->getDimX() * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_a, a->getDimX() * a->getDimY() * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_b, b->getDimX() * b->getDimY() * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a->getArrayPointer(), a->getDimX() * a->getDimY() * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_b, b->getArrayPointer(), b->getDimX() * b->getDimY() * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	dim3 gridSize(b->getDimX(),a->getDimY(),1);

    // Launch a kernel on the GPU with one thread for each element.
    mulKernel<<<gridSize, 1>>>(dev_c, dev_a, dev_b, a->getDimX());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
	c = new Matrix(b->getDimX(),a->getDimY());
	cudaStatus = cudaMemcpy(c->getArrayPointer(), dev_c, a->getDimY() * b->getDimX() * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
