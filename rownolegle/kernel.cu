
#include "cuda_runtime.h"
#include <nvtx3/nvToolsExt.h>
#include "device_launch_parameters.h"

#include <chrono>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <vector>


const int N = 2000;
const int R = 2;
const int OUTSIZE = N - 2 * R;

float sumLocalWithCuda(float *tab, float *out, int block_size, int k, char * name);

__global__ void localKernel(float* tab, float* out, int* kkk)
{
    int i = (threadIdx.x + blockIdx.x * blockDim.x) * *kkk;
    int j = (threadIdx.y + blockIdx.y * blockDim.y);

    for (int k = 0; k < *kkk; k++) {
        int ik = i + k;
        if (ik < OUTSIZE) {
			float sum = 0;
            for (int y = 0; y <= 2*R; y++) {
                int jy = (j + y)*N;
				for (int x = 0; x <= 2*R; x++) {
					sum += tab[jy + (ik + x)];
				}
            }
            out[(j) * (OUTSIZE) + ik] = sum;
        }
    }
}

void sequential(float tab[N*N], float out[(N-2*R)*(N-2*R)])
{
	for (int i = R; i < N - R; i++) {
		for (int j = R; j < N - R; j++) {
			float sum = 0;
			for (int x = i - R; x <= i + R; x++) {
				for (int y = j - R; y <= j + R; y++) {
					sum += tab[x * N + y];
				}
			}
			out[(i - R) * (N - 2 * R) + j - R] = sum;
		}
	}
}

void print(float tab[N*N])
{
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            printf("%f\t", tab[j * N + i]);
        }
        printf("\n");
    }
}

void print_out(float tab[(N-2*R)*(N-2*R)])
{
    for (int j = 0; j < N-2*R; j++) {
        for (int i = 0; i < N-2*R; i++) {
            printf("%f\t", tab[j * (N - 2 * R) + i]);
        }
        printf("\n");
    }
}

int main()
{
    srand(time(NULL));

    float* tab = (float*)malloc(N * N * sizeof(float));


    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            tab[j * N + i] = rand() % 10;
        }
    }

    float* out_seq = (float*)malloc(OUTSIZE * OUTSIZE * sizeof(float));

    auto start = std::chrono::high_resolution_clock::now();
    sequential(tab, out_seq);
    auto end = std::chrono::high_resolution_clock::now();
    auto timeSeq = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    int field = 2 * R + 1;
    double fps = OUTSIZE * OUTSIZE / static_cast<double>(timeSeq.count() / 1000.f) * field * field;


    printf("-,-,%.4f,%.4f\n", timeSeq.count() / 1000.0f, fps / 1e9);

    float* out_local = (float*)malloc(OUTSIZE * OUTSIZE * sizeof(float));
    start = std::chrono::high_resolution_clock::now();
    sumLocalWithCuda(tab, out_local, 1, 8, "warmup");
    end = std::chrono::high_resolution_clock::now();
    timeSeq = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::vector<int> ks = { 1, 4, 16 };
    std::vector<int> bs = { 8, 16, 32 };

    long long memory_traffic_elem = field * field * sizeof(float) + sizeof(float);
    long long full_memory_traffic = (OUTSIZE) * (OUTSIZE) * memory_traffic_elem;
    for (auto k : ks) {
        for (auto b : bs) {
			cudaDeviceReset();
            char text[30] = "";
            sprintf(text, "b: %d, k: %d", b, k);
			start = std::chrono::high_resolution_clock::now();
			auto ftime = sumLocalWithCuda(tab, out_local, b, k, text);
			end = std::chrono::high_resolution_clock::now();
			timeSeq = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            double fps = OUTSIZE * OUTSIZE / static_cast<double>(ftime / 1000.f) * field * field ;
            double fpb = 0 / static_cast<double>(full_memory_traffic);

			printf("%d,%d,%.4f,%.4f\n", b, k, timeSeq.count() / 1000.0f, fps / 1e9);
            
            for (int i = 0; i < OUTSIZE * OUTSIZE; i++) {
                if (out_local[i] != out_seq[i]) {
                    printf("dupa blada\n");
                    break;
                }
            }
        }
    }
}

float sumLocalWithCuda(float* tab, float* out, int block_size, int k, char* name) 
{
    float* dev_tab = 0;
    float* dev_out = 0;
    int* dev_k = 0;
    cudaError_t cudaStatus;

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_k, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_k, &k, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_tab, N * N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_out, (OUTSIZE) * (OUTSIZE) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_tab, tab, N*N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    dim3 threadsMatrix(block_size, block_size);
    dim3 blocksMatrix(ceil((OUTSIZE) / (float)block_size / k), ceil((OUTSIZE) / (float)block_size));

    cudaEventRecord(start, nullptr);

    nvtxRangePush(name);
    localKernel<<< blocksMatrix, threadsMatrix >>>(dev_tab, dev_out, dev_k);
    nvtxRangePop();

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "local launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaEventRecord(stop, nullptr);


    cudaStatus = cudaMemcpy(out, dev_out, (OUTSIZE) * (OUTSIZE) * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaEventElapsedTime(&time, start, stop);

    Error:
	cudaFree(dev_tab);
	cudaFree(dev_out);
    
    return time;
}
