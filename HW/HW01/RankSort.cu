%%cu

//-------------------------------------------------------------------------------------------
/**
    Purpose:    Parallel Ranksort Algorithm
    Author:     Iman Ghadimi
    Date:       March 2022
*/

//-------------------------------------------------------------------------------------------

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <cuda_profiler_api.h>
#include <sys/time.h>

// Iterations for a given parallel execution size
const int REP_PAR = 32;
// Lenght of the random sequence numbers
const int LENGHT = 2;
// End size
const int N_MAX = 4194304;
// Initial size
const int N = 1024;
// Max threads per block
const int THREAD_MAX = 1024;
const int BLOCK_MAX = 65535;

// GPU Parallel algorithm

/**
 * [rankSort description] GPU function. Parallel ranksort algorithm function.
 *
 * @param array  : Array to be sorted.
 * @param result : Result array: it's the sorted final array.
 * @param k      : Size of the array.
 */

__global__ void rankSort(int * array, int * result, int k) {
	// How many numbers have a block to RANK.
	int a = k / gridDim.x;
	// How many numbers have a thread to COMPARE with the RANK number.
	int b = k / blockDim.x;
	__shared__ int tamBlocks; // Extra (rest) number(s) that lower blocks will have to RANK after.
	__shared__ int tamThreads; // Extra (rest) number(s) that lower threads will have to COMPARE with the actual RANK.
	__shared__ int miNumero; // Number to RANK.
	__shared__ int rank; // Rank (index) accumulation to sort an array position.
	int localRank; // Local thread Rank (index) accumulation (will be sum to the global one at the end of the thread comparissions)
	int comparador; // Number to compare with the RANK (miNumero).
	int range2 = threadIdx.x * b; // Second loop range distribution (comparissions)

	if(threadIdx.x == 0) {
		// Rest of the numbers (indexes) that don't fit with the block distribution to be RANK after with lower blocks IDS
		tamBlocks = k - (a * gridDim.x);
		tamThreads = k - (b * blockDim.x);
	}

	int range1 = blockIdx.x * a; // First loop range distribution (numbers to RANK)
	for(int i = range1; i < range1 + a; i++) {
		if(threadIdx.x == 0) {
			miNumero = array[i]; // We get the RANK number so we will let threads to make their comparissions
			rank = 0; // Initial shared rank
		}
		__syncthreads();

		localRank = 0; // Initial thead local rank
		for(int j = range2; j < range2 + b; j++) {
			comparador = array[j];
			if(comparador < miNumero || (comparador == miNumero && (j < i)))
				localRank += 1; // Local rank accumulation
		}

		// Let the lower threads ID's compute the 'rest' of the comparissions //
		if(threadIdx.x < tamThreads) {
			comparador = array[(blockDim.x * b) + threadIdx.x];
			if(comparador < miNumero || (comparador == miNumero && (((blockDim.x * b) + threadIdx.x) < i)))
				localRank += 1; // Local rank accumulation
		}

		atomicAdd(&rank, localRank); // Atomic shared rank accumulation

		__syncthreads();

		if(threadIdx.x == 0) {
			result[rank] = miNumero; // Placing the number in its sorted position
		}

		__syncthreads();
	}

	// Let the lower blocks ID's compute the 'rest' of the RANKS //
	if(blockIdx.x < tamBlocks) {
		if(threadIdx.x == 0) {
			miNumero = array[gridDim.x * a  + blockIdx.x];
			rank = 0;
		}

		__syncthreads();

		localRank = 0;
		for(int j = range2; j < range2 + b; j++) {
			comparador = array[j];
			if(comparador < miNumero || (comparador == miNumero && (j < (gridDim.x * a + blockIdx.x))))
				localRank += 1; // Local rank accumulation
		}

		// Let the lower threads ID's compute the 'rest' of the comparissions //
		if(threadIdx.x < tamThreads) {
			comparador = array[(blockDim.x * b) + threadIdx.x];
			if(comparador < miNumero || (comparador == miNumero && (((blockDim.x * b) + threadIdx.x) < gridDim.x * a  + blockIdx.x)))
				localRank += 1; // Local rank accumulation
		}

		atomicAdd(&rank, localRank); // Atomic shared rank accumulation

		__syncthreads();

		if(threadIdx.x == 0) {
			result[rank] = miNumero; // Placing the number in its sorted position
		}
	}
}

// CPU :

// Main function :

int main( int argc, char* argv[] )
{

    // Time structures variables for sequentiall and parallel calculation
	struct timeval t1, t2, t1_seq, t2_seq;
    double elapsedTime = 0, elapsedTimeSec = 0;

    for(int k = N; k < N_MAX; k*=2) {

    	// Total number of blocks in the GPU
	    int numBlocks = k;

	    // Threads per each block in the GPU
	    int threadsPerBlock = k;

	    if(threadsPerBlock > THREAD_MAX)
	    	threadsPerBlock = THREAD_MAX;

		if(numBlocks > BLOCK_MAX)
			numBlocks = BLOCK_MAX;

		// Elements data arrays (CPU and (h_) GPU (d_))
			k = 100;
	    int h_array[k];
	    int *d_array;
	    int *d_result;

	    // Random array initialization

			int i ;
   		for(i=0;i<k;i++)
				h_array[i]=rand()%100;   //Generate number between 0 to 99

			printf("Array Before RankSort:\t");
			for(i=0; i<k; i++)
				{
					printf("\nElement number %d::%d",i+1,h_array[i]);
				}

	    // Array memory allocation
	    cudaMalloc((void**)&d_array, k * sizeof(int));
	    cudaMalloc((void**)&d_result, k * sizeof(int));

        // PARALELL Part :

	    for(int m = 0; m < REP_PAR; m++) {
	    	cudaMemcpy(d_array, &h_array, k * sizeof(int), cudaMemcpyHostToDevice);
	    	rankSort<<<numBlocks, threadsPerBlock>>>(d_array, d_result, k);
		    cudaThreadSynchronize(); // Synchronization
	    }

			printf("Array After RankSort:\t");
			for(i=0; i<k; i++)
				{
					printf("\nElement number %d::%d",i+1,d_result[i]);
				}

 		// Free GPU memory
	    cudaFree(d_array);
	    cudaFree(d_result);
	}

	cudaDeviceSynchronize();

    return 0;
}
