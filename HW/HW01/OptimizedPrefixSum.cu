
%%cu

#include<stdio.h>
#include<cuda.h>

//-------------------------------------------------------------------------------------------
/**
    Purpose:    Optimized Prefix Sum
    Author:     Iman Ghadimi
    Date:       March 2022
*/

//-------------------------------------------------------------------------------------------

// GPU :

__global__ void optimizedPrefixSum(float *out_data, float *in_data, int n) {

    extern __shared__ float temp[];

      // allocated on invocation

      int thid = threadIdx.x;
      int offset = 1;
      temp[2*thid] = in_data[2*thid];

      // load input into shared memory

      temp[2*thid+1] = in_data[2*thid+1];
      for (int d = n>>1; d > 0; d >>= 1)

        // build sum in place up the tree
            {
                __syncthreads();
             if (thid < d){
                 int ai = offset*(2*thid+1)-1;
                 int bi = offset*(2*thid+2)-1;
                 float t = temp[ai];
                 temp[ai] = temp[bi];
                 temp[bi] += t;
                  }
             }
              __syncthreads();

      out_data[2*thid] = temp[2*thid];

      // write results to device memory

      out_data[2*thid+1] = temp[2*thid+1];

      }

// CPU


int main()
{
  int n = 100;
  int i;
	float a[n],c[n];
  float *d;

  for(i=0;i<n;i++){
       a[i]=rand()%100;   //Generate number between 0 to 99
   }

	printf("Before Optimized Prefix Sum:\t");
	for(i=0; i<n; i++)
	{
		printf("\nElement number %d::%d",i+1,a[i]);
	}

	cudaMalloc(&d, n*sizeof(float));
	cudaMemcpy(d,&a,n*sizeof(float),cudaMemcpyHostToDevice);

		// Calling Function:
		optimizedPrefixSum<<<1,1>>>(d,a,n);

	cudaMemcpy(&c,d,n*sizeof(float), cudaMemcpyDeviceToHost);
	printf("\n After Optimized Prefix Sum:\t");
	for(i=0; i<n; i++)
	{
		printf("\nElement number %d::%d",i+1,c[i]);
	}
	cudaFree(d);
	return 0;
}
