
%%cu

//-------------------------------------------------------------------------------------------
/**
    Purpose:    Odd/Even Transposition Sorting
    Author:     Iman Ghadimi
    Date:       March 2022
*/

//-------------------------------------------------------------------------------------------


#include<stdio.h>
#include<cuda.h>

// GPU

__global__ void oddeven(int* x,int I,int n)
{
	int id=blockIdx.x;
	if(I==0 && ((id*2+1)< n)){
		if(x[id*2]>x[id*2+1]){
			int X=x[id*2];
			x[id*2]=x[id*2+1];
			x[id*2+1]=X;
		}
	}
	if(I==1 && ((id*2+2)< n)){
		if(x[id*2+1]>x[id*2+2]){
			int X=x[id*2+1];
			x[id*2+1]=x[id*2+2];
			x[id*2+2]=X;
		}
	}
}

// CPU

int main()
{
	int a[100],n,c[100],i;
	int *d;

  n = 100 ;
   int randArray[n];
   for(i=0;i<n;i++)
     a[i]=rand()%100;   //Generate number between 0 to 99

		printf("Array:\t");
	for(i=0; i<n; i++)
	{
		printf("\nElement number %d::%d",i+1,a[i]);
	}

	cudaMalloc((void**)&d, n*sizeof(int));
	cudaMemcpy(d,a,n*sizeof(int),cudaMemcpyHostToDevice);

	for(i=0;i<n;i++){

		// Calling Function:
		oddeven<<<n/2,1>>>(d,i%2,n);
	}

	cudaMemcpy(c,d,n*sizeof(int), cudaMemcpyDeviceToHost);
	printf("\nSorted Array:\t");
	for(i=0; i<n; i++)
	{
		printf("\nElement number %d::%d",i+1,c[i]);
	}
	cudaFree(d);
	return 0;
}
