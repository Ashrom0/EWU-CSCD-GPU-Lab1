#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef unsigned long long bignum;
//return 1 if it is a prime else return 0
//save as main.cu 
// CUDA kernel. Each thread takes care of one element of c
__host__ __device__ bignum checkIfValIsPrime(bignum number)
{
    if(number ==1) return (bignum) 0;	
    if (number == 2) return (bignum) 0;
    if (number % 2 == 0) return (bignum) 0;
    for (long divisor = 3; divisor < (number / 2); divisor += 2)
    {
        if (number % divisor == 0)
        {
            return (bignum) 0;
        }
    }
    return (bignum) 1;
}

__global__ void isPrime(double *a, bignum length)
{
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<length){
	  a[id] = checkIfValIsPrime((bignum) id);
	}
}


int main( int argc, char* argv[] )
{
   if(argc < 2)
   {
       printf("Usage: prime upbound\n");
       exit(-1);
   }
   bignum N = (bignum) atoi(argv[1]);
   bignum blockSize  = (bignum) atoi(argv[2]);
   if(N <= 0)
   {
       printf("Usage: prime upbound, you input invalid upbound number!\n");
       exit(-1);
   }
 
    // Host input
    double *h_a;
    // Host output
    double *h_c;
    
    // Device input
    double *d_a;
    // Device output
    double *d_c;
 
    // Size, in bytes, of each vector
    size_t bytes = N*sizeof(double);

    // Allocate memory for vector on host
    h_a = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);
 
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);
    printf("Made it past the allocation of memory\n"); 
    int i;
    // Initialize array with 0's to show that it is empty

    printf("Initialize array with 0's\n"); 
    for( i = 0; i < N; i++ ) {
        h_a[i] = 0;
    }
 
    // Copy host vectors to device
    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
	
    //Number of threads blocks in grid.  
    int gridSize = (int)ceil((float)(N+1)/2/blockSize);

    // Execute the kernel
    isPrime<<<gridSize, blockSize>>>(d_a, N);
 
    // Copy array back to host
    cudaMemcpy( h_a, d_a, bytes, cudaMemcpyDeviceToHost );
 
    // Sum up vector c and print result divided by n, this should equal 1 without error
    double sum = 0;
    printf("In the for block adding up sum\n");

    for(i=0; i<N; i++){
        sum += h_a[i];
	printf("In position %d ", i);
	printf("We have %f\n", h_a[i]);
    }
    printf("Final result: %f\n", sum);
 
    // Release device memory
    cudaFree(d_a);
    cudaFree(d_c);
 
    // Release host memory
    free(h_a);
    free(h_c);
 
    return 0;
}
