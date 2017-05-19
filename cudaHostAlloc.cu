#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>#include<malloc.h>
float cuda_malloc_test(int size, bool up)
{ cudaEvent_t start, stop; 
int *a, *dev_a; 
float elapseTime; 
cudaEventCreate(&start); 
cudaEventCreate(&stop); 
a = (int*)malloc(size*sizeof(*a)); 
cudaMalloc((void**)&dev_a, size*sizeof(*dev_a)); 
cudaEventRecord(start, 0); 
for (size_t i = 0; i < 100; i++) 
{  if (up)   
   cudaMemcpy(dev_a, a, size*sizeof(*dev_a), cudaMemcpyHostToDevice);  
   else  {   cudaMemcpy(a, dev_a, size*sizeof(*dev_a), cudaMemcpyDeviceToHost); 
   }
 }
 cudaEventRecord(stop, 0); 
 cudaEventSynchronize(stop); 
 cudaEventElapsedTime(&elapseTime, start, stop); 
 free(a); 
 cudaFree(dev_a); 
 cudaEventDestroy(start); 
 cudaEventDestroy(stop);
 return elapseTime;
}


float cuda_host_alloc_test(int size, bool up)
{ cudaEvent_t start, stop; 
int *a, *dev_a; 
float elapseTime; 
cudaEventCreate(&start); 
cudaEventCreate(&stop); 
cudaHostAlloc((void**)&a,size*sizeof(*a),cudaHostAllocDefault);  
cudaMalloc((void**)&dev_a, size*sizeof(*dev_a)); 
cudaEventRecord(start, 0); 
for (size_t i = 0; i < 100; i++) 
{  if (up)   
   cudaMemcpy(dev_a, a, size*sizeof(*dev_a), cudaMemcpyHostToDevice);  
	 else  {   
	 cudaMemcpy(a, dev_a, size*sizeof(*dev_a), cudaMemcpyDeviceToHost);
  }
 } 
 cudaEventRecord(stop, 0); 
 cudaEventSynchronize(stop); 
 cudaEventElapsedTime(&elapseTime, start, stop);
 cudaFreeHost(a); 
 cudaFree(dev_a); 
 cudaEventDestroy(start); 
 cudaEventDestroy(stop);
 return elapseTime;
 }
#define SIZE (10*1024*1024)

int main()
{ float elapsedTime; 
  float MB = (float)100 * SIZE*sizeof(int) / 1024 / 1024; 
	//elapsedTime = cuda_host_alloc_test(SIZE, false); 
	elapsedTime = cuda_malloc_test(SIZE, false); 
	printf("Time using cudamalloc:%3.1fms\n", elapsedTime); 
	printf("\tMB/s during copy up: %3.1f\n", MB / (elapsedTime / 1000));
}




