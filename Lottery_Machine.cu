// Simulator a lottery system
//
// Author: Yili Zou
// 
// For the GPU Programming class, NDSU Spring '14


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>

#define FILE_CREATE_ERROR -1
#define Number_Max 9
#define Number_Min 0


#define THREADS_PER_BLOCK 10 // Setting the grid up
#define BLOCKS_PER_GRID 1
#define OFFSET 0 // No offset

__global__ void Setup_RNG(curandState *state, int seed)
{
	// Setup of the random number generator. It seeds it, sets the sequence according to the thread id
	curand_init(seed, threadIdx.x + blockIdx.x * THREADS_PER_BLOCK, OFFSET, &state[threadIdx.x + blockIdx.x * THREADS_PER_BLOCK]);
	
}

__global__ void RNG(curandState *state,  int *result)
{
	int id_k = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK; // Here we calculate the id_k as to save calculations

	curandState localState = state[id_k]; // Copy it to local memory to save global memory accesses (faster)

	result[id_k] = curand(&localState)/(RAND_MAX/5); // Use the state to generate the random number AND updates the state,the range will be from 0 to 9, which is a dice

	state[id_k] = localState; // Update the state in global memory. This allows the next generation to be uncorrelated to this generation

}
__global__ void Number_Matching(int *lucky_numbers, int *user_numbers, int *Matching_numbers)
{
	//set up a counter to see how many numbers are matching
	int counter=0;  //initialize the counter, 0 is not matching, 1 is matching. 

	if(lucky_numbers[threadIdx.x]==user_numbers[threadIdx.x])
	{
		counter++;
	}

	Matching_numbers[threadIdx.x]= counter; //for every index that is matching, counter becomes 1, so this is a array of where these matching numbers are
	
}

int main()
{	
	//the array to store users number
	int user_number[10];
	//the array to store the lucky number
	int price_number[10];
	//define a address to store randomnumbers on the device
	int *randomnumbers_d;
	//how many numbers are matching
	int numbers_matching[10];
	//define stuff in device
	int *price_number_d;
	int *user_number_d;
	int *numbers_matching_d;
	
	//States
    curandState *states_d;
	
	// Allocate memory on the device
    cudaMalloc((void **)&randomnumbers_d, THREADS_PER_BLOCK*sizeof( int)); 
	cudaMalloc((void **)&states_d, THREADS_PER_BLOCK*sizeof(curandState));
    
	// Set up grid and block
	dim3 dimGrid(BLOCKS_PER_GRID);
	dim3 dimBlock(THREADS_PER_BLOCK); 
	
	// Set up RNG
	Setup_RNG<<<dimGrid, dimBlock>>>(states_d, time(NULL)); 

	RNG<<<dimGrid, dimBlock>>>(states_d, randomnumbers_d); // Launch RNG
	
	//copy results back
	cudaMemcpy(price_number, randomnumbers_d, THREADS_PER_BLOCK*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	
	//user interface
	printf("\nThe 10 lucky number have been generated, please input your lucky numbers!!(from 0 to 9)\n");
	//encourage user to input numbers
	int input; //the input by users
	for(int i=0; i<10; i++)
	{
		while(1)
		{
			scanf("%d", &input);  //scan the input
			if(input<0 || input >9)
			{
				printf("\n Please enter numbers within 0 to 9!\n"); //encourage to input valide number
			}
			else
			break;
		}
		user_number[i]=input;
	}
	printf("\nYour lucky numbers have been picked, waiting for results\n");
	// Allocate memory on the device
    cudaMalloc((void **)&numbers_matching_d, 10*sizeof( int)); 
    cudaMalloc((void **)&price_number_d, THREADS_PER_BLOCK*sizeof( int)); 
	cudaMalloc((void **)&user_number_d, THREADS_PER_BLOCK*sizeof( int)); 
	
	//copy the parameters in the device
	cudaMemcpy(user_number_d, user_number, THREADS_PER_BLOCK*sizeof(int ), cudaMemcpyHostToDevice);
	cudaMemcpy(price_number_d, price_number, THREADS_PER_BLOCK*sizeof(int ), cudaMemcpyHostToDevice);
	
	//Launch number matching kernel
	Number_Matching<<<dimGrid, dimBlock>>>(price_number_d, user_number_d, numbers_matching_d);
	//copy the result
	cudaMemcpy(numbers_matching, numbers_matching_d, 10*sizeof(int), cudaMemcpyDeviceToHost);
	//clean up memory
	 cudaFree(numbers_matching_d);
	 cudaFree(price_number_d);
	 cudaFree(user_number_d);
	 
	 //how many numbers matching
	 int nMatch=0;
	 //show result
	 
	 printf("lucky numbers:\n");
	 for(int i=0; i<10; i++)
	 {
		nMatch+=numbers_matching[i];   //numbers_matching[i] is going to be either 1 or 0, so sum them all up we can get the totally numbers matching
		printf("%d\n",price_number[i]);
	 }
	 printf("\n You have %d numbers matching!", nMatch);
	 if(nMatch==10) //when all matches, win the price, which is not likely to happen
	 {
		printf("\n Conflagrations! You have won 1 Million dollars! \n");
	 }
	 return EXIT_SUCCESS;
}
