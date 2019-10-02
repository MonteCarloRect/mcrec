#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <curand.h>
#include "../mcrec.h"


//transfer data from host to devices
int double_box_host_to_device(options &config, hDoubleBox &doubleBox, gDoublebox gDBox, gDoublebox hDBox, gSingleBox &hostData, molecules* initMol, int deviceCount){
    cudaError_t cuErr;
    int sum;
    
    for(int curDev = 0; curDev < deviceCount; curDev++){
        cuErr = cudaSetDevice(curDev);  //set to current device
        if(cuErr != cudaSuccess){
            printf("Cannot swtich to device %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        //allocate total number of plates
        hDBox.pltNum = (int*) malloc(deviceCount * sizeof(int));
        for(int i = 0; i < deviceCount; i++){
            hDBox.pltNum[i] = doubleBox.devicePlates[i];
        }
        cuErr = cudaMalloc(&gDBox.pltNum, deviceCount*sizeof(int));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate gDBox.molNum %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMemcpy(gDBox.pltNum, hDBox.pltNum, deviceCount*sizeof(int), cudaMemcpyHostToDevice);
        if(cuErr != cudaSuccess){
            printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        //allocate global number of plates for device
        
        
        //allocate total molecules per plate
        hDBox.molNum = (int*) malloc(doubleBox.devicePlates[curDev] * sizeof(int));
        for(int i = 0; i < doubleBox.devicePlates[curDev]; i++){
            hDBox.molNum[i] = doubleBox.molNum[doubleBox.platesPerDevice[curDev][i]];
        }
        cuErr = cudaMalloc(&gDBox.molNum, doubleBox.devicePlates[curDev] * sizeof(int));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate gDBox.molNum %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMemcpy(gDBox.molNum, hDBox.molNum, deviceCount*sizeof(int), cudaMemcpyHostToDevice);
        if(cuErr != cudaSuccess){
            printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        //allocate molecules coordinates
        sum = 0;
        for(int i = 0; i < doubleBox.devicePlates[curDev]; i++){
            sum += doubleBox.molNum[doubleBox.platesPerDevice[curDev][i]];
        }
        hDBox.xm = (float*) malloc(sum * sizeof(float));
        hDBox.ym = (float*) malloc(sum * sizeof(float));
        hDBox.zm = (float*) malloc(sum * sizeof(float));
        hDBox.mType = (int*) malloc(sum * sizeof(int));
        hDBox.nVap = (int*) malloc(sum * sizeof(int));
        hDBox.nLiq = (int*) malloc(sum * sizeof(int));
        
        
        cuErr = cudaMalloc(&gDBox.xm, sum * sizeof(float));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate gDBox.xm %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMalloc(&gDBox.ym, sum * sizeof(float));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate gDBox.ym %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMalloc(&gDBox.zm, sum * sizeof(float));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate gDBox.zm %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        
        
        
        
        //free host arrays
        free(hDBox.pltNum);
        free(hDBox.molNum);
        free(hDBox.xm);
        free(hDBox.ym);
        free(hDBox.zm);
        printf("\n free %d done\n", curDev);
    }
    
    return 1;
}


//first time allocation plates to devices
int double_box_init_allocate(options &config, hDoubleBox &doubleBox, int deviceCount){
    int id;
    //allocate plates evenly
    printf("distribution of plates\n");
    for(int i = 0; i < config.plateNum; i++){
        doubleBox.plateDevice[i] = (int) i * deviceCount / config.plateNum;
        printf(" plate %d device %d\n", i, doubleBox.plateDevice[i]);
    }
    for(int i = 0; i < deviceCount; i++){
        doubleBox.devicePlates[i] = 0;
        for(int j = 0; j < config.plateNum; j++){
            doubleBox.platesPerDevice[i][j] = 0;
        }
    }
    for(int i = 0; i < config.plateNum; i++){
        doubleBox.devicePlates[doubleBox.plateDevice[i]]++;
    }
    for(int i = 0; i < deviceCount; i++){
        id = 0;
        for(int j = 0; j < config.plateNum; j++){
            if(doubleBox.plateDevice[j] == i){
                doubleBox.platesPerDevice[i][id] = j;
                id++;
            }
        }
    }
    
    for(int i = 0; i < deviceCount; i++){
        printf(" device %d calculate %d plates \n ", i, doubleBox.devicePlates[i]);
        for(int j = 0; j < doubleBox.devicePlates[i]; j++){
            printf(" pl  %d   \n", doubleBox.platesPerDevice[i][j]);
        }
    }
    
    return 1;
}


