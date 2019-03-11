#include <stdio.h>
#include "../mcrec.h"
#include "../global.h"

int data_from_device(gSingleBox &gData, gSingleBox &hData, options config){
    //int sum;
    cudaError_t cuErr;
    
    
    //copy molecules
    cuErr = cudaMemcpy(hData.xm, gData.xm, hData.tMol*sizeof(float), cudaMemcpyDeviceToHost);
    if(cuErr != cudaSuccess){
        printf("Cannot copy from device box.xm file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(hData.ym, gData.ym, hData.tMol*sizeof(float), cudaMemcpyDeviceToHost);
    if(cuErr != cudaSuccess){
        printf("Cannot copy from device box.ym file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(hData.zm, gData.zm, hData.tMol*sizeof(float), cudaMemcpyDeviceToHost);
    if(cuErr != cudaSuccess){
        printf("Cannot copy from device box.zm file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    
    //copy atoms
    cuErr = cudaMemcpy(hData.xa, gData.xa, hData.tAtom*sizeof(float), cudaMemcpyDeviceToHost);
    if(cuErr != cudaSuccess){
        printf("Cannot copy from device box.xa file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(hData.ya, gData.ya, hData.tAtom*sizeof(float), cudaMemcpyDeviceToHost);
    if(cuErr != cudaSuccess){
        printf("Cannot copy from device box.ya file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(hData.za, gData.za, hData.tAtom*sizeof(float), cudaMemcpyDeviceToHost);
    if(cuErr != cudaSuccess){
        printf("Cannot copy from device box.za file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    //copy averages energy
    cuErr = cudaMemcpy(hData.avEnergy, gData.avEnergy, config.flowNum * sizeof(float), cudaMemcpyDeviceToHost);
    if(cuErr != cudaSuccess){
        printf("Cannot copy from device box.eqEnergy file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(hData.avPressure, gData.avPressure, config.flowNum * sizeof(float), cudaMemcpyDeviceToHost);
    if(cuErr != cudaSuccess){
        printf("Cannot copy from device box.eqPressure file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    //
    cuErr = cudaMemcpy(hData.eqBlockEnergy, gData.eqBlockEnergy, config.flowNum * EQBLOCKS *sizeof(float), cudaMemcpyDeviceToHost);
    if(cuErr != cudaSuccess){
        printf("Cannot copy from device box.eqBlockEnergy file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(hData.eqBlockPressure, gData.eqBlockPressure, config.flowNum * EQBLOCKS * sizeof(float), cudaMemcpyDeviceToHost);
    if(cuErr != cudaSuccess){
        printf("Cannot copy from device box.eqBlockPressure file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    //copy total accept/rejected
    cuErr = cudaMemcpy(hData.tAccept, gData.tAccept, config.flowNum * sizeof(int), cudaMemcpyDeviceToHost);
    if(cuErr != cudaSuccess){
        printf("Cannot copy from device box.eqPressure file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(hData.tReject, gData.tReject, config.flowNum * sizeof(int), cudaMemcpyDeviceToHost);
    if(cuErr != cudaSuccess){
        printf("Cannot copy from device box.eqPressure file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    return 0;
}
