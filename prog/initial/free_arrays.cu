//
#include <stdio.h>
#include "../mcrec.h"

int freeAll(singleBox* &gpuSingleBox,singleBox* &initFlows,options config){

cudaFree(gpuSingleBox);

for(int i=0; i<config.flowNum;i++){
    for(int j=0;j<initFlows[i].molNum;j++){
        free(initFlows[i].xa[j]);
        free(initFlows[i].ya[j]);
        free(initFlows[i].za[j]);
    }
    free(initFlows[i].xm);
    free(initFlows[i].ym);
    free(initFlows[i].zm);
    free(initFlows[i].typeMolNum);
    free(initFlows[i].type);
}
free(initFlows);
return 1;
}
