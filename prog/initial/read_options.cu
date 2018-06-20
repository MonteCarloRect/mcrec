//
#include <stdio.h>
#include "../mcrec.h"

int read_options(options &config){
    //varaibles
    FILE* fileId;
    
    fileId = fopen("data.mcr", "r");
    fscanf(fileId, "%d", &config.subNum);
    printf("Substance number: %d\n", config.subNum);
    for (int i = 0; i < config.subNum; i++) {
        fscanf(fileId, "%s", config.subFile[i]);
    }
//read flow data
    fscanf(fileId, "%d", &config.flowNum);
    config.flowT = (float *) malloc(config.flowNum * sizeof(float));
    config.flowN = (float *) malloc(config.flowNum * sizeof(float));
    config.flowP = (float *) malloc(config.flowNum * sizeof(float));
    config.flowIns = (int *) malloc(config.flowNum * sizeof(int));
    for (int i = 0; i < config.flowNum; i++) {
        fscanf(fileId, "%f", &config.flowT[i]);
    }
    for (int i = 0; i < config.flowNum; i++) {
        fscanf(fileId, "%f", &config.flowN[i]);
    }
    for (int i = 0; i < config.flowNum; i++) {
        fscanf(fileId, "%d", &config.flowIns[i]);
    }
    config.flowX = (float **) malloc(config.flowNum * sizeof(float *));
    for (int i = 0; i < config.flowNum; i++) {
        config.flowX[i] = (float *) malloc(config.subNum * sizeof(float));
    }
    for (int i = 0; i < config.flowNum; i++) {
        for (int j = 0; j < config.subNum; j++) {
            fscanf(fileId, "%f", &config.flowX[i][j]);
        }
    }
    fclose(fileId);
    return 1;
}
