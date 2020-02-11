#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../mcrec.h"
#include "../global.h"
//#include "../initial.h"

int plates_initial_state(options &config, hDoubleBox &doubleBox, gSingleBox &hostData, molecules* initMol, int deviceCount){
int insert; //number of inserted molecules
int latticeRow; //rows on cubic lattice

//temporally lattice for molecules insert
float* tempXm;
float* tempYm;
float* tempZm;
int* tempFree;  //1 if place is free
float latticeDelta; //lattice delta
float volFrac;  //fraction of liqud volume

int id; //tmporally id
int maxArrSize; //maximum size of temp array
int randMol;    //random molecules


if(config.plateInit == VAK){    //vakuum at all plates
    printf("%s \n Initial plate state: vakuum %s\n",ANSI_COLOR_GREEN, ANSI_COLOR_RESET);
    //allocate arrays
    //.flowEns = (int *) malloc(config.flowNum * sizeof(int));
    
    doubleBox.molNum = (int*) malloc(config.plateNum * sizeof(int));    //numbers of molecules per plate
    //set initial coordinates of molecules and atoms
    doubleBox.xm = (float**) malloc(config.plateNum * sizeof(float*));
    doubleBox.ym = (float**) malloc(config.plateNum * sizeof(float*));
    doubleBox.zm = (float**) malloc(config.plateNum * sizeof(float*));
    doubleBox.mType = (int**) malloc(config.plateNum * sizeof(int*));
    doubleBox.gpuIndex = (int**) malloc(config.plateNum * sizeof(int*));
    
    doubleBox.nVap = (int*) malloc(config.plateNum * sizeof(int));
    doubleBox.nLiq = (int*) malloc(config.plateNum * sizeof(int));
    doubleBox.liqList = (int**) malloc(config.plateNum * sizeof(int*));
    doubleBox.vapList = (int**) malloc(config.plateNum * sizeof(int*));
    
    doubleBox.xa = (float***) malloc(config.plateNum * sizeof(float**));
    doubleBox.ya = (float***) malloc(config.plateNum * sizeof(float**));
    doubleBox.za = (float***) malloc(config.plateNum * sizeof(float**));
    
    doubleBox.refEnergy = (float*) malloc(config.plateNum * sizeof(float)); //referance energy of plate
    for(int i = 0; i < config.plateNum; i++){
        doubleBox.nVap[i] = 0; //initial molecules in vapor
        doubleBox.nLiq[i] = 0;  //initial molecules in liquid 
        doubleBox.molNum[i] = 0;    //if vak initial number is 0
        doubleBox.refEnergy[i] = 0.0;   //set energy of phases to zero
        //doubleBox.liqEnergy[i] = 0.0;
    }
    printf("\n test 0 \n");
    for(int i = 0; i < config.flowNum; i++){
        for(int j = 0; j < config.subNum; j++){
            //set molecules in inputs of molecules
            doubleBox.molNum[config.plateIn[i]] = doubleBox.molNum[config.plateIn[i]] + config.flowIns[i][j];
            //set reference energy of plate
            doubleBox.refEnergy[config.plateIn[i]] = doubleBox.refEnergy[config.plateIn[i]] + config.flowIns[i][j] * hostData.avEnergy[i];
        }
        doubleBox.nLiq[i] += doubleBox.molNum[i];   //all molecules goes to liquid phase
    }
    
    //set numbers of molecules
    doubleBox.molNumType = (int**) malloc(config.plateNum * sizeof(int*));
    for(int i = 0; i < config.plateNum; i++){
        doubleBox.molNumType[i] = (int*) malloc(config.subNum * sizeof(int));
        for(int j = 0; j < config.subNum; j++){
            doubleBox.molNumType[i][j] = 0;
        }
    }
    
    //set atoms
    for(int i = 0; i < config.flowNum; i++){
        for(int j = 0; j < config.subNum; j++){
            printf(" i %d j %d inplate %d ", i, j, config.plateIn[i]);
            doubleBox.molNumType[config.plateIn[i]][j] = config.flowIns[i][j];
            printf(" molecules %d \n", doubleBox.molNumType[config.plateIn[i]][j]);
        }
    }
    
    for(int i = 0; i < config.plateNum; i++){
        doubleBox.xm[i] = (float*) malloc(doubleBox.molNum[i] * sizeof(float) + 1);
        doubleBox.ym[i] = (float*) malloc(doubleBox.molNum[i] * sizeof(float) + 1);
        doubleBox.zm[i] = (float*) malloc(doubleBox.molNum[i] * sizeof(float) + 1);
        doubleBox.mType[i] = (int*) malloc(doubleBox.molNum[i] * sizeof(int) +1);
        doubleBox.gpuIndex[i] = (int*) malloc(doubleBox.molNum[i] * sizeof(int) +1);
        doubleBox.vapList[i] = (int*) malloc(doubleBox.molNum[i] * sizeof(int) +1);
        doubleBox.liqList[i] = (int*) malloc(doubleBox.molNum[i] * sizeof(int) +1);
        
        doubleBox.xa[i] = (float**) malloc(doubleBox.molNum[i] * sizeof(float*) + 1);
        doubleBox.ya[i] = (float**) malloc(doubleBox.molNum[i] * sizeof(float*) + 1);
        doubleBox.za[i] = (float**) malloc(doubleBox.molNum[i] * sizeof(float*) + 1);
    }
    //allocate volumes of boxes
    doubleBox.liqVol = (float*) malloc(config.plateNum * sizeof(float));
    doubleBox.vapVol = (float*) malloc(config.plateNum * sizeof(float));
    //allocate type of atoms
    printf("\n test 2 \n");
    doubleBox.type = (int**) malloc(config.plateNum * sizeof(int*));
    for(int i = 0; i < config.plateNum; i++){
        doubleBox.type[i] = (int*) malloc(doubleBox.molNum[i] * sizeof(int));
//        id = 0;
//        for(int j = 0; j < config.subNum; j++){
//            for(int k = 0; k < doubleBox.molNumType[i][j]; k++){
//                doubleBox.type[i][id] = j; //initMol[j].atomNum
//                id++;
//            }
//        }
    }
    
    volFrac = 0.1;  //fraction of liquid phase
    //molecules inserted
    for(int i = 0; i < config.plateNum; i++){   //for each plate
        //set initial lattice for molecules insert
        printf(" insert molecules to plate %d \n", i);
        latticeRow = pow(doubleBox.molNum[i], 1.0/3.0) + 2;
        tempXm = (float*) malloc(latticeRow * latticeRow * latticeRow * sizeof(float));
        tempYm = (float*) malloc(latticeRow * latticeRow * latticeRow * sizeof(float));
        tempZm = (float*) malloc(latticeRow * latticeRow * latticeRow * sizeof(float));
        tempFree = (int*) malloc(latticeRow * latticeRow * latticeRow * sizeof(int));
        //initial volume of gas and liqud phase
        doubleBox.liqVol[i] = volFrac * config.plateVol;
        doubleBox.vapVol[i] = config.plateVol - doubleBox.liqVol[i];
        
        id = 0;
        for(int ii = 0; ii < latticeRow; ii++){
            for(int jj = 0; jj < latticeRow; jj++){
                for(int kk = 0; kk < latticeRow; kk++){
                    tempFree[id] = 1;
                    tempXm[id] = (ii + 0.5) * latticeDelta;
                    tempYm[id] = (jj + 0.5) * latticeDelta;
                    tempZm[id] = (kk + 0.5) * latticeDelta;
                    id++;
                }
            }
        }
        srand(time(0));
        maxArrSize = latticeRow * latticeRow * latticeRow;
        id = 0;
        for(int j = 0; j < config.subNum; j++){ //for each substance type
            printf("plate %d substance %d molecules to insert %d \n", i, j, doubleBox.molNumType[i][j]);
            insert = 0;
            while(insert < doubleBox.molNumType[i][j]){   //insert molecule at random plase
                randMol = rand() % maxArrSize;
                if(tempFree[randMol] == 1){
                    //set coordinates of molecules
                    doubleBox.xm[i][id] = tempXm[randMol];  
                    doubleBox.ym[i][id] = tempYm[randMol];
                    doubleBox.zm[i][id] = tempZm[randMol];
                    doubleBox.mType[i][id] = j; //set type of molecules
                    doubleBox.liqList[i][id] = id;  //set current molecule to liquid phase
                    //doubleBox.nLiq[i]++;    //add one more molecule to liquid phase
                    //doubleBox.nVap[i]++;    //summ molecules DO THIS UPPER
                    
                    doubleBox.xa[i][id] = (float*) malloc(initMol[j].atomNum * sizeof(float));
                    doubleBox.ya[i][id] = (float*) malloc(initMol[j].atomNum * sizeof(float));
                    doubleBox.za[i][id] = (float*) malloc(initMol[j].atomNum * sizeof(float));
                    //set coordinates of atoms
                    for(int k = 0; k < initMol[j].atomNum; k++){
                        doubleBox.xa[i][id][k] = initMol[j].x[k];
                        doubleBox.ya[i][id][k] = initMol[j].y[k];
                        doubleBox.za[i][id][k] = initMol[j].z[k];
                    }
                    id++;
                    insert++;
                    tempFree[randMol] = 0;  //plase not free
                }
                
            }
        }
        free(tempXm);
        free(tempYm);
        free(tempZm);
        free(tempFree);
    }
    //allocate devices plate
    doubleBox.plateDevice = (int*) malloc(config.plateNum * sizeof(int));
    doubleBox.devicePlates = (int*) malloc(deviceCount * sizeof(int));
    doubleBox.platesPerDevice = (int **) malloc(deviceCount * sizeof(int*));
    for(int i = 0; i < deviceCount; i++){
        doubleBox.platesPerDevice[i] = (int*) malloc(config.plateNum * sizeof(int));
    }
    
}
printf("%s initial state of plate done %s\n",ANSI_COLOR_GREEN, ANSI_COLOR_RESET);
return 0;
}
