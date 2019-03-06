#include <stdio.h>
#include "../mcrec.h"
#include "../global.h"

//calculate cut radius energy and pressure correction

int rcut(gSingleBox &hostData, options config, gMolecula hostTop){
    float sumE;
    float sumP;
    int id1;
    int id2;
    
    sumE=0.0;
    sumP=0.0;
    for(int flowN=0; flowN < config.flowNum; flowN++){  //flows
        for(int i=0; i < config.subNum; i++){   //molecules
            for(int j=0; j < config.subNum; j++){
                for(int a = 0; a < hostTop.aNum[i]; a++){  //atoms
                    for(int b = 0; b < hostTop.aNum[j]; b++){
                        id1 = config.flowNum*flowN + i;
                        id2 = config.flowNum*flowN + j;
                        hostData.typeMolNum
                        sumE += 8.0/9.0 * PI * hostData.typeMolNum[id] * 
                    }
                }
            }
        }
    }
    return 0;
}


