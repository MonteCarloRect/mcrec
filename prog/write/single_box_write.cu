//
#include <stdio.h>
#include "../mcrec.h"

int write_singlebox_log(FILE* logFile, gSingleBox &hData){
    float sumE;
    float sumP;
    //print on sreeen
    printf("%s Singlebox results %s\n", ANSI_COLOR_BLUE, ANSI_COLOR_RESET);
    printf("energy %f pressure %f \n", hData.avEnergy[0], hData.avPressure[0]);
    printf("energy correction %f pressure correction %f \n", hData.energyCorr[0], hData.pressureCorr[0]);
    sumE = 0.0f;
    sumP = 0.0f;
    for(int i=0; i<EQBLOCKS; i++){
        sumE += (hData.eqBlockEnergy[0 * EQBLOCKS +i] - hData.avEnergy[0]) * (hData.eqBlockEnergy[0 * EQBLOCKS +i] - hData.avEnergy[0]);
        sumP += (hData.eqBlockPressure[0 * EQBLOCKS +i] - hData.avPressure[0]) * (hData.eqBlockPressure[0 * EQBLOCKS +i] - hData.avPressure[0]);
    }
    sumE = sqrt(1.0 / (EQBLOCKS - 1.0) * sumE);
    sumP = sqrt(1.0 / (EQBLOCKS - 1.0) * sumP);
    printf("total energy correction %f +/- %f total pressure %f +/- %f \n", hData.avEnergy[0] + hData.energyCorr[0], sumE, hData.avPressure[0] + hData.pressureCorr[0], sumP);
    //write log
    
    return 0;
}

