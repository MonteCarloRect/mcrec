//
#include <stdio.h>
#include "../mcrec.h"

int write_singlebox_log(FILE* logFile, gSingleBox &hData){
    //print on sreeen
    printf("%s Singlebox results %s\n", ANSI_COLOR_BLUE, ANSI_COLOR_RESET);
    printf("energy %f pressure %f \n", hData.eqBlockEnergy[0], hData.eqBlockPressure[0]);
    //write log
    
    return 0;
}

