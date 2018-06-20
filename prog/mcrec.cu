#include <stdio.h>
#include <stdlib.h>
#include "global.h"
#include "mcrec.h"
#include "initial.h"


int main (int argc, char * argv[]){

//openlog file
    logFile=fopen("calculation.log","w");
    //
    get_device_prop(deviceCount, deviceProp);
    write_prop_log(deviceCount,deviceProp,logFile);
    //read initial data
    read_options(config);
    write_config_log(config,logFile);
    //read gro data for each molecules
    initMol = (molecules *) malloc(config.subNum * sizeof(molecules));
    read_init_gro(config, initMol);
//close log file
    fclose(logFile);


}
