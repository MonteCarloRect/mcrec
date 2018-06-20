#include <stdio.h>
#include "../mcrec.h"

int write_prop_log(int deviceCount, cudaDeviceProp* deviceProp, FILE* logFile){
	fprintf(logFile,"Device count %d:\n", deviceCount);
	for(int i=0;i<deviceCount;i++){
		fprintf(logFile, " -- %s\n",deviceProp[i].name);
	}
	return 0;
}

int write_config_log(options con,FILE* logFile){
	fprintf(logFile,"Number of substances %d, files:\n",con.subNum);
	for(int i=0;i<con.subNum;i++){
		fprintf(logFile,"%d -- %s\n",i, con.subFile[i]);
	}
	return 0;
}
